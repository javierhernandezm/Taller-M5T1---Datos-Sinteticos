"""
data_loading.py — Carga del universo PIT y del panel de precios del TFM.

Responsabilidad única de este módulo: pasar de los ficheros crudos del TFM
(`universo_pit_rics_2400.csv` + `sp1500_daily_prices.parquet`) a un panel
largo de precios LIMPIO y point-in-time, restringido al universo
"perfectamente mapeado". Todo lo que sea transformación estadística
(retornos, ventanas, target) vive en `preprocessing.py`.

Decisiones de diseño documentadas aquí porque son decisiones de DATOS,
no de modelado:

1.  Universo = tramos con mapping_status == READY y status == OK.
    Según la auditoría del maestro, READY exige: CIK válido, RIC resuelto
    de forma unívoca y ningún control de mapeo activado; OK añade que la
    cobertura de precios del tramo es completa. Es la definición operativa
    de "activo perfectamente mapeado".

2.  Filtro de serie completa (cfg.serie_completa = True, el modo del taller).
    Se conservan SOLO los activos con histórico de precios completo de la
    primera a la última fecha del panel, estén o no en el índice en cada
    momento: nada de recorte por pertenencia, y fuera deslistados y listados
    tardíos. "Completa" = empieza a más tardar tolerancia_bordes_dias tras el
    inicio del panel, termina como muy pronto esa tolerancia antes del final,
    y sin huecos internos > max_gap_days días naturales.
    ADVERTENCIA deliberada: este filtro introduce sesgo de supervivencia
    (solo empresas que sobrevivieron todo el periodo). Se acepta a cambio de
    un panel rectangular y homogéneo, y queda documentado como limitación en
    el notebook 01. Con serie_completa = False se recupera el comportamiento
    anterior (recorte PIT por tramo, deslistados incluidos).

3.  El identificador de trabajo es el CIK (texto de 10 dígitos). En el
    universo filtrado la relación CIK↔activo↔RIC es 1:1:1 (verificado en
    la función `load_universe`), por lo que no hay ambigüedad.
"""

from __future__ import annotations

import pandas as pd

from .config import Config


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------

def load_universe(cfg: Config) -> pd.DataFrame:
    """Carga el universo PIT y lo filtra a los tramos perfectamente mapeados.

    Devuelve un DataFrame con una fila por TRAMO de pertenencia (un activo
    puede tener varios tramos si entró y salió del índice), con columnas:
    ``project_asset_id, cik, ric, analysis_start, analysis_end, spell_id`` y
    metadatos de diagnóstico (delisted, cobertura...).

    Además verifica invariantes que, de romperse, invalidarían el resto del
    pipeline (fail fast):
      * CIK únicos por activo dentro del filtro,
      * cada CIK mapea a un único RIC,
      * cobertura de precios ≈ 100 % en todos los tramos.
    """
    u = pd.read_csv(
        cfg.universe_path,
        dtype={"cik": str},
        parse_dates=["analysis_start", "analysis_end", "membership_start", "membership_end"],
    )
    n_total = len(u)

    mask = (u["mapping_status"] == cfg.universe_mapping_status) & (
        u["status"] == cfg.universe_status
    )
    uf = u.loc[mask].copy()
    uf["cik"] = uf["cik"].str.zfill(10)

    # Identificador de tramo: necesario porque un mismo activo puede tener
    # varios periodos de pertenencia y una ventana jamás debe cruzar de uno
    # a otro (habría un hueco temporal en medio).
    uf = uf.sort_values(["cik", "analysis_start"]).reset_index(drop=True)
    uf["spell_id"] = uf.groupby("cik").cumcount()

    # --- Invariantes -------------------------------------------------------
    assert uf["coverage_pct"].min() > 99.0, "Hay tramos READY/OK con cobertura baja"
    ric_per_cik = uf.groupby("cik")["ric"].nunique()
    multi_ric = int((ric_per_cik > 1).sum())
    # Los (poquísimos) activos con 2 tramos usan RIC distinto por tramo
    # (p. ej. cambio de listing); el spell_id ya los separa, no es ambigüedad.
    asset_per_cik = uf.groupby("cik")["project_asset_id"].nunique()
    assert (asset_per_cik == 1).all(), "Un CIK mapea a más de un activo: revisar maestro"

    print(
        f"[universe] {n_total} tramos en el universo PIT -> "
        f"{len(uf)} tramos perfectamente mapeados "
        f"({uf['cik'].nunique()} activos únicos, {multi_ric} con RIC múltiple entre tramos)"
    )
    return uf


# ---------------------------------------------------------------------------
# Precios
# ---------------------------------------------------------------------------

#: Columnas del parquet que realmente necesitamos. Leer solo estas reduce
#: memoria y tiempo de carga (el parquet es columnar).
_PRICE_COLS = ["cik", "date", "close", "volume", "resolved_ric", "sector", "match_confidence"]


def load_prices(cfg: Config, universe: pd.DataFrame) -> pd.DataFrame:
    """Carga precios diarios y aplica el recorte PIT por tramo de pertenencia.

    Pasos:
      1. Lee el parquet (solo columnas necesarias) y normaliza el CIK.
      2. Restringe a los CIK del universo filtrado.
      3. Une cada fila de precios con su tramo de pertenencia y descarta las
         fechas fuera de [analysis_start, analysis_end]  (recorte PIT).
      4. Elimina filas con cierre nulo (huecos de la fuente, ~1e-4 del panel);
         el control de huecos fino se hace en preprocessing sobre el
         calendario de cada activo.

    Devuelve el panel largo ordenado por (cik, date) con la columna spell_id.
    """
    px = pd.read_parquet(cfg.prices_path, columns=_PRICE_COLS)
    px["cik"] = px["cik"].str.zfill(10)
    n0 = len(px)

    px = px[px["cik"].isin(set(universe["cik"]))]
    n1 = len(px)

    if cfg.serie_completa:
        # --- Filtro de serie completa (sin recorte por pertenencia) --------
        # Nos quedamos con los activos que cotizan de punta a punta del panel
        # y sin huecos internos. Un único "tramo" por activo (spell_id = 0).
        tol = pd.Timedelta(days=cfg.tolerancia_bordes_dias)
        cal_ini, cal_fin = px["date"].min(), px["date"].max()

        px = px.sort_values(["cik", "date"])
        g = px.groupby("cik")["date"]
        rango = g.agg(ini="min", fin="max")
        # hueco interno máximo (días naturales) entre sesiones consecutivas
        rango["hueco_max"] = g.diff().dt.days.groupby(px["cik"]).max()

        empieza_a_tiempo = rango["ini"] <= cal_ini + tol
        llega_al_final = rango["fin"] >= cal_fin - tol
        sin_huecos = rango["hueco_max"] <= cfg.max_gap_days
        completos = rango.index[empieza_a_tiempo & llega_al_final & sin_huecos]

        print(
            f"[serie completa] panel {cal_ini.date()} -> {cal_fin.date()} | "
            f"{len(rango)} activos candidatos: "
            f"{int((~empieza_a_tiempo).sum())} listados tarde, "
            f"{int((~llega_al_final).sum())} deslistados/terminan antes, "
            f"{int((empieza_a_tiempo & llega_al_final & ~sin_huecos).sum())} con huecos "
            f"> {cfg.max_gap_days} dias -> {len(completos)} con serie completa"
        )
        px = px[px["cik"].isin(set(completos))].copy()
        px["spell_id"] = 0
        n2 = len(px)
    else:
        # merge por cik y filtrado por rango de fechas del tramo (modo PIT) --
        spells = universe[["cik", "spell_id", "analysis_start", "analysis_end"]]
        px = px.merge(spells, on="cik", how="left")
        in_spell = (px["date"] >= px["analysis_start"]) & (px["date"] <= px["analysis_end"])
        px = px.loc[in_spell].drop(columns=["analysis_start", "analysis_end"])
        n2 = len(px)

    n_null = int(px["close"].isna().sum())
    px = px.dropna(subset=["close"])

    # Unicidad (cik, date): documentada aguas arriba, pero verificar es gratis
    # comparado con depurar un duplicado silencioso más adelante.
    assert not px.duplicated(["cik", "date"]).any(), "Duplicados (cik, date) en precios"

    px = px.sort_values(["cik", "date"]).reset_index(drop=True)
    filtro = "serie completa" if cfg.serie_completa else "recorte PIT por tramo"
    print(
        f"[prices] {n0:,} filas totales -> {n1:,} en universo filtrado -> "
        f"{n2:,} tras {filtro} -> {len(px):,} tras eliminar "
        f"{n_null} cierres nulos"
    )
    return px

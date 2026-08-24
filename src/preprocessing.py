"""
preprocessing.py — Del panel de precios limpio al dataset supervisado.

Cadena de transformaciones (cada función es pura: DataFrame/arrays dentro,
DataFrame/arrays fuera, sin estado global):

    panel de precios (cik, date, close, volume)
        └─ compute_returns()      retornos log por activo + control de huecos
        └─ build_windows()        ventanas X de 60 retornos + target y = log σ_fwd
        └─ temporal_split()       train/val/test por fecha, con purga y embargo
        └─ Standardizer           normalización ajustada SOLO con train

Definiciones clave
------------------
Retorno log:        r_t = ln(P_t / P_{t-1}),  solo entre sesiones consecutivas
                    del calendario del propio activo separadas ≤ max_gap_days
                    días naturales. Un hueco mayor (suspensión, salida temporal
                    del histórico) invalida ese retorno y, con él, toda ventana
                    que lo contenga: preferimos perder muestras a fabricar un
                    retorno multi-día disfrazado de diario.

Target (volatilidad realizada forward):
                    σ_fwd(t) = sqrt( 252/H · Σ_{k=1..H} r_{t+k}² ),  H = 21
                    y(t)     = ln σ_fwd(t)
                    Se modela en logaritmo porque σ es estrictamente positiva
                    y aproximadamente lognormal: en logs el target es casi
                    gaussiano y el MSE no queda dominado por las colas.
                    No se demean: es la volatilidad realizada estándar (la
                    media diaria es despreciable frente a σ a 21 días).

Anti-fuga de información
------------------------
* Split TEMPORAL por fechas, nunca aleatorio: dos ventanas que comparten 59
  de 60 días jamás pueden caer una en train y otra en test.
* Purga estructural: una ventana se asigna a una partición solo si TODO su
  soporte temporal [inicio de X, fin del target] cae dentro de la partición.
  Las ventanas que cruzan una frontera se descartan.
* Embargo adicional de `embargo_days` días naturales tras cada frontera,
  para cortar la dependencia serial residual (la volatilidad es persistente).
* El Standardizer se ajusta exclusivamente con train y se aplica congelado
  a validación y test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


# ---------------------------------------------------------------------------
# 1. Retornos
# ---------------------------------------------------------------------------

def compute_returns(px: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Añade la columna ``ret`` (retorno log diario) al panel de precios.

    Reglas:
      * el retorno se calcula dentro de cada (cik, spell_id): nunca entre
        tramos de pertenencia distintos;
      * si entre dos sesiones consecutivas hay más de ``cfg.max_gap_days``
        días naturales, el retorno se marca NaN (hueco);
      * sesiones con cierre < ``cfg.min_price`` invalidan el retorno de ese
        día y del siguiente (ruido de microestructura en penny stocks).

    Devuelve el panel con columnas nuevas: ``ret`` y ``gap_days``.
    """
    px = px.sort_values(["cik", "spell_id", "date"]).copy()
    g = px.groupby(["cik", "spell_id"], sort=False)

    px["ret"] = np.log(px["close"]).groupby([px["cik"], px["spell_id"]]).diff()
    px["gap_days"] = g["date"].diff().dt.days

    # hueco temporal: retorno inválido
    px.loc[px["gap_days"] > cfg.max_gap_days, "ret"] = np.nan

    # precio bajo el umbral: el retorno que ENTRA y el que SALE del día quedan
    # invalidados (ambos usan ese cierre contaminado)
    low = px["close"] < cfg.min_price
    low_next = low.groupby([px["cik"], px["spell_id"]]).shift(1, fill_value=False)
    px.loc[low | low_next, "ret"] = np.nan

    n_nan = int(px["ret"].isna().sum())
    print(
        f"[returns] {len(px):,} filas; retornos inválidos (primer día de tramo, "
        f"huecos > {cfg.max_gap_days}d o precio < {cfg.min_price}$): {n_nan:,} "
        f"({n_nan / len(px):.2%})"
    )
    return px


# ---------------------------------------------------------------------------
# 2. Ventanas y target
# ---------------------------------------------------------------------------

def build_windows(px: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construye el dataset supervisado de ventanas por activo.

    Para cada activo (cik, spell) y cada posición t tomada con paso
    ``cfg.stride`` sobre SU propio calendario de sesiones:

        X = [r_{t-59}, ..., r_t]                       (cfg.window_len valores)
        y = ln sqrt( 252/H · Σ_{k=1..H} r_{t+k}² )     (cfg.horizon días futuros)

    Una ventana solo es válida si sus 60+21 retornos existen (sin NaN): la
    contigüidad ya viene garantizada por el control de huecos de
    `compute_returns`.

    Devuelve:
      X    float32 (N, window_len)
      y    float32 (N,)
      meta DataFrame (N filas): cik, sector, fecha inicio de X, fecha t
           (fin de X = "hoy"), fecha fin del target. El meta es la pieza que
           permite el split temporal purgado y cualquier análisis posterior
           (por sector, por régimen, por activo).

    Implementación vectorizada con sliding_window_view por activo: evita el
    doble bucle Python sobre (activo, t), que sería ~100x más lento.
    """
    W, H, S = cfg.window_len, cfg.horizon, cfg.stride
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []

    for (cik, spell), grp in px.groupby(["cik", "spell_id"], sort=False):
        r = grp["ret"].to_numpy(np.float64)
        dates = grp["date"].to_numpy()
        n = len(r)
        if n < W + H + 1:
            continue  # tramo demasiado corto para una sola ventana

        # todas las sub-secuencias de longitud W+H que terminan en cada t+H
        sw = np.lib.stride_tricks.sliding_window_view(r, W + H)  # (n-W-H+1, W+H)
        valid = ~np.isnan(sw).any(axis=1)

        # posiciones muestreadas con stride (ancladas al final del tramo para
        # aprovechar siempre los datos más recientes)
        idx = np.arange(sw.shape[0] - 1, -1, -S)[::-1]
        idx = idx[valid[idx]]
        if idx.size == 0:
            continue

        block = sw[idx]                       # (m, W+H)
        Xw = block[:, :W]                     # retornos de la ventana de entrada
        fwd = block[:, W:]                    # retornos del horizonte del target
        sigma = np.sqrt(cfg.ann_factor * np.mean(fwd**2, axis=1))
        y = np.log(sigma)

        # posiciones absolutas en el calendario del activo:
        # la ventana i empieza en idx[i], "hoy" es idx[i]+W-1, el target
        # termina en idx[i]+W+H-1
        metas.append(
            pd.DataFrame(
                {
                    "cik": cik,
                    "spell_id": spell,
                    "sector": grp["sector"].iat[0],
                    "date_x_start": dates[idx],
                    "date_t": dates[idx + W - 1],
                    "date_y_end": dates[idx + W + H - 1],
                }
            )
        )
        Xs.append(Xw.astype(np.float32))
        ys.append(y.astype(np.float32))

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = pd.concat(metas, ignore_index=True)
    assert len(X) == len(y) == len(meta)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    print(
        f"[windows] {len(X):,} ventanas ({meta['cik'].nunique()} activos) | "
        f"X: {X.shape} | y: media={y.mean():.3f}, sd={y.std():.3f} "
        f"(σ mediana anualizada = {np.exp(np.median(y)):.1%})"
    )
    return X, y, meta


# ---------------------------------------------------------------------------
# 3. Split temporal con purga y embargo
# ---------------------------------------------------------------------------

def temporal_split(meta: pd.DataFrame, cfg: Config) -> pd.Series:
    """Asigna cada ventana a train / val / test / (descartada).

    Regla de asignación — sobre el SOPORTE COMPLETO de la ventana:
      train:  date_y_end   <= train_end
      val:    date_x_start >= train_end + embargo   y   date_y_end <= val_end
      test:   date_x_start >= val_end  + embargo
      resto:  'purged' (cruza una frontera o cae dentro del embargo)

    Devuelve una Series categórica alineada con `meta`.
    """
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    emb = pd.Timedelta(days=cfg.embargo_days)

    split = pd.Series("purged", index=meta.index, dtype=object)
    split[meta["date_y_end"] <= train_end] = "train"
    split[(meta["date_x_start"] >= train_end + emb) & (meta["date_y_end"] <= val_end)] = "val"
    split[meta["date_x_start"] >= val_end + emb] = "test"

    counts = split.value_counts()
    print("[split]", dict(counts), f"| purga+embargo = {counts.get('purged', 0):,} ventanas")
    return split


# ---------------------------------------------------------------------------
# 4. Estandarización (ajustada solo con train)
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    """Estandarización z-score con estadísticos congelados de train.

    * X (retornos): media global ~0 por construcción; se escala por la
      desviación típica GLOBAL de train (un único escalar, no por columna:
      las 60 posiciones de la ventana son la misma variable física y un
      escalado por columna rompería la estacionariedad temporal interna).
    * y (log σ): z-score escalar clásico.

    `fit` devuelve self para permitir el encadenado fit(...).transform(...).
    """

    x_mu: float = 0.0
    x_sd: float = 1.0
    y_mu: float = 0.0
    y_sd: float = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Standardizer":
        self.x_mu = float(X_train.mean())
        self.x_sd = float(X_train.std())
        self.y_mu = float(y_train.mean())
        self.y_sd = float(y_train.std())
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mu) / self.x_sd

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mu) / self.y_sd

    def inverse_y(self, y_std: np.ndarray) -> np.ndarray:
        return y_std * self.y_sd + self.y_mu

    def to_dict(self) -> dict:
        return {"x_mu": self.x_mu, "x_sd": self.x_sd, "y_mu": self.y_mu, "y_sd": self.y_sd}

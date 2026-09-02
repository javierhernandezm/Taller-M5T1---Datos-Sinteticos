"""
malla.py — Barrido experimental real × sintético.

Responde la pregunta del taller: ¿cuántos datos sintéticos ayudan, en qué
régimen de escasez, y qué generador los produce mejor?

Diseño del experimento
----------------------
Para cada combinación (N_real, generador, ratio, semilla):

  1. Se submuestrean N_real ventanas del conjunto de train.
  2. El generador se **reentrena con esas N_real ventanas y solo esas**.
  3. Se generan N_synth = ratio × N_real ventanas sintéticas.
  4. Se entrena la arquitectura downstream CONGELADA (notebook 02) sobre
     la mezcla real+sintético.
  5. Se evalúa en el test real, intocado y siempre el mismo.

Las cuatro reglas que hacen el experimento interpretable
--------------------------------------------------------
* **Anti-fuga de escasez.** El generador NO puede ver más datos reales de
  los que el escenario declara tener. Ajustarlo con las 102k ventanas
  completas y luego simular escasez haría que los sintéticos
  transportasen información de datos que el escenario dice no poseer, y
  toda la curva saldría optimista. Por eso se reentrena en cada celda.
* **Submuestreo por FECHAS, no por filas.** En una fecha dada, todas las
  ventanas del panel cuentan la misma historia de mercado: un muestreo
  aleatorio de filas daría un N nominal muy superior al efectivo. Se
  sortean fechas y se toman sus ventanas.
* **Validación siempre real y completa.** El early-stopping / selección de
  pesos nunca mira datos sintéticos: si lo hiciera, optimizaríamos la
  distribución equivocada. Es idéntica en todas las celdas.
* **Arquitectura e hiperparámetros congelados.** Se reconstruyen desde
  `downstream_reference.json`. Si una celda rinde distinto, la única
  explicación posible son los datos.

Reanudabilidad
--------------
Cada celda se escribe al CSV de resultados en cuanto termina. Al relanzar,
las celdas ya presentes se saltan. Un corte de corriente cuesta una celda,
no la malla entera.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .baselines import regression_metrics
from .diffusion_ts import DiffusionTSR61Generator
from .eda import PALETTE, _style
from .generators import (
    BlockBootstrapGenerator,
    GaussianGenerator,
    JitterGenerator,
    RealNVPGenerator,
    VAEGenerator,
    WGANGPGenerator,
)
from .models import build_model
from .training import predict, train_model

#: Columnas del CSV de resultados. El orden es el de lectura humana.
COLUMNAS = [
    "n_real",
    "generador",
    "ratio",
    "seed",
    "n_synth",
    "n_train",
    "val_mse",
    "test_mse",
    "test_mae",
    "test_r2",
    "epoca_mejor",
    "segundos",
    "receta",
]


def firma_receta(ref: dict) -> str:
    """Huella corta de la configuración congelada (arquitectura + entrenamiento).

    Existe porque la clave de reanudación —(n_real, generador, n_synth, seed)—
    NO identifica con QUÉ modelo se calculó la celda. Sin esta huella, cambiar
    la campeona en el notebook 02 y relanzar la malla daba por hechas las filas
    de la ejecución anterior: el CSV acababa mezclando dos modelos distintos y
    las tablas promediaban sobre la mezcla, sin un solo aviso. Es el mismo fallo
    que tenía la reanudación del TSTR en el notebook 03.

    Entra también `train_kwargs`: cambiar el horizonte de épocas invalida los
    resultados igual que cambiar la arquitectura.
    """
    canon = json.dumps(
        {
            "arch": ref["arch"],
            "arch_kwargs": ref["arch_kwargs"],
            "train_kwargs": ref["train_kwargs"],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:8]


#: Etiqueta de la celda sin datos sintéticos (el punto de referencia de cada N).
SOLO_REAL = "ninguno"

#: Los seis brazos sintéticos del experimento vigente. WGAN-GP permanece en la
#: fábrica únicamente para reproducir los CSV históricos de la investigación.
GENERADORES_ACTIVOS = (
    "jitter",
    "gaussiana",
    "block_bootstrap",
    "vae",
    "diffusion_ts",
    "realnvp",
)


# --------------------------------------------------------------------------- #
# Generadores: fábrica
# --------------------------------------------------------------------------- #


def construir_generador(nombre: str):
    """Instancia un generador por nombre, con la configuración del notebook 03.

    Se instancia uno NUEVO en cada celda: los generadores guardan estado
    ajustado y reutilizarlos entre celdas sería precisamente la fuga que
    el diseño evita.
    """
    fabrica = {
        "jitter": lambda: JitterGenerator(noise=0.10),
        "gaussiana": lambda: GaussianGenerator(),
        "block_bootstrap": lambda: BlockBootstrapGenerator(mean_block=10),
        "vae": lambda: VAEGenerator(latent=16, hidden=256, beta=1.0, epochs=40),
        "wgan_gp": lambda: WGANGPGenerator(
            latent=32, hidden=256, epochs=150, lr=2e-4, n_critic=5
        ),
        "diffusion_ts": lambda: DiffusionTSR61Generator(
            train_steps=3_000, sample_steps=50
        ),
        "realnvp": lambda: RealNVPGenerator(n_layers=6, hidden=128, epochs=30),
    }
    if nombre not in fabrica:
        raise KeyError(
            f"Generador desconocido: {nombre!r}. Disponibles: {list(fabrica)}"
        )
    return fabrica[nombre]()


#: Los generadores que necesitan semilla en fit() (las redes).
NEURONALES = {"vae", "wgan_gp", "diffusion_ts", "realnvp"}


# --------------------------------------------------------------------------- #
# Submuestreo de escasez
# --------------------------------------------------------------------------- #


def submuestrear_por_fechas(meta_train: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Devuelve n índices de train sorteando FECHAS completas.

    Se barajan las fechas de train y se van tomando todas las ventanas de
    cada fecha hasta alcanzar n. Frente a un muestreo aleatorio de filas,
    esto respeta que las ventanas de una misma fecha son casi la misma
    observación (todas describen el mismo día de mercado en distintos
    activos), de modo que N refleja mejor el tamaño muestral efectivo.
    """
    rng = np.random.default_rng(seed)
    por_fecha = meta_train.groupby("date_t").indices  # dict fecha -> índices
    fechas = rng.permutation(list(por_fecha.keys()))
    elegidos: list[np.ndarray] = []
    total = 0
    for f in fechas:
        idx = por_fecha[f]
        elegidos.append(idx)
        total += len(idx)
        if total >= n:
            break
    todos = np.concatenate(elegidos)
    return rng.permutation(todos)[:n]


# --------------------------------------------------------------------------- #
# Una celda del experimento
# --------------------------------------------------------------------------- #


def resolver_presupuesto(
    n_real: int, ratio: float | None, n_synth: int | None
) -> tuple[float, int]:
    """Traduce el presupuesto de sintéticos a (ratio, n_synth), venga como venga.

    La malla admite dos diseños experimentales que son la misma celda vista
    desde dos lados:

    * **ratio fijo** (`plan_de_malla`): "3 sintéticos por cada real". Responde
      ¿cuánto añade el sintético *en proporción* al presupuesto real?
    * **conteo fijo** (`plan_de_curvas`): "20.000 sintéticos, haya los reales
      que haya". Responde ¿cómo cae el error al añadir reales, para un
      presupuesto sintético dado?

    El que se pase manda; el otro se deriva. Así una fila del CSV siempre
    lleva los dos y las dos mallas se comparan sin traducir nada a mano.
    """
    if n_synth is not None:
        n_synth = int(n_synth)
        return n_synth / n_real, n_synth
    if ratio is not None:
        return float(ratio), int(round(ratio * n_real))
    raise ValueError("Hay que dar 'ratio' o 'n_synth' (uno de los dos, no ninguno).")


def ejecutar_celda(
    *,
    n_real: int,
    generador: str,
    seed: int,
    ratio: float | None = None,
    n_synth: int | None = None,
    Xs_train,
    ys_train,
    meta_train,
    Xs_val,
    ys_val,
    Xs_test,
    y_test_fisico,
    std,
    ref,
    device,
) -> dict:
    """Ejecuta una celda completa y devuelve su fila de resultados.

    El presupuesto de sintéticos se declara con `ratio` **o** con `n_synth`
    (ver `resolver_presupuesto`); la fila devuelta lleva siempre los dos.

    `Xs_*`/`ys_*` llegan YA estandarizados; `y_test_fisico` es el target
    de test SIN estandarizar, para reportar métricas en unidades de ln σ.
    """
    t0 = time.time()
    ratio, n_synth = resolver_presupuesto(n_real, ratio, n_synth)

    # 1) escasez: qué ventanas reales "existen" en este escenario
    idx = submuestrear_por_fechas(meta_train, n_real, seed)
    Xr, yr = Xs_train[idx], ys_train[idx]

    # 2-3) generador reentrenado SOLO con esas ventanas, y muestreo
    if generador == SOLO_REAL or n_synth == 0:
        X_mix, y_mix, ratio, n_synth = Xr, yr, 0.0, 0
    else:
        XY = np.column_stack([Xr, yr]).astype(np.float32)
        gen = construir_generador(generador)
        gen.fit(XY, seed=seed) if generador in NEURONALES else gen.fit(XY)
        XY_s = gen.sample(n_synth, seed=seed + 1000)
        X_mix = np.vstack([Xr, XY_s[:, :-1]]).astype(np.float32)
        y_mix = np.concatenate([yr, XY_s[:, -1]]).astype(np.float32)

    # 4) downstream con la arquitectura CONGELADA del notebook 02
    kw = dict(ref["train_kwargs"])
    kw["patience"] = kw["epochs"]  # horizonte completo, mejor estado
    modelo = build_model(ref["arch"], **ref["arch_kwargs"])
    res = train_model(
        modelo, X_mix, y_mix, Xs_val, ys_val, seed=seed, device=device, **kw
    )

    # 5) evaluación en test real, des-estandarizando al espacio de ln σ
    p = predict(modelo, Xs_test, device) * std["y_sd"] + std["y_mu"]
    m = regression_metrics(y_test_fisico, p)

    return {
        "n_real": n_real,
        "generador": generador,
        "ratio": ratio,
        "seed": seed,
        "receta": firma_receta(ref),
        "n_synth": n_synth,
        "n_train": len(X_mix),
        "val_mse": res.best_val,
        "test_mse": m["mse"],
        "test_mae": m["mae"],
        "test_r2": m["r2"],
        "epoca_mejor": res.best_epoch,
        "segundos": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# La malla completa, reanudable
# --------------------------------------------------------------------------- #


def plan_de_malla(n_reales, generadores, ratios, seeds) -> list[dict]:
    """Lista de celdas a ejecutar, en orden de coste creciente.

    Ordenar de barato a caro tiene una ventaja práctica: si la malla se
    interrumpe, lo que ya está calculado es el régimen de escasez, que es
    justo el que más importa.
    """
    celdas = []
    for n in n_reales:
        for s in seeds:
            celdas.append(
                {
                    "n_real": n,
                    "generador": SOLO_REAL,
                    "ratio": 0.0,
                    "n_synth": 0,
                    "seed": s,
                }
            )
        for r in ratios:
            if r == 0:
                continue
            for g in generadores:
                for s in seeds:
                    celdas.append(
                        {
                            "n_real": n,
                            "generador": g,
                            "ratio": float(r),
                            "n_synth": int(round(r * n)),
                            "seed": s,
                        }
                    )
    return sorted(celdas, key=lambda c: (c["n_real"], c["n_synth"]))


def plan_de_curvas(n_reales, generadores, n_sinteticos, seeds) -> list[dict]:
    """Plan con el presupuesto sintético en **conteo absoluto**, no en proporción.

    Es el diseño que hace falta para la lectura "error frente al número de
    reales": cada curva de la figura es un `n_synth` que se mantiene constante
    mientras el eje x barre los reales. Con `plan_de_malla` eso es imposible,
    porque allí `n_synth = ratio × n_real` cambia en cada punto de la curva.

    El nivel `n_synth = 0` se resuelve con celdas de `SOLO_REAL`: es la curva
    de referencia, la única compartida por todos los generadores.
    """
    celdas = []
    for n in n_reales:
        for ns in sorted(set(n_sinteticos)):
            if ns == 0:
                for s in seeds:
                    celdas.append(
                        {
                            "n_real": n,
                            "generador": SOLO_REAL,
                            "ratio": 0.0,
                            "n_synth": 0,
                            "seed": s,
                        }
                    )
                continue
            for g in generadores:
                for s in seeds:
                    celdas.append(
                        {
                            "n_real": n,
                            "generador": g,
                            "ratio": ns / n,
                            "n_synth": int(ns),
                            "seed": s,
                        }
                    )
    return sorted(celdas, key=lambda c: (c["n_real"], c["n_synth"]))


def ejecutar_malla(
    plan: list[dict], ruta_csv: Path, *, verbose: bool = True, **contexto
) -> pd.DataFrame:
    """Ejecuta el plan escribiendo cada celda al CSV en cuanto termina.

    Reanudable: las celdas cuya clave (n_real, generador, ratio, seed) ya
    figura en el CSV **y fueron calculadas con la misma receta congelada** se
    saltan sin recalcular. Las filas de otra receta se descartan y se
    recalculan: ver `firma_receta`. Un CSV anterior a esta columna se trata
    como "receta desconocida", que es lo prudente.
    """
    ruta_csv = Path(ruta_csv)
    firma = firma_receta(contexto["ref"])
    if ruta_csv.exists():
        hechas = pd.read_csv(ruta_csv)
        if "receta" not in hechas.columns:
            hechas["receta"] = None
        vigentes = hechas[hechas["receta"] == firma]
        ajenas = len(hechas) - len(vigentes)
        if ajenas:
            # Se reescribe el CSV: dejar conviviendo dos recetas haría que
            # resumir() promediase sobre modelos distintos.
            print(
                f"  ! {ajenas} filas de {ruta_csv.name} son de otra receta "
                f"congelada (la de ahora es {firma}): se descartan y recalculan."
            )
            vigentes.to_csv(ruta_csv, index=False)
        hechas = vigentes
        clave = set(zip(hechas.n_real, hechas.generador, hechas.n_synth, hechas.seed))
    else:
        hechas, clave = pd.DataFrame(columns=COLUMNAS), set()
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        hechas.to_csv(ruta_csv, index=False)

    # La clave es n_synth y no ratio: es la magnitud que las dos mallas
    # comparten. Dentro de un mismo n_real ambas son biyectivas, así que los
    # CSV escritos por la malla de ratios se siguen reanudando igual.
    pendientes = [
        c
        for c in plan
        if (
            c["n_real"],
            c["generador"],
            resolver_presupuesto(c["n_real"], c.get("ratio"), c.get("n_synth"))[1],
            c["seed"],
        )
        not in clave
    ]
    if verbose:
        print(
            f"Malla: {len(plan)} celdas | ya hechas: {len(plan) - len(pendientes)} | "
            f"pendientes: {len(pendientes)}"
        )

    t_inicio = time.time()
    for i, celda in enumerate(pendientes, 1):
        fila = ejecutar_celda(**celda, **contexto)
        pd.DataFrame([fila])[COLUMNAS].to_csv(
            ruta_csv, mode="a", header=False, index=False
        )
        if verbose:
            transcurrido = time.time() - t_inicio
            restante = transcurrido / i * (len(pendientes) - i)
            print(
                f"[{i:>3}/{len(pendientes)}] N={celda['n_real']:>6} "
                f"{celda['generador']:<16} synth={fila['n_synth']:>6} "
                f"s={celda['seed']} -> R² {fila['test_r2']:.4f} "
                f"({fila['segundos']:.0f}s) | ETA {restante / 60:.0f} min"
            )
    return pd.read_csv(ruta_csv)


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega sobre semillas: media y desviación por (n_real, generador, ratio)."""
    g = df.groupby(["n_real", "generador", "ratio"])
    out = g.agg(
        test_r2_media=("test_r2", "mean"),
        test_r2_sd=("test_r2", "std"),
        test_mse_media=("test_mse", "mean"),
        n_train=("n_train", "first"),
        n_seeds=("seed", "count"),
    )
    return out.reset_index()


def delta_vs_solo_real(resumen: pd.DataFrame) -> pd.DataFrame:
    """Ganancia de R² de cada celda frente a su referencia de solo-real.

    Es la métrica que responde literalmente al enunciado: para el mismo
    presupuesto de datos reales, ¿cuánto añade el sintético?
    """
    base = (
        resumen[resumen.generador == SOLO_REAL]
        .set_index("n_real")["test_r2_media"]
        .rename("r2_solo_real")
    )
    out = resumen[resumen.generador != SOLO_REAL].merge(base, on="n_real")
    out["delta_r2"] = out["test_r2_media"] - out["r2_solo_real"]
    return out.sort_values(["n_real", "generador", "ratio"])


def delta_pareado(
    df: pd.DataFrame,
    metrica: str = "test_r2",
    presupuesto: str = "ratio",
    mas_es_mejor: bool = True,
) -> pd.DataFrame:
    """Delta **pareado por semilla** de `metrica`, con su significancia.

    Por qué pareado y no diferencia de medias: la semilla fija el submuestreo
    de ventanas reales, así que comparar una celda contra la celda de solo-real
    de *su misma semilla* elimina la varianza del submuestreo — que es la que
    domina en régimen de escasez. Con N=250 la desviación entre semillas del
    modelo solo-real es 0,063, once veces el umbral de ±0,006 que fijó el
    notebook 02 con 102k ventanas: aplicar aquel umbral aquí declararía
    significativa casi cualquier diferencia. El pareado resuelve eso.

    Devuelve, por (n_real, generador, `presupuesto`): media del delta, su
    desviación, el error estándar y el estadístico t = media / error estándar.
    Con solo 3 semillas se exige |t| >= 2,5 para hablar de efecto; por debajo,
    la celda se declara no concluyente en lugar de inventar una conclusión.

    Parámetros
    ----------
    metrica, mas_es_mejor
        Qué columna se compara y en qué dirección. Por defecto `test_r2`, donde
        más es mejor; para un error (`val_mse`, `test_mse`) hay que pasar
        `mas_es_mejor=False` o el veredicto sale del revés.
    presupuesto
        La columna que identifica cuánto sintético lleva la celda: `ratio` para
        la malla de proporciones, `n_synth` para la de conteos absolutos.
    """
    base = (
        df[df.generador == SOLO_REAL]
        .set_index(["n_real", "seed"])[metrica]
        .rename("base")
    )
    d = df[df.generador != SOLO_REAL].join(base, on=["n_real", "seed"])
    d = d.assign(delta=d[metrica] - d["base"])

    out = (
        d.groupby(["n_real", "generador", presupuesto])["delta"]
        .agg(delta_medio="mean", delta_sd="std", n_seeds="count")
        .reset_index()
    )
    out["ee"] = out.delta_sd / np.sqrt(out.n_seeds)
    out["t"] = out.delta_medio / out.ee.replace(0, np.nan)
    mejora = out.delta_medio > 0 if mas_es_mejor else out.delta_medio < 0
    out["veredicto"] = np.where(
        out.t.abs() >= 2.5, np.where(mejora, "mejora", "empeora"), "no concluyente"
    )
    return out


def umbral_ruido_por_n(df: pd.DataFrame) -> pd.Series:
    """Desviación entre semillas del modelo solo-real, por nivel de N.

    Es el umbral de relevancia HONESTO en cada régimen de escasez, y sustituye
    al ±0,006 del notebook 02, que solo vale para el modelo entrenado con las
    102k ventanas completas.
    """
    return df[df.generador == SOLO_REAL].groupby("n_real")["test_r2"].std()


# --------------------------------------------------------------------------- #
# Figuras: error frente al número de reales, por presupuesto sintético
# --------------------------------------------------------------------------- #

#: El color codifica el PRESUPUESTO SINTÉTICO, no el generador. Es la decisión
#: que hace legibles las dos figuras: el generador se lee por panel (fig. 27) o
#: por grosor de línea (fig. 28), y el color siempre significa lo mismo.
_COLORES_SYNTH = ("blue", "orange", "green", "vermillion", "purple", "sky")


def _serie_por_n_real(
    df: pd.DataFrame, generador: str, n_synth: int, metrica: str
) -> pd.Series:
    """Media sobre semillas de `metrica`, indexada por n_real.

    El nivel n_synth=0 no lo produce ningún generador: es la celda de solo
    real, y por eso se lee siempre de `SOLO_REAL` sea cual sea el generador
    que se pida. Es lo que permite dibujar la misma curva de referencia en
    los seis paneles.
    """
    g = SOLO_REAL if n_synth == 0 else generador
    s = df[(df.generador == g) & (df.n_synth == n_synth)]
    return s.groupby("n_real")[metrica].mean().sort_index()


def _eje_n_real(ax, n_reales) -> None:
    """Eje x logarítmico con los niveles reales como ticks, en miles legibles.

    El LogLocator por defecto pondría 10³ y 10⁴, que no son los puntos que
    hemos medido: el lector no sabría dónde cae N=3.000. Los ticks menores se
    apagan **solo en x** (`minorticks_off` los apagaría también en y, y ahí
    son los únicos que hay — ver `_eje_error`).
    """
    ax.set_xscale("log")
    ax.set_xticks(list(n_reales))
    ax.set_xticklabels([f"{n // 1000}k" if n >= 1000 else str(n) for n in n_reales])
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def _eje_error(ax, metrica: str) -> None:
    """Eje y logarítmico etiquetado dentro de una sola década.

    Todo el rango observado cae entre 0,35 y 0,8, así que no hay ni un tick
    **mayor** (potencia de diez) dentro de la vista: con el formateo por defecto
    el eje sale mudo. Se etiquetan los menores —0,4 · 0,5 · 0,6 · 0,8— que es
    lo mismo que hace la gráfica de referencia.
    """
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    )
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.set_ylabel(f"{metrica} (escala log)")


def plot_curvas_error_por_generador(
    df: pd.DataFrame,
    n_reales,
    n_sinteticos,
    generadores,
    metrica: str = "val_mse",
    ncols: int = 3,
) -> plt.Figure:
    """Un panel por generador; una curva por presupuesto sintético absoluto.

    Responde: para un presupuesto sintético dado, ¿cómo cae el error al
    disponer de más datos reales, y qué generador lo hace caer más? La curva
    de 0 sintéticos (solo real) se repite en los seis paneles: es la línea que
    hay que batir, y verla en cada panel evita tener que cruzar la vista.
    """
    nrows = int(np.ceil(len(generadores) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.3 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    niveles = sorted(set(n_sinteticos))

    for ax, gen in zip(axes, generadores):
        for color, ns in zip(_COLORES_SYNTH, niveles):
            s = _serie_por_n_real(df, gen, ns, metrica)
            if not len(s):
                continue
            ax.plot(
                s.index,
                s.values,
                "o-",
                color=PALETTE[color],
                lw=1.4,
                ms=4,
                label=f"{ns:,} sintéticos".replace(",", "."),
            )
        ax.set_title(gen, fontsize=10)
        _eje_error(ax, metrica)
        _eje_n_real(ax, n_reales)
        _style(ax)

    for ax in axes[len(generadores) :]:  # huecos si la rejilla no cuadra
        ax.set_visible(False)
    for ax in axes[len(generadores) - ncols : len(generadores)]:
        ax.set_xlabel("ventanas reales disponibles")
    for k, ax in enumerate(axes):  # el ylabel solo en la columna izquierda
        if k % ncols:
            ax.set_ylabel("")

    manejadores, etiquetas = axes[0].get_legend_handles_labels()
    fig.legend(
        manejadores,
        etiquetas,
        loc="lower center",
        ncol=len(niveles),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Error frente al número de reales, por presupuesto sintético")
    fig.tight_layout()
    return fig


def plot_curvas_error_todos(
    df: pd.DataFrame, n_reales, n_sinteticos, generadores, metrica: str = "val_mse"
) -> plt.Figure:
    """Los seis generadores en un solo eje, resumidos por presupuesto sintético.

    Cada color es un presupuesto: seis líneas finas (un generador cada una) y
    encima la **mediana** de las seis en grueso. Mediana y no media porque el
    WGAN-GP colapsa en régimen de escasez y una media arrastraría el haz
    entero; las líneas finas siguen mostrando esa cola sin dejar que la domine.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    niveles = sorted(set(n_sinteticos))

    for color, ns in zip(_COLORES_SYNTH, niveles):
        series = [
            _serie_por_n_real(df, g, ns, metrica)
            for g in (["-"] if ns == 0 else generadores)
        ]
        series = [s for s in series if len(s)]
        if not series:
            continue
        if ns > 0:  # el haz: un generador por línea fina
            for s in series:
                ax.plot(
                    s.index,
                    s.values,
                    "-",
                    color=PALETTE[color],
                    lw=0.9,
                    alpha=0.35,
                    zorder=1,
                )
        resumen_ns = pd.concat(series, axis=1).median(axis=1)
        ax.plot(
            resumen_ns.index,
            resumen_ns.values,
            "o-",
            color=PALETTE[color],
            lw=2.4,
            ms=5,
            zorder=3,
            label=f"{ns:,} sintéticos".replace(",", "."),
        )

    _eje_error(ax, metrica)
    _eje_n_real(ax, n_reales)
    ax.set_xlabel("ventanas reales disponibles")
    ax.set_title(
        "TODOS — error frente al número de reales\n"
        "línea fina = un generador · línea gruesa = mediana de los seis",
        fontsize=11,
    )
    ax.legend(fontsize=8, title="presupuesto sintético", title_fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig

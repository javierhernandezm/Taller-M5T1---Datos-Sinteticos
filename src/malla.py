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

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .generators import (BlockBootstrapGenerator, GaussianGenerator, JitterGenerator,
                         RealNVPGenerator, VAEGenerator, WGANGPGenerator)
from .models import build_model
from .baselines import regression_metrics
from .training import predict, train_model

#: Columnas del CSV de resultados. El orden es el de lectura humana.
COLUMNAS = ["n_real", "generador", "ratio", "seed", "n_synth", "n_train",
            "val_mse", "test_mse", "test_mae", "test_r2", "epoca_mejor", "segundos"]

#: Etiqueta de la celda sin datos sintéticos (el punto de referencia de cada N).
SOLO_REAL = "ninguno"


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
        "wgan_gp": lambda: WGANGPGenerator(latent=32, hidden=256, epochs=150,
                                           lr=2e-4, n_critic=5),
        "realnvp": lambda: RealNVPGenerator(n_layers=6, hidden=128, epochs=30),
    }
    if nombre not in fabrica:
        raise KeyError(f"Generador desconocido: {nombre!r}. Disponibles: {list(fabrica)}")
    return fabrica[nombre]()


#: Los generadores que necesitan semilla en fit() (las redes).
NEURONALES = {"vae", "wgan_gp", "realnvp"}


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

def ejecutar_celda(*, n_real: int, generador: str, ratio: float, seed: int,
                   Xs_train, ys_train, meta_train, Xs_val, ys_val,
                   Xs_test, y_test_fisico, std, ref, device) -> dict:
    """Ejecuta una celda completa y devuelve su fila de resultados.

    `Xs_*`/`ys_*` llegan YA estandarizados; `y_test_fisico` es el target
    de test SIN estandarizar, para reportar métricas en unidades de ln σ.
    """
    t0 = time.time()

    # 1) escasez: qué ventanas reales "existen" en este escenario
    idx = submuestrear_por_fechas(meta_train, n_real, seed)
    Xr, yr = Xs_train[idx], ys_train[idx]

    # 2-3) generador reentrenado SOLO con esas ventanas, y muestreo
    n_synth = int(round(ratio * n_real))
    if generador == SOLO_REAL or n_synth == 0:
        X_mix, y_mix, n_synth = Xr, yr, 0
    else:
        XY = np.column_stack([Xr, yr]).astype(np.float32)
        gen = construir_generador(generador)
        gen.fit(XY, seed=seed) if generador in NEURONALES else gen.fit(XY)
        XY_s = gen.sample(n_synth, seed=seed + 1000)
        X_mix = np.vstack([Xr, XY_s[:, :-1]]).astype(np.float32)
        y_mix = np.concatenate([yr, XY_s[:, -1]]).astype(np.float32)

    # 4) downstream con la arquitectura CONGELADA del notebook 02
    kw = dict(ref["train_kwargs"])
    kw["patience"] = kw["epochs"]          # horizonte completo, mejor estado
    modelo = build_model(ref["arch"], **ref["arch_kwargs"])
    res = train_model(modelo, X_mix, y_mix, Xs_val, ys_val,
                      seed=seed, device=device, **kw)

    # 5) evaluación en test real, des-estandarizando al espacio de ln σ
    p = predict(modelo, Xs_test, device) * std["y_sd"] + std["y_mu"]
    m = regression_metrics(y_test_fisico, p)

    return {"n_real": n_real, "generador": generador, "ratio": ratio, "seed": seed,
            "n_synth": n_synth, "n_train": len(X_mix),
            "val_mse": res.best_val, "test_mse": m["mse"], "test_mae": m["mae"],
            "test_r2": m["r2"], "epoca_mejor": res.best_epoch,
            "segundos": round(time.time() - t0, 1)}


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
            celdas.append({"n_real": n, "generador": SOLO_REAL, "ratio": 0.0, "seed": s})
        for r in ratios:
            if r == 0:
                continue
            for g in generadores:
                for s in seeds:
                    celdas.append({"n_real": n, "generador": g, "ratio": r, "seed": s})
    return sorted(celdas, key=lambda c: (c["n_real"], c["ratio"]))


def ejecutar_malla(plan: list[dict], ruta_csv: Path, *, verbose: bool = True,
                   **contexto) -> pd.DataFrame:
    """Ejecuta el plan escribiendo cada celda al CSV en cuanto termina.

    Reanudable: las celdas cuya clave (n_real, generador, ratio, seed) ya
    figura en el CSV se saltan sin recalcular.
    """
    ruta_csv = Path(ruta_csv)
    if ruta_csv.exists():
        hechas = pd.read_csv(ruta_csv)
        clave = set(zip(hechas.n_real, hechas.generador, hechas.ratio, hechas.seed))
    else:
        hechas, clave = pd.DataFrame(columns=COLUMNAS), set()
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        hechas.to_csv(ruta_csv, index=False)

    pendientes = [c for c in plan
                  if (c["n_real"], c["generador"], c["ratio"], c["seed"]) not in clave]
    if verbose:
        print(f"Malla: {len(plan)} celdas | ya hechas: {len(plan) - len(pendientes)} | "
              f"pendientes: {len(pendientes)}")

    t_inicio = time.time()
    for i, celda in enumerate(pendientes, 1):
        fila = ejecutar_celda(**celda, **contexto)
        pd.DataFrame([fila])[COLUMNAS].to_csv(ruta_csv, mode="a", header=False, index=False)
        if verbose:
            transcurrido = time.time() - t_inicio
            restante = transcurrido / i * (len(pendientes) - i)
            print(f"[{i:>3}/{len(pendientes)}] N={celda['n_real']:>6} "
                  f"{celda['generador']:<16} r={celda['ratio']:<4} s={celda['seed']} "
                  f"-> R² {fila['test_r2']:.4f} ({fila['segundos']:.0f}s) "
                  f"| ETA {restante/60:.0f} min")
    return pd.read_csv(ruta_csv)


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #

def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega sobre semillas: media y desviación por (n_real, generador, ratio)."""
    g = df.groupby(["n_real", "generador", "ratio"])
    out = g.agg(test_r2_media=("test_r2", "mean"), test_r2_sd=("test_r2", "std"),
                test_mse_media=("test_mse", "mean"), n_train=("n_train", "first"),
                n_seeds=("seed", "count"))
    return out.reset_index()


def delta_vs_solo_real(resumen: pd.DataFrame) -> pd.DataFrame:
    """Ganancia de R² de cada celda frente a su referencia de solo-real.

    Es la métrica que responde literalmente al enunciado: para el mismo
    presupuesto de datos reales, ¿cuánto añade el sintético?
    """
    base = (resumen[resumen.generador == SOLO_REAL]
            .set_index("n_real")["test_r2_media"].rename("r2_solo_real"))
    out = resumen[resumen.generador != SOLO_REAL].merge(base, on="n_real")
    out["delta_r2"] = out["test_r2_media"] - out["r2_solo_real"]
    return out.sort_values(["n_real", "generador", "ratio"])


def delta_pareado(df: pd.DataFrame) -> pd.DataFrame:
    """Δ R² **pareado por semilla**, con su significancia.

    Por qué pareado y no diferencia de medias: la semilla fija el submuestreo
    de ventanas reales, así que comparar una celda contra la celda de solo-real
    de *su misma semilla* elimina la varianza del submuestreo — que es la que
    domina en régimen de escasez. Con N=250 la desviación entre semillas del
    modelo solo-real es 0,063, once veces el umbral de ±0,006 que fijó el
    notebook 02 con 102k ventanas: aplicar aquel umbral aquí declararía
    significativa casi cualquier diferencia. El pareado resuelve eso.

    Devuelve, por (n_real, generador, ratio): media del delta, su desviación,
    el error estándar y el estadístico t = media / error estándar. Con solo 3
    semillas se exige |t| >= 2,5 para hablar de efecto; por debajo, la celda
    se declara no concluyente en lugar de inventar una conclusión.
    """
    base = (df[df.generador == SOLO_REAL]
            .set_index(["n_real", "seed"])["test_r2"].rename("r2_base"))
    d = df[df.generador != SOLO_REAL].join(base, on=["n_real", "seed"])
    d = d.assign(delta=d.test_r2 - d.r2_base)

    out = (d.groupby(["n_real", "generador", "ratio"])["delta"]
             .agg(delta_medio="mean", delta_sd="std", n_seeds="count").reset_index())
    out["ee"] = out.delta_sd / np.sqrt(out.n_seeds)
    out["t"] = out.delta_medio / out.ee.replace(0, np.nan)
    out["veredicto"] = np.where(out.t.abs() >= 2.5,
                                np.where(out.delta_medio > 0, "mejora", "empeora"),
                                "no concluyente")
    return out


def umbral_ruido_por_n(df: pd.DataFrame) -> pd.Series:
    """Desviación entre semillas del modelo solo-real, por nivel de N.

    Es el umbral de relevancia HONESTO en cada régimen de escasez, y sustituye
    al ±0,006 del notebook 02, que solo vale para el modelo entrenado con las
    102k ventanas completas.
    """
    return df[df.generador == SOLO_REAL].groupby("n_real")["test_r2"].std()

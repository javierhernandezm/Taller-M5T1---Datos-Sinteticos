"""
gen_utility.py — Auditoría de UTILIDAD de los generadores: TSTR frente a TRTR.

`gen_audit.py` responde "¿se parecen las ventanas sintéticas a las reales?".
Este módulo responde la otra mitad del marco: **"¿sirven para entrenar?"**.

    TSTR  Train on Synthetic, Test on Real
    TRTR  Train on Real,      Test on Real   (la referencia)

El ratio R²_TSTR / R²_TRTR es el número que resume la utilidad: 1,0 significa
que el dataset sintético conserva TODA la información que el modelo necesita;
0 significa que no conserva ninguna; negativo, que además mete sesgo.

POR QUÉ NO SE REUTILIZA `malla.ejecutar_celda`
----------------------------------------------
La malla del notebook 04 responde una pregunta distinta y más rica —cuánto
añade el sintético SOBRE N ventanas reales— y por eso su mezcla de
entrenamiento siempre incluye el bloque real, y su early stopping mira la
validación real. Ninguna de esas dos cosas vale aquí:

  * un TSTR con datos reales en la mezcla no es un TSTR;
  * y un TSTR que elige la mejor época mirando datos reales tampoco lo es,
    porque el dato real habría entrado por la puerta de atrás, en la
    selección de modelo.

De ahí el 10 % interno: cada brazo aparta una porción de su PROPIO material
—sintético en el brazo TSTR, real en el TRTR— para elegir la época. El brazo
sintético no ve un solo dato real en ningún punto del bucle de entrenamiento.

LA SIMETRÍA ES LO QUE HACE QUE EL RATIO SIGNIFIQUE ALGO
------------------------------------------------------
Los dos brazos reciben el mismo presupuesto de ventanas, apartan la misma
fracción interna, usan la MISMA arquitectura congelada con los MISMOS
hiperparámetros (los de `downstream_reference.json`) y se evalúan sobre el
mismo conjunto real. Si un brazo rinde distinto, la única explicación posible
son los datos — que es la tesis del taller entero.

Se evalúa sobre VALIDACIÓN, no sobre test: comparar generadores es selección
de modelo, y el test se reserva intacto para el resultado final del notebook
04.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import regression_metrics
from .models import build_model
from .training import predict, train_model

COLUMNAS = ["brazo", "seed", "n_train", "val_mse", "val_mae", "val_r2",
            "epoca_mejor", "segundos"]

BRAZO_REAL = "real"   # etiqueta del brazo TRTR, la referencia del ratio


def entrenar_y_evaluar(X_tr: np.ndarray, y_tr: np.ndarray,
                       X_eval: np.ndarray, y_eval_fisico: np.ndarray, *,
                       ref: dict, std: dict, device, seed: int,
                       val_frac: float = 0.1) -> dict:
    """Entrena la arquitectura congelada sobre (X_tr, y_tr) y evalúa en real.

    `X_tr`/`y_tr` vienen YA estandarizados (es el espacio en el que viven
    tanto las ventanas reales como las de los generadores). De ellos se aparta
    una fracción `val_frac` para elegir la mejor época: es la única función de
    esa porción, porque `ref["train_kwargs"]` trae `patience == epochs` y por
    tanto el early stopping nunca dispara.

    `y_eval_fisico` es el target SIN estandarizar (ln σ anualizada), de modo
    que el R² sale en las mismas unidades que el notebook 02 y el 04.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(1, int(round(val_frac * len(X_tr))))
    i_val, i_tr = perm[:n_val], perm[n_val:]

    kw = dict(ref["train_kwargs"])
    modelo = build_model(ref["arch"], **ref["arch_kwargs"])
    res = train_model(modelo, X_tr[i_tr], y_tr[i_tr], X_tr[i_val], y_tr[i_val],
                      seed=seed, device=device, **kw)

    p = predict(modelo, X_eval, device) * std["y_sd"] + std["y_mu"]
    m = regression_metrics(y_eval_fisico, p)
    return {"n_train": len(i_tr), "val_mse": m["mse"], "val_mae": m["mae"],
            "val_r2": m["r2"], "epoca_mejor": res.best_epoch,
            "segundos": res.seconds}


def tstr_trtr(fitted: dict, XY_train: np.ndarray,
              X_eval: np.ndarray, y_eval_fisico: np.ndarray, *,
              ref: dict, std: dict, device,
              seeds: tuple[int, ...] = (42, 43, 44),
              n_muestras: int | None = None,
              ruta_csv: Path | None = None,
              verbose: bool = True) -> pd.DataFrame:
    """Un brazo TRTR más un brazo TSTR por generador, sobre varias semillas.

    `fitted` es el dict {nombre: generador ya ajustado} del notebook 03. Los
    generadores NO se reajustan por semilla: reajustar los tres neuronales
    tres veces cuesta cerca de una hora en CPU y aportaría poco, porque la
    pregunta aquí es la utilidad del generador ya entrenado. La dispersión
    que se mide es, por tanto, la del muestreo sintético más la del
    entrenamiento downstream — la misma convención que el notebook 02, donde
    solo varía la semilla de entrenamiento.

    Con `ruta_csv` el barrido es reanudable: las filas cuya clave
    (brazo, seed) ya está en el CSV se saltan sin recalcular.

    CUIDADO: esa clave NO incluye la arquitectura de `ref`. Si la campeona
    del notebook 02 cambia y el CSV de la ejecución anterior sigue en disco,
    la reanudación da por buenas filas calculadas con OTRO modelo, en
    silencio. Reanudar solo es seguro dentro de una misma configuración
    congelada; si `ref` puede haber cambiado, pasar `ruta_csv=None` y
    escribir el CSV desde fuera.
    """
    n = n_muestras or len(XY_train)
    plan = [{"brazo": b, "seed": s}
            for b in [BRAZO_REAL, *fitted] for s in seeds]

    clave: set = set()
    if ruta_csv is not None:
        ruta_csv = Path(ruta_csv)
        try:
            hechas = pd.read_csv(ruta_csv)
            clave = set(zip(hechas.brazo, hechas.seed))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # No existe, o quedó vacío al interrumpir la primera escritura:
            # en ambos casos no hay nada hecho y se (re)crea la cabecera.
            ruta_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=COLUMNAS).to_csv(ruta_csv, index=False)

    pendientes = [c for c in plan if (c["brazo"], c["seed"]) not in clave]
    if verbose:
        print(f"TSTR/TRTR: {len(plan)} entrenamientos | "
              f"ya hechos: {len(plan) - len(pendientes)} | "
              f"pendientes: {len(pendientes)}")

    filas, t_inicio = [], time.time()
    for i, celda in enumerate(pendientes, 1):
        brazo, seed = celda["brazo"], celda["seed"]
        if brazo == BRAZO_REAL:
            # TRTR: el mismo presupuesto de ventanas, pero reales
            idx = np.random.default_rng(seed).choice(
                len(XY_train), min(n, len(XY_train)), replace=False)
            XY = XY_train[idx]
        else:
            # El desfase +1000 descorrelaciona el muestreo de la semilla de
            # ajuste, igual que en malla.ejecutar_celda.
            XY = fitted[brazo].sample(n, seed=seed + 1000)

        fila = {"brazo": brazo, "seed": seed,
                **entrenar_y_evaluar(XY[:, :-1], XY[:, -1], X_eval, y_eval_fisico,
                                     ref=ref, std=std, device=device, seed=seed)}
        filas.append(fila)
        if ruta_csv is not None:
            pd.DataFrame([fila])[COLUMNAS].to_csv(
                ruta_csv, mode="a", header=False, index=False)
        if verbose:
            restante = (time.time() - t_inicio) / i * (len(pendientes) - i)
            print(f"[{i:>2}/{len(pendientes)}] {brazo:<16} s={seed} "
                  f"-> R² val {fila['val_r2']:.4f} ({fila['segundos']:.0f}s) "
                  f"| ETA {restante/60:.0f} min")

    if ruta_csv is not None:
        return pd.read_csv(ruta_csv)
    return pd.DataFrame(filas)[COLUMNAS]


def resumir_tstr(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega sobre semillas y añade el ratio TSTR/TRTR.

    El ratio se calcula sobre las MEDIAS, no promediando ratios por semilla:
    con un denominador cerca de cero un ratio por semilla puede explotar y
    contaminar la media. Se ordena de mayor a menor utilidad.
    """
    out = (df.groupby("brazo")
             .agg(val_r2_media=("val_r2", "mean"), val_r2_sd=("val_r2", "std"),
                  val_mse_media=("val_mse", "mean"),
                  epoca_mejor=("epoca_mejor", "mean"),
                  n_seeds=("seed", "nunique"), segundos=("segundos", "sum"))
             .reset_index())
    trtr = float(out.loc[out.brazo == BRAZO_REAL, "val_r2_media"].iloc[0])
    out["ratio_tstr_trtr"] = out["val_r2_media"] / trtr
    return out.sort_values("val_r2_media", ascending=False).reset_index(drop=True)

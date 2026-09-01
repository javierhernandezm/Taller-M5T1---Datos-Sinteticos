"""
extender_malla.py — Malla extendida a ratios altos (0, 1, 3, 10, 30).

Por qué existe
--------------
La malla del notebook 04 llega hasta ratio 3x. A ese rango el dato sintético
todavía está en la rama ascendente de la curva: se ve que ayuda, pero no dónde
deja de hacerlo. El deterioro que se discute en clase aparece bastante más
arriba (el profesor llegaba a 40x). Este script fija N_real = 1.000 —el
escenario de escasez donde el sintético tiene margen para aportar— y barre
hasta 30x para localizar el punto de giro.

Salida: data/processed/malla_extendida.csv (reanudable: si el CSV ya existe,
solo ejecuta las celdas que falten).

Uso
---
    uv run python scripts/extender_malla.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.malla import ejecutar_malla, plan_de_malla
from src.training import get_device

# En CPU conviene limitar hilos: las redes son pequeñas y el oversubscribing
# de BLAS penaliza más de lo que aporta.
if not torch.cuda.is_available():
    torch.set_num_threads(2)


def main() -> int:
    cfg = Config()
    dev = get_device()
    proc = cfg.out_dir

    d = np.load(proc / "windows_dataset.npz")
    std = json.loads((proc / "standardizer.json").read_text())
    ref = json.loads((proc / "downstream_reference.json").read_text())
    meta = pd.read_parquet(proc / "windows_meta.parquet")
    meta_train = meta[meta.split == "train"].reset_index(drop=True)

    def S(a, mu, sd):
        return ((a - std[mu]) / std[sd]).astype(np.float32)

    # Validación real fija y submuestreada: 5.000 ventanas bastan para el early
    # stopping y recortan mucho el coste de 51 entrenamientos.
    rng = np.random.default_rng(cfg.seed)
    iv = rng.choice(len(d["X_val"]), 5000, replace=False)

    ctx = dict(
        Xs_train=S(d["X_train"], "x_mu", "x_sd"),
        ys_train=S(d["y_train"], "y_mu", "y_sd"),
        meta_train=meta_train,
        Xs_val=S(d["X_val"], "x_mu", "x_sd")[iv],
        ys_val=S(d["y_val"], "y_mu", "y_sd")[iv],
        Xs_test=S(d["X_test"], "x_mu", "x_sd"),
        y_test_fisico=d["y_test"],
        std=std, ref=ref, device=dev,
    )

    # Cuatro generadores que cubren el rango de fidelidad observado en el nb03:
    # jitter (AUC 0,50) y realnvp (0,65) arriba, block_bootstrap (0,76) en medio,
    # gaussiana (0,96) como control negativo.
    plan = plan_de_malla(
        n_reales=[1000],
        generadores=["jitter", "realnvp", "block_bootstrap", "gaussiana"],
        ratios=[0, 1, 3, 10, 30],
        seeds=[0, 1, 2],
    )
    print(f"{len(plan)} celdas | device={dev}", flush=True)
    df = ejecutar_malla(plan, proc / "malla_extendida.csv", **ctx)
    print(f"Terminado: {len(df)} filas en {proc / 'malla_extendida.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

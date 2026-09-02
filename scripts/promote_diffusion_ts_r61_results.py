"""Promueve las mallas R61 validadas a los artefactos canonicos de la rama."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.malla import GENERADORES_ACTIVOS, delta_pareado, delta_vs_solo_real, resumir

SOURCE = REPO_ROOT / "reports/diffusion_ts_r61/mallas"
PROCESSED = REPO_ROOT / "data/processed"
FIGURES = REPO_ROOT / "reports/figures"
ACTIVE = {*GENERADORES_ACTIVOS, "ninguno"}


def validate(frame: pd.DataFrame, *, expected_rows: int, diffusion_rows: int) -> None:
    keys = ["n_real", "generador", "n_synth", "seed"]
    if len(frame) != expected_rows:
        raise RuntimeError(f"Malla incompleta: {len(frame)} != {expected_rows}")
    if set(frame.generador) != ACTIVE:
        raise RuntimeError(f"Generadores inesperados: {sorted(set(frame.generador))}")
    if int((frame.generador == "diffusion_ts").sum()) != diffusion_rows:
        raise RuntimeError("Numero incorrecto de celdas Diffusion-TS")
    if frame.duplicated(keys).any():
        raise RuntimeError("Hay celdas duplicadas en la malla")
    counts = frame.groupby(["n_real", "generador", "n_synth"]).seed.nunique()
    if not (counts == 3).all():
        raise RuntimeError("Alguna celda no contiene exactamente tres semillas")
    numeric = frame.select_dtypes(include="number")
    if numeric.isna().any().any():
        raise RuntimeError("La malla contiene valores numericos ausentes")


def preserve_wgan_history(name: str) -> None:
    source = PROCESSED / name
    if not source.exists():
        return
    historical = pd.read_csv(source)
    wgan = historical.loc[historical.generador == "wgan_gp"]
    if len(wgan):
        wgan.to_csv(SOURCE / f"historical_wgan_{name}", index=False)


def main() -> None:
    ratios = pd.read_csv(SOURCE / "malla_ratios_finos.csv")
    curves = pd.read_csv(SOURCE / "malla_curvas.csv")
    validate(ratios, expected_rows=333, diffusion_rows=54)
    validate(curves, expected_rows=465, diffusion_rows=75)

    preserve_wgan_history("malla_ratios_finos.csv")
    preserve_wgan_history("malla_curvas.csv")

    ratios.to_csv(PROCESSED / "malla_ratios_finos.csv", index=False)
    resumir(ratios).to_csv(PROCESSED / "malla_ratios_finos_resumen.csv", index=False)
    delta_vs_solo_real(resumir(ratios)).to_csv(
        PROCESSED / "malla_ratios_finos_delta.csv", index=False
    )
    delta_pareado(ratios, presupuesto="ratio").to_csv(
        PROCESSED / "malla_ratios_finos_delta_pareado.csv", index=False
    )

    curves.to_csv(PROCESSED / "malla_curvas.csv", index=False)
    resumir(curves).to_csv(PROCESSED / "malla_curvas_resumen.csv", index=False)
    delta_pareado(curves, presupuesto="n_synth").to_csv(
        PROCESSED / "malla_curvas_delta_pareado.csv", index=False
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        SOURCE / "figures/curvas_error_por_generador.png",
        FIGURES / "27_curvas_error_por_generador.png",
    )
    shutil.copy2(
        SOURCE / "figures/curvas_error_todos.png",
        FIGURES / "28_curvas_error_todos.png",
    )
    print("Artefactos R61 promovidos: 333 celdas de ratios y 465 de curvas")


if __name__ == "__main__":
    main()

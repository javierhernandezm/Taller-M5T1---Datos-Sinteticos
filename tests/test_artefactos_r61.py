"""
test_artefactos_r61.py — Coherencia de los artefactos canónicos R61.

No recalcula nada del experimento: comprueba que los CSV versionados que
sostienen los notebooks 03/04 y `reports/diffusion_ts_r61/` cuentan la misma
historia. Son las invariantes que ya se rompieron una vez:

  * la auditoría del informe y la canónica del notebook 03 divergieron porque
    el notebook se ejecutó DESPUÉS del runner y solo reescribió `data/processed`;
  * un `wgan_gp` heredado puede reaparecer en una tabla activa sin que nada
    falle, porque el pipeline es tolerante a generadores de más.

Los tests se saltan solos en un clon sin los artefactos versionados.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.malla import GENERADORES_ACTIVOS

PROCESSED = REPO_ROOT / "data/processed"
R61 = REPO_ROOT / "reports/diffusion_ts_r61"

#: (canónico en data/processed, copia en el informe R61)
PAREJAS = [
    ("auditoria_nb03.csv", "nb03/auditoria.csv"),
    ("tstr_nb03.csv", "nb03/tstr.csv"),
    ("tstr_nb03_resumen.csv", "nb03/tstr_resumen.csv"),
    ("malla_ratios_finos.csv", "mallas/malla_ratios_finos.csv"),
    ("malla_curvas.csv", "mallas/malla_curvas.csv"),
]


def _par(canonico: str, informe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = PROCESSED / canonico, R61 / informe
    if not a.exists() or not b.exists():
        pytest.skip(f"faltan artefactos versionados ({canonico})")
    index_col = 0 if canonico.startswith("auditoria") else None
    return pd.read_csv(a, index_col=index_col), pd.read_csv(b, index_col=index_col)


@pytest.mark.parametrize(("canonico", "informe"), PAREJAS)
def test_el_informe_r61_y_el_canonico_no_divergen(canonico, informe):
    """El runner escribe ambas copias; si divergen, una de las dos miente."""
    esperado, obtenido = _par(canonico, informe)
    pd.testing.assert_frame_equal(esperado, obtenido, check_dtype=False)


def test_la_auditoria_cubre_exactamente_los_generadores_activos():
    """Ni un wgan_gp heredado de más, ni un generador activo de menos."""
    aud, _ = _par("auditoria_nb03.csv", "nb03/auditoria.csv")
    assert list(aud.index) == list(GENERADORES_ACTIVOS)


@pytest.mark.parametrize(
    ("archivo", "filas", "celdas_diffusion"),
    [("malla_ratios_finos.csv", 333, 54), ("malla_curvas.csv", 465, 75)],
)
def test_las_mallas_canonicas_conservan_su_forma(archivo, filas, celdas_diffusion):
    ruta = PROCESSED / archivo
    if not ruta.exists():
        pytest.skip(f"falta {archivo}")
    malla = pd.read_csv(ruta)
    assert len(malla) == filas
    assert set(malla.generador) == {*GENERADORES_ACTIVOS, "ninguno"}
    assert int((malla.generador == "diffusion_ts").sum()) == celdas_diffusion
    # tres semillas por celda: es lo que hace comparables los deltas pareados
    assert (malla.groupby(["n_real", "generador", "n_synth"]).seed.nunique() == 3).all()
    assert not malla.duplicated(["n_real", "generador", "n_synth", "seed"]).any()


def test_el_tstr_oficial_tiene_los_21_brazos_activos():
    ruta = PROCESSED / "tstr_nb03.csv"
    if not ruta.exists():
        pytest.skip("falta tstr_nb03.csv")
    tstr = pd.read_csv(ruta)
    assert len(tstr) == 21
    assert set(tstr.brazo) == {*GENERADORES_ACTIVOS, "real"}
    assert (tstr.groupby("brazo").seed.nunique() == 3).all()


def test_el_manifiesto_describe_la_receta_r61():
    ruta = PROCESSED / "tstr_nb03_manifest.json"
    if not ruta.exists():
        pytest.skip("falta tstr_nb03_manifest.json")
    import json

    manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    assert manifiesto["complete"] is True
    assert manifiesto["rows"] == 21
    assert manifiesto["representation"].startswith("R61")
    assert manifiesto["train_steps"] == 3_000
    assert manifiesto["sample_steps"] == 50
    assert manifiesto["fit_seed"] == 42
    assert len(manifiesto["checkpoint_sha1"]) == 40

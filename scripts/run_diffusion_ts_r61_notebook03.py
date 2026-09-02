"""Regenera la auditoria canonica R61 ejecutando el notebook 03.

Quinto y ultimo paso de la reproduccion. Los cuatro runners anteriores calculan
Diffusion-TS y promueven las mallas, pero NINGUNO produce la tabla de auditoria
publicada: `run_diffusion_ts_r61_nb03.py` solo recalcula su propia fila y hereda
las otras cinco de la ejecucion previa. La tabla de fidelidad del informe sale
de aqui, del notebook, que es el unico sitio donde los seis generadores se
ajustan en una misma corrida y por tanto son comparables entre si.

Coste
-----
El notebook reajusta los seis generadores (VAE, Diffusion-TS y RealNVP son los
caros) pero NO repite los 21 entrenamientos downstream: reutiliza el TSTR
cacheado si el manifiesto que dejo el paso 1 sigue siendo valido. En una GPU
son minutos, no horas.

Espejo del informe
------------------
El notebook escribe en `data/processed/`. `reports/diffusion_ts_r61/nb03/` debe
reflejar exactamente lo mismo: son dos copias del mismo hecho, y ya divergieron
una vez porque el notebook se ejecuto despues del runner y solo reescribio el
canonico. Este script las vuelve a igualar y verifica las invariantes.

Aviso: reescribe las cifras publicadas
--------------------------------------
El ajuste en GPU no es reproducible bit a bit. El runner y el notebook ajustan
Diffusion-TS con la MISMA semilla 42, los mismos datos y la misma receta de
auditoria, y aun asi sus filas difieren (sd_x 0,6919 frente a 0,6925). Ejecutar
este paso sobre un arbol ya publicado regenera la tabla de fidelidad con valores
ligeramente distintos a los que citan FINAL_CONCLUSION.md, el README y la
narrativa del notebook. En una reproduccion desde cero es lo correcto; sobre los
resultados ya versionados, revisa el diff antes de confirmarlo. `--solo-espejo`
re-sincroniza el informe sin tocar ningun numero.

Uso
---
    uv run python scripts/run_diffusion_ts_r61_notebook03.py
    uv run python scripts/run_diffusion_ts_r61_notebook03.py --solo-espejo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_all import run

from src.malla import GENERADORES_ACTIVOS

NOTEBOOK = REPO_ROOT / "notebooks" / "03_generadores.ipynb"
PROCESSED = REPO_ROOT / "data/processed"
OUTPUT = REPO_ROOT / "reports/diffusion_ts_r61/nb03"

#: canonico en data/processed -> copia en el informe R61
ESPEJO = {
    "auditoria_nb03.csv": "auditoria.csv",
    "tstr_nb03.csv": "tstr.csv",
    "tstr_nb03_resumen.csv": "tstr_resumen.csv",
    "tstr_nb03_manifest.json": "manifest.json",
}


def validar() -> None:
    """Comprueba lo que el notebook debe haber dejado antes de espejarlo."""
    aud = pd.read_csv(PROCESSED / "auditoria_nb03.csv", index_col=0)
    if list(aud.index) != list(GENERADORES_ACTIVOS):
        raise RuntimeError(
            f"La auditoria no cubre los generadores activos: {list(aud.index)}"
        )
    if aud.isna().any().any():
        raise RuntimeError("La auditoria contiene valores ausentes")

    tstr = pd.read_csv(PROCESSED / "tstr_nb03.csv")
    if len(tstr) != 21 or set(tstr.brazo) != {*GENERADORES_ACTIVOS, "real"}:
        raise RuntimeError(
            f"TSTR inesperado: {len(tstr)} filas, brazos={sorted(set(tstr.brazo))}"
        )

    manifiesto = json.loads(
        (PROCESSED / "tstr_nb03_manifest.json").read_text(encoding="utf-8")
    )
    if not manifiesto.get("complete") or manifiesto.get("rows") != 21:
        raise RuntimeError("El manifiesto TSTR no describe una corrida completa")


def espejar() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for canonico, copia in ESPEJO.items():
        origen = PROCESSED / canonico
        if not origen.exists():
            raise RuntimeError(f"Falta el artefacto canonico {canonico}")
        shutil.copy2(origen, OUTPUT / copia)
        print(f"  = {canonico} -> reports/diffusion_ts_r61/nb03/{copia}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--solo-espejo",
        action="store_true",
        help="no ejecuta el notebook; solo re-sincroniza el informe R61",
    )
    ap.add_argument(
        "--timeout", type=int, default=43_200, help="segundos por celda (43200)"
    )
    args = ap.parse_args()

    if not args.solo_espejo:
        print(f"Ejecutando {NOTEBOOK.name} (reajusta los seis generadores)")
        if run(NOTEBOOK, args.timeout):
            print("El notebook 03 termino con celdas en error; no se espeja nada.")
            return 1

    validar()
    espejar()
    print("\nAuditoria canonica regenerada y espejada en el informe R61.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

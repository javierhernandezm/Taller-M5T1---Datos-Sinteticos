"""
make_arch_figures.py — Regenera los diagramas de arquitectura (PlotNeuralNet).

Produce las figuras 16 a 23 de reports/figures/: las cuatro candidatas
downstream del notebook 02, los tres generadores neuronales del 03 y el
pipeline de datos del taller. Los diagramas se derivan por introspección de los
`nn.Module` reales (ver src/netviz.py), así que basta cambiar una arquitectura
en src/ y volver a lanzar esto para que las figuras se pongan al día.

Uso
---
    uv run python scripts/make_arch_figures.py            # solo lo que cambió
    uv run python scripts/make_arch_figures.py --force    # recompila todo
    uv run python scripts/make_arch_figures.py --list     # ver qué genera
    uv run python scripts/make_arch_figures.py 19 23      # solo esas figuras

Notas
-----
* LaTeX con `pdflatex` (MiKTeX en Windows, TeX Live en Linux/macOS) y
  `pdftoppm` producen la versión vectorial. Son dependencias OPCIONALES: sin
  ellas se reutilizan los PNG versionados y, para una figura nueva sin caché,
  `netviz.render` genera un PNG 2-D equivalente con Matplotlib.
* El `.tex` de cada figura es la clave de caché: si el generado coincide con el
  versionado y el PNG existe, no se recompila nada. `--force` salta el caché.
* Los .tex y .pdf intermedios quedan en reports/figures/tex/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.netviz import diagramas_taller, render


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "prefijos",
        nargs="*",
        help="prefijos numéricos de figura (p. ej. 19 23); vacío = todas",
    )
    ap.add_argument(
        "--force", action="store_true", help="recompila aunque el .tex no haya cambiado"
    )
    ap.add_argument("--list", action="store_true", help="lista las figuras y sale")
    ap.add_argument("--dpi", type=int, default=300, help="resolución del PNG (300)")
    args = ap.parse_args()

    cfg = Config()
    cfg.ensure_dirs()
    todos = diagramas_taller(cfg)

    if args.list:
        for d in todos:
            print(f"{d.nombre:22s}  {d.titulo}")
        return 0

    objetivo = [d for d in todos if not args.prefijos or d.nombre[:2] in args.prefijos]
    if not objetivo:
        print(
            f"Ninguna figura coincide con {args.prefijos}. Disponibles: "
            f"{[d.nombre[:2] for d in todos]}"
        )
        return 1

    if shutil.which("pdflatex") is None:
        print(
            "AVISO: no hay pdflatex en el PATH. Se reutilizarán los PNG "
            "versionados y las figuras nuevas usarán el fallback Matplotlib; "
            "instala MiKTeX/TeX Live para obtener también PDF vectorial."
        )

    print(f"Generando {len(objetivo)} diagrama(s) en {cfg.fig_dir}")
    fallos = 0
    for d in objetivo:
        try:
            render(d, fig_dir=cfg.fig_dir, force=args.force, dpi=args.dpi)
        except Exception as e:  # noqa: BLE001 - se informa y se sigue
            fallos += 1
            print(f"  X {d.nombre}: {e}")

    print(
        "\nTerminado."
        if not fallos
        else f"\nTerminado con {fallos} figura(s) en error."
    )
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())

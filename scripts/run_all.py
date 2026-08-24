"""
run_all.py — Ejecuta los notebooks del taller de principio a fin.

Punto de entrada de reproducibilidad: el enunciado exige que "el código genere
todas las gráficas y tablas reportadas". Este script lo hace sin intervención
manual, dejando cada notebook con sus salidas ya calculadas y las figuras
escritas en reports/figures/.

Uso
---
    uv run python scripts/run_all.py              # todos los notebooks
    uv run python scripts/run_all.py 02 03        # solo los indicados
    python scripts/run_all.py --list              # ver qué hay

    # con pip en lugar de uv:
    python scripts/run_all.py

Notas
-----
* El notebook 01 necesita los datos crudos del TFM (ver README). Los demás
  parten de data/processed/, que va versionado: en un clon limpio,
  `python scripts/run_all.py 02 03` funciona sin acceso al Drive privado.
* La ejecución es secuencial y puede tardar bastante en CPU (el notebook 03
  entrena tres redes generativas). En GPU es uno o dos órdenes de magnitud
  más rápido.
* Cada notebook se sobrescribe con sus salidas. Es deliberado: los .ipynb del
  repo son el entregable y deben mostrar las curvas de loss.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"


def notebooks() -> list[Path]:
    """Notebooks del repo, en orden de dependencia (por prefijo numérico)."""
    return sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))


def run(path: Path, timeout: int) -> int:
    """Ejecuta un notebook in-place. Devuelve el número de celdas con error."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    # cwd = carpeta del notebook, para que las rutas relativas se comporten
    # igual que al abrirlo en JupyterLab.
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbformat.write(nb, path)

    errores = [o for c in nb.cells if c.cell_type == "code"
               for o in c.get("outputs", []) if o.get("output_type") == "error"]
    estado = "OK" if not errores else f"{len(errores)} ERRORES"
    print(f"  -> {path.name}: {estado} ({time.time() - t0:.0f}s)")
    for e in errores[:3]:
        print(f"     {e.get('ename')}: {e.get('evalue')}")
    return len(errores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("prefijos", nargs="*",
                    help="prefijos numéricos a ejecutar (p. ej. 02 03); vacío = todos")
    ap.add_argument("--timeout", type=int, default=7200,
                    help="segundos máximos por celda (por defecto 7200)")
    ap.add_argument("--list", action="store_true", help="listar notebooks y salir")
    args = ap.parse_args()

    todos = notebooks()
    if args.list:
        for p in todos:
            print(p.name)
        return 0

    objetivo = [p for p in todos if not args.prefijos or p.name[:2] in args.prefijos]
    if not objetivo:
        print(f"Ningún notebook coincide con {args.prefijos}. Disponibles: "
              f"{[p.name for p in todos]}")
        return 1

    print(f"Ejecutando {len(objetivo)} notebook(s) desde {NB_DIR}")
    fallos = 0
    for p in objetivo:
        print(f"[{p.name}] ...")
        fallos += run(p, args.timeout)
    print("\nTerminado." if not fallos else f"\nTerminado con {fallos} celdas en error.")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())

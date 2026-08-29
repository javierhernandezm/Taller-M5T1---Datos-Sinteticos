"""
test_netviz.py — Pruebas de los diagramas de arquitectura.

No compilan LaTeX: comprueban el `.tex` que produce `src/netviz.py`, que es la
parte que puede romperse en silencio. Corren en cualquier máquina, con o sin
MiKTeX, y por eso valen como red de seguridad en un clon limpio.

Lo que se protege aquí:
  * que el walker deduzca las MISMAS formas que un forward de verdad — es la
    garantía de que la figura no puede desviarse del código;
  * que el TikZ sea sintácticamente coherente (nodos únicos, sin flechas a
    nodos inexistentes, documento cerrado);
  * que el escapado de LaTeX no vuelva a dejar pasar los dos casos que ya
    reventaron una compilación: el guion bajo y el `[` tras un salto de línea.

Uso
---
    uv run pytest tests/ -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import Config
from src.models import build_model
from src.netviz import (
    CANDIDATAS, Bloque, Diagrama, _escape, bloques_desde_modelo, diagramas_taller,
    tikz_source,
)

CFG = Config()


@pytest.fixture(scope="module")
def diagramas():
    return diagramas_taller(CFG)


# --------------------------------------------------------------------------- #
# El walker frente a la realidad
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("clave", list(CANDIDATAS))
def test_walker_coincide_con_el_forward(clave):
    """Las formas deducidas deben ser las que produce el modelo de verdad.

    Es la prueba que da sentido a todo el módulo: si el walker se equivoca, la
    figura miente sobre la arquitectura congelada.
    """
    arch, kwargs = CANDIDATAS[clave]
    modelo = (build_model(arch, in_len=CFG.window_len, **kwargs) if arch == "mlp"
              else build_model(arch, **kwargs))
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.zeros(2, CFG.window_len))
    assert salida.shape == (2,)

    bloques = bloques_desde_modelo(modelo, CFG.window_len, prefijo=clave)
    assert bloques[0].kind == "dato"
    assert bloques[-1].kind == "salida"
    # la última capa lineal del modelo fija la anchura de la caja de salida
    ultima = [m for m in modelo.modules() if isinstance(m, nn.Linear)][-1]
    assert bloques[-1].canales == ultima.out_features == 1


def test_cnn_deduce_la_cadena_de_longitudes():
    """60 -> 30 -> 15 -> 7 debe salir de la aritmética, no de una constante."""
    modelo = build_model("cnn", channels=(32, 64, 128), fc=128)
    bloques = bloques_desde_modelo(modelo, 60, prefijo="c")
    pools = [b.largo for b in bloques if b.kind == "pool"]
    assert pools == [30, 15, 7]
    convs = [b.canales for b in bloques if b.kind == "conv"]
    assert convs == [32, 64, 128]


def test_walker_rechaza_capas_desconocidas():
    """Mejor fallar que dibujar una capa que el walker no entiende."""
    modelo = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
    with pytest.raises(TypeError, match="BatchNorm1d"):
        bloques_desde_modelo(modelo, 4)


# --------------------------------------------------------------------------- #
# Coherencia del TikZ generado
# --------------------------------------------------------------------------- #

def test_se_generan_las_ocho_figuras(diagramas):
    assert [d.nombre for d in diagramas] == [
        "16_arq_mlp_s", "17_arq_mlp_l", "18_arq_cnn_s", "19_arq_cnn_l",
        "20_arq_vae", "21_arq_wgan_gp", "22_arq_realnvp", "23_pipeline_datos",
    ]


def test_documento_bien_formado(diagramas):
    for d in diagramas:
        tex = tikz_source(d)
        assert tex.count(r"\begin{document}") == 1
        assert tex.count(r"\end{document}") == 1
        assert tex.count(r"\begin{tikzpicture}") == tex.count(r"\end{tikzpicture}") == 1


def test_nombres_de_nodo_unicos(diagramas):
    for d in diagramas:
        nombres = [b.name for b in d.bloques]
        assert len(nombres) == len(set(nombres)), f"nodos repetidos en {d.nombre}"
        assert all(n.isalnum() for n in nombres), f"nodo no alfanumérico en {d.nombre}"


def test_las_flechas_apuntan_a_nodos_existentes(diagramas):
    """Una flecha a un nodo inexistente es un error de compilación de LaTeX."""
    for d in diagramas:
        definidos = {b.name for b in d.bloques}
        tex = tikz_source(d)
        for origen, destino in re.findall(
                r"\\draw \[connection\] \((\w+)-east\) -- node \{\\midarrow\} \((\w+)-west\)",
                tex):
            assert origen in definidos, f"{d.nombre}: origen {origen} no existe"
            assert destino in definidos, f"{d.nombre}: destino {destino} no existe"


def test_todo_bloque_queda_conectado(diagramas):
    """Ningún bloque debe quedar suelto: sería un despiste de layout."""
    for d in diagramas:
        if not d.conexiones:
            continue        # cadena implícita: todos conectados por construcción
        tocados = {n for par in d.conexiones for n in par}
        sueltos = {b.name for b in d.bloques} - tocados
        assert not sueltos, f"{d.nombre}: bloques sin conectar {sueltos}"


def test_la_campeona_lleva_el_sello_de_congelada(diagramas):
    """El sello se deduce de downstream_reference.json, no está cableado."""
    ref = CFG.out_dir / "downstream_reference.json"
    if not ref.exists():
        pytest.skip("no hay downstream_reference.json en este clon")
    sellados = [d.nombre for d in diagramas if "CONGELADA" in d.subtitulo]
    assert len(sellados) == 1, f"se esperaba exactamente una campeona, hay {sellados}"


# --------------------------------------------------------------------------- #
# Escapado de LaTeX (regresiones ya vividas)
# --------------------------------------------------------------------------- #

def test_escape_protege_el_guion_bajo():
    assert _escape("mlp_s") == r"mlp\_s"


def test_escape_neutraliza_el_corchete_inicial():
    """`\\[X | y]` se lee como `\\\\[dimensión]` y rompe la compilación."""
    assert _escape("[X | y]").startswith("{}")


def test_ningun_rotulo_deja_un_corchete_tras_un_salto(diagramas):
    for d in diagramas:
        assert r"\\ [" not in tikz_source(d), f"{d.nombre}: corchete tras \\\\"


def test_los_rotulos_de_los_bloques_van_escapados(diagramas):
    """Un guion bajo sin escapar en un pie tumbaría pdflatex."""
    for d in diagramas:
        tex = tikz_source(d)
        for linea in tex.splitlines():
            if linea.startswith(r"\node[anchor=north"):
                assert not re.search(r"(?<!\\)_", linea), f"{d.nombre}: {linea[:80]}"


def test_bloque_sin_pie_no_emite_rotulo():
    tex = tikz_source(Diagrama(
        nombre="x", titulo="t", bloques=[Bloque(kind="densa", name="a", canales=4)]))
    assert r"\node[anchor=north" not in tex


# --------------------------------------------------------------------------- #
# El espejo de los hiperparámetros del notebook 03
# --------------------------------------------------------------------------- #

def _generadores_del_notebook_03() -> dict[str, dict]:
    """Extrae el literal `GENERADORES = {...}` del notebook 03 con `ast`.

    Leer el notebook en vez de fiarse de una copia es lo que convierte
    `GENERADORES_03` en un espejo comprobable y no en otra fuente de verdad.
    """
    import ast
    import json

    nb = json.loads((REPO_ROOT / "notebooks" / "03_generadores.ipynb")
                    .read_text(encoding="utf-8"))
    fuente = next(("".join(c["source"]) for c in nb["cells"]
                   if c["cell_type"] == "code" and "GENERADORES = {" in "".join(c["source"])),
                  None)
    assert fuente is not None, "no se encontró GENERADORES en el notebook 03"

    arbol = ast.parse(fuente)
    asignacion = next(n for n in ast.walk(arbol)
                      if isinstance(n, ast.Assign)
                      and getattr(n.targets[0], "id", None) == "GENERADORES")
    salida: dict[str, dict] = {}
    for clave, llamada in zip(asignacion.value.keys, asignacion.value.values):
        salida[clave.value] = {kw.arg: ast.literal_eval(kw.value)
                               for kw in llamada.keywords}
    return salida


def test_los_hiperparametros_dibujados_son_los_del_notebook_03():
    """La figura debe describir lo que el notebook entrena, no los defaults.

    El notebook usa RealNVP con 6 capas y hidden 128, frente al (8, 256) por
    defecto de la clase: dibujar los defaults inventaría dos capas de
    acoplamiento inexistentes.
    """
    from src.netviz import GENERADORES_03

    del_notebook = _generadores_del_notebook_03()
    for nombre, kwargs in GENERADORES_03.items():
        assert nombre in del_notebook, f"{nombre} ya no está en el notebook 03"
        assert kwargs == del_notebook[nombre], (
            f"{nombre}: netviz.GENERADORES_03 dice {kwargs} pero el notebook 03 "
            f"construye {del_notebook[nombre]}")


def test_realnvp_dibuja_las_capas_que_se_entrenan(diagramas):
    from src.netviz import GENERADORES_03

    flow = next(d for d in diagramas if d.nombre == "22_arq_realnvp")
    acoplamientos = [b for b in flow.bloques if b.name.startswith("rc")]
    assert len(acoplamientos) == GENERADORES_03["realnvp"]["n_layers"]

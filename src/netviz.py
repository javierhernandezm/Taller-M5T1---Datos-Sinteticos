"""
netviz.py — Diagramas de arquitectura (PlotNeuralNet) derivados del código.

Genera las fichas visuales de las redes del taller con
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) (MIT, vendorizado
en `vendor/PlotNeuralNet/`): Python emite TikZ, `pdflatex` lo compila a PDF y
`pdftoppm` lo rasteriza a PNG.

Principio de diseño
-------------------
Los diagramas se DERIVAN POR INTROSPECCIÓN de los `nn.Module` reales; no se
teclean a mano. En un repo cuya tesis es "la arquitectura queda congelada y solo
cambian los datos", una figura capaz de desviarse en silencio del código sería
una mentira documental. Cada recuento de canales, de unidades y de longitud de
secuencia sale de recorrer el modelo instanciado, y las longitudes 60->30->15->7
se CALCULAN con la aritmética real de la convolución y el pooling.

El walker (`bloques_desde_modelo`) cubre `MLP`, `ConvNet1D` y todo lo que
devuelve `generators._mlp()`, porque los tres son cadenas de `nn.Sequential`.
Las topologías con ramas (las dos cabezas mu/logvar del VAE, la descomposición
de Diffusion-TS y el acoplamiento de RealNVP) llevan un layout escrito a mano,
pero sus CIFRAS siguen saliendo del módulo vivo.

Degradación sin LaTeX
---------------------
El `.tex` se escribe siempre (es Python puro). La compilación solo ocurre si el
`.tex` cambió respecto al versionado o si falta el PNG. Si no hay `pdflatex`,
se avisa y se reutiliza el PNG cacheado en `reports/figures/`; para una figura
nueva sin caché se dibuja una ficha 2-D equivalente con Matplotlib. Así el repo
sigue siendo legible para quien no tenga MiKTeX/TeX Live, y `scripts/run_all.py`
no se rompe en headless. `render()` nunca lanza por falta de LaTeX.

Uso
---
    from src.netviz import diagramas_taller, render
    for diag in diagramas_taller():
        render(diag, fig_dir=cfg.fig_dir)

o, desde la línea de órdenes, `scripts/make_arch_figures.py`.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path

from torch import nn

from .config import Config
from .eda import PALETTE
from .models import build_model, count_params

#: Raíz del repo (src/..), para localizar `vendor/PlotNeuralNet/layers/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _REPO_ROOT / "vendor" / "PlotNeuralNet"
_LAYERS = _VENDOR / "layers"


# =========================================================================== #
# 1. Representación intermedia
# =========================================================================== #


@dataclass(frozen=True)
class Bloque:
    """Una caja del diagrama.

    `kind` decide color y geometría; `etiqueta` es el número que va impreso en
    la cara frontal (canales o unidades) y `pie` el rótulo de varias líneas que
    se dibuja debajo. La geometría NO se guarda aquí: la calcula `_geometria`
    a partir de `canales` y `largo`, para que todas las figuras del taller
    compartan la misma escala y se lean como una familia.
    """

    kind: str  # dato|conv|pool|gap|densa|salida|latente|ruido
    name: str  # nodo TikZ (solo [A-Za-z0-9])
    etiqueta: str = ""  # xlabel: canales o unidades
    largo: int = 1  # longitud de la secuencia (eje z)
    canales: int = 1  # canales o unidades (eje x)
    pie: tuple[str, ...] = ()  # rótulo bajo la caja, una entrada por línea
    activado: bool = False  # dibuja la franja de activación (RightBandedBox)
    #: colocación explícita; None = encadenar tras el bloque anterior
    at: str | None = None
    offset: str | None = None


@dataclass
class Diagrama:
    """Un diagrama completo: bloques, flechas y títulos."""

    nombre: str  # p.ej. "16_arq_mlp_s"
    titulo: str
    subtitulo: str = ""
    bloques: list[Bloque] = field(default_factory=list)
    #: flechas explícitas (origen, destino); vacío = encadenar consecutivos
    conexiones: list[tuple[str, str]] = field(default_factory=list)
    #: TikZ extra inyectado antes de \end{tikzpicture} (llaves, notas)
    extra: str = ""


# =========================================================================== #
# 2. Estilo: colores y geometría
# =========================================================================== #

#: Paleta Okabe-Ito de `eda.PALETTE`, para que los diagramas casen con las
#: figuras de datos y resultados ya versionadas.
_COLORES: dict[str, tuple[str, str]] = {
    #  kind      relleno                banda de activación (tono claro del mismo)
    "dato": (PALETTE["sky"], "#A6DAF5"),
    "conv": (PALETTE["blue"], PALETTE["sky"]),
    "pool": (PALETTE["grey"], "#B0B0B0"),
    "gap": (PALETTE["purple"], "#E4B4CF"),
    "densa": (PALETTE["green"], "#66C7AC"),
    "salida": (PALETTE["vermillion"], "#F0956B"),
    "latente": (PALETTE["orange"], "#F3C86B"),
    "ruido": (PALETTE["yellow"], "#F7F0A0"),
    # Okabe-Ito tiene 8 tonos y los 8 anteriores ya estan asignados. El bloque
    # recurrente usa un tono OSCURO del azul de "conv" en lugar de un color nuevo:
    # conv y GRU son la misma cosa en la lectura de la figura -el tronco que
    # recorre la secuencia- y el tono los separa sin salirse de la paleta.
    "recurrente": ("#004E7C", PALETTE["sky"]),
}


def _geometria(b: Bloque) -> tuple[float, float, float]:
    """(width, height, depth) de una caja, en unidades sin escalar.

    Todo en este taller es 1-D (una ventana de 60 retornos, un vector de 61),
    así que todas las cajas son losas planas de altura fija: el grosor codifica
    canales/unidades y el fondo codifica la longitud.

    El fondo usa una potencia 0,45 y no un logaritmo: en log, 64 y 128 unidades
    salen con cajas indistinguibles (22,7 frente a 21,3) y la figura deja de
    informar. La potencia comprime lo justo para que 512 no aplaste a 16 pero
    una duplicación siga siendo visible.
    """
    if b.kind == "pool":
        width = 1.0
    elif b.kind in ("conv", "recurrente"):
        width = round(1.0 + 0.9 * math.log2(max(b.canales, 1)), 2)
    else:
        width = 1.6
    # fondo: longitud de secuencia (conv) o nº de unidades (vectores)
    n = b.largo if b.kind in ("conv", "pool", "dato", "recurrente") else b.canales
    depth = round(4.0 + 26.0 * (max(n, 1) / 512.0) ** 0.45, 2)
    return width, 3.0, max(depth, 3.0)


def _escape(s: str) -> str:
    """Escapa lo que rompería LaTeX en un rótulo (guiones bajos, %, &, ...)."""
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(a, b)
    # Tras un `\\`, LaTeX lee un `[` inicial como el opcional de `\\[dim]` y la
    # compilación revienta ("Illegal unit of measure"). Un `{}` delante es
    # invisible en la salida y desactiva esa lectura.
    return "{}" + s if s.startswith("[") else s


def _nodo(s: str) -> str:
    """Nombre de nodo TikZ seguro: solo letras y dígitos."""
    return "".join(c for c in s if c.isalnum()) or "n"


# =========================================================================== #
# 3. Emisores propios sobre PlotNeuralNet
# =========================================================================== #
# Upstream es CNN-first: no existe `to_FullyConnected`, aunque `to_cor()` define
# \FcColor y \FcReluColor sin usarlos nunca. Además sus cajas traen `caption`
# con `text width=15*width`, que para una losa fina son ~24pt: cualquier rótulo
# se parte en confeti. Por eso aquí las cajas van SIN caption y el rótulo se
# dibuja como nodo propio, con ancho y tipografía bajo control.
#
# Restricción verificada contra vendor/PlotNeuralNet/layers/: la clave
# /boxblock/ de Box.sty NO admite `bandfill` ni `zlabelposition` (solo
# RightBandedBox, bajo /block/, tiene `bandfill`/`bandopacity`). El snippet de
# `to_FullyConnected` que circula por los issues asume un Box.sty parcheado y
# no compila contra master. Estos emisores respetan la restricción.


def _pic_caja(b: Bloque) -> str:
    """Una caja: `RightBandedBox` si lleva activación, `Box` si no."""
    width, height, depth = _geometria(b)
    at = b.at if b.at is not None else "(0,0,0)"
    offset = b.offset if b.offset is not None else "(0,0,0)"
    xlabel = '{{" ","dummy"}}'  # ver nota de arriba: la forma va en el pie
    comun = (
        f"        name={b.name},\n"
        f"        caption= ,\n"
        f"        xlabel={xlabel},\n"
        f"        zlabel= ,\n"
        f"        fill=c{b.kind},\n"
    )
    if b.activado:
        cuerpo = comun + (
            f"        bandfill=b{b.kind},\n"
            f"        opacity=0.75,\n"
            f"        bandopacity=0.9,\n"
        )
        tipo = "RightBandedBox"
    else:
        cuerpo = comun + "        opacity=0.75,\n"
        tipo = "Box"
    return (
        f"\n\\pic[shift={{{offset}}}] at {at}\n"
        f"    {{{tipo}={{\n{cuerpo}"
        f"        height={height},\n"
        f"        width={width},\n"
        f"        depth={depth}\n"
        f"        }}\n    }};\n"
    )


def _pie(b: Bloque) -> str:
    """Rótulo multilínea bajo la caja, con ancho fijo (no el de upstream).

    La primera línea es la forma del tensor y va en negrita: es el dato que se
    consulta de un vistazo. El fondo blanco semiopaco permite que una flecha
    cruce por detrás sin dejar el texto ilegible — inevitable en cuanto el
    diagrama tiene ramas.
    """
    if not b.pie:
        return ""
    lineas = [r"\textbf{" + _escape(b.pie[0]) + "}"]
    lineas += [_escape(l) for l in b.pie[1:]]
    texto = r" \\ ".join(lineas)
    return (
        f"\\node[anchor=north, align=center, font=\\scriptsize, "
        f"text width=2.2cm, inner sep=2pt, fill=white, fill opacity=0.82, "
        f"text opacity=1] "
        f"at ([yshift=-26pt] {b.name}-south) {{{texto}}};\n"
    )


def _colores_tex() -> str:
    """Define las macros de color a partir de la paleta Okabe-Ito del repo."""
    out = ["\n% Paleta Okabe-Ito, importada de src/eda.py PALETTE"]
    for kind, (relleno, banda) in _COLORES.items():
        out.append(f"\\definecolor{{c{kind}}}{{HTML}}{{{relleno.lstrip('#')}}}")
        out.append(f"\\definecolor{{b{kind}}}{{HTML}}{{{banda.lstrip('#')}}}")
    return "\n".join(out) + "\n"


def tikz_source(diag: Diagrama) -> str:
    """El `.tex` completo y autocontenido del diagrama.

    `\\subimport{./layers/}` es relativo a propósito: `render()` copia
    `vendor/PlotNeuralNet/layers/` junto al `.tex` en el directorio temporal de
    compilación. Meter ahí una ruta absoluta de Windows rompería en cuanto
    alguien clonase el repo en una carpeta con espacios.
    """
    partes = [
        "\\documentclass[border=10pt, multi, tikz]{standalone}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage{import}\n",
        "\\subimport{./layers/}{init}\n",
        "\\usetikzlibrary{positioning}\n",
        "\\usetikzlibrary{3d}\n",
        _colores_tex(),
        "\n\\begin{document}\n\\begin{tikzpicture}\n",
        (
            "\\tikzstyle{connection}=[ultra thick,every node/.style="
            "{sloped,allow upside down},draw=\\edgecolor,opacity=0.7]\n"
        ),
    ]

    # Orden de dibujo: cajas, flechas y AL FINAL los rótulos. En un diagrama con
    # ramas las flechas cruzan por fuerza la zona de algún rótulo; dibujando el
    # texto el último (y con fondo blanco semiopaco) gana el texto, que es lo
    # que hay que poder leer.
    previo: str | None = None
    pies: list[str] = []
    for b in diag.bloques:
        colocado = replace(
            b,
            at=b.at
            if b.at is not None
            else (f"({previo}-east)" if previo else "(0,0,0)"),
            offset=b.offset
            if b.offset is not None
            else ("(2.15,0,0)" if previo else "(0,0,0)"),
        )
        partes.append(_pic_caja(colocado))
        pies.append(_pie(colocado))
        previo = b.name

    conexiones = diag.conexiones or [
        (a.name, b.name) for a, b in zip(diag.bloques, diag.bloques[1:])
    ]
    for origen, destino in conexiones:
        partes.append(
            f"\\draw [connection] ({origen}-east) -- "
            f"node {{\\midarrow}} ({destino}-west);\n"
        )
    partes.extend(pies)

    if diag.extra:
        partes.append(diag.extra)

    # Título anclado al bounding box ya dibujado (arriba a la izquierda).
    rotulo = f"\\bfseries\\large {_escape(diag.titulo)}"
    if diag.subtitulo:
        rotulo += f" \\\\ \\mdseries\\small {_escape(diag.subtitulo)}"
    partes.append(
        "\\node[anchor=south west, align=left, inner sep=0pt] at "
        "([yshift=14pt] current bounding box.north west) "
        f"{{\\begin{{tabular}}{{@{{}}l@{{}}}} {rotulo} \\end{{tabular}}}};\n"
    )
    partes.append("\\end{tikzpicture}\n\\end{document}\n")
    return "".join(partes)


# =========================================================================== #
# 4. Introspección: de nn.Module a bloques
# =========================================================================== #


def bloques_desde_modelo(
    model: nn.Module, in_len: int, prefijo: str = "L"
) -> list[Bloque]:
    """Recorre un modelo secuencial y devuelve sus bloques, con formas reales.

    Cubre `MLP` (un único `nn.Sequential`) y `ConvNet1D` (`conv` + `head`), y
    por extensión cualquier cadena de `nn.Sequential` — incluidas las que
    fabrica `generators._mlp()`.

    Las activaciones y el dropout NO generan caja: se funden en el bloque
    anterior (franja bicolor y anotación `p=...`), que es como se leen de verdad
    esas capas. La forma se propaga con la aritmética real de cada módulo, así
    que 60->30->15->7 sale calculado y no escrito a mano.
    """
    hijos: list[nn.Module] = []
    for m in model.children():
        hijos.extend(m.children() if isinstance(m, nn.Sequential) else [m])
    if not hijos:
        raise ValueError(f"{type(model).__name__} no expone submódulos recorribles")

    bloques: list[Bloque] = []
    # Conv1d y GRU comparten forma de entrada: una secuencia de `in_len` pasos con
    # 1 canal/feature. El MLP, en cambio, recibe la ventana aplanada.
    es_mapa = isinstance(hijos[0], (nn.Conv1d, nn.GRU, nn.LSTM, nn.RNN))
    canales, largo = (1, in_len) if es_mapa else (in_len, 1)

    bloques.append(
        Bloque(
            kind="dato",
            name=_nodo(f"{prefijo}entrada"),
            etiqueta=str(canales if es_mapa else ""),
            largo=largo if es_mapa else canales,
            canales=canales,
            pie=(
                f"({canales}, {largo})" if es_mapa else f"{canales} valores",
                "entrada",
            ),
        )
    )

    for i, capa in enumerate(hijos):
        nombre = _nodo(f"{prefijo}{i}")
        if isinstance(capa, nn.Conv1d):
            largo = (
                largo
                + 2 * capa.padding[0]
                - capa.dilation[0] * (capa.kernel_size[0] - 1)
                - 1
            ) // capa.stride[0] + 1
            canales = capa.out_channels
            bloques.append(
                Bloque(
                    kind="conv",
                    name=nombre,
                    etiqueta=str(canales),
                    largo=largo,
                    canales=canales,
                    pie=(f"({canales}, {largo})", f"Conv1d k={capa.kernel_size[0]}"),
                )
            )
        elif isinstance(capa, (nn.MaxPool1d, nn.AvgPool1d)):
            k = (
                capa.kernel_size
                if isinstance(capa.kernel_size, int)
                else capa.kernel_size[0]
            )
            s = capa.stride if isinstance(capa.stride, int) else capa.stride[0]
            largo = (largo - k) // s + 1
            bloques.append(
                Bloque(
                    kind="pool",
                    name=nombre,
                    largo=largo,
                    canales=canales,
                    pie=(f"({canales}, {largo})", f"MaxPool {k}"),
                )
            )
        elif isinstance(capa, nn.AdaptiveAvgPool1d):
            largo = (
                capa.output_size
                if isinstance(capa.output_size, int)
                else capa.output_size[0]
            )
            bloques.append(
                Bloque(
                    kind="gap",
                    name=nombre,
                    largo=largo,
                    canales=canales,
                    pie=(f"({canales}, {largo})", "GlobalAvgPool"),
                )
            )
        elif isinstance(capa, (nn.GRU, nn.LSTM, nn.RNN)):
            # Una caja POR CAPA recurrente, no una sola por el modulo: apilar dos
            # GRU es una decision de arquitectura y tiene que verse, igual que se
            # ven los tres bloques conv de la cnn_l. Cada capa devuelve la
            # secuencia entera de estados ocultos, asi que la forma es (H, L): el
            # largo NO se reduce, que es justo lo que distingue a la recurrente de
            # la convolucional con pooling.
            tipo = type(capa).__name__
            canales = capa.hidden_size
            for j in range(capa.num_layers):
                pie = [
                    f"({canales}, {largo})",
                    f"{tipo} capa {j + 1}/{capa.num_layers}",
                ]
                # nn.GRU aplica su dropout ENTRE capas, no tras la ultima.
                if capa.dropout and j < capa.num_layers - 1:
                    pie.append(f"Dropout p={capa.dropout:g}")
                bloques.append(
                    Bloque(
                        kind="recurrente",
                        name=_nodo(f"{prefijo}{i}r{j}"),
                        etiqueta=str(canales),
                        largo=largo,
                        canales=canales,
                        pie=tuple(pie),
                    )
                )
            # El modelo se queda con el ultimo paso: (H, L) -> (H, 1). Merece caja
            # propia porque es donde la secuencia deja de serlo, y es el analogo
            # exacto del GlobalAvgPool de la CNN: mismo papel, distinto criterio
            # (el ultimo estado en vez de la media sobre el tiempo).
            largo = 1
            bloques.append(
                Bloque(
                    kind="gap",
                    name=_nodo(f"{prefijo}{i}ult"),
                    largo=largo,
                    canales=canales,
                    pie=(f"({canales}, 1)", "último estado", "h[:, -1]"),
                )
            )
        elif isinstance(capa, nn.Flatten):
            canales, largo = canales * largo, 1  # sin caja: solo contabilidad
        elif isinstance(capa, nn.Linear):
            canales, largo = capa.out_features, 1
            bloques.append(
                Bloque(
                    kind="densa",
                    name=nombre,
                    etiqueta=str(canales),
                    canales=canales,
                    pie=(
                        f"{canales} unidad" + ("es" if canales != 1 else ""),
                        f"Linear {capa.in_features}->{canales}",
                    ),
                )
            )
        elif isinstance(capa, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.GELU, nn.SiLU)):
            if bloques:
                acto = type(capa).__name__
                bloques[-1] = replace(
                    bloques[-1], activado=True, pie=bloques[-1].pie + (acto,)
                )
        elif isinstance(capa, nn.Dropout):
            if bloques:
                bloques[-1] = replace(
                    bloques[-1], pie=bloques[-1].pie + (f"Dropout p={capa.p:g}",)
                )
        else:
            raise TypeError(f"Capa no contemplada por el walker: {type(capa).__name__}")

    # La última caja es la salida de la red: color y rótulo propios.
    ultimo = bloques[-1]
    bloques[-1] = replace(
        ultimo,
        kind="salida",
        pie=(ultimo.pie[0], "salida") + tuple(l for l in ultimo.pie if "Linear" in l),
    )
    return bloques


# =========================================================================== #
# 5. Constructores de diagrama
# =========================================================================== #

#: Mismas candidatas que la sección 3 del notebook 02. Se declaran aquí para que
#: el script de figuras no dependa de ejecutar el notebook.
CANDIDATAS: dict[str, tuple[str, dict]] = {
    "mlp_s": ("mlp", {"hidden": (128, 64)}),
    "mlp_l": ("mlp", {"hidden": (256, 128, 64)}),
    "cnn_s": ("cnn", {"channels": (32, 64)}),
    "cnn_l": ("cnn", {"channels": (32, 64, 128), "fc": 128}),
    "gru_s": ("gru", {}),
    "gru_l": ("gru", {"hidden": 72, "layers": 3, "fc": 72}),
}


#: Nombre legible de cada familia, para el titulo de la ficha.
_FAMILIA = {"mlp": "MLP", "cnn": "CNN 1-D", "gru": "GRU (recurrente)"}


def _campeona(cfg: Config) -> tuple[str, dict] | None:
    """(arch, kwargs) de la arquitectura congelada, o None si aún no existe.

    Se lee de `downstream_reference.json` en lugar de cablearse: si algún día
    la campeona cambia, el sello "CONGELADA" se mueve de figura solo.
    """
    ref = cfg.out_dir / "downstream_reference.json"
    if not ref.exists():
        return None
    d = json.loads(ref.read_text(encoding="utf-8"))
    return d["arch"], d["arch_kwargs"]


def _misma_arquitectura(a: tuple[str, dict], b: tuple[str, dict] | None) -> bool:
    """Compara (arch, kwargs) tolerando listas frente a tuplas del JSON."""
    if b is None:
        return False
    norm = lambda kw: {
        k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in kw.items()
    }
    return a[0] == b[0] and norm(a[1]) == norm(b[1])


def diagrama_candidata(
    nombre_fig: str, clave: str, cfg: Config, in_len: int
) -> Diagrama:
    """Ficha de una de las seis candidatas del notebook 02."""
    arch, kwargs = CANDIDATAS[clave]
    model = (
        build_model(arch, in_len=in_len, **kwargs)
        if arch == "mlp"
        else build_model(arch, **kwargs)
    )
    congelada = _misma_arquitectura((arch, kwargs), _campeona(cfg))
    sub = f"{count_params(model):,} parámetros · entrada (B, {in_len}) · salida (B,)"
    if congelada:
        sub += " · CONGELADA: referencia de la malla real×sintético"
    return Diagrama(
        nombre=nombre_fig,
        titulo=f"{clave} — {_FAMILIA[arch]}",
        subtitulo=sub,
        bloques=bloques_desde_modelo(model, in_len, prefijo=clave),
    )


def diagrama_vae(gen, d: int) -> Diagrama:
    """VAE: encoder, cuello con dos cabezas mu/logvar, reparametrización, decoder."""
    from .generators import _VAENet

    net = _VAENet(d, gen.cfg["latent"], gen.cfg["hidden"])
    h, lat = gen.cfg["hidden"], gen.cfg["latent"]
    # el walker marca su última caja como "salida"; aquí es una capa oculta más
    enc = bloques_desde_modelo(nn.Sequential(net.enc), d, prefijo="ve")
    enc[-1] = replace(
        enc[-1],
        kind="densa",
        activado=True,
        pie=(f"{h} unidades", f"Linear {h}->{h}", "ReLU"),
    )
    dec = bloques_desde_modelo(nn.Sequential(net.dec), lat, prefijo="vd")[1:]

    mu = Bloque(
        kind="latente",
        name="vmu",
        etiqueta=str(lat),
        canales=lat,
        pie=(f"{lat} unidades", "mu", f"Linear {h}->{lat}"),
        at=f"({enc[-1].name}-east)",
        offset="(2.6,2.9,0)",
    )
    lv = Bloque(
        kind="latente",
        name="vlogvar",
        etiqueta=str(lat),
        canales=lat,
        pie=(f"{lat} unidades", "log var", f"Linear {h}->{lat}"),
        at=f"({enc[-1].name}-east)",
        offset="(2.6,-2.9,0)",
    )
    z = Bloque(
        kind="ruido",
        name="vz",
        etiqueta=str(lat),
        canales=lat,
        pie=(f"{lat} unidades", "muestreo de z", "reparametrización"),
        at="(vmu-east)",
        offset="(2.6,-2.9,0)",
    )
    dec = [replace(dec[0], at="(vz-east)", offset="(2.7,0,0)")] + dec[1:]
    dec[-1] = replace(dec[-1], kind="salida", pie=("reconstrucción", f"{d} valores"))

    bloques = enc + [mu, lv, z] + dec
    conexiones = [(a.name, b.name) for a, b in pairwise(enc)]
    conexiones += [
        (enc[-1].name, "vmu"),
        (enc[-1].name, "vlogvar"),
        ("vmu", "vz"),
        ("vlogvar", "vz"),
        ("vz", dec[0].name),
    ]
    conexiones += [(a.name, b.name) for a, b in pairwise(dec)]
    return Diagrama(
        nombre="20_arq_vae",
        titulo="VAE — familia latente variacional",
        subtitulo=(
            f"entrada [X | y] de {d} dims · latente {lat} · "
            "z = mu + sigma·eps · loss = MSE + beta·KL(q(z|x) || N(0,I))"
        ),
        bloques=bloques,
        conexiones=conexiones,
    )


def diagrama_diffusion_ts(gen, d: int) -> Diagrama:
    """Diffusion-TS R61: tokens, Transformer y cabezas de reconstrucción."""
    c = gen.cfg
    window_len = c["window_len"]
    d_model = c["d_model"]
    blocks = [
        Bloque(
            kind="ruido",
            name="dnoise",
            etiqueta=str(d),
            canales=d,
            pie=("R61 con ruido", "paso t", "[X60 | y]"),
        ),
        Bloque(
            kind="densa",
            name="dtokens",
            etiqueta=str(d_model),
            canales=d_model,
            activado=True,
            pie=(f"{window_len} retornos", "1 token target", f"d_model {d_model}"),
        ),
        Bloque(
            kind="conv",
            name="dtransformer",
            etiqueta=str(c["n_layers"]),
            canales=d_model,
            largo=window_len + 1,
            pie=(
                "Transformer bidireccional",
                f"{c['n_layers']} capas · {c['n_heads']} heads",
                "atención conjunta X-y",
            ),
        ),
        Bloque(
            kind="densa",
            name="ddecomp",
            etiqueta=str(window_len),
            canales=window_len,
            activado=True,
            pie=(
                "tendencia + Fourier",
                f"grado {c['trend_degree']} · top-{c['seasonal_k']}",
                "residuo temporal",
            ),
        ),
        Bloque(
            kind="salida",
            name="dout",
            etiqueta=str(d),
            canales=d,
            pie=("x0 reconstruido", f"{window_len} retornos + y", "DDIM"),
        ),
    ]
    return Diagrama(
        nombre="21_arq_diffusion_ts",
        titulo="Diffusion-TS R61 — familia de difusión",
        subtitulo=(
            f"{c['diffusion_steps']} pasos de difusión · {c['sample_steps']} pasos DDIM · "
            "loss L1 temporal + Fourier solo sobre X · y es un token especial"
        ),
        bloques=blocks,
        conexiones=[(a.name, b.name) for a, b in pairwise(blocks)],
    )


def diagrama_realnvp(gen, d: int) -> Diagrama:
    """RealNVP: la cadena de acoplamientos, más el zoom de uno solo."""
    n, h = gen.cfg["n_layers"], gen.cfg["hidden"]

    cadena = [
        Bloque(
            kind="dato",
            name="rx",
            etiqueta=str(d),
            canales=d,
            pie=("x = [X | y]", f"{d} valores"),
        )
    ]
    for i in range(n):
        cadena.append(
            Bloque(
                kind="densa",
                name=f"rc{i}",
                etiqueta=str(d),
                canales=d,
                activado=True,
                pie=(
                    f"acoplamiento {i + 1}",
                    f"máscara {'par' if i % 2 == 0 else 'impar'}",
                ),
            )
        )
    cadena.append(
        Bloque(
            kind="salida",
            name="rz",
            etiqueta=str(d),
            canales=d,
            pie=("z ~ N(0, I)", "verosimilitud exacta"),
        )
    )

    # Zoom: qué hay dentro de una capa de acoplamiento. Se ancla muy por debajo.
    interior = nn.Sequential(
        nn.Linear(d, h),
        nn.ReLU(),
        nn.Linear(h, h),
        nn.ReLU(),
        nn.Linear(h, d),
        nn.Tanh(),
    )
    s = bloques_desde_modelo(nn.Sequential(interior), d, prefijo="rs")
    s[0] = replace(
        s[0],
        kind="dato",
        pie=("x parte a", "mitad congelada"),
        at="(rx-east)",
        offset="(-1.6,-4.2,0)",
    )
    s[-1] = replace(s[-1], kind="salida", pie=("s y t", "escala y traslación"))

    extra = (
        "\\node[anchor=west, align=left, font=\\scriptsize] at "
        "([xshift=22pt, yshift=0pt] " + s[-1].name + "-east) "
        "{$y_a = x_a$ \\\\ $y_b = x_b \\cdot e^{s(x_a)} + t(x_a)$ \\\\[2pt] "
        "$\\log|\\det J| = \\sum s(x_a)$};\n"
    )
    conexiones = [(a.name, b.name) for a, b in pairwise(cadena)]
    conexiones += [(a.name, b.name) for a, b in pairwise(s)]
    return Diagrama(
        nombre="22_arq_realnvp",
        titulo="RealNVP — familia biyectiva",
        subtitulo=(
            f"{n} capas de acoplamiento afín · MLP interno {d}->{h}->{h}->{d} · "
            "entrenado por NLL exacta (abajo, el interior de una capa)"
        ),
        bloques=cadena + s,
        conexiones=conexiones,
        extra=extra,
    )


def diagrama_pipeline(cfg: Config) -> Diagrama:
    """El recorrido del dato: de los precios PIT al target, con la rama sintética."""
    camino = [
        Bloque(
            kind="dato",
            name="pprecios",
            canales=256,
            pie=("precios PIT", "S&P 1500", "2012–2026"),
        ),
        Bloque(
            kind="dato",
            name="pret",
            canales=180,
            pie=("log-retornos", "diarios", "huecos > 5d fuera"),
        ),
        Bloque(
            kind="conv",
            name="pvent",
            canales=64,
            largo=cfg.window_len,
            etiqueta=str(cfg.window_len),
            pie=(
                f"ventanas {cfg.window_len}d",
                f"stride {cfg.stride}",
                "split temporal",
            ),
        ),
        Bloque(
            kind="densa",
            name="pstd",
            canales=cfg.window_len,
            activado=False,
            etiqueta=str(cfg.window_len),
            pie=("estandarizar", "mu, sd de train", "(congelados)"),
        ),
        Bloque(
            kind="gap",
            name="pred",
            canales=96,
            offset="(5.4,0,0)",
            pie=("red congelada", "downstream", "reference.json"),
        ),
        Bloque(
            kind="salida",
            name="py",
            canales=2,
            etiqueta="1",
            pie=("y = ln sigma", f"{cfg.horizon}d adelante", "anualizada"),
        ),
    ]
    # Rama sintética: los generadores se entrenan con las ventanas reales de
    # train y devuelven ventanas nuevas a la mezcla. Es la tesis del taller.
    rama = [
        Bloque(
            kind="latente",
            name="pgen",
            canales=110,
            pie=("generador", "VAE / Diffusion-TS", "RealNVP"),
            at="(pstd-east)",
            offset="(1.2,-3.6,0)",
        ),
        Bloque(
            kind="ruido",
            name="psint",
            canales=110,
            pie=("ventanas", "sintéticas", "[X | y]"),
            at="(pgen-east)",
            offset="(2.2,0,0)",
        ),
    ]
    conexiones = [(a.name, b.name) for a, b in pairwise(camino)]
    conexiones += [("pstd", "pgen"), ("pgen", "psint"), ("psint", "pred")]
    return Diagrama(
        nombre="23_pipeline_datos",
        titulo="Pipeline del taller — del precio al target, con la rama sintética",
        subtitulo=(
            f"ventana {cfg.window_len}d -> horizonte {cfg.horizon}d · "
            f"anualización {cfg.ann_factor} · embargo {cfg.embargo_days}d · "
            "la red del centro es la MISMA en las 798 celdas de las dos mallas"
        ),
        bloques=camino + rama,
        conexiones=conexiones,
    )


#: Hiperparámetros con los que el notebook 03 construye los generadores. NO son
#: los valores por defecto de las clases: el notebook entrena el RealNVP con 6
#: capas y hidden 128, frente al (8, 256) por defecto. Dibujar los defectos daría
#: una figura con dos capas de acoplamiento que no existen — el tipo exacto de
#: mentira que este módulo existe para evitar. `tests/test_netviz.py` compara
#: este diccionario con el literal del propio notebook, así que si allí cambia
#: un hiperparámetro y aquí no, el test falla.
GENERADORES_03: dict[str, dict] = {
    "vae": {"latent": 16, "hidden": 256, "beta": 1.0, "epochs": 40},
    "diffusion_ts": {"train_steps": 3_000, "sample_steps": 50},
    "realnvp": {"n_layers": 6, "hidden": 128, "epochs": 30},
}


def diagramas_taller(
    cfg: Config | None = None, generadores: dict | None = None
) -> list[Diagrama]:
    """Los diez diagramas del taller, en el orden de numeración de figuras.

    `generadores` acepta el diccionario de instancias YA construidas del
    notebook 03 (`{"vae": VAEGenerator(...), ...}`), que es la forma de
    garantizar que la figura describe exactamente lo que se entrenó. Si no se
    pasa, se reconstruyen desde `GENERADORES_03`.
    """
    from .diffusion_ts import DiffusionTSR61Generator
    from .generators import RealNVPGenerator, VAEGenerator

    cfg = cfg or Config()
    in_len = cfg.window_len
    d = in_len + 1  # el par conjunto [X | y] que ven los generadores
    # Las dos GRU van al final de la numeracion (29, 30) y no a continuacion de
    # las otras candidatas: 20-28 ya estan ocupados por los generadores, el
    # pipeline y las figuras de auditoria de los notebooks 03 y 04. Meterlas en
    # bloque con 16-19 obligaria a renumerar media carpeta -y a tocar los cuatro
    # notebooks-, que es un cambio propio y no de esta rama. Queda como deuda.
    nombres = [
        "16_arq_mlp_s",
        "17_arq_mlp_l",
        "18_arq_cnn_s",
        "19_arq_cnn_l",
        "29_arq_gru_s",
        "30_arq_gru_l",
    ]
    assert len(nombres) == len(CANDIDATAS), "un nombre de figura por candidata"
    diags = [diagrama_candidata(n, k, cfg, in_len) for n, k in zip(nombres, CANDIDATAS)]

    gens = generadores or {}
    vae = gens.get("vae") or VAEGenerator(**GENERADORES_03["vae"])
    diffusion = gens.get("diffusion_ts") or DiffusionTSR61Generator(
        **GENERADORES_03["diffusion_ts"]
    )
    flow = gens.get("realnvp") or RealNVPGenerator(**GENERADORES_03["realnvp"])

    diags.append(diagrama_vae(vae, d))
    diags.append(diagrama_diffusion_ts(diffusion, d))
    diags.append(diagrama_realnvp(flow, d))
    diags.append(diagrama_pipeline(cfg))
    return diags


# =========================================================================== #
# 6. Compilación con caché y degradación
# =========================================================================== #


class LatexNoDisponible(RuntimeError):
    """No hay `pdflatex` en el PATH."""


def _render_fallback_matplotlib(diag: Diagrama, png_path: Path, dpi: int) -> None:
    """Dibuja una versión 2-D informativa cuando LaTeX no está disponible.

    No pretende imitar la perspectiva de PlotNeuralNet. Su función es que una
    arquitectura nueva nunca desaparezca del pipeline documental por depender
    de una instalación externa de TeX. Los textos y conexiones proceden de la
    misma representación :class:`Diagrama` que alimenta el TikZ.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    n = len(diag.bloques)
    fig, ax = plt.subplots(figsize=(max(10.0, 2.25 * n), 4.8))
    posiciones = {b.name: i for i, b in enumerate(diag.bloques)}

    for i, bloque in enumerate(diag.bloques):
        color, _ = _COLORES[bloque.kind]
        caja = FancyBboxPatch(
            (i - 0.42, -0.42),
            0.84,
            0.84,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.1,
        )
        ax.add_patch(caja)
        ax.text(
            i,
            0.04,
            bloque.etiqueta or bloque.kind,
            ha="center",
            va="center",
            color="white" if bloque.kind not in {"ruido"} else "#222222",
            fontsize=11,
            fontweight="bold",
        )
        if bloque.pie:
            ax.text(
                i,
                -0.58,
                "\n".join(bloque.pie),
                ha="center",
                va="top",
                fontsize=8,
                linespacing=1.3,
            )

    conexiones = diag.conexiones or [
        (a.name, b.name) for a, b in pairwise(diag.bloques)
    ]
    for origen, destino in conexiones:
        x0, x1 = posiciones[origen], posiciones[destino]
        ax.annotate(
            "",
            xy=(x1 - 0.46, 0),
            xytext=(x0 + 0.46, 0),
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.4},
        )

    ax.set_title(diag.titulo, fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.01,
        diag.subtitulo,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.set_xlim(-0.65, max(n - 0.35, 0.65))
    ax.set_ylim(-1.45, 0.65)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _compilar(tex: str, nombre: str, dpi: int) -> tuple[bytes, bytes]:
    """Compila el `.tex` en un temporal y devuelve (pdf, png) en memoria.

    Se copia `layers/` junto al `.tex` para que `\\subimport{./layers/}` resuelva
    sin rutas absolutas: así funciona igual en Windows, en Linux y en carpetas
    con espacios en el nombre.
    """
    if shutil.which("pdflatex") is None:
        raise LatexNoDisponible("pdflatex no está en el PATH")
    with tempfile.TemporaryDirectory(prefix="netviz_") as tmp:
        td = Path(tmp)
        shutil.copytree(_LAYERS, td / "layers")
        (td / f"{nombre}.tex").write_text(tex, encoding="utf-8", newline="\n")
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{nombre}.tex"],
            cwd=td,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
            check=False,  # el returncode se inspecciona abajo para dar un log útil
        )
        pdf = td / f"{nombre}.pdf"
        if r.returncode != 0 or not pdf.exists():
            # El log acaba en cientos de líneas de "LaTeX Font Info", así que
            # quedarse con la cola esconde el error: lo que importa son las
            # líneas que empiezan por "!" y su contexto inmediato.
            log = td / f"{nombre}.log"
            texto = log.read_text(errors="replace") if log.exists() else r.stdout
            lineas = texto.splitlines()
            fallos = [
                "\n".join(lineas[i : i + 6])
                for i, l in enumerate(lineas)
                if l.startswith("!")
            ]
            detalle = "\n\n".join(fallos[:3]) if fallos else texto[-2000:]
            raise RuntimeError(f"pdflatex falló en {nombre}:\n{detalle}")
        png_b = b""
        if shutil.which("pdftoppm") is not None:
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    str(dpi),
                    "-singlefile",
                    f"{nombre}.pdf",
                    nombre,
                ],
                cwd=td,
                capture_output=True,
                timeout=300,
                check=True,
            )
            salida = td / f"{nombre}.png"
            if salida.exists():
                png_b = salida.read_bytes()
        return pdf.read_bytes(), png_b


def render(
    diag: Diagrama,
    *,
    fig_dir: Path | None = None,
    force: bool = False,
    dpi: int = 300,
    verbose: bool = True,
) -> Path | None:
    """Genera (o reutiliza) el PNG del diagrama. Devuelve su ruta, o None.

    El `.tex` es la clave de caché: si el que se acaba de generar es idéntico al
    versionado y el PNG existe, no se compila nada. Nunca lanza por falta de
    LaTeX — se degrada al PNG cacheado, para no romper `scripts/run_all.py`.
    """
    fig_dir = Path(fig_dir or Config().fig_dir)
    tex_dir = fig_dir / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path, png_path = tex_dir / f"{diag.nombre}.tex", fig_dir / f"{diag.nombre}.png"
    pdf_path = tex_dir / f"{diag.nombre}.pdf"

    nuevo = tikz_source(diag)
    previo = tex_path.read_text(encoding="utf-8") if tex_path.exists() else None
    if not force and previo == nuevo and png_path.exists():
        if verbose:
            print(f"  = {diag.nombre}: sin cambios, se reutiliza el PNG cacheado")
        return png_path

    # Con un PNG viejo, el `.tex` solo se actualiza si LaTeX compila: evita que
    # la clave de caché describa una arquitectura distinta a la imagen. Si no
    # existe PNG, el fallback Matplotlib sí puede escribir ambos artefactos de
    # forma atómica desde la misma representación intermedia.
    try:
        pdf_b, png_b = _compilar(nuevo, diag.nombre, dpi)
    except LatexNoDisponible:
        if png_path.exists():
            print(
                f"  ! {diag.nombre}: sin pdflatex; se usa el PNG cacheado "
                "(instala MiKTeX/TeX Live para regenerarlo)"
            )
            return png_path
        _render_fallback_matplotlib(diag, png_path, dpi)
        tex_path.write_text(nuevo, encoding="utf-8", newline="\n")
        print(f"  + {diag.nombre}: sin pdflatex; PNG 2-D generado con Matplotlib")
        return png_path
    tex_path.write_text(nuevo, encoding="utf-8", newline="\n")
    pdf_path.write_bytes(pdf_b)
    if png_b:
        png_path.write_bytes(png_b)
        if verbose:
            print(f"  + {diag.nombre}: {len(png_b) // 1024} KB")
        return png_path
    _render_fallback_matplotlib(diag, png_path, dpi)
    print(
        f"  + {diag.nombre}: PDF generado sin pdftoppm; PNG 2-D generado con Matplotlib"
    )
    return png_path

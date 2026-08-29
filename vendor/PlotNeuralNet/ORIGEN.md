# PlotNeuralNet (vendorizado)

Copia parcial de [HarisIqbal88/PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet),
rama `master`, descargada el 2026-08-29 desde `raw.githubusercontent.com`.

## Por qué vendorizado y no instalado

PlotNeuralNet **no está publicado en PyPI** (`pip install plotneuralnet` da 404):
la única vía oficial es clonar el repositorio. Son seis ficheros sin ninguna
dependencia de Python, así que copiarlos aquí sale más barato que un submódulo
—que obligaría a `git submodule update --init` a todo el que clone— y garantiza
que las figuras se puedan regenerar sin acceso a la red.

## Qué se ha copiado

| Fichero | Para qué |
|---|---|
| `pycore/tikzeng.py` | emisores `to_*` que generan el TikZ |
| `pycore/blocks.py` | bloques compuestos (no lo usa este repo; se copia por integridad) |
| `layers/init.tex` | preámbulo que carga los tres estilos (ojo: `.tex`, no `.sty`) |
| `layers/Box.sty` | caja 3-D simple |
| `layers/RightBandedBox.sty` | caja con franja de activación |
| `layers/Ball.sty` | nodo esférico (operadores tipo suma) |
| `LICENSE` | MIT, Copyright (c) 2018 HarisIqbal88 |

Los `__init__.py` los añade este repo para poder importar
`vendor.PlotNeuralNet.pycore` como paquete normal.

## Qué NO se ha copiado

`examples/`, `pyexamples/`, `tikzmake.sh` y el README de upstream. El script
`tikzmake.sh` hace `rm *.tex` en el directorio de trabajo, lo que borraría
cualquier otro `.tex` presente; aquí la compilación la hace `src/netviz.py` en
un directorio temporal.

## Modificaciones

**Ninguna.** Los ficheros están tal cual vienen de upstream. Todo lo propio
—los emisores de capa densa, los rótulos, la paleta Okabe-Ito, la geometría—
vive en `src/netviz.py`, precisamente para que actualizar esta carpeta sea
volver a descargar y nada más.

Nota técnica que condiciona `src/netviz.py`: la clave `/boxblock/` de `Box.sty`
admite solo `width, height, depth, scale, xlabel, ylabel, zlabel, caption,
name, fill, opacity`. **No admite `bandfill` ni `zlabelposition`** (eso es de
`RightBandedBox`, bajo `/block/`). El `to_FullyConnected` que circula por los
issues de upstream asume un `Box.sty` parcheado y no compila contra `master`.

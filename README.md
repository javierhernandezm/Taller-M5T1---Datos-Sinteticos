# Taller B5-T1 · Generación de datos financieros sintéticos

**MIAX 14 · Módulo 5: IA Generativa** — ¿mejoran los datos sintéticos el entrenamiento
de un modelo de predicción financiera, y qué generador los produce mejor?

## El problema

Predicción de **volatilidad realizada a 21 días** (`y = ln σ_fwd`) a partir de una ventana
de 60 retornos log diarios, por activo, sobre el universo S&P 1500 point-in-time
(2012–2026) restringido a los activos **perfectamente mapeados** (auditoría
`mapping_status == READY` y `status == OK` del maestro de identificadores del TFM:
1.494 activos, 30 % de ellos delisted y retenidos hasta su salida real del índice).

Se entrenan varios modelos generativos sobre las ventanas reales de train, se generan
datasets sintéticos y se entrena la misma arquitectura downstream con distintas mezclas
real/sintético y distintos niveles de escasez de datos reales, comparando el error en un
test real intocado.

## Estructura del repositorio

```
taller_b5t1/
├── src/                       # toda la lógica, con docstrings que justifican cada decisión
│   ├── config.py              #   parámetros del experimento + autodetección de entorno
│   ├── data_loading.py        #   universo PIT filtrado + panel de precios limpio
│   ├── preprocessing.py       #   retornos, ventanas, target, splits purgados, estandarización
│   ├── eda.py                 #   hechos estilizados y figuras del análisis exploratorio
│   ├── baselines.py           #   predictores clásicos (media, persistencia, HAR-RV) y métricas
│   ├── models.py              #   arquitecturas candidatas del modelo downstream
│   ├── training.py            #   harness único de entrenamiento (clipping, cosine, best-state)
│   ├── generators.py          #   los 6 generadores (3 baselines + VAE, WGAN-GP, RealNVP)
│   ├── gen_audit.py           #   auditoría de generadores contra los hechos estilizados
│   ├── malla.py               #   barrido real×sintético, reanudable con checkpoint
│   └── netviz.py              #   diagramas de arquitectura derivados de los nn.Module
├── notebooks/
│   ├── 01_datos_y_eda.ipynb          # datos, preprocesamiento y EDA crítico
│   ├── 02_downstream_baselines.ipynb # baselines + arquitectura downstream congelada
│   ├── 03_generadores.ipynb          # generadores, curvas de convergencia y auditoría
│   └── 04_malla_sintetica.ipynb      # malla real×sintético y contraste de hipótesis
├── data/processed/            # artefactos generados (VERSIONADOS: 16 MB)
│   ├── windows_dataset.npz           #   X/y de train, val y test (sin estandarizar)
│   ├── windows_meta.parquet          #   cik, sector, fechas y split de cada ventana
│   ├── standardizer.json             #   estadísticos de estandarización (solo train)
│   ├── downstream_reference.json     #   arquitectura congelada + métricas de referencia
│   └── downstream_reference.pt       #   pesos de la campeona
├── reports/figures/           # figuras exportadas por los notebooks (versionadas)
│   └── tex/                   #   fuente TikZ y PDF vectorial de los diagramas 16-23
├── vendor/PlotNeuralNet/      # PlotNeuralNet (MIT), vendorizado: no está en PyPI
├── tests/test_netviz.py       # pruebas de los diagramas (no requieren LaTeX)
├── scripts/
│   ├── run_all.py             # ejecuta todos los notebooks de principio a fin
│   └── make_arch_figures.py   # regenera los diagramas de arquitectura
├── pyproject.toml             # entorno uv (torch desde el índice cu128)
├── requirements.txt           # ruta pip, para quien no use uv y para Colab
└── .python-version            # Python 3.13
```

**Qué se versiona y qué no.** `data/processed/` (16 MB en total) **sí** entra en el repo:
contiene el dataset de ventanas ya construido y el modelo de referencia congelado, de modo
que cualquiera puede ejecutar los notebooks 02–04 sin acceso al Drive privado del TFM. Los
datos **crudos** (`data/raw/`, 115 MB) no entran: son privados y se referencian por ruta.
Las figuras y las salidas de los notebooks también se versionan, porque el enunciado exige
poder ver las curvas de loss y las gráficas de resultados en el propio repositorio.

Los notebooks **orquestan y narran**; las funciones viven en `src/` y son puras
(DataFrame/arrays dentro → DataFrame/arrays fuera). Cualquier parámetro del experimento se
cambia en `src/config.py`, en un único sitio.

## Puesta en marcha

Hay dos rutas equivalentes. La primera es la canónica (reproducible bit a bit); la segunda
es la portable, para quien no use `uv` y para Google Colab.

### Opción A — uv (recomendada)

```powershell
git clone https://github.com/<usuario>/taller-b5t1-datos-sinteticos.git
cd taller-b5t1-datos-sinteticos
uv sync                  # crea .venv desde uv.lock; torch GPU (~3 GB) la primera vez
uv run jupyter lab
```

`uv.lock` fija las versiones exactas de todas las dependencias transitivas: dos personas
que ejecuten `uv sync` obtienen entornos idénticos. `pyproject.toml` enruta `torch` al
índice CUDA 12.8 de PyTorch en Windows (necesario para GPUs Blackwell / RTX 50xx).

### Opción B — pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

> **Aviso GPU en Windows con pip.** El `torch` de PyPI es **CPU-only** en Windows. Todo el
> proyecto funciona en CPU (más lento), pero para usar una GPU NVIDIA hay que instalar
> torch aparte **antes** del `requirements.txt`:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```
> `cu128` es CUDA 12.8, obligatorio para GPUs Blackwell (RTX 50xx, `sm_120`); para tarjetas
> anteriores sirven `cu121` o `cu124`. En Linux y en Colab el `torch` de PyPI ya trae CUDA
> y no hace falta nada especial.

### Comprobaciones

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Cualquiera de las dos rutas ejecuta los notebooks **02, 03 y 04** de principio a fin sin
datos crudos, partiendo de los artefactos versionados en `data/processed/`.

### Reproducir todo sin abrir Jupyter

```bash
uv run python scripts/run_all.py          # todos los notebooks, en orden
uv run python scripts/run_all.py 02 03    # solo los indicados
```

Ejecuta cada notebook in-place y regenera todas las figuras de `reports/figures/`. Es el
punto de entrada de reproducibilidad que pide el enunciado. En CPU tarda bastante (el
notebook 03 entrena tres redes generativas); en GPU, uno o dos órdenes de magnitud menos.

### Diagramas de arquitectura

Las figuras 16–23 de `reports/figures/` son las fichas visuales de las redes: las cuatro
candidatas downstream, los tres generadores neuronales y el pipeline de datos. Se dibujan
con [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) (MIT), vendorizado en
`vendor/PlotNeuralNet/` porque no está publicado en PyPI.

```bash
uv run python scripts/make_arch_figures.py           # solo lo que haya cambiado
uv run python scripts/make_arch_figures.py --force   # recompila las ocho
uv run python scripts/make_arch_figures.py --list    # ver qué genera
```

**No son dibujos.** `src/netviz.py` recorre los `nn.Module` de verdad y calcula cada cifra
con la aritmética real de la convolución y el pooling: la cadena 60 → 30 → 15 → 7 de la CNN
campeona no está escrita en ninguna parte. En un repo cuya tesis es *"la arquitectura queda
congelada y solo cambian los datos"*, una figura capaz de desviarse en silencio del código
sería una mentira documental. Por lo mismo, el sello **CONGELADA** se decide leyendo
`downstream_reference.json`, y el notebook 03 pasa sus generadores ya entrenados para que
el diagrama describa lo que se ejecutó y no los valores por defecto de las clases.

**LaTeX es una dependencia opcional.** La compilación necesita `pdflatex` (MiKTeX en
Windows, TeX Live en Linux/macOS) y `pdftoppm` para el PNG. Sin ellos no se rompe nada: los
`.png` y los `.tex` van versionados, y tanto el script como las celdas de los notebooks
avisan y reutilizan la figura cacheada. `scripts/run_all.py` sigue funcionando igual.

```bash
uv run pytest tests/ -q      # las pruebas de netviz NO necesitan LaTeX
```

### Notebook 01: datos crudos

Solo el notebook 01 necesita los ficheros originales del TFM
(`PRECIOS/sp1500_daily_prices.parquet` y `MAESTRO/universo_pit_rics_2400.csv`). Tres formas
de dárselos, por orden de preferencia:

1. Copiar o enlazar ambas carpetas dentro de `data/raw/` del repo (se detecta solo).
2. Definir la variable de entorno `TALLER_DATA_ROOT` con la carpeta que las contiene:
   ```powershell
   $env:TALLER_DATA_ROOT = "G:\...\TFM\03_Datos"     # PowerShell
   export TALLER_DATA_ROOT=/ruta/a/TFM/03_Datos         # bash
   ```
3. En Colab, montar Drive: `from google.colab import drive; drive.mount('/content/drive')`.

Si faltan, `Config()` **no** falla: solo falla el notebook 01 al pedirlos, con un mensaje
que explica estas tres opciones.

## Trabajo en equipo

Repositorio compartido por **Javi**, **Diego** y **Javier**. Reglas mínimas para que tres
personas no se pisen:

- **Rama por trabajo, nunca commits directos a `main`.** `git switch -c 04-malla-sintetica`,
  se trabaja, se abre Pull Request, se revisa y se mezcla.
- **Un notebook, un dueño a la vez.** Los `.ipynb` guardan sus salidas (imágenes en base64),
  así que dos personas editando el mismo notebook producen un conflicto de miles de líneas
  que git no sabe resolver. Repartid por notebook, no por celda.
- **La lógica va a `src/`, no a las celdas.** Los módulos sí se mezclan bien. Cuanto más
  fino sea el notebook, menos conflictos.
- **Si aparece un conflicto en un `.ipynb`**: no lo edites a mano. Quédate con una versión
  (`git checkout --ours notebooks/0X.ipynb` o `--theirs`) y vuelve a ejecutarla.
- **Nunca comitear `.venv/`** — ya está en `.gitignore`, pero conviene saber por qué: son
  gigabytes y es específico de cada máquina.
- **El repo vive fuera de OneDrive.** La sincronización pelea con `.venv` y con `.git`, y
  produce corrupciones que parecen bugs de paquetes. La copia en OneDrive, si se mantiene,
  es un espejo del entregable, no el repositorio de trabajo.

## Decisiones metodológicas clave

- **Ventanas por activo** (60 retornos → 1 target), no transversales: ~153k muestras de
  dimensión 61 en lugar de ~9k de dimensión 1.380; la escasez de datos reales pasa a ser
  una variable experimental controlable.
- **Target en logaritmo**: σ es ~lognormal; en logs el target es casi gaussiano y el MSE
  no queda dominado por los episodios de pánico (verificado empíricamente en el EDA).
- **Split temporal con purga y embargo** (train ≤ 2021 / val 2022-23 / test 2024-26,
  embargo 21 días): ninguna ventana cruza fronteras; sin fuga por solapamiento. Los
  generadores solo ven train; la validación es siempre real; el test no se toca.
- **Sin winsorización**: las colas son parte de la distribución que los generadores deben
  aprender. Los retornos solo se invalidan por huecos > 5 días o precio < 1 $.
- **Gradient clipping en el entrenamiento**: con colas de |z| ≈ 7,7 en el target, un lote
  extremo produce un pico de gradiente de dos órdenes de magnitud. Recortar la norma
  estabiliza sin tocar el objetivo — winsorizar habría cambiado el problema.
- **Generadores del par conjunto `[X | y]`** (opción OPT2 de las transparencias): generar
  solo `X` obligaría a etiquetar después con otro modelo, contaminando el experimento.
- **Auditoría de generadores previa al downstream**: colas, clustering, apalancamiento,
  preservación de corr(X,y) y discriminative AUC. Sin ella no se puede distinguir un
  generador que ayuda por realismo de otro que ayuda por mera regularización.

## Resultados hasta ahora

**Modelo downstream congelado** (notebook 02): CNN 1-D de 68k parámetros, R² en test
0,456 ± 0,006 (3 semillas), frente a 0,359 de HAR-RV y 0,120 de persistencia.

**Auditoría de generadores** (notebook 03; referencia real: curtosis 25,2 · ACF|r| 0,062):

| Generador | Curtosis | ACF\|r\| lag1 | AUC discrim. |
|---|---:|---:|---:|
| jitter | 23,6 | 0,058 | 0,51 |
| gaussiana | 0,03 | −0,015 | 0,96 |
| block_bootstrap | 26,2 | 0,125 | 0,76 |
| VAE | 1,62 | −0,007 | 0,90 |
| WGAN-GP | 5,10 | −0,012 | 0,81 |
| **RealNVP** | **13,9** | **0,046** | **0,65** |

**Malla real×sintético** (notebook 04; 117 celdas, Δ R² pareado por semilla):

> Los datos sintéticos **sí** mejoran el modelo, pero solo por encima de un umbral mínimo
> de datos reales (entre 250 y 1.000 ventanas) y solo si el generador reproduce las colas.
> Por debajo de ese umbral, y con generadores que gaussianizan la distribución, el dato
> sintético **destruye** el modelo. El generador más sofisticado (WGAN-GP) es el más
> peligroso —deja el R² en negativo con N=250— y el más trivial (ruido sobre datos reales)
> nunca hace daño.

Y el hallazgo que cierra el círculo: el **discriminative AUC** medido en el notebook 03,
sin mirar el modelo downstream, **ordena perfectamente** a los seis generadores por su
ganancia real (Spearman −1,00 con N=1.000). Se puede elegir generador midiendo fidelidad,
en minutos, en lugar de barrer una malla entera.

---

El flow es el mejor generador neuronal en los tres ejes, pero **sobreajusta**: su ventaja
en log-verosimilitud sobre una N(0, I) trivial cae de **+24,0 nats en train** a **+4,3 en
validación**, un 82 % menos. Sigue batiendo al listón fuera de muestra, pero por mucho
menos de lo que su ajuste sugiere. Parecerse a train no garantiza generalizar.

## Estado

| Fase | Estado |
|---|---|
| 01 · Datos, preprocesamiento y EDA crítico | ✅ |
| 02 · Arquitectura downstream y baselines no generativos | ✅ |
| 03 · Generadores (baselines simples + familias neuronales) | ✅ |
| 04 · Malla real×sintético y análisis de resultados | ✅ (malla REDUCIDA: 117 celdas) |
| 05 · Presentación (PDF) | pendiente |

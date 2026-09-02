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
│   ├── generators.py          #   baselines + VAE/RealNVP; WGAN legado reproducible
│   ├── diffusion_ts.py        #   Diffusion-TS R61: denoiser, calendario de ruido y DDIM
│   ├── gen_audit.py           #   auditoría de FIDELIDAD (hechos estilizados, KS/W1, corr)
│   ├── gen_utility.py         #   auditoría de UTILIDAD (TSTR frente a TRTR)
│   ├── malla.py               #   barrido real×sintético, reanudable con checkpoint
│   └── netviz.py              #   diagramas de arquitectura derivados de los nn.Module
├── notebooks/
│   ├── 01_datos_y_eda.ipynb          # datos, preprocesamiento y EDA crítico
│   ├── 02_downstream_baselines.ipynb # baselines + arquitectura downstream congelada
│   ├── 03_generadores.ipynb          # generadores, convergencia y auditoría (fidelidad+utilidad)
│   └── 04_malla_sintetica.ipynb      # malla real×sintético y contraste de hipótesis
├── data/processed/            # artefactos generados (VERSIONADOS: 16 MB)
│   ├── windows_dataset.npz           #   X/y de train, val y test (sin estandarizar)
│   ├── windows_meta.parquet          #   cik, sector, fechas y split de cada ventana
│   ├── standardizer.json             #   estadísticos de estandarización (solo train)
│   ├── downstream_reference.json     #   arquitectura congelada + métricas de referencia
│   ├── downstream_reference.pt       #   pesos de la campeona
│   ├── auditoria_nb03.csv            #   tabla de fidelidad de los 6 generadores
│   └── tstr_nb03.csv                 #   utilidad TSTR/TRTR, una fila por brazo y semilla
├── reports/figures/           # figuras exportadas por los notebooks (versionadas)
│   └── tex/                   #   fuente TikZ y PDF vectorial de los diagramas de arquitectura
├── vendor/PlotNeuralNet/      # PlotNeuralNet (MIT), vendorizado: no está en PyPI
├── tests/
│   ├── test_netviz.py         # pruebas de los diagramas (no requieren LaTeX)
│   ├── test_diffusion_ts.py   # contrato R61, reproducibilidad y save/load
│   ├── test_artefactos_r61.py # coherencia de los CSV canónicos y su espejo en el informe
│   └── test_gen_audit.py      # pruebas de las métricas de fidelidad (sin GPU, segundos)
├── scripts/
│   ├── run_all.py             # ejecuta todos los notebooks de principio a fin
│   ├── run_diffusion_ts_r61_mallas.py  # mallas R61 reanudables
│   ├── run_diffusion_ts_r61_nb03.py    # TSTR oficial alineado
│   ├── promote_diffusion_ts_r61_results.py # valida y promueve resultados canónicos
│   ├── update_diffusion_ts_r61_notebooks.py # sincroniza narrativa 03/04
│   ├── run_diffusion_ts_r61_notebook03.py # regenera la auditoría canónica (paso 5)
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

Las figuras 16–23 y 29–30 de `reports/figures/` son las fichas visuales de las redes: las
seis candidatas downstream (16–19 y 29–30), los tres generadores neuronales (20–22) y el
pipeline de datos (23). Se dibujan con
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) (MIT), vendorizado en
`vendor/PlotNeuralNet/` porque no está publicado en PyPI.

Las dos GRU llevan número 29–30 y no 20–21 por deuda de numeración: esos huecos ya
estaban ocupados cuando se añadieron, y renumerar la carpeta obliga a tocar los cuatro
notebooks. Pendiente de un cambio propio.

```bash
uv run python scripts/make_arch_figures.py           # solo lo que haya cambiado
uv run python scripts/make_arch_figures.py --force   # recompila las diez
uv run python scripts/make_arch_figures.py --list    # ver qué genera
```

**No son dibujos.** `src/netviz.py` recorre los `nn.Module` de verdad y calcula cada cifra
con la aritmética real de la convolución, el pooling y la recurrencia: la cadena
60 → 30 → 15 → 7 de la CNN grande no está escrita en ninguna parte, y que la GRU mantenga
la secuencia en 60 hasta el último estado tampoco — sale de recorrer las capas. En un repo cuya tesis es *"la arquitectura queda
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
- **Auditoría de generadores previa al downstream**, en los dos ejes que el marco estándar
  de datos sintéticos exige. **Fidelidad**: colas, clustering, apalancamiento, preservación
  de corr(X,y), distancias marginales (KS y Wasserstein), distancia entre matrices de
  correlación y discriminative AUC. **Utilidad**: TSTR frente a TRTR. Sin la primera no se
  puede distinguir un generador que ayuda por realismo de otro que ayuda por mera
  regularización; sin la segunda, parecerse al dato real se confunde con servir para
  entrenar — y el jitter demuestra que no es lo mismo.
- **Privacidad, fuera de alcance y declarado como tal**: los datos de partida son precios de
  mercado públicos, no registros personales, así que no hay secreto que filtrar. En un caso
  de uso con datos de clientes esa tercera auditoría (duplicados, vecinos más cercanos,
  inferencia de membresía, presupuesto DP) sería obligatoria.

## Resultados hasta ahora

**Modelo downstream congelado** (notebook 02): **GRU de 85k parámetros** (3 capas, h=72),
R² en test 0,472 ± 0,005 (3 semillas), frente a 0,359 de HAR-RV y 0,120 de persistencia.

La búsqueda compara **tres familias en dos tallas cada una** — comparar familias con una
sola talla confunde "esta familia es mejor" con "esta red tenía el tamaño adecuado":

| candidata | params | val MSE |
|---|---:|---:|
| mlp_s | 16.129 | 0,1104 |
| mlp_l | 56.833 | 0,1163 |
| cnn_s | 14.721 | 0,0949 |
| cnn_l | 68.225 | 0,0946 |
| gru_s | 42.049 | 0,0918 |
| **gru_l** | **84.601** | **0,0912** |

> **Las dos recurrentes baten a las cuatro anteriores.** El orden de los retornos lleva
> señal que ni el MLP (invariante al orden) ni la CNN (solo vecindarios locales)
> capturan. Ese resultado es sólido: la brecha con `cnn_l` (0,003) es varias veces la
> dispersión entre semillas.
>
> **Cuál de las dos GRU gana, en cambio, no lo es.** Las separan 0,0006, del orden de esa
> misma dispersión, y el orden se invierte con solo cambiar de versión de torch (con
> 2.6.0+cu124 gana `gru_s` por 0,0002; con la 2.11.0+cu128 del `pyproject`, `gru_l` por
> 0,0006). Se congela `gru_l` porque es la que gana en el entorno declarado del repo, no
> porque haya evidencia de que sea mejor. La lectura honesta es que **añadir capacidad
> dentro de la familia no compra nada medible**: lo que decide es el sesgo inductivo
> recurrente, no el tamaño.

**Auditoría de generadores R61** (notebook 03; referencia real: curtosis 25,24 ·
ACF|r| 0,062). La comparación activa es `[X60 | y]` para todos: Diffusion-TS usa 60 tokens
temporales y un token especial para `y`. Una exploración previa (R81) quedó descartada
porque habría recibido 20 retornos que ningún rival ve; no se versiona.

| Generador | sd(X) | Curtosis | ACF\|r\| lag1 | AUC | ratio TSTR/TRTR |
|---|---:|---:|---:|---:|---:|
| jitter | 1,000 | 23,57 | **0,058** | **0,507** | 0,9233 |
| gaussiana | 1,001 | 0,03 | −0,015 | 0,958 | **−0,3876** |
| block bootstrap | 0,990 | 26,16 | 0,125 | 0,760 | 0,6237 |
| VAE | 0,909 | 1,50 | −0,005 | 0,900 | 0,6633 |
| **Diffusion-TS R61** | **0,693** | 20,32 | −0,013 | 0,784 | **0,9472** |
| RealNVP | 0,940 | 22,48 | **0,042** | **0,662** | **0,9467** |

> Diffusion-TS encabeza el TSTR activo con R² 0,4773, pero aventaja a RealNVP solo en
> **0,00025 R²**: es un empate práctico. Su fidelidad es peor —contrae la escala un 31 % y
> no reproduce el clustering—, de modo que no basta con mirar TSTR para declararlo ganador.
>
> Frente al WGAN-GP retirado (R² TSTR histórico 0,1250), Diffusion mejora **+0,3523 R²** y
> gana en las tres semillas. Esta es la evidencia que justifica la sustitución.

**Malla real×sintético** (notebook 04; 333 celdas de ratios + 465 de curvas, ΔR² pareado
por semilla). La malla de ratios activa queda:

| N_real | R² solo real | sd | mejoras | empeoras | no concluyentes |
|---:|---:|---:|---:|---:|---:|
| 250 | 0,151 | **0,115** | 2 | 3 | 31 |
| 1.000 | 0,199 | **0,161** | 0 | 0 | 36 |
| 3.000 | 0,338 | 0,022 | **19** | 4 | 13 |

Para Diffusion-TS, los ratios dan **7 mejoras, 0 empeoramientos y 11 resultados no
concluyentes**. Con N=3.000 mejoran las seis dosis, desde +0,0136 R² con 0,25× hasta
+0,0629 con 5×. Con N=1.000 todas las medias son positivas, pero tres semillas no bastan
para separarlas de la inestabilidad del downstream. Con N=250 solo concluye 5× (+0,0765).

En la malla de presupuestos absolutos, Diffusion obtiene **9 mejoras, 1 empeoramiento y 15
no concluyentes** sobre el test. Empata globalmente con RealNVP (+0,0003 R² medio) y jitter
(−0,0009), y supera al WGAN histórico en +0,1276 R² medio, ganando 69 de 75 pares. A
partir de 10.000–20.000 reales, el beneficio práctico se reduce a milésimas.

**Conclusión:** sí, Diffusion-TS R61 merece sustituir a WGAN-GP en el pipeline activo. No
es un campeón absoluto: RealNVP conserva mejor la distribución y el jitter es más barato;
los tres empatan en la malla larga. El caso de uso más claro de Diffusion es alrededor de
3.000 ventanas reales, con presupuestos sintéticos de 1× a 6,7×.

---

RealNVP sigue siendo el generador neuronal de **mejor fidelidad**, aunque **sobreajusta**:
su ventaja en log-verosimilitud sobre una N(0, I) trivial cae de **+24,0 nats en train** a
**+4,3 en validación**, un 82 % menos. Diffusion-TS enseña mejor a la GRU en TSTR pese a
ser más fácil de distinguir; parecerse a train y conservar la señal downstream son ejes
relacionados, no equivalentes.

## Estado

| Fase | Estado |
|---|---|
| 01 · Datos, preprocesamiento y EDA crítico | ✅ |
| 02 · Arquitectura downstream y baselines no generativos | ✅ |
| 03 · Generadores (baselines simples + familias neuronales) | ✅ |
| 04 · Malla real×sintético y análisis de resultados | ✅ (RATIOS_FINOS 333 celdas + curvas 465) |
| 05 · Presentación (PDF) | pendiente |

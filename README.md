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
│   └── test_gen_audit.py      # pruebas de las métricas de fidelidad (sin GPU, segundos)
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

**Auditoría de generadores** (notebook 03; referencia real: curtosis 25,2 · ACF|r| 0,062).
Dos ejes: **fidelidad** (¿se parecen?) y **utilidad** (¿sirven para entrenar?). El ratio
TSTR/TRTR es el R² de un modelo entrenado **solo con sintético** dividido por el del mismo
modelo entrenado con las 92k ventanas reales de train; 1,0 sería conservar toda la
información. Los dos brazos reciben el MISMO presupuesto de ventanas: sin esa simetría el
ratio no significa nada.

| Generador | Curtosis | ACF\|r\| lag1 | AUC discrim. | W1 por columna | ratio TSTR/TRTR |
|---|---:|---:|---:|---:|---:|
| jitter | 23,6 | 0,058 | 0,50 | 0,026 | 0,92 |
| gaussiana | 0,03 | −0,015 | 0,96 | 0,236 | **−0,39** |
| block_bootstrap | 26,2 | 0,125 | 0,76 | 0,057 | 0,62 |
| VAE | 1,43 | −0,006 | 0,89 | 0,158 | 0,66 |
| WGAN-GP | 4,43 | −0,013 | 0,83 | 0,095 | 0,25 |
| **RealNVP** | **23,2** | **0,042** | **0,65** | **0,057** | **0,95** |

> **La gaussiana tiene utilidad negativa:** un modelo entrenado con sus muestras es *peor
> que predecir la media constante*. La fidelidad no es un lujo estético — un generador que
> no reproduce las colas no es neutro, es tóxico.
>
> El orden de la columna TSTR reproduce **casi exactamente** el del AUC discriminativo, que
> se mide sin entrenar un solo modelo downstream: se puede elegir generador en minutos, en
> lugar de barrer una malla entera.
>
> **RealNVP conserva el 95 % de la utilidad y es el único que bate al jitter.** Que el
> listón sea el jitter —ruido sobre el dato real, quince líneas de código— sigue siendo el
> dato incómodo de la tabla: las otras dos redes generativas, VAE y WGAN-GP, quedan muy por
> detrás de él pese a ser mucho más caras.
>
> **Nota de vigencia.** Estos números son con la campeona `gru_l` a 100 épocas. Con la
> anterior (`cnn_l` a 40) el WGAN-GP salía con utilidad **negativa** (−0,21) y aquí sale en
> +0,25; y el orden entre RealNVP y jitter también depende del downstream. El brazo TSTR
> mide "cuánto sabe enseñar el generador *a este modelo*", no una propiedad del generador
> en abstracto — y es la razón de que el TSTR se recalcule entero cuando cambia la
> arquitectura congelada, en vez de reanudarse desde el CSV.

**Malla real×sintético** (notebook 04; 333 celdas de ratios + 465 de curvas, Δ R² pareado
por semilla):

| N_real | R² solo real | sd entre semillas | mejoras | empeoras | no concluyentes |
|---:|---:|---:|---:|---:|---:|
| 250 | 0,151 | **0,115** | 1 | 3 | 32 |
| 1.000 | 0,199 | **0,161** | 0 | 0 | 36 |
| 3.000 | 0,338 | 0,022 | **13** | 5 | 18 |

> **Existe un suelo mínimo de datos reales por debajo del cual generar datos no sirve de
> nada.** Con 250 y 1.000 ventanas la dispersión entre semillas del propio modelo sin
> sintéticos (0,12–0,16 de R²) es mayor que cualquier efecto buscado: 68 de 72 celdas salen
> no concluyentes. El beneficio aparece con 3.000 y lo firman `realnvp` (+0,081 con ratio
> 5×, t=10,6) y `jitter`. Ese umbral **depende del modelo downstream** —con la campeona
> `cnn_l` anterior caía entre 250 y 1.000— y no es una constante del problema.

> **No hay óptimo intermedio de proporción, tampoco por abajo.** Los ratios fraccionarios
> (0,25× · 0,5× · 0,75×) se añadieron para cubrir el tramo que `[0, 1, 3]` se saltaba
> entero. Con N=3.000 el efecto ya es significativo en **0,25×** (`jitter` +0,020, t=5,4) y
> crece de forma monótona hasta 5×: una dosis pequeña ya ayuda, y más ayuda más. Lo que
> cambia con el ratio no es el óptimo sino el **signo**, y lo fija la fidelidad del
> generador: la `gaussiana` empeora *más* cuanto más sintético se añade (−0,180 con 5×), que
> es la firma de un sesgo y no de ruido.

**La auditoría del notebook 03 predice el resultado de la malla** sin mirar el modelo
downstream: la correlación de rangos entre `err_corr(X,X)` en rangos y la ganancia pareada
es −0,89, estable en los tres regímenes, y el ratio TSTR/TRTR llega a +0,89 en N=3.000. Se
puede elegir generador en minutos en lugar de barrer 333 celdas. Con la cautela de que son
correlaciones sobre **n = 6** generadores: la señal está en que el orden se repite en los
tres regímenes, no en el valor puntual.

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
| 04 · Malla real×sintético y análisis de resultados | ✅ (RATIOS_FINOS 333 celdas + curvas 465) |
| 05 · Presentación (PDF) | pendiente |

"""
config.py — Configuración central del proyecto (Taller B5-T1, MIAX 14).

Único punto de verdad para rutas, parámetros de preprocesamiento y
particiones temporales. Todos los notebooks y módulos importan de aquí:
cambiar un parámetro del experimento nunca debe requerir tocar más de
un fichero.

Diseño:
  * `find_data_root()` autodetecta el entorno (data/raw del repo, Google
    Drive en Windows, Colab con Drive montado) probando una lista de rutas
    candidatas. La variable de entorno TALLER_DATA_ROOT tiene prioridad
    sobre todas ellas, de modo que cualquier entorno no previsto se
    resuelve sin editar código.
  * La resolución es PEREZOSA: `Config()` no toca el disco. Solo al acceder
    a `cfg.data_root` (o a las rutas derivadas) se busca la carpeta. Así el
    notebook 01 —el único que necesita los datos crudos— falla con un
    mensaje claro si faltan, mientras que los notebooks 02-04, que parten
    de `data/processed/`, funcionan en cualquier máquina del equipo aunque
    no tenga acceso al Drive privado del TFM.
  * `Config` es un dataclass inmutable (frozen): los parámetros quedan
    fijados al inicio de la ejecución y no pueden mutarse por accidente
    a mitad de un experimento.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Raíz del repositorio, derivada de la posición de este fichero (src/..).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Localización de los datos (robusto local / Colab / cloud)
# ---------------------------------------------------------------------------

#: Rutas candidatas donde puede vivir la carpeta con los datos crudos del TFM,
#: por orden de preferencia. La variable de entorno TALLER_DATA_ROOT tiene
#: prioridad sobre todas: es el mecanismo pensado para el resto del equipo y
#: para cualquier máquina no listada aquí. Añadir rutas es opcional.
_CANDIDATE_DATA_ROOTS: tuple[str, ...] = (
    # data/raw dentro del propio repo (convención portable: cada miembro del
    # equipo copia o enlaza ahí PRECIOS/ y MAESTRO/)
    str(Path(__file__).resolve().parent.parent / "data" / "raw"),
    # Google Drive para escritorio en Windows (unidad G:)
    r"G:\.shortcut-targets-by-id\1cw7giL3w4JYzPN06zpt0vfrvbSekulla\TFM\03_Datos",
    # Google Colab con Drive montado
    "/content/drive/MyDrive/TFM/03_Datos",
    "/content/drive/.shortcut-targets-by-id/1cw7giL3w4JYzPN06zpt0vfrvbSekulla/TFM/03_Datos",
)


def find_data_root() -> Path:
    """Devuelve la primera ruta candidata existente que contenga los datos.

    Prioridad: variable de entorno TALLER_DATA_ROOT > lista de candidatas.
    Lanza FileNotFoundError con un mensaje accionable si ninguna existe:
    preferimos fallar pronto y claro a arrastrar rutas rotas.
    """
    env = os.environ.get("TALLER_DATA_ROOT")
    candidates = ([env] if env else []) + list(_CANDIDATE_DATA_ROOTS)
    for cand in candidates:
        p = Path(cand)
        if (p / "PRECIOS" / "sp1500_daily_prices.parquet").exists():
            return p
    raise FileNotFoundError(
        "No se encontraron los datos crudos del TFM. Se buscó "
        f"PRECIOS/sp1500_daily_prices.parquet bajo: {candidates}\n\n"
        "Opciones:\n"
        "  1) Define TALLER_DATA_ROOT apuntando a la carpeta que contiene "
        "PRECIOS/ y MAESTRO/.\n"
        "  2) Copia esas dos carpetas dentro de data/raw/ del repo.\n"
        "  3) Si solo quieres ejecutar los notebooks 02-04, NO necesitas los "
        "datos crudos: usa los artefactos versionados en data/processed/."
    )


# ---------------------------------------------------------------------------
# 2. Parámetros del experimento
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Parámetros del pipeline de datos y del experimento.

    Los valores por defecto definen el experimento 'canónico' del taller;
    cualquier variante (p. ej. otra longitud de ventana) se crea con
    `dataclasses.replace(cfg, window_len=90)` para dejar rastro explícito.
    """

    # --- Rutas -------------------------------------------------------------
    #: Ruta a los datos crudos. None = autodetectar al primer acceso (perezoso).
    data_root_override: Path | None = None
    #: Salidas ancladas a la raíz del REPO (no al cwd): así el notebook puede
    #: ejecutarse desde notebooks/, desde la raíz o desde un runner externo y
    #: los artefactos caen siempre en el mismo sitio.
    out_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "processed")
    fig_dir: Path = field(default_factory=lambda: _REPO_ROOT / "reports" / "figures")

    # --- Universo ----------------------------------------------------------
    #: Criterio de "perfectamente mapeado": tramos del universo PIT con
    #: mapping_status == READY y status == OK (auditoría del maestro TFM).
    universe_mapping_status: str = "READY"
    universe_status: str = "OK"

    # --- Construcción de ventanas ------------------------------------------
    window_len: int = 60          #: días de retornos en la ventana de entrada X
    horizon: int = 21             #: días hábiles del horizonte del target
    stride: int = 21              #: paso entre ventanas consecutivas de un activo.
                                  #  21 = targets sin solapamiento: reduce la
                                  #  autocorrelación entre muestras del mismo activo.
    max_gap_days: int = 5         #: hueco máximo (días naturales) permitido entre
                                  #  sesiones consecutivas; por encima, el retorno
                                  #  se invalida y la ventana que lo cruce se descarta.
    min_price: float = 1.0        #: precio mínimo (USD): por debajo, microestructura
                                  #  (tick size) domina y contamina la volatilidad.
    ann_factor: int = 252         #: factor de anualización de la volatilidad.

    # --- Particiones temporales (por fecha de FIN del target) ---------------
    #: El split es temporal, nunca aleatorio: ventanas cuyo target termina
    #: antes de train_end van a train; las de validación/test deben EMPEZAR
    #: después del embargo. Ninguna ventana puede cruzar una frontera.
    train_end: str = "2021-12-31"
    val_end: str = "2023-12-31"
    #: Embargo adicional (días naturales) entre particiones, sumado a la
    #: purga estructural que ya impone descartar ventanas que crucen fronteras.
    embargo_days: int = 21

    # --- Reproducibilidad ----------------------------------------------------
    seed: int = 42

    def ensure_dirs(self) -> None:
        """Crea las carpetas de salida si no existen (idempotente)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # Rutas derivadas -------------------------------------------------------
    @property
    def data_root(self) -> Path:
        """Carpeta de datos crudos, resuelta la primera vez que se pide.

        Construir `Config()` nunca falla aunque no haya datos crudos en la
        máquina: solo falla quien realmente los necesita, y con un mensaje
        que dice cómo arreglarlo.
        """
        return self.data_root_override or find_data_root()

    @property
    def prices_path(self) -> Path:
        return self.data_root / "PRECIOS" / "sp1500_daily_prices.parquet"

    @property
    def universe_path(self) -> Path:
        return self.data_root / "MAESTRO" / "universo_pit_rics_2400.csv"


def set_global_seed(seed: int = 42) -> None:
    """Fija todas las semillas relevantes para reproducibilidad.

    torch se importa de forma perezosa: los notebooks de datos/EDA no lo
    necesitan y así el módulo funciona en entornos sin PyTorch instalado.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - opcional
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Las semillas no bastan en CUDA si cuDNN puede escoger algoritmos
        # distintos entre ejecuciones. La búsqueda de arquitectura necesita
        # que una misma semilla produzca realmente el mismo resultado.
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        pass

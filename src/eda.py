"""
eda.py — Funciones de análisis exploratorio y hechos estilizados.

Cada función produce UNA figura (o un DataFrame de resumen) y devuelve el
objeto matplotlib para que el notebook decida si mostrarla, guardarla o
ambas. Ninguna función modifica sus argumentos.

El EDA de este proyecto no es decorativo: cada figura responde a una
pregunta que condiciona el diseño del experimento de datos sintéticos.
La pregunta se documenta en el docstring de cada función y se retoma en
el análisis crítico del notebook.

Estilo: paleta Okabe-Ito (apta para daltonismo), un solo eje por figura,
rejilla recesiva, una idea por figura.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Paleta categórica fija (Okabe-Ito). El orden es fijo: nunca se recicla.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}
SPLIT_COLORS = {"train": PALETTE["blue"], "val": PALETTE["orange"], "test": PALETTE["green"]}


def _style(ax: plt.Axes) -> None:
    """Estilo recesivo común: la tinta se la lleva el dato, no el marco."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Cobertura del universo
# ---------------------------------------------------------------------------

def plot_coverage(px: pd.DataFrame, cfg) -> plt.Figure:
    """Nº de activos con precio por fecha, con las fronteras del split.

    Pregunta: ¿el tamaño del universo es estable en el tiempo o el panel está
    desequilibrado? Un panel decreciente indicaría fuga de delisted (sesgo de
    supervivencia); uno estable con rotación es lo esperable en un índice.
    """
    counts = px.groupby("date")["cik"].nunique()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(counts.index, counts.values, color=PALETTE["blue"], lw=1.2)
    for name, d in [("fin train", cfg.train_end), ("fin val", cfg.val_end)]:
        ax.axvline(pd.Timestamp(d), color=PALETTE["grey"], ls="--", lw=1)
        ax.text(pd.Timestamp(d), counts.max(), f" {name}", fontsize=8, color=PALETTE["grey"])
    ax.set_title("Activos con precio por fecha (universo perfectamente mapeado, PIT)")
    ax.set_ylabel("nº activos")
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Hechos estilizados de los retornos
# ---------------------------------------------------------------------------

def stylized_stats(px: pd.DataFrame) -> pd.DataFrame:
    """Momentos de los retornos diarios, por activo y agregados.

    Pregunta: ¿cuánto se alejan los retornos de la gaussiana que asume el
    generador baseline? La curtosis media por activo es el número contra el
    que luego se evaluará cada generador sintético.
    """
    g = px.dropna(subset=["ret"]).groupby("cik")["ret"]
    per_asset = pd.DataFrame(
        {
            "media_diaria": g.mean(),
            "sd_diaria": g.std(),
            "asimetria": g.skew(),
            "curtosis_exceso": g.apply(lambda s: stats.kurtosis(s, fisher=True)),
            "n_obs": g.size(),
        }
    )
    return per_asset


def plot_fat_tails(px: pd.DataFrame) -> plt.Figure:
    """Densidad de retornos estandarizados vs N(0,1), en escala log + QQ.

    Pregunta: ¿dónde vive la no-gaussianidad? La escala log del panel
    izquierdo hace visibles las colas; el QQ cuantifica a partir de qué
    cuantil la gaussiana deja de describir los datos.
    """
    r = px["ret"].dropna().to_numpy()
    # estandarizamos por activo para no mezclar niveles de vol distintos
    z = px.dropna(subset=["ret"]).groupby("cik")["ret"].transform(
        lambda s: (s - s.mean()) / s.std()
    ).to_numpy()
    z = z[np.isfinite(z)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    xs = np.linspace(-10, 10, 400)
    ax.hist(z, bins=400, range=(-10, 10), density=True, color=PALETTE["blue"], alpha=0.8)
    ax.plot(xs, stats.norm.pdf(xs), color=PALETTE["vermillion"], lw=1.5, label="N(0,1)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_title("Retornos estandarizados vs gaussiana (escala log)")
    ax.legend()
    _style(ax)

    ax = axes[1]
    sample = np.random.default_rng(0).choice(z, size=min(200_000, len(z)), replace=False)
    osm, osr = stats.probplot(sample, dist="norm", fit=False)
    ax.plot(osm, osr, ".", ms=1.5, color=PALETTE["blue"])
    lim = [-5, 5]
    ax.plot(lim, lim, color=PALETTE["vermillion"], lw=1)
    ax.set_xlim(lim), ax.set_ylim(-15, 15)
    ax.set_title("QQ-plot vs normal")
    ax.set_xlabel("cuantiles teóricos"), ax.set_ylabel("cuantiles empíricos")
    _style(ax)

    fig.suptitle(f"Colas gruesas: curtosis exceso global = {stats.kurtosis(r):.1f}", y=1.02)
    fig.tight_layout()
    return fig


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF simple por producto de desviaciones (suficiente para EDA)."""
    x = x - x.mean()
    denom = float((x**2).sum())
    return np.array([(x[: len(x) - k] * x[k:]).sum() / denom for k in range(1, max_lag + 1)])


def plot_vol_clustering(px: pd.DataFrame, max_lag: int = 60, n_assets: int = 300) -> plt.Figure:
    """ACF media de r_t y de |r_t| sobre una muestra de activos.

    Pregunta: ¿hay clustering de volatilidad? Si ACF(r)≈0 pero ACF(|r|)>0 y
    persistente, existe estructura NO LINEAL que una gaussiana multivariante
    no puede capturar por construcción — es la justificación empírica de
    usar generadores neuronales, y también el hecho estilizado con el que
    luego los auditaremos.
    """
    rng = np.random.default_rng(0)
    ciks = set(rng.choice(px["cik"].unique(), size=n_assets, replace=False))
    acf_r, acf_a = [], []
    # una sola pasada groupby en lugar de un escaneo del panel por activo
    for cik, grp in px[px["cik"].isin(ciks)].groupby("cik", sort=False):
        r = grp["ret"].dropna().to_numpy()
        if len(r) < 500:
            continue
        acf_r.append(_acf(r, max_lag))
        acf_a.append(_acf(np.abs(r), max_lag))
    lags = np.arange(1, max_lag + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, np.mean(acf_r, axis=0), color=PALETTE["blue"], lw=1.5, label="ACF(r)  — señal lineal")
    ax.plot(lags, np.mean(acf_a, axis=0), color=PALETTE["orange"], lw=1.5, label="ACF(|r|) — clustering de vol")
    ax.axhline(0, color=PALETTE["grey"], lw=0.8)
    ax.set_xlabel("retardo (días)"), ax.set_ylabel("autocorrelación media")
    ax.set_title(f"Clustering de volatilidad (media sobre {len(acf_r)} activos)")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def plot_leverage(px: pd.DataFrame, max_lag: int = 20, n_assets: int = 300) -> plt.Figure:
    """Efecto apalancamiento: corr(r_t, |r_{t+k}|) para k = 1..max_lag.

    Pregunta: ¿las caídas anticipan más volatilidad que las subidas? Una
    correlación negativa y persistente es el tercer hecho estilizado clásico
    y otra asimetría invisible para el generador gaussiano.
    """
    rng = np.random.default_rng(1)
    ciks = set(rng.choice(px["cik"].unique(), size=n_assets, replace=False))
    corrs = []
    for cik, grp in px[px["cik"].isin(ciks)].groupby("cik", sort=False):
        r = grp["ret"].dropna().to_numpy()
        if len(r) < 500:
            continue
        a = np.abs(r)
        corrs.append(
            [np.corrcoef(r[:-k], a[k:])[0, 1] for k in range(1, max_lag + 1)]
        )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    m = np.nanmean(corrs, axis=0)
    ax.bar(np.arange(1, max_lag + 1), m, color=PALETTE["blue"], width=0.7)
    ax.axhline(0, color=PALETTE["grey"], lw=0.8)
    ax.set_xlabel("retardo k (días)"), ax.set_ylabel("corr(r_t, |r_{t+k}|)")
    ax.set_title(f"Efecto apalancamiento (media sobre {len(corrs)} activos)")
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. El target
# ---------------------------------------------------------------------------

def plot_target_distribution(y: np.ndarray, split: pd.Series) -> plt.Figure:
    """Distribución del target y = log σ_fwd por partición.

    Pregunta doble: (i) ¿el log convierte σ en algo aproximadamente
    gaussiano, como asumimos al elegir MSE?; (ii) ¿hay desplazamiento de
    distribución entre train, val y test? Si lo hay, parte del error de
    test NO será atribuible a los datos sintéticos y habrá que decirlo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for name in ["train", "val", "test"]:
        vals = y[(split == name).to_numpy()]
        ax.hist(vals, bins=80, density=True, alpha=0.55, label=f"{name} (n={len(vals):,})",
                color=SPLIT_COLORS[name])
    ax.set_title("y = ln σ_fwd por partición")
    ax.set_xlabel("ln σ (anualizada)")
    ax.legend()
    _style(ax)

    ax = axes[1]
    for name in ["train", "val", "test"]:
        vals = np.exp(y[(split == name).to_numpy()])
        ax.hist(vals, bins=120, range=(0, 1.5), density=True, alpha=0.55,
                label=name, color=SPLIT_COLORS[name])
    ax.set_title("σ_fwd en niveles (por qué modelamos el log)")
    ax.set_xlabel("σ anualizada")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def plot_vol_regimes(meta: pd.DataFrame, y: np.ndarray, cfg) -> plt.Figure:
    """Mediana transversal de σ_fwd en el tiempo, con fronteras del split.

    Pregunta: ¿qué regímenes de volatilidad caen en cada partición? Es LA
    figura del análisis crítico: si el test no contiene ningún episodio de
    estrés comparable a los de train, la generalización medida es optimista;
    si contiene uno inédito, pesimista. En cualquier caso, condiciona la
    lectura de los resultados con sintéticos.
    """
    df = pd.DataFrame({"date": meta["date_t"].values, "sigma": np.exp(y)})
    q = df.groupby(pd.Grouper(key="date", freq="ME"))["sigma"].quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.25, color=PALETTE["blue"], label="p25–p75")
    ax.plot(q.index, q[0.5], color=PALETTE["blue"], lw=1.4, label="mediana")
    for d in [cfg.train_end, cfg.val_end]:
        ax.axvline(pd.Timestamp(d), color=PALETTE["grey"], ls="--", lw=1)
    ax.set_title("σ_fwd transversal (mensual): regímenes de volatilidad y fronteras del split")
    ax.set_ylabel("σ anualizada")
    ax.legend()
    _style(ax)
    fig.tight_layout()
    return fig


def plot_sector_vol(meta: pd.DataFrame, y: np.ndarray) -> plt.Figure:
    """Distribución del target por sector (boxplot ordenado por mediana).

    Pregunta: ¿la heterogeneidad transversal es sectorial? Si lo es, un
    generador condicional podría explotar el sector como covariable, y un
    generador incondicional deberá al menos cubrir todos los niveles.
    """
    df = pd.DataFrame({"sector": meta["sector"].values, "y": y})
    order = df.groupby("sector")["y"].median().sort_values().index
    data = [df.loc[df["sector"] == s, "y"].values for s in order]
    fig, ax = plt.subplots(figsize=(9, 4))
    # matplotlib 3.11: `labels` se renombro a `tick_labels` y `vert` quedo
    # deprecado en favor de `orientation`.
    bp = ax.boxplot(data, tick_labels=order, showfliers=False,
                    patch_artist=True, orientation="vertical")
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["blue"])
        patch.set_alpha(0.55)
    ax.set_ylabel("y = ln σ_fwd")
    ax.set_title("Target por sector (clasificación SIC del panel SEC)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_sample_windows(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                        n: int = 9, seed: int = 0) -> plt.Figure:
    """Muestras aleatorias de ventanas (lo que verán los generadores).

    Pregunta de control visual: ¿las ventanas parecen retornos diarios
    plausibles (media ~0, rachas de volatilidad, algún salto)? Es el mismo
    control que luego aplicaremos, en espejo, a las ventanas sintéticas.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True)
    for ax, i in zip(axes.ravel(), idx):
        ax.plot(X[i], lw=0.8, color=PALETTE["blue"])
        ax.axhline(0, color=PALETTE["grey"], lw=0.5)
        ax.set_title(
            f"{meta['cik'].iat[i]} | {pd.Timestamp(meta['date_t'].iat[i]):%Y-%m} | σ={np.exp(y[i]):.0%}",
            fontsize=7,
        )
        _style(ax)
    fig.suptitle("Ventanas de entrada X (60 retornos diarios) — muestras reales")
    fig.tight_layout()
    return fig

"""
test_gen_audit.py — Pruebas de las métricas de fidelidad de los generadores.

No cargan el dataset del repo ni entrenan nada: fabrican ventanas sintéticas
con una estructura conocida y comprueban que las métricas dicen lo que deben.
Corren en segundos, sin GPU y sin LaTeX.

Lo que se protege aquí es la propiedad que hace útiles a estas métricas: que
DISCRIMINEN. Una distancia que devuelve cero cuando las dos muestras son
iguales, pero que también devuelve casi cero cuando son distintas, no sirve
para ordenar generadores — y ese fallo es silencioso en una tabla de números.

Uso
---
    uv run pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.gen_audit import _corr_dist, _ks_w1, audit_generator

D = 21          # 20 retornos + target: suficiente para la estructura, rápido
N = 3000
SEED = 0


def _ventanas_reales(n: int = N, seed: int = SEED) -> np.ndarray:
    """Ventanas con colas gruesas y correlación temporal, como las del taller.

    Se usa una t de Student (colas) sobre un AR(1) (dependencia entre columnas)
    para que las métricas tengan algo real que medir: con ruido i.i.d. gaussiano
    la matriz de correlación sería la identidad y `_corr_dist` no distinguiría
    nada.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=4, size=(n, D))
    W = np.empty_like(z)
    W[:, 0] = z[:, 0]
    for j in range(1, D):
        W[:, j] = 0.6 * W[:, j - 1] + z[:, j]
    return W.astype(np.float32)


def _gaussiana_ajustada(XY: np.ndarray, n: int = N, seed: int = SEED + 1) -> np.ndarray:
    """Muestra de la normal multivariante con la media y covarianza de `XY`.

    Es el generador "honesto pero equivocado" del taller: captura toda la
    dependencia LINEAL por construcción y no puede reproducir las colas.
    """
    rng = np.random.default_rng(seed)
    mu, C = XY.mean(0), np.cov(XY, rowvar=False)
    return rng.multivariate_normal(mu, C, size=n).astype(np.float32)


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #

def test_ks_w1_identicas_son_cero():
    """Una muestra contra sí misma: ambas distancias son exactamente 0."""
    a = _ventanas_reales()[:, 0]
    ks, w1 = _ks_w1(a, a, np.random.default_rng(SEED))
    assert ks == pytest.approx(0.0, abs=1e-12)
    assert w1 == pytest.approx(0.0, abs=1e-12)


def test_ks_w1_detectan_desplazamiento():
    """Desplazar la muestra tiene que mover W1 aproximadamente ese desplazamiento.

    W1 entre F(x) y F(x - c) vale exactamente |c|; es la comprobación que ancla
    la métrica a una unidad interpretable en lugar de a un número sin escala.
    """
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=20_000)
    ks, w1 = _ks_w1(a, a + 0.5, rng)
    assert w1 == pytest.approx(0.5, abs=0.05)
    assert ks > 0.15


def test_corr_dist_identicas_son_cero():
    W = _ventanas_reales()
    assert _corr_dist(W, W, "pearson") == pytest.approx(0.0, abs=1e-12)
    assert _corr_dist(W, W, "spearman") == pytest.approx(0.0, abs=1e-12)


def test_corr_dist_detecta_perdida_de_estructura():
    """Barajar cada columna destruye la dependencia y la métrica debe acusarlo.

    El barajado conserva EXACTAMENTE las marginales de cada columna: es el caso
    que ninguna métrica marginal (KS, Wasserstein, curtosis) puede detectar, y
    por tanto justifica que la distancia de matriz de correlación exista.

    Con la segunda matriz de correlación en la identidad, la distancia tiene que
    recuperar por identidad el RMS de las correlaciones reales fuera de la
    diagonal. Comprobarlo contra esa cantidad —y no contra un umbral inventado—
    ancla la métrica a algo verificable a mano.
    """
    rng = np.random.default_rng(SEED)
    W = _ventanas_reales()
    barajada = np.column_stack([rng.permutation(W[:, j]) for j in range(W.shape[1])])

    # las marginales son idénticas columna a columna...
    ks, _ = _ks_w1(W[:, 3], barajada[:, 3], rng)
    assert ks == pytest.approx(0.0, abs=1e-12)

    # ...pero la estructura de dependencia ha desaparecido por completo
    C = np.corrcoef(W, rowvar=False)
    off = ~np.eye(C.shape[0], dtype=bool)
    rms_real = float(np.sqrt((C[off] ** 2).mean()))
    assert _corr_dist(W, barajada, "pearson") == pytest.approx(rms_real, rel=0.15)


# --------------------------------------------------------------------------- #
# audit_generator
# --------------------------------------------------------------------------- #

def test_auditar_contra_si_mismo():
    """Dos muestras del MISMO proceso: todo indistinguible, AUC en 0,5.

    No se compara una muestra consigo misma (sería trivial y el clasificador
    memorizaría): son dos extracciones independientes del mismo generador, que
    es el techo alcanzable por cualquier generador perfecto.
    """
    real = _ventanas_reales(seed=SEED)
    otra = _ventanas_reales(seed=SEED + 100)
    a = audit_generator(real, otra, lags=(1, 5), seed=SEED)

    assert a["ks_x"] < 0.03
    assert a["w1_x"] < 0.10
    assert a["err_corr_xx_pearson"] < 0.05
    assert a["err_corr_xx_spearman"] < 0.05
    assert a["discriminative_auc"] == pytest.approx(0.5, abs=0.08)


def test_gaussiana_acierta_la_covarianza_y_falla_la_forma():
    """El caso que verifica que las métricas nuevas miden cosas DISTINTAS.

    Una gaussiana ajustada reproduce la matriz de correlación de Pearson por
    construcción, así que `err_corr_xx_pearson` se queda en el suelo de ruido
    muestral: si ese fuera el único diagnóstico, la gaussiana pasaría por buen
    generador. Lo que la delata es la marginal (KS, Wasserstein, curtosis) y la
    correlación de RANGOS, que sí es sensible a la no linealidad.

    Es exactamente el patrón que el notebook 03 observa con el dato real, y por
    eso el marco de auditoría pide las dos familias de métricas y no una.
    """
    real = _ventanas_reales()
    referencia = _ventanas_reales(seed=SEED + 100)
    gauss = _gaussiana_ajustada(real)

    base = audit_generator(real, referencia, lags=(1, 5), seed=SEED)
    g = audit_generator(real, gauss, lags=(1, 5), seed=SEED)

    # Acierta la dependencia lineal: su error NO es peor que el de dos muestras
    # independientes del mismo proceso, o sea que está en el suelo de ruido.
    assert g["err_corr_xx_pearson"] < 1.2 * base["err_corr_xx_pearson"]
    # Y falla todo lo demás, con holgura.
    assert g["ks_x"] > 3 * base["ks_x"]
    assert g["w1_x"] > 3 * base["w1_x"]
    assert g["curtosis_x"] < 0.5 * g["curtosis_x_real"]
    assert g["discriminative_auc"] > 0.60


def test_claves_y_tipos():
    """El dict debe ser plano y de escalares finitos: alimenta un DataFrame directo.

    No se exige `float` de Python: las métricas heredadas devuelven `np.float32`
    porque operan sobre las ventanas sin promocionar, y pandas las acepta sin
    problema. Lo que sí hay que garantizar es que no haya arrays anidados ni
    NaN, que es lo que rompería la tabla del notebook en silencio.
    """
    real = _ventanas_reales()
    a = audit_generator(real, _ventanas_reales(seed=SEED + 100), lags=(1, 5), seed=SEED)

    esperadas = {"ks_x", "w1_x", "ks_y", "w1_y", "ks_x_col_media", "w1_x_col_media",
                 "err_corr_xx_pearson", "err_corr_xx_spearman",
                 "curtosis_x", "acf_abs_lag1", "err_corr_xy", "discriminative_auc"}
    assert esperadas <= set(a)
    assert all(np.ndim(v) == 0 for v in a.values())
    assert np.isfinite(np.array(list(a.values()), dtype=float)).all()


def test_reproducible_con_la_misma_semilla():
    """Misma semilla, mismo resultado: sin esto la tabla del notebook no es citable."""
    real, synth = _ventanas_reales(), _ventanas_reales(seed=SEED + 100)
    a = audit_generator(real, synth, lags=(1, 5), seed=SEED)
    b = audit_generator(real, synth, lags=(1, 5), seed=SEED)
    assert a == b

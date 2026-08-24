"""
gen_audit.py — Auditoría de generadores contra los hechos estilizados.

El enunciado solo exige medir el efecto *downstream* de los datos sintéticos.
Este módulo añade la pregunta anterior y más informativa: **¿se parecen las
ventanas sintéticas a las reales, y en qué dejan de parecerse?** Sin esto, un
generador que mejore el modelo por puro efecto regularizador es
indistinguible de otro que lo mejore por realismo — y son conclusiones
opuestas.

Las tres varas de medir son los hechos estilizados documentados en el EDA
(notebook 01), porque son exactamente lo que una gaussiana NO puede capturar:

  1. Colas gruesas          -> curtosis en exceso de los retornos
  2. Clustering de vol.     -> ACF de |r| a varios retardos
  3. Efecto apalancamiento  -> corr(r_t, |r_{t+k}|)

Se añaden dos métricas globales estándar en la literatura de datos sintéticos:

  4. Discriminative score   -> AUC de un clasificador real-vs-sintético.
                               0,5 = indistinguibles; 1,0 = trivialmente
                               separables. Es la métrica de Yoon et al.
                               (TimeGAN) y resume en un número lo que las
                               tres anteriores dicen por separado.
  5. Correlación del par    -> error de la correlación entre cada retorno de
                               la ventana y el target. Mide si el generador
                               preserva la RELACIÓN X-y, que es justamente lo
                               que el modelo downstream tiene que aprender.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _acf_abs(W: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """ACF media de |r| a los retardos dados, promediando sobre ventanas."""
    A = np.abs(W)
    A = A - A.mean(axis=1, keepdims=True)
    denom = (A**2).sum(axis=1)
    out = []
    for k in lags:
        num = (A[:, :-k] * A[:, k:]).sum(axis=1)
        out.append(np.mean(num / np.maximum(denom, 1e-12)))
    return np.array(out)


def _leverage(W: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """corr(r_t, |r_{t+k}|) media sobre ventanas, por retardo."""
    out = []
    for k in lags:
        a, b = W[:, :-k], np.abs(W[:, k:])
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        num = (a * b).sum(1)
        den = np.sqrt((a**2).sum(1) * (b**2).sum(1))
        out.append(np.nanmean(num / np.maximum(den, 1e-12)))
    return np.array(out)


def discriminative_score(XY_real: np.ndarray, XY_synth: np.ndarray,
                         seed: int = 0, max_n: int = 8000) -> float:
    """AUC de un clasificador real-vs-sintético (0,5 = indistinguibles).

    Se usa un boosting de árboles sobre las ventanas aplanadas, entrenado en
    una mitad y evaluado en la otra. Un AUC cercano a 0,5 significa que el
    generador es bueno; cercano a 1,0, que un modelo trivial separa lo real
    de lo sintético — y entonces el downstream también notará la diferencia.
    """
    rng = np.random.default_rng(seed)
    n = min(len(XY_real), len(XY_synth), max_n)
    R = XY_real[rng.choice(len(XY_real), n, replace=False)]
    S = XY_synth[rng.choice(len(XY_synth), n, replace=False)]
    X = np.vstack([R, S])
    y = np.r_[np.zeros(n), np.ones(n)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def audit_generator(XY_real: np.ndarray, XY_synth: np.ndarray,
                    lags: tuple[int, ...] = (1, 5, 10, 21),
                    seed: int = 0) -> dict[str, float]:
    """Compara ventanas sintéticas con reales en los hechos estilizados.

    Devuelve un dict plano de métricas listo para construir un DataFrame:
    momentos de los retornos, ACF de |r|, apalancamiento, preservación de la
    correlación X-y y discriminative score. Las claves con sufijo `_real` y
    `_synth` permiten mostrar el valor de referencia junto al obtenido.
    """
    Wr, Wg = XY_real[:, :-1], XY_synth[:, :-1]
    yr, yg = XY_real[:, -1], XY_synth[:, -1]

    acf_r, acf_g = _acf_abs(Wr, lags), _acf_abs(Wg, lags)
    lev_r, lev_g = _leverage(Wr, (1, 5)), _leverage(Wg, (1, 5))

    # correlación de cada posición de la ventana con el target
    corr_r = np.array([np.corrcoef(Wr[:, j], yr)[0, 1] for j in range(Wr.shape[1])])
    corr_g = np.array([np.corrcoef(Wg[:, j], yg)[0, 1] for j in range(Wg.shape[1])])

    return {
        "sd_x": Wg.std(), "sd_x_real": Wr.std(),
        "curtosis_x": stats.kurtosis(Wg.ravel()), "curtosis_x_real": stats.kurtosis(Wr.ravel()),
        "asimetria_x": stats.skew(Wg.ravel()), "asimetria_x_real": stats.skew(Wr.ravel()),
        "sd_y": yg.std(), "sd_y_real": yr.std(),
        "acf_abs_lag1": acf_g[0], "acf_abs_lag1_real": acf_r[0],
        "acf_abs_lag21": acf_g[-1], "acf_abs_lag21_real": acf_r[-1],
        "err_acf_abs": float(np.abs(acf_g - acf_r).mean()),
        "leverage_lag1": lev_g[0], "leverage_lag1_real": lev_r[0],
        "err_corr_xy": float(np.abs(corr_g - corr_r).mean()),
        "discriminative_auc": discriminative_score(XY_real, XY_synth, seed=seed),
    }

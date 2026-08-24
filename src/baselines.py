"""
baselines.py — Predictores no generativos de referencia.

Estos modelos fijan el SUELO del experimento: cualquier red neuronal — y, más
adelante, cualquier mezcla real+sintético — tiene que justificarse frente a
ellos. Son deliberadamente clásicos y fuertes en este dominio:

  * media global        el peor caso informativo (R² = 0 por definición en train)
  * persistencia RV     la volatilidad realizada de los últimos 21 días como
                        predicción de la de los próximos 21 — explota la
                        persistencia del proceso y es difícil de batir
  * HAR-RV (OLS)        Corsi (2009): regresión lineal sobre la RV realizada en
                        horizontes diario / semanal / mensual / trimestral.
                        Estándar de la literatura de forecasting de volatilidad.

Convención: todos reciben las ventanas X SIN estandarizar (los retornos crudos),
porque calculan volatilidades realizadas con unidades físicas, y devuelven
predicciones en el espacio del target y = ln σ_anualizada.
"""

from __future__ import annotations

import numpy as np

#: suelo numérico para evitar log(0) en ventanas de volatilidad nula
_EPS = 1e-8


def _ln_rv(X: np.ndarray, last_k: int, ann_factor: int = 252) -> np.ndarray:
    """ln de la vol realizada anualizada sobre los últimos `last_k` días de cada ventana."""
    seg = X[:, -last_k:]
    rv = np.sqrt(ann_factor * np.mean(seg**2, axis=1))
    return np.log(np.maximum(rv, _EPS))


def predict_global_mean(y_train: np.ndarray, n: int) -> np.ndarray:
    """Predicción constante: la media del target en train."""
    return np.full(n, float(y_train.mean()), dtype=np.float32)


def predict_persistence(X_raw: np.ndarray, horizon: int = 21) -> np.ndarray:
    """Persistencia: σ futura = σ realizada de los últimos `horizon` días."""
    return _ln_rv(X_raw, horizon).astype(np.float32)


class HarOLS:
    """HAR-RV lineal: y ~ 1 + ln RV_1 + ln RV_5 + ln RV_21 + ln RV_60.

    Se ajusta por mínimos cuadrados (lstsq) sobre train. Sin regularización:
    con 4 regresores y >100k observaciones no la necesita, y así el baseline
    queda libre de hiperparámetros que hubiera que barrer.
    """

    HORIZONS = (1, 5, 21, 60)

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    @classmethod
    def _features(cls, X_raw: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(X_raw))] + [_ln_rv(X_raw, k) for k in cls.HORIZONS]
        return np.column_stack(cols)

    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> "HarOLS":
        F = self._features(X_raw)
        self.coef_, *_ = np.linalg.lstsq(F, y.astype(np.float64), rcond=None)
        return self

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "HarOLS sin ajustar: llama a fit() primero"
        return (self._features(X_raw) @ self.coef_).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MSE, MAE y R² en el espacio del target (ln σ).

    El R² se calcula contra la varianza de y_true en la MISMA partición: mide
    qué fracción de la variación transversal+temporal del target explica el
    modelo dentro de esa partición, que es la comparación honesta cuando hay
    distribution shift entre particiones.
    """
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - mse / np.var(y_true)),
    }

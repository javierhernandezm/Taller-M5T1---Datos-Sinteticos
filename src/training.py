"""
training.py — Harness de entrenamiento único para todo el taller.

Un solo bucle de entrenamiento sirve para la búsqueda de arquitectura (nb 02)
y para los cientos de entrenamientos de la malla real×sintético (nb 04): así
cualquier diferencia entre celdas de la malla es atribuible a los DATOS, nunca
al procedimiento.

Decisiones del harness, todas al servicio de la comparabilidad:
  * Datos completos residentes en el device como tensores; mini-batches por
    permutación de índices, sin DataLoader (con ~100k×60 floats el overhead
    de un DataLoader multiplica por varias veces el tiempo de época).
  * Early stopping por MSE de validación con paciencia fija y restauración
    del mejor estado: las "curvas de loss donde se vea la convergencia" que
    exige el enunciado salen del history que devuelve este módulo.
  * La validación es SIEMPRE real: los datos sintéticos solo pueden aparecer
    en el conjunto de entrenamiento (se asume aguas arriba; este módulo no
    distingue — recibe arrays).
  * Semilla explícita por entrenamiento: la malla repetirá cada celda con
    varias semillas para poner barras de error a las conclusiones.

  * **Gradient clipping por norma** (`grad_clip`). No es cosmético: el target
    estandarizado tiene colas de hasta |z| ≈ 7,7 (y las entradas hasta |z| ≈ 30).
    Con MSE, un lote que contenga una de esas muestras aporta una pérdida ~50
    frente a la típica ~0,4 — un pico de gradiente de dos órdenes de magnitud
    que desestabiliza el entrenamiento y produce curvas de validación con
    dientes de sierra. Recortar la norma acota el paso sin tocar el objetivo
    (winsorizar el target sí cambiaría el problema, y no es admisible en un
    taller cuyo objeto son precisamente las colas).

  * **Cosine annealing** del learning rate: hace que las últimas épocas
    refinen en lugar de rebotar, de modo que las curvas muestren convergencia
    de forma inequívoca, tal y como exige el enunciado.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    """cuda (RTX 5070 / Colab GPU) > mps (Mac) > cpu, en ese orden."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainResult:
    """Resultado de un entrenamiento: historia, mejor época y tiempos."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    seconds: float = 0.0


def train_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 12,
    grad_clip: float = 1.0,
    cosine: bool = True,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Entrena con Adam + early stopping y deja en `model` los mejores pesos.

    Espera entradas YA estandarizadas (X e y); la des-estandarización para
    métricas en unidades físicas es responsabilidad del llamador.

    `grad_clip` acota la norma del gradiente (ver docstring del módulo) y
    `cosine` activa el annealing del learning rate hasta ~0 en la última época.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    loss_fn = nn.MSELoss()
    res = TrainResult()
    best_state: dict | None = None
    since_best = 0
    t0 = time.time()
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        # --- una época de entrenamiento -----------------------------------
        model.train()
        perm = torch.as_tensor(rng.permutation(len(Xt)), device=device)
        ep_loss, n_batches = 0.0, 0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1
        res.train_loss.append(ep_loss / n_batches)
        if sched is not None:
            sched.step()

        # --- validación (real) --------------------------------------------
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xv), yv))
        res.val_loss.append(vl)

        if vl < res.best_val:
            res.best_val, res.best_epoch, since_best = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
        if verbose and epoch % 10 == 0:
            print(f"  época {epoch:3d}  train {res.train_loss[-1]:.4f}  val {vl:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)  # restaurar el mejor estado
    res.seconds = time.time() - t0
    return res


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device | None = None,
            batch_size: int = 8192) -> np.ndarray:
    """Predicción por lotes; devuelve un array numpy en el mismo espacio que y."""
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)

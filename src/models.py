"""
models.py — Arquitecturas candidatas para el modelo downstream.

El taller exige fijar UNA arquitectura "válida" con datos reales y reutilizarla
idéntica en todos los experimentos con sintéticos. Este módulo define las
candidatas de la búsqueda; la elegida queda congelada vía su nombre + kwargs
en un json de referencia (notebook 02) y se reconstruye siempre con
`build_model(name, **kwargs)`.

Criterios de diseño de las candidatas:
  * PEQUEÑAS a propósito (10k–100k parámetros). El objeto del experimento es
    el efecto de los datos, no la arquitectura: un modelo grande añade
    varianza de optimización que enmascara la señal que queremos medir, y la
    malla real×sintético requiere cientos de entrenamientos.
  * Entrada (B, 60): la ventana de retornos estandarizada. La red debe poder
    aprender por sí sola transformaciones tipo |r| o r² (las que un experto
    codificaría a mano en HAR): dárselas hechas contaminaría la comparación
    con los baselines.
  * Salida (B,): el target estandarizado ŷ. Regresión pura, sin activación.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Perceptrón multicapa sobre la ventana aplanada."""

    def __init__(self, in_len: int = 60, hidden: tuple[int, ...] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_len
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 60) -> (B,)
        return self.net(x).squeeze(-1)


class ConvNet1D(nn.Module):
    """CNN 1-D causal-agnóstica: bloques Conv+ReLU+Pool y cabeza densa.

    El global average pooling final hace la cabeza independiente de la
    longitud de la ventana: la misma arquitectura serviría con ventanas de
    otra longitud sin tocar código.
    """

    def __init__(self, channels: tuple[int, ...] = (32, 64), kernel: int = 5,
                 fc: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = 1
        for ch in channels:
            blocks += [nn.Conv1d(prev, ch, kernel, padding=kernel // 2),
                       nn.ReLU(), nn.MaxPool1d(2)]
            prev = ch
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(prev, fc), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 60) -> (B,)
        h = self.conv(x.unsqueeze(1))  # (B, 1, 60) -> (B, C, L')
        return self.head(h).squeeze(-1)


#: Registro de arquitecturas: el nombre + kwargs es todo lo que se persiste
#: para reconstruir el modelo congelado en notebooks posteriores.
_REGISTRY = {"mlp": MLP, "cnn": ConvNet1D}


def build_model(name: str, **kwargs) -> nn.Module:
    """Construye una arquitectura del registro por nombre (fail-fast si no existe)."""
    if name not in _REGISTRY:
        raise KeyError(f"Arquitectura desconocida: {name!r}. Disponibles: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

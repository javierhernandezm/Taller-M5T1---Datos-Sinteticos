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


class ResidualBlock1D(nn.Module):
    """Bloque residual 1-D con normalización por grupos y dilatación opcional."""

    def __init__(self, in_ch: int, out_ch: int, *, kernel: int = 3,
                 dilation: int = 1, stride: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel // 2)
        groups = min(8, out_ch)
        while out_ch % groups:
            groups -= 1
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=padding,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_ch),
        )
        self.skip = (nn.Identity() if in_ch == out_ch and stride == 1 else
                     nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResidualConvNet1D(nn.Module):
    """CNN residual para ventanas cortas, con reducción progresiva y GAP."""

    def __init__(self, widths: tuple[int, ...] = (32, 64, 128),
                 blocks_per_stage: int = 1, kernel: int = 3,
                 dilated: bool = False, fc: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, widths[0], kernel, padding=kernel // 2)
        blocks: list[nn.Module] = []
        prev = widths[0]
        for stage, width in enumerate(widths):
            for block in range(blocks_per_stage):
                blocks.append(ResidualBlock1D(
                    prev, width, kernel=kernel,
                    dilation=(2 ** block if dilated else 1),
                    stride=(2 if stage > 0 and block == 0 else 1),
                    dropout=dropout,
                ))
                prev = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(prev, fc), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fc, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x.unsqueeze(1)))).squeeze(-1)


#: Registro de arquitecturas: el nombre + kwargs es todo lo que se persiste
#: para reconstruir el modelo congelado en notebooks posteriores.
_REGISTRY = {"mlp": MLP, "cnn": ConvNet1D, "rescnn": ResidualConvNet1D}


def build_model(name: str, **kwargs) -> nn.Module:
    """Construye una arquitectura del registro por nombre (fail-fast si no existe)."""
    if name not in _REGISTRY:
        raise KeyError(f"Arquitectura desconocida: {name!r}. Disponibles: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

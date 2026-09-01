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


class ConvNetBN(nn.Module):
    """CNN 1-D "sofisticada sin pasarse": BatchNorm + Dropout en cada bloque.

    Frente a ConvNet1D añade tres cosas con motivación concreta:
      * **BatchNorm1d** tras cada convolución: estabiliza las activaciones y
        permite lr más agresivos; con colas de |z| ≈ 30 en la entrada, evita
        que un lote extremo desplace la escala interna de la red.
      * **Dropout también en el tronco convolucional** (no solo en la cabeza):
        regularización que importará cuando entrenemos con pocas ventanas.
      * **Un bloque más y cabeza densa de dos capas**: más capacidad
        (~×4 parámetros que cnn_l) sin cambiar la familia del modelo.
    """

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 128),
                 kernel: int = 5, fc: tuple[int, ...] = (128, 64),
                 dropout: float = 0.2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = 1
        for i, ch in enumerate(channels):
            blocks += [nn.Conv1d(prev, ch, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(ch), nn.ReLU()]
            if i < 3:                      # 60 -> 30 -> 15 -> 7; no agotar la longitud
                blocks.append(nn.MaxPool1d(2))
            blocks.append(nn.Dropout(dropout))
            prev = ch
        self.conv = nn.Sequential(*blocks)
        head: list[nn.Module] = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        for h in fc:
            head += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 60) -> (B,)
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


class _TCNBlock(nn.Module):
    """Bloque residual con convolución dilatada (estilo TCN)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(c_out), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(c_out),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class TCN(nn.Module):
    """Red convolucional temporal: dilataciones crecientes + conexiones residuales.

    La dilatación duplica el campo receptivo por bloque sin apilar pooling:
    con dilations (1, 2, 4, 8) y kernel 3, la última capa "ve" toda la ventana
    de 60 días. Es la arquitectura convolucional canónica para series
    temporales y el candidato natural cuando la hipótesis es que el ORDEN de
    los retornos (rachas, clustering) lleva señal más allá de su magnitud.
    """

    def __init__(self, channels: int = 64, kernel: int = 3,
                 dilations: tuple[int, ...] = (1, 2, 4, 8),
                 fc: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        blocks, prev = [], 1
        for d in dilations:
            blocks.append(_TCNBlock(prev, channels, kernel, d, dropout))
            prev = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, fc), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 60) -> (B,)
        return self.head(self.tcn(x.unsqueeze(1))).squeeze(-1)


class GRUNet(nn.Module):
    """Recurrente (GRU) de 2 capas: el contraste secuencial de la comparativa.

    Procesa la ventana paso a paso manteniendo un estado oculto; en teoría es
    la familia más natural para memoria de rachas. En la práctica, con
    ventanas cortas (60) y señal dominada por la escala reciente, las
    convolucionales suelen igualarla o superarla con menos coste — la
    comparativa lo mide en lugar de suponerlo.
    """

    def __init__(self, hidden: int = 64, layers: int = 2, fc: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, fc), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 60) -> (B,)
        out, _ = self.gru(x.unsqueeze(-1))               # (B, 60, H)
        return self.head(out[:, -1]).squeeze(-1)         # último estado


#: Registro de arquitecturas: el nombre + kwargs es todo lo que se persiste
#: para reconstruir el modelo congelado en notebooks posteriores.
_REGISTRY = {"mlp": MLP, "cnn": ConvNet1D, "cnn_bn": ConvNetBN,
             "tcn": TCN, "gru": GRUNet}


def build_model(name: str, **kwargs) -> nn.Module:
    """Construye una arquitectura del registro por nombre (fail-fast si no existe)."""
    if name not in _REGISTRY:
        raise KeyError(f"Arquitectura desconocida: {name!r}. Disponibles: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

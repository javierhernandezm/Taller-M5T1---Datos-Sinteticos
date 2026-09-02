"""Diffusion-TS adaptado al experimento de volatilidad del taller.

La implementación activa, :class:`DiffusionTSR61Generator`, recibe exactamente
el contrato común estandarizado ``[X60 | y]``. Los retornos son tokens
temporales y el target es un token especial con proyección y cabeza propias;
participa en la atención conjunta, pero no se interpreta como un retorno. Así
la comparación con VAE, WGAN-GP y RealNVP mantiene idéntica información.

El módulo conserva también el prototipo exploratorio
:class:`DiffusionTSGenerator`, que reconstruye trayectorias R81 y deriva el
target de sus 21 retornos futuros. Sirve para reproducir la investigación
inicial, pero no entra en las tablas definitivas porque tiene una ventaja de
representación frente a los generadores R61.

La red conserva los elementos centrales de Diffusion-TS: predicción directa
de x0, Transformer bidireccional, tendencia polinómica, componente Fourier,
pérdida temporal + espectral, calendario coseno, DDIM y EMA. Es una adaptación
univariante al problema local, no una copia literal del notebook externo.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .training import get_device


@dataclass(frozen=True)
class ReconstructionResult:
    """Trayectorias recuperadas y trazabilidad de las filas utilizadas."""

    paths: np.ndarray
    anchor_indices: np.ndarray
    next_indices: np.ndarray
    candidates: int
    rejected_overlap: int
    target_max_abs_error: float | None

    @property
    def coverage(self) -> float:
        return len(self.paths) / max(self.candidates, 1)


def reconstruct_return_paths(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    y: np.ndarray | None = None,
    horizon: int = 21,
    annualization: int = 252,
    atol: float = 1e-7,
) -> ReconstructionResult:
    """Reconstruye trayectorias contiguas a partir de ventanas solapadas.

    Con ``stride == horizon == 21``, la fila siguiente del mismo activo
    contiene en sus últimas 21 posiciones exactamente los retornos usados para
    calcular el target de la fila actual. No se confía solo en el orden: se
    exige coincidencia de activo, tramo y fechas, y además se verifica el
    solapamiento numérico de los otros 39 retornos.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] <= horizon:
        raise ValueError("X debe tener forma (N, W) con W > horizon")
    if len(X) != len(meta):
        raise ValueError("X y meta deben estar alineados y tener la misma longitud")
    required = {"cik", "spell_id", "date_t", "date_y_end"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Faltan columnas de trazabilidad: {sorted(missing)}")

    m = meta.reset_index(drop=True)
    same_entity = (
        m["cik"].astype(str).to_numpy()[:-1] == m["cik"].astype(str).to_numpy()[1:]
    ) & (m["spell_id"].to_numpy()[:-1] == m["spell_id"].to_numpy()[1:])
    date_t_next = pd.to_datetime(m["date_t"].iloc[1:]).to_numpy()
    date_y_end = pd.to_datetime(m["date_y_end"].iloc[:-1]).to_numpy()
    consecutive = same_entity & (date_t_next == date_y_end)

    anchors = np.flatnonzero(consecutive)
    followers = anchors + 1
    overlap = np.isclose(
        X[anchors, horizon:],
        X[followers, : X.shape[1] - horizon],
        rtol=0.0,
        atol=atol,
    ).all(axis=1)
    good, next_good = anchors[overlap], followers[overlap]
    paths = np.concatenate([X[good], X[next_good, -horizon:]], axis=1).astype(
        np.float32
    )

    target_error: float | None = None
    if y is not None and len(paths):
        y = np.asarray(y, dtype=np.float32)
        if len(y) != len(X):
            raise ValueError("y y X deben tener la misma longitud")
        y_rebuilt = np.log(
            np.sqrt(annualization * np.mean(paths[:, -horizon:] ** 2, axis=1))
        )
        target_error = float(np.max(np.abs(y_rebuilt - y[good])))

    return ReconstructionResult(
        paths=paths,
        anchor_indices=good,
        next_indices=next_good,
        candidates=len(X),
        rejected_overlap=int((~overlap).sum()),
        target_max_abs_error=target_error,
    )


def paths_to_xy(
    paths: np.ndarray,
    *,
    window_len: int,
    horizon: int,
    x_mu: float,
    x_sd: float,
    y_mu: float,
    y_sd: float,
    annualization: int = 252,
) -> np.ndarray:
    """Convierte trayectorias físicas a la representación estandarizada [X|y]."""
    paths = np.asarray(paths, dtype=np.float32)
    if paths.ndim != 2 or paths.shape[1] != window_len + horizon:
        raise ValueError(
            f"Se esperaban trayectorias de longitud {window_len + horizon}"
        )
    if x_sd <= 0 or y_sd <= 0:
        raise ValueError("Las desviaciones de estandarización deben ser positivas")
    X = (paths[:, :window_len] - x_mu) / x_sd
    sigma = np.sqrt(annualization * np.mean(paths[:, window_len:] ** 2, axis=1))
    y = (np.log(sigma) - y_mu) / y_sd
    XY = np.column_stack([X, y]).astype(np.float32)
    if not np.isfinite(XY).all():
        raise FloatingPointError("El muestreo produjo valores no finitos")
    return XY


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Calendario coseno de Improved DDPM, estable para pocos pasos."""
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def _sinusoidal_embedding(index: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10_000) / max(half - 1, 1)
    freq = torch.exp(-scale * torch.arange(half, device=index.device))
    angles = index.float()[:, None] * freq[None]
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(emb, (0, dim - emb.shape[1]))


def _positional_encoding(length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32)
    return _sinusoidal_embedding(pos, dim).unsqueeze(0)


class _DiffusionTSDenoiser(nn.Module):
    """Transformer no causal con salidas de tendencia, Fourier y residuo."""

    def __init__(
        self,
        seq_len: int,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        trend_degree: int,
        seasonal_k: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible por n_heads")
        self.seq_len = seq_len
        self.trend_degree = trend_degree
        self.seasonal_k = seasonal_k
        self.input_projection = nn.Linear(1, d_model)
        self.register_buffer(
            "position", _positional_encoding(seq_len, d_model), persistent=False
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.trend_head = nn.Linear(d_model, trend_degree + 1)
        self.seasonal_head = nn.Linear(d_model, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )
        grid = torch.linspace(-1.0, 1.0, seq_len)
        powers = torch.stack([grid**p for p in range(trend_degree + 1)], dim=0)
        self.register_buffer("trend_basis", powers, persistent=False)

    def _seasonality(self, raw: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(raw.float(), dim=1, norm="ortho")
        if spectrum.shape[1] <= 1 or self.seasonal_k <= 0:
            return torch.zeros_like(raw, dtype=torch.float32)
        amplitude = spectrum.abs()
        amplitude[:, 0] = -torch.inf
        k = min(self.seasonal_k, spectrum.shape[1] - 1)
        keep = amplitude.topk(k, dim=1).indices
        filtered = torch.zeros_like(spectrum)
        selected = spectrum.gather(1, keep)
        filtered.scatter_(1, keep, selected)
        return torch.fft.irfft(filtered, n=self.seq_len, dim=1, norm="ortho")

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x_t.unsqueeze(-1))
        time = self.time_mlp(_sinusoidal_embedding(timestep, h.shape[-1]))
        h = h + self.position + time[:, None, :]
        h = self.norm(self.encoder(h))

        coefficients = self.trend_head(h.mean(dim=1))
        trend = coefficients @ self.trend_basis
        seasonal = self._seasonality(self.seasonal_head(h).squeeze(-1))
        residual = self.residual_head(h).squeeze(-1)
        return trend + seasonal + residual


class _DiffusionTSJointDenoiser(nn.Module):
    """Denoiser para el contrato R61: 60 retornos y un token objetivo especial.

    El target no ocupa una posición temporal ficticia. Tiene proyección, tipo y
    cabeza propios, pero participa en la atención bidireccional para que la red
    aprenda la distribución conjunta ``p(X, y)`` igual que VAE, WGAN-GP y
    RealNVP. La descomposición tendencia/Fourier se aplica exclusivamente a los
    60 retornos, donde sí tiene significado temporal.
    """

    def __init__(
        self,
        window_len: int,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        trend_degree: int,
        seasonal_k: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible por n_heads")
        self.window_len = window_len
        self.seasonal_k = seasonal_k
        self.return_projection = nn.Linear(1, d_model)
        self.target_projection = nn.Linear(1, d_model)
        self.token_type = nn.Embedding(2, d_model)
        self.register_buffer(
            "position", _positional_encoding(window_len + 1, d_model), persistent=False
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.trend_head = nn.Linear(d_model, trend_degree + 1)
        self.seasonal_head = nn.Linear(d_model, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )
        self.target_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )
        grid = torch.linspace(-1.0, 1.0, window_len)
        powers = torch.stack([grid**p for p in range(trend_degree + 1)], dim=0)
        self.register_buffer("trend_basis", powers, persistent=False)

    def _seasonality(self, raw: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(raw.float(), dim=1, norm="ortho")
        if spectrum.shape[1] <= 1 or self.seasonal_k <= 0:
            return torch.zeros_like(raw, dtype=torch.float32)
        amplitude = spectrum.abs()
        amplitude[:, 0] = -torch.inf
        k = min(self.seasonal_k, spectrum.shape[1] - 1)
        keep = amplitude.topk(k, dim=1).indices
        filtered = torch.zeros_like(spectrum)
        filtered.scatter_(1, keep, spectrum.gather(1, keep))
        return torch.fft.irfft(filtered, n=self.window_len, dim=1, norm="ortho")

    def forward(self, xy_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        returns = self.return_projection(xy_t[:, : self.window_len].unsqueeze(-1))
        target = self.target_projection(xy_t[:, self.window_len :].unsqueeze(-1))
        tokens = torch.cat([returns, target], dim=1)
        types = torch.cat(
            [
                torch.zeros(self.window_len, dtype=torch.long, device=xy_t.device),
                torch.ones(1, dtype=torch.long, device=xy_t.device),
            ]
        )
        time = self.time_mlp(_sinusoidal_embedding(timestep, tokens.shape[-1]))
        hidden = tokens + self.position + self.token_type(types)[None] + time[:, None]
        hidden = self.norm(self.encoder(hidden))

        return_hidden = hidden[:, : self.window_len]
        target_hidden = hidden[:, self.window_len]
        coefficients = self.trend_head(return_hidden.mean(dim=1))
        trend = coefficients @ self.trend_basis
        seasonal = self._seasonality(self.seasonal_head(return_hidden).squeeze(-1))
        residual = self.residual_head(return_hidden).squeeze(-1)
        pooled = return_hidden.mean(dim=1)
        predicted_target = self.target_head(torch.cat([target_hidden, pooled], dim=1))
        return torch.cat([trend + seasonal + residual, predicted_target], dim=1)


class DiffusionTSGenerator:
    """Generador temporal de trayectorias y pares estandarizados ``[X | y]``."""

    name = "diffusion_ts_path81"

    def __init__(
        self,
        *,
        window_len: int = 60,
        horizon: int = 21,
        annualization: int = 252,
        diffusion_steps: int = 500,
        sample_steps: int = 50,
        train_steps: int = 3_000,
        batch_size: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.05,
        trend_degree: int = 3,
        seasonal_k: int = 8,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        spectral_weight: float = 0.1,
        ema_decay: float = 0.995,
        warmup_steps: int = 200,
        grad_clip: float = 1.0,
    ) -> None:
        self.cfg = {
            "window_len": window_len,
            "horizon": horizon,
            "annualization": annualization,
            "diffusion_steps": diffusion_steps,
            "sample_steps": sample_steps,
            "train_steps": train_steps,
            "batch_size": batch_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "ff_mult": ff_mult,
            "dropout": dropout,
            "trend_degree": trend_degree,
            "seasonal_k": seasonal_k,
            "lr": lr,
            "weight_decay": weight_decay,
            "spectral_weight": spectral_weight,
            "ema_decay": ema_decay,
            "warmup_steps": warmup_steps,
            "grad_clip": grad_clip,
        }
        self.history_: dict[str, list[float]] = {}

    @property
    def seq_len(self) -> int:
        return int(self.cfg["window_len"] + self.cfg["horizon"])

    def _build(self, device: torch.device) -> None:
        c = self.cfg
        self.model_ = _DiffusionTSDenoiser(
            self.seq_len,
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_layers=c["n_layers"],
            ff_mult=c["ff_mult"],
            dropout=c["dropout"],
            trend_degree=c["trend_degree"],
            seasonal_k=c["seasonal_k"],
        ).to(device)
        self.ema_model_ = copy.deepcopy(self.model_).eval().requires_grad_(False)
        betas = _cosine_beta_schedule(c["diffusion_steps"]).to(device)
        alphas = 1.0 - betas
        self.alpha_bar_ = torch.cumprod(alphas, dim=0)
        self.device_ = device

    def _loss(
        self, clean: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        alpha = self.alpha_bar_[timestep][:, None]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
        predicted = self.model_(noisy, timestep)
        temporal = F.l1_loss(predicted, clean)
        pred_fft = torch.view_as_real(
            torch.fft.rfft(predicted.float(), dim=1, norm="ortho")
        )
        real_fft = torch.view_as_real(
            torch.fft.rfft(clean.float(), dim=1, norm="ortho")
        )
        spectral = F.l1_loss(pred_fft, real_fft)
        total = temporal + self.cfg["spectral_weight"] * spectral
        return total, temporal, spectral

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.cfg["ema_decay"]
        for ema, current in zip(self.ema_model_.parameters(), self.model_.parameters()):
            ema.mul_(decay).add_(current, alpha=1.0 - decay)
        for ema, current in zip(self.ema_model_.buffers(), self.model_.buffers()):
            ema.copy_(current)

    def fit_paths(
        self,
        paths: np.ndarray,
        *,
        x_mu: float,
        x_sd: float,
        y_mu: float,
        y_sd: float,
        seed: int = 42,
        verbose: bool = True,
    ) -> DiffusionTSGenerator:
        """Ajusta el modelo únicamente con trayectorias físicas de train."""
        paths = np.asarray(paths, dtype=np.float32)
        if paths.ndim != 2 or paths.shape[1] != self.seq_len:
            raise ValueError(f"paths debe tener forma (N, {self.seq_len})")
        if not np.isfinite(paths).all() or x_sd <= 0 or y_sd <= 0:
            raise ValueError("Datos o estadísticos de estandarización inválidos")

        self.x_mu_, self.x_sd_ = float(x_mu), float(x_sd)
        self.y_mu_, self.y_sd_ = float(y_mu), float(y_sd)
        device = get_device()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        self._build(device)

        data = torch.as_tensor(
            (paths - x_mu) / x_sd, dtype=torch.float32, device=device
        )
        opt = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        total_steps = self.cfg["train_steps"]
        warmup = min(self.cfg["warmup_steps"], total_steps)

        def lr_factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        self.history_ = {"loss": [], "temporal": [], "spectral": [], "lr": []}
        started = time.perf_counter()

        self.model_.train()
        for step in range(total_steps):
            current_lr = self.cfg["lr"] * lr_factor(step)
            for group in opt.param_groups:
                group["lr"] = current_lr
            idx = rng.integers(
                0, len(data), size=min(self.cfg["batch_size"], len(data))
            )
            batch = data[torch.as_tensor(idx, device=device)]
            timestep = torch.randint(
                0, self.cfg["diffusion_steps"], (len(batch),), device=device
            )
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                loss, temporal, spectral = self._loss(batch, timestep)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(self.model_.parameters(), self.cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            self._update_ema()

            self.history_["loss"].append(float(loss.detach()))
            self.history_["temporal"].append(float(temporal.detach()))
            self.history_["spectral"].append(float(spectral.detach()))
            self.history_["lr"].append(float(current_lr))
            if verbose and (step == 0 or (step + 1) % max(total_steps // 10, 1) == 0):
                print(
                    f"  Diffusion-TS {step + 1:>5}/{total_steps} "
                    f"loss={self.history_['loss'][-1]:.4f} "
                    f"({time.perf_counter() - started:.0f}s)"
                )

        self.training_seconds_ = time.perf_counter() - started
        self.fit_seed_ = int(seed)
        self.n_fit_ = len(paths)
        return self

    @torch.no_grad()
    def sample_paths(
        self, n: int, seed: int = 0, *, batch_size: int = 512
    ) -> np.ndarray:
        """Muestrea trayectorias físicas mediante DDIM determinista (eta=0)."""
        if not hasattr(self, "ema_model_"):
            raise RuntimeError("El generador no está ajustado")
        if n <= 0:
            return np.empty((0, self.seq_len), dtype=np.float32)
        model = self.ema_model_.to(self.device_).eval()
        generator = torch.Generator(device=self.device_).manual_seed(seed)
        grid = np.linspace(
            self.cfg["diffusion_steps"] - 1,
            0,
            min(self.cfg["sample_steps"], self.cfg["diffusion_steps"]),
            dtype=int,
        )
        timesteps = np.unique(grid)[::-1].copy()
        out: list[np.ndarray] = []
        amp = self.device_.type == "cuda"

        for start in range(0, n, batch_size):
            size = min(batch_size, n - start)
            x = torch.randn(
                (size, self.seq_len), generator=generator, device=self.device_
            )
            for position, t_value in enumerate(timesteps):
                t = torch.full(
                    (size,), int(t_value), device=self.device_, dtype=torch.long
                )
                with torch.autocast(
                    device_type=self.device_.type, dtype=torch.float16, enabled=amp
                ):
                    x0 = model(x, t).float()
                if position == len(timesteps) - 1:
                    x = x0
                    continue
                next_t = int(timesteps[position + 1])
                alpha_t = self.alpha_bar_[int(t_value)]
                alpha_next = self.alpha_bar_[next_t]
                eps = (x - alpha_t.sqrt() * x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-8)
                x = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * eps
            out.append(x.cpu().numpy().astype(np.float32))

        standardized = np.concatenate(out)
        paths = standardized * self.x_sd_ + self.x_mu_
        if not np.isfinite(paths).all():
            raise FloatingPointError("El muestreo DDIM produjo valores no finitos")
        return paths.astype(np.float32)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Devuelve ``[X | y]`` estandarizado, compatible con TSTR."""
        paths = self.sample_paths(n, seed)
        return paths_to_xy(
            paths,
            window_len=self.cfg["window_len"],
            horizon=self.cfg["horizon"],
            annualization=self.cfg["annualization"],
            x_mu=self.x_mu_,
            x_sd=self.x_sd_,
            y_mu=self.y_mu_,
            y_sd=self.y_sd_,
        )

    def save(self, path: Path | str) -> None:
        """Guarda lo necesario para reanudar evaluación sin reentrenar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.ema_model_.state_dict().items()}
        torch.save(
            {
                "cfg": self.cfg,
                "ema_state": state,
                "x_mu": self.x_mu_,
                "x_sd": self.x_sd_,
                "y_mu": self.y_mu_,
                "y_sd": self.y_sd_,
                "history": self.history_,
                "training_seconds": self.training_seconds_,
                "fit_seed": self.fit_seed_,
                "n_fit": self.n_fit_,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: Path | str, device: torch.device | None = None
    ) -> DiffusionTSGenerator:
        """Carga un checkpoint generado por :meth:`save`."""
        device = device or get_device()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(**payload["cfg"])
        obj._build(device)
        obj.ema_model_.load_state_dict(payload["ema_state"])
        obj.model_.load_state_dict(payload["ema_state"])
        obj.x_mu_, obj.x_sd_ = float(payload["x_mu"]), float(payload["x_sd"])
        obj.y_mu_, obj.y_sd_ = float(payload["y_mu"]), float(payload["y_sd"])
        obj.history_ = payload.get("history", {})
        obj.training_seconds_ = float(payload.get("training_seconds", 0.0))
        obj.fit_seed_ = int(payload.get("fit_seed", -1))
        obj.n_fit_ = int(payload.get("n_fit", 0))
        return obj


class DiffusionTSR61Generator(DiffusionTSGenerator):
    """Diffusion-TS con el mismo contrato R61 que los generadores existentes.

    La entrada y la salida son matrices estandarizadas ``[X60 | y]``. Los 60
    retornos forman la secuencia temporal y ``y`` es un token especial que
    participa en la atención, no el retorno número 61. Esto mantiene la
    comparación de información exactamente alineada con VAE, WGAN-GP y
    RealNVP, a diferencia del prototipo exploratorio de trayectorias R81.
    """

    name = "diffusion_ts"

    @property
    def seq_len(self) -> int:
        return int(self.cfg["window_len"] + 1)

    def _build(self, device: torch.device) -> None:
        c = self.cfg
        self.model_ = _DiffusionTSJointDenoiser(
            c["window_len"],
            d_model=c["d_model"],
            n_heads=c["n_heads"],
            n_layers=c["n_layers"],
            ff_mult=c["ff_mult"],
            dropout=c["dropout"],
            trend_degree=c["trend_degree"],
            seasonal_k=c["seasonal_k"],
        ).to(device)
        self.ema_model_ = copy.deepcopy(self.model_).eval().requires_grad_(False)
        betas = _cosine_beta_schedule(c["diffusion_steps"]).to(device)
        alphas = 1.0 - betas
        self.alpha_bar_ = torch.cumprod(alphas, dim=0)
        self.device_ = device

    def _loss(
        self, clean: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        alpha = self.alpha_bar_[timestep][:, None]
        noise = torch.randn_like(clean)
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
        predicted = self.model_(noisy, timestep)
        temporal = F.l1_loss(predicted, clean)
        window_len = self.cfg["window_len"]
        pred_fft = torch.view_as_real(
            torch.fft.rfft(predicted[:, :window_len].float(), dim=1, norm="ortho")
        )
        real_fft = torch.view_as_real(
            torch.fft.rfft(clean[:, :window_len].float(), dim=1, norm="ortho")
        )
        spectral = F.l1_loss(pred_fft, real_fft)
        total = temporal + self.cfg["spectral_weight"] * spectral
        return total, temporal, spectral

    def fit(
        self, XY: np.ndarray, seed: int = 0, verbose: bool = False
    ) -> DiffusionTSR61Generator:
        """Ajusta la distribución conjunta estandarizada sin cambiar de representación."""
        XY = np.asarray(XY, dtype=np.float32)
        if XY.ndim != 2 or XY.shape[1] != self.seq_len:
            raise ValueError(f"XY debe tener forma (N, {self.seq_len}) = [X60 | y]")
        return self.fit_paths(
            XY,
            x_mu=0.0,
            x_sd=1.0,
            y_mu=0.0,
            y_sd=1.0,
            seed=seed,
            verbose=verbose,
        )

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Muestrea directamente pares R61 estandarizados ``[X60 | y]``."""
        return self.sample_paths(n, seed)

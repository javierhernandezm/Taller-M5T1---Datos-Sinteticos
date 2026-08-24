"""
generators.py — Modelos generativos de ventanas (X, y) sintéticas.

Todos los generadores comparten la MISMA interfaz, para que el notebook 04
pueda barrer la malla real×sintético sin condicionales por modelo:

    gen = XxxGenerator(...)
    gen.fit(XY_train)                 # XY: (N, 61) = [60 retornos | 1 target]
    XY_synth = gen.sample(n, seed)    # (n, 61)

Convención de datos: los generadores trabajan sobre el par CONJUNTO [X | y]
estandarizado. Modelar la distribución conjunta (y no X sola) es lo que permite
generar muestras supervisadas — es la "OPT2" de las transparencias del profesor:
z -> Generador -> (X_g, y_g). Un generador que solo produjera X obligaría a
etiquetar después, y esa etiqueta sería inventada por otro modelo.

Familias implementadas
----------------------
Baselines sin red neuronal (§1 del enunciado: "un cuarto modelo simple"):
  * JitterGenerator       muestras reales + ruido gaussiano
  * GaussianGenerator     N(mu, Sigma) ajustada al conjunto (Ledoit-Wolf opcional)
  * BlockBootstrapGenerator  remuestreo por bloques contiguos

Redes neuronales (las tres familias distintas que pide el enunciado):
  * VAEGenerator          autoencoder variacional (latente variacional)
  * WGANGPGenerator       Wasserstein GAN con gradient penalty (adversarial)
  * RealNVPGenerator      normalizing flow con verosimilitud exacta (biyectiva)

Cada red expone además `history_` (listas de pérdidas por época) para poder
dibujar las curvas de convergencia que exige el enunciado.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from .training import get_device


# =========================================================================== #
# Interfaz común
# =========================================================================== #

class BaseGenerator(ABC):
    """Contrato mínimo: fit(XY) y sample(n) -> (n, D)."""

    name: str = "base"

    @abstractmethod
    def fit(self, XY: np.ndarray) -> "BaseGenerator": ...

    @abstractmethod
    def sample(self, n: int, seed: int = 0) -> np.ndarray: ...

    #: historia de entrenamiento {clave: [valores por época]}; vacío en los no-neuronales
    history_: dict[str, list[float]]


# =========================================================================== #
# 1. Baselines sin red
# =========================================================================== #

class JitterGenerator(BaseGenerator):
    """Muestras reales + ruido gaussiano isotrópico.

    El generador trivial que pide el enunciado. No aprende ninguna distribución
    nueva: para sigma pequeño equivale a añadir regularización de Tikhonov al
    modelo downstream. `noise` se expresa en unidades de la desviación típica
    de cada columna, de modo que el mismo valor significa lo mismo en X y en y.
    """

    name = "jitter"

    def __init__(self, noise: float = 0.1) -> None:
        self.noise = noise
        self.history_ = {}

    def fit(self, XY: np.ndarray) -> "JitterGenerator":
        self.XY_ = XY.astype(np.float32)
        self.sd_ = XY.std(axis=0).astype(np.float32)
        return self

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.XY_), size=n)
        base = self.XY_[idx]
        return base + rng.normal(0, self.noise, size=base.shape).astype(np.float32) * self.sd_


class GaussianGenerator(BaseGenerator):
    """Gaussiana multivariante ajustada al par conjunto [X | y].

    Captura TODA la estructura de dependencia lineal: autocorrelación de los
    retornos dentro de la ventana y su covarianza con el target. No captura
    colas gruesas, asimetría ni clustering de volatilidad (dependencias no
    lineales invisibles a la matriz de covarianza) — que es exactamente lo que
    lo convierte en el rival honesto de las redes.

    `shrinkage` aplica encogimiento tipo Ledoit-Wolf hacia una diagonal: con
    pocas muestras reales (el régimen de escasez de la malla) la covarianza
    muestral de 61x61 es singular o casi, y sin encogimiento el muestreo
    degenera.
    """

    name = "gaussian"

    def __init__(self, shrinkage: float | None = None) -> None:
        self.shrinkage = shrinkage
        self.history_ = {}

    def fit(self, XY: np.ndarray) -> "GaussianGenerator":
        X = XY.astype(np.float64)
        self.mu_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        d = cov.shape[0]
        # encogimiento automático si no se especifica: crece cuando N ~ d
        lam = self.shrinkage if self.shrinkage is not None else min(1.0, d / max(len(X), 1))
        target = np.eye(d) * np.trace(cov) / d
        self.cov_ = (1 - lam) * cov + lam * target
        self.lam_ = lam
        return self

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mu_, self.cov_, size=n).astype(np.float32)


class BlockBootstrapGenerator(BaseGenerator):
    """Bootstrap estacionario por bloques sobre la ventana de retornos.

    Reconstruye cada ventana sintética concatenando bloques contiguos tomados
    al azar de ventanas reales. Al reutilizar tramos reales preserva colas
    gruesas y parte del clustering POR CONSTRUCCIÓN, sin aprender nada: es el
    baseline que un tribunal financiero exigiría antes de creerse una GAN.

    El target se toma del de la ventana donante del último bloque — el tramo
    más cercano al momento de predicción, que es el que más determina la
    volatilidad futura.
    """

    name = "block_bootstrap"

    def __init__(self, mean_block: int = 10) -> None:
        self.mean_block = mean_block
        self.history_ = {}

    def fit(self, XY: np.ndarray) -> "BlockBootstrapGenerator":
        self.X_ = XY[:, :-1].astype(np.float32)
        self.y_ = XY[:, -1].astype(np.float32)
        self.win_ = self.X_.shape[1]
        return self

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        N, W = len(self.X_), self.win_
        out = np.empty((n, W + 1), dtype=np.float32)
        p = 1.0 / self.mean_block  # longitud de bloque ~ Geométrica(p)
        for i in range(n):
            pos, donor = 0, 0
            while pos < W:
                donor = rng.integers(0, N)
                L = min(int(rng.geometric(p)), W - pos)
                start = rng.integers(0, W - L + 1)
                out[i, pos : pos + L] = self.X_[donor, start : start + L]
                pos += L
            out[i, W] = self.y_[donor]  # target del donante del último bloque
        return out


# =========================================================================== #
# 2. Redes: utilidades comunes
# =========================================================================== #

def _mlp(sizes: list[int], out_act: nn.Module | None = None) -> nn.Sequential:
    """MLP ReLU a partir de una lista de anchuras."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:-1] + [sizes[-1]]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers = layers[:-1]  # sin ReLU final
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    """Itera índices de mini-lote sobre una permutación."""
    perm = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


# =========================================================================== #
# 3. VAE — familia latente variacional
# =========================================================================== #

class _VAENet(nn.Module):
    def __init__(self, d: int, latent: int, hidden: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)   # truco de reparametrización
        return self.dec(z), mu, logvar


class VAEGenerator(BaseGenerator):
    """Autoencoder variacional sobre el par conjunto [X | y].

    Loss = MSE de reconstrucción + beta * KL(q(z|x) || N(0,I)). El término KL
    empuja el latente hacia una normal estándar, y por eso muestrear z ~ N(0,I)
    y decodificar produce muestras nuevas.

    Patología conocida y esperable en este dominio: el decoder gaussiano
    produce muestras SUAVIZADAS, es decir colas y curtosis por debajo de las
    reales. Es una hipótesis contrastable — la auditamos con QQ-plots.

    **Ruido de observación.** El decoder devuelve E[x|z], no una muestra de
    p(x|z). Muestrear solo la media pierde el segundo término de
    Var(x) = Var(E[x|z]) + E[Var(x|z)] y produce sintéticos con desviación
    típica muy por debajo de la real (medido: 0,39 frente a 1,00). Tras
    entrenar estimamos la desviación residual por dimensión y la reinyectamos
    al muestrear: eso equivale a muestrear de un decoder gaussiano con varianza
    aprendida, que es el modelo generativo que el ELBO define realmente.
    Desactivable con `observation_noise=False` para exhibir el efecto.
    """

    name = "vae"

    def __init__(self, latent: int = 16, hidden: int = 256, beta: float = 1.0,
                 epochs: int = 60, batch_size: int = 512, lr: float = 1e-3,
                 observation_noise: bool = True) -> None:
        self.cfg = dict(latent=latent, hidden=hidden, beta=beta, epochs=epochs,
                        batch_size=batch_size, lr=lr,
                        observation_noise=observation_noise)
        self.history_ = {}

    def fit(self, XY: np.ndarray, seed: int = 0, verbose: bool = False) -> "VAEGenerator":
        c = self.cfg
        dev = get_device()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        d = XY.shape[1]
        self.net_ = _VAENet(d, c["latent"], c["hidden"]).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=c["lr"])
        Xt = torch.as_tensor(XY, dtype=torch.float32, device=dev)
        self.history_ = {"loss": [], "recon": [], "kl": []}

        for ep in range(c["epochs"]):
            tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            nb = 0
            for idx in _batches(len(Xt), c["batch_size"], rng):
                xb = Xt[torch.as_tensor(idx, device=dev)]
                xr, mu, logvar = self.net_(xb)
                recon = ((xr - xb) ** 2).sum(1).mean()
                kl = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1)).mean()
                loss = recon + c["beta"] * kl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                opt.step()
                tot["loss"] += float(loss.detach()); tot["recon"] += float(recon.detach())
                tot["kl"] += float(kl.detach())
                nb += 1
            for k in tot:
                self.history_[k].append(tot[k] / nb)
            if verbose and ep % 10 == 0:
                print(f"  VAE ep{ep:3d} loss {self.history_['loss'][-1]:.3f}")

        # Desviación residual por dimensión = sigma del decoder gaussiano (ver docstring)
        self.net_.eval()
        with torch.no_grad():
            xr, _, _ = self.net_(Xt)
            self.resid_sd_ = (Xt - xr).std(0).cpu().numpy().astype(np.float32)
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        dev = get_device()
        torch.manual_seed(seed)
        self.net_.eval()
        z = torch.randn(n, self.cfg["latent"], device=dev)
        out = self.net_.dec(z).cpu().numpy().astype(np.float32)
        if self.cfg["observation_noise"]:
            rng = np.random.default_rng(seed)
            out = out + rng.normal(0, 1, out.shape).astype(np.float32) * self.resid_sd_
        return out


# =========================================================================== #
# 4. WGAN-GP — familia adversarial
# =========================================================================== #

class WGANGPGenerator(BaseGenerator):
    """Wasserstein GAN con gradient penalty.

    Por qué WGAN-GP y no una GAN clásica: la loss de una GAN vanilla oscila
    alrededor de un equilibrio y NO demuestra convergencia — choca de frente
    con el requisito del enunciado de aportar curvas de loss convergentes. La
    pérdida del crítico en WGAN-GP aproxima la distancia de Wasserstein entre
    la distribución real y la generada, de modo que sí es una curva
    interpretable y decreciente; además reduce drásticamente el colapso de
    modos, la patología que en datos financieros se manifiesta como pérdida
    de las colas.
    """

    name = "wgan_gp"

    def __init__(self, latent: int = 32, hidden: int = 256, epochs: int = 60,
                 batch_size: int = 512, lr: float = 1e-4, n_critic: int = 5,
                 gp_weight: float = 10.0) -> None:
        self.cfg = dict(latent=latent, hidden=hidden, epochs=epochs, batch_size=batch_size,
                        lr=lr, n_critic=n_critic, gp_weight=gp_weight)
        self.history_ = {}

    def _gradient_penalty(self, critic, real, fake, dev):
        eps = torch.rand(len(real), 1, device=dev)
        mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
        score = critic(mix)
        grad = torch.autograd.grad(score, mix, torch.ones_like(score),
                                   create_graph=True, retain_graph=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, XY: np.ndarray, seed: int = 0, verbose: bool = False) -> "WGANGPGenerator":
        c = self.cfg
        dev = get_device()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        d = XY.shape[1]
        h = c["hidden"]
        self.gen_ = _mlp([c["latent"], h, h, d]).to(dev)
        critic = _mlp([d, h, h, 1]).to(dev)
        og = torch.optim.Adam(self.gen_.parameters(), lr=c["lr"], betas=(0.5, 0.9))
        oc = torch.optim.Adam(critic.parameters(), lr=c["lr"], betas=(0.5, 0.9))
        Xt = torch.as_tensor(XY, dtype=torch.float32, device=dev)
        self.history_ = {"wasserstein": [], "critic": [], "gen": []}

        for ep in range(c["epochs"]):
            acc = {"wasserstein": 0.0, "critic": 0.0, "gen": 0.0}
            nb = 0
            for step, idx in enumerate(_batches(len(Xt), c["batch_size"], rng)):
                real = Xt[torch.as_tensor(idx, device=dev)]
                # --- crítico -------------------------------------------------
                z = torch.randn(len(real), c["latent"], device=dev)
                fake = self.gen_(z).detach()
                gp = self._gradient_penalty(critic, real, fake, dev)
                w_est = critic(real).mean() - critic(fake).mean()
                loss_c = -w_est + c["gp_weight"] * gp
                oc.zero_grad(); loss_c.backward(); oc.step()
                # --- generador (1 de cada n_critic pasos) --------------------
                if step % c["n_critic"] == 0:
                    z = torch.randn(len(real), c["latent"], device=dev)
                    loss_g = -critic(self.gen_(z)).mean()
                    og.zero_grad(); loss_g.backward(); og.step()
                    acc["gen"] += float(loss_g.detach())
                acc["wasserstein"] += float(w_est.detach()); acc["critic"] += float(loss_c.detach())
                nb += 1
            for k in acc:
                self.history_[k].append(acc[k] / max(nb, 1))
            if verbose and ep % 10 == 0:
                print(f"  WGAN ep{ep:3d} W~{self.history_['wasserstein'][-1]:.3f}")
        return self

    @torch.no_grad()
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        dev = get_device()
        torch.manual_seed(seed)
        self.gen_.eval()
        z = torch.randn(n, self.cfg["latent"], device=dev)
        return self.gen_(z).cpu().numpy().astype(np.float32)


# =========================================================================== #
# 5. RealNVP — familia biyectiva (verosimilitud exacta)
# =========================================================================== #

class _CouplingLayer(nn.Module):
    """Capa de acoplamiento afín: transforma media dimensión condicionada a la otra.

    y_a = x_a                              (identidad)
    y_b = x_b * exp(s(x_a)) + t(x_a)       (escala y traslación)

    El jacobiano es triangular, así que su log-determinante es simplemente
    sum(s(x_a)) — de ahí que la verosimilitud sea exacta y barata.
    """

    def __init__(self, d: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.s = _mlp([d, hidden, hidden, d], out_act=nn.Tanh())
        self.t = _mlp([d, hidden, hidden, d])

    def forward(self, x):
        xa = x * self.mask
        s = self.s(xa) * (1 - self.mask)
        t = self.t(xa) * (1 - self.mask)
        y = xa + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(1)

    def inverse(self, y):
        ya = y * self.mask
        s = self.s(ya) * (1 - self.mask)
        t = self.t(ya) * (1 - self.mask)
        return ya + (1 - self.mask) * ((y - t) * torch.exp(-s))


class RealNVPGenerator(BaseGenerator):
    """Normalizing flow RealNVP sobre el par conjunto [X | y].

    Aprende una biyección hacia una normal estándar maximizando la
    verosimilitud exacta. Es el único generador del taller que puede reportar
    la **log-verosimilitud de datos reales no vistos**: una métrica de calidad
    del generador independiente del efecto downstream, que ni VAE ni GAN
    ofrecen. Además su loss es una NLL monótona, sin adversario.
    """

    name = "realnvp"

    def __init__(self, n_layers: int = 8, hidden: int = 256, epochs: int = 60,
                 batch_size: int = 512, lr: float = 1e-3) -> None:
        self.cfg = dict(n_layers=n_layers, hidden=hidden, epochs=epochs,
                        batch_size=batch_size, lr=lr)
        self.history_ = {}

    def _build(self, d: int, dev):
        layers = []
        for i in range(self.cfg["n_layers"]):
            mask = torch.zeros(d, device=dev)
            mask[i % 2 :: 2] = 1.0          # máscaras alternas: par / impar
            layers.append(_CouplingLayer(d, self.cfg["hidden"], mask).to(dev))
        return nn.ModuleList(layers)

    def _log_prob(self, x):
        logdet = torch.zeros(len(x), device=x.device)
        z = x
        for layer in self.layers_:
            z, ld = layer(z)
            logdet = logdet + ld
        # log N(z; 0, I) + log|det J|
        base = -0.5 * (z**2 + np.log(2 * np.pi)).sum(1)
        return base + logdet

    def fit(self, XY: np.ndarray, seed: int = 0, verbose: bool = False) -> "RealNVPGenerator":
        c = self.cfg
        dev = get_device()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.d_ = XY.shape[1]
        self.layers_ = self._build(self.d_, dev)
        params = [p for l in self.layers_ for p in l.parameters()]
        opt = torch.optim.Adam(params, lr=c["lr"])
        Xt = torch.as_tensor(XY, dtype=torch.float32, device=dev)
        self.history_ = {"nll": []}

        for ep in range(c["epochs"]):
            tot, nb = 0.0, 0
            for idx in _batches(len(Xt), c["batch_size"], rng):
                xb = Xt[torch.as_tensor(idx, device=dev)]
                loss = -self._log_prob(xb).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                tot += float(loss.detach()); nb += 1
            self.history_["nll"].append(tot / nb)
            if verbose and ep % 10 == 0:
                print(f"  RealNVP ep{ep:3d} NLL {self.history_['nll'][-1]:.3f}")
        return self

    @torch.no_grad()
    def log_likelihood(self, XY: np.ndarray) -> float:
        """log-verosimilitud media de datos reales no vistos (nats por muestra)."""
        dev = get_device()
        Xt = torch.as_tensor(XY, dtype=torch.float32, device=dev)
        return float(self._log_prob(Xt).mean())

    @torch.no_grad()
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        dev = get_device()
        torch.manual_seed(seed)
        z = torch.randn(n, self.d_, device=dev)
        for layer in reversed(self.layers_):
            z = layer.inverse(z)
        return z.cpu().numpy().astype(np.float32)

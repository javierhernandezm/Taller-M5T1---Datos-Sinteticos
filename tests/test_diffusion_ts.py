"""Pruebas rápidas de Diffusion-TS R81/R61; no usan los datos del proyecto."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.diffusion_ts as diffusion_module
from src.diffusion_ts import (
    DiffusionTSGenerator,
    DiffusionTSR61Generator,
    paths_to_xy,
    reconstruct_return_paths,
)


def test_reconstruccion_usa_solo_continuidad_demostrada():
    returns = np.linspace(-0.04, 0.05, 102, dtype=np.float32)
    X = np.stack([returns[0:60], returns[21:81], returns[42:102]])
    horizon = 21
    paths = np.stack([returns[0:81], returns[21:102]])
    y = np.log(np.sqrt(252 * np.mean(paths[:, -horizon:] ** 2, axis=1)))
    y_all = np.r_[y, 0.0].astype(np.float32)
    meta = pd.DataFrame(
        {
            "cik": ["1", "1", "1"],
            "spell_id": [0, 0, 0],
            "date_t": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "date_y_end": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"]),
        }
    )

    result = reconstruct_return_paths(X, meta, y=y_all, horizon=horizon)

    np.testing.assert_allclose(result.paths, paths, atol=1e-7)
    np.testing.assert_array_equal(result.anchor_indices, [0, 1])
    assert result.target_max_abs_error == pytest.approx(0.0, abs=1e-6)


def test_reconstruccion_rechaza_cambio_de_activo_y_solape_falso():
    base = np.arange(100, dtype=np.float32)
    X = np.stack([base[:10], base[3:13], base[6:16]])
    X[1, 0] += 1.0  # rompe el solape de la primera pareja
    meta = pd.DataFrame(
        {
            "cik": ["1", "1", "2"],
            "spell_id": [0, 0, 0],
            "date_t": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "date_y_end": pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"]),
        }
    )
    result = reconstruct_return_paths(X, meta, horizon=3)
    assert len(result.paths) == 0
    assert result.rejected_overlap == 1


def test_paths_to_xy_calcula_el_target_y_estandariza():
    paths = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    xy = paths_to_xy(
        paths,
        window_len=3,
        horizon=2,
        x_mu=1.0,
        x_sd=2.0,
        y_mu=0.5,
        y_sd=0.25,
        annualization=1,
    )
    np.testing.assert_allclose(xy[0, :3], [0.0, 0.5, 1.0])
    expected_y = (np.log(np.sqrt((4.0**2 + 5.0**2) / 2)) - 0.5) / 0.25
    assert xy[0, -1] == pytest.approx(expected_y)


def test_modelo_minimo_entrena_muestrea_y_es_reproducible(monkeypatch):
    monkeypatch.setattr(diffusion_module, "get_device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(7)
    paths = rng.normal(0.0, 0.02, size=(32, 12)).astype(np.float32)
    future = paths[:, -4:]
    y = np.log(np.sqrt(252 * np.mean(future**2, axis=1)))
    gen = DiffusionTSGenerator(
        window_len=8,
        horizon=4,
        diffusion_steps=8,
        sample_steps=3,
        train_steps=2,
        batch_size=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_mult=2,
        dropout=0.0,
        seasonal_k=2,
        warmup_steps=1,
    ).fit_paths(
        paths,
        x_mu=float(paths.mean()),
        x_sd=float(paths.std()),
        y_mu=float(y.mean()),
        y_sd=float(y.std()),
        seed=11,
        verbose=False,
    )

    a = gen.sample(5, seed=99)
    b = gen.sample(5, seed=99)
    assert a.shape == (5, 9)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert len(gen.history_["loss"]) == 2


def test_r61_respeta_el_contrato_comun_y_el_target_es_token_especial(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(diffusion_module, "get_device", lambda: torch.device("cpu"))
    XY = np.random.default_rng(17).normal(size=(32, 9)).astype(np.float32)
    gen = DiffusionTSR61Generator(
        window_len=8,
        diffusion_steps=8,
        sample_steps=3,
        train_steps=2,
        batch_size=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_mult=2,
        dropout=0.0,
        seasonal_k=2,
        warmup_steps=1,
    ).fit(XY, seed=13)

    synth = gen.sample(7, seed=101)
    assert gen.name == "diffusion_ts"
    assert gen.seq_len == 9
    assert synth.shape == (7, 9)
    assert np.isfinite(synth).all()
    np.testing.assert_array_equal(synth, gen.sample(7, seed=101))
    assert gen.sample(0, seed=101).shape == (0, 9)
    assert gen.model_.return_projection is not gen.model_.target_projection

    checkpoint = tmp_path / "r61.pt"
    gen.save(checkpoint)
    restored = DiffusionTSR61Generator.load(checkpoint, device=torch.device("cpu"))
    np.testing.assert_array_equal(synth, restored.sample(7, seed=101))

    with pytest.raises(ValueError, match=r"\[X60 \| y\]"):
        gen.fit(np.zeros((5, 12), dtype=np.float32))
    with pytest.raises(ValueError, match="inválidos"):
        invalid = XY.copy()
        invalid[0, -1] = np.nan
        gen.fit(invalid)


def test_factory_de_malla_usa_r61_y_la_configuracion_oficial():
    from src.malla import GENERADORES_ACTIVOS, construir_generador

    gen = construir_generador("diffusion_ts")
    assert isinstance(gen, DiffusionTSR61Generator)
    assert gen.cfg["window_len"] == 60
    assert gen.cfg["train_steps"] == 3_000
    assert gen.cfg["sample_steps"] == 50
    assert "diffusion_ts" in GENERADORES_ACTIVOS
    assert "wgan_gp" not in GENERADORES_ACTIVOS

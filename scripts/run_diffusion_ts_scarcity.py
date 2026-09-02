"""Gate de escasez para Diffusion-TS: 3.000 trayectorias y ratio sintético 5x.

Este experimento complementa el TSTR puro. Reajusta el generador desde cero en
cada semilla usando solo las trayectorias declaradas y compara el GRU entrenado
con esas observaciones frente al mismo GRU entrenado con reales + sintéticos.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines import regression_metrics
from src.diffusion_ts import DiffusionTSGenerator, reconstruct_return_paths
from src.malla import submuestrear_por_fechas
from src.models import build_model
from src.training import get_device, predict, train_model

N_REAL = 3_000
RATIO = 5
TRAIN_STEPS = 3_000
SEEDS = (42, 43, 44)
OUTPUT = REPO_ROOT / "reports/diffusion_ts_experiment/scarcity_3000_x5"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_result(path: Path, row: dict) -> None:
    try:
        old = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        old = pd.DataFrame()
    if not old.empty:
        mask = (old["brazo"] == row["brazo"]) & (old["seed"] == row["seed"])
        old = old.loc[~mask]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test_physical: np.ndarray,
    ref: dict,
    std: dict,
    seed: int,
    device: torch.device,
) -> dict:
    started = time.perf_counter()
    kwargs = dict(ref["train_kwargs"])
    kwargs["patience"] = kwargs["epochs"]
    model = build_model(ref["arch"], **ref["arch_kwargs"])
    result = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        seed=seed,
        device=device,
        **kwargs,
    )
    prediction = predict(model, X_test, device) * std["y_sd"] + std["y_mu"]
    metrics = regression_metrics(y_test_physical, prediction)
    return {
        "n_train": len(X_train),
        "val_mse": result.best_val,
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "epoca_mejor": result.best_epoch,
        "downstream_seconds": time.perf_counter() - started,
    }


def render_report(results: pd.DataFrame, config: dict) -> None:
    pivot = results.pivot(index="seed", columns="brazo", values="test_r2").reset_index()
    pivot["delta_r2"] = pivot["diffusion_ts_path81"] - pivot["solo_real_path81"]
    deltas = pivot["delta_r2"]
    mean = float(deltas.mean())
    sd = float(deltas.std())
    half = float(stats.t.ppf(0.975, len(deltas) - 1) * sd / np.sqrt(len(deltas)))
    ci = [mean - half, mean + half]
    historical = pd.read_csv(
        REPO_ROOT / "data/processed/malla_ratios_finos_resumen.csv"
    )
    comparable = historical.loc[
        (historical["n_real"] == N_REAL)
        & (historical["ratio"] == RATIO)
        & historical["generador"].isin(
            ["ninguno", "wgan_gp", "vae", "realnvp", "jitter"]
        ),
        ["generador", "test_r2_media", "test_r2_sd", "n_seeds"],
    ]
    conclusion = (
        "supera el control real en las tres semillas: merece entrar en la malla completa"
        if ci[0] > 0
        else "no demuestra una mejora estable sobre el control real"
    )
    summary = {
        "config": config,
        "delta_r2_mean": mean,
        "delta_r2_sd": sd,
        "delta_r2_ci95": ci,
        "wins": int((deltas > 0).sum()),
        "pairs": len(deltas),
        "conclusion": conclusion,
        "historical_same_cell": comparable.to_dict(orient="records"),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def table(frame: pd.DataFrame) -> str:
        header = "| " + " | ".join(map(str, frame.columns)) + " |"
        separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        rows = []
        for row in frame.itertuples(index=False, name=None):
            values = [
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in row
            ]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *rows])

    report = [
        "# Gate de escasez Diffusion-TS",
        "",
        f"**Conclusión:** {conclusion}.",
        "",
        f"Cada réplica usa {N_REAL:,} trayectorias reales y añade {N_REAL * RATIO:,} sintéticas.",
        "El generador se reajusta desde cero y no ve otras filas de train.",
        "",
        "## Comparación emparejada",
        "",
        table(pivot),
        "",
        f"Delta R² medio: **{mean:+.4f}**, IC95 t de Student [{ci[0]:+.4f}, {ci[1]:+.4f}].",
        "",
        "## Celda histórica del notebook 04",
        "",
        table(comparable),
        "",
        (
            "La referencia histórica usa ventanas [X|y], mientras Diffusion-TS usa "
            "trayectorias de 81 retornos. Sirve para contextualizar magnitudes, no "
            "como contraste emparejado."
        ),
        "",
    ]
    (OUTPUT / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT / "synthetic").mkdir(exist_ok=True)
    data = np.load(REPO_ROOT / "data/processed/windows_dataset.npz")
    meta = pd.read_parquet(REPO_ROOT / "data/processed/windows_meta.parquet")
    meta_train = meta.loc[meta["split"] == "train"].reset_index(drop=True)
    std = read_json(REPO_ROOT / "data/processed/standardizer.json")
    ref = read_json(REPO_ROOT / "data/processed/downstream_reference.json")
    reconstructed = reconstruct_return_paths(
        data["X_train"], meta_train, y=data["y_train"], horizon=21
    )
    path_meta = meta_train.iloc[reconstructed.anchor_indices].reset_index(drop=True)
    X_train_z = ((data["X_train"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    y_train_z = ((data["y_train"] - std["y_mu"]) / std["y_sd"]).astype(np.float32)
    X_val_z = ((data["X_val"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    y_val_z = ((data["y_val"] - std["y_mu"]) / std["y_sd"]).astype(np.float32)
    X_test_z = ((data["X_test"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    device = get_device()
    config = {
        "n_real": N_REAL,
        "ratio": RATIO,
        "n_synth": N_REAL * RATIO,
        "train_steps": TRAIN_STEPS,
        "sample_steps": 50,
        "seeds": list(SEEDS),
        "path_coverage": len(reconstructed.paths) / len(data["X_train"]),
        "sampling_unit": "trayectoria física de 81 retornos",
    }
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    (OUTPUT / "config.json").write_text(
        json.dumps({**config, "experiment_id": digest}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    results_path = OUTPUT / "results.csv"

    for seed in SEEDS:
        existing = (
            pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
        )
        selected = submuestrear_por_fechas(path_meta, N_REAL, seed)
        anchor_rows = reconstructed.anchor_indices[selected]
        X_real, y_real = X_train_z[anchor_rows], y_train_z[anchor_rows]

        if (
            existing.empty
            or not (
                (existing["brazo"] == "solo_real_path81") & (existing["seed"] == seed)
            ).any()
        ):
            print(f"[seed {seed}] downstream solo real", flush=True)
            metrics = evaluate(
                X_real,
                y_real,
                X_val=X_val_z,
                y_val=y_val_z,
                X_test=X_test_z,
                y_test_physical=data["y_test"],
                ref=ref,
                std=std,
                seed=seed,
                device=device,
            )
            append_result(
                results_path,
                {
                    "brazo": "solo_real_path81",
                    "seed": seed,
                    "n_real": N_REAL,
                    "n_synth": 0,
                    **metrics,
                },
            )

        checkpoint = OUTPUT / "checkpoints" / f"seed{seed}_{digest}.pt"
        if checkpoint.exists():
            generator = DiffusionTSGenerator.load(checkpoint, device=device)
        else:
            print(
                f"[seed {seed}] ajuste Diffusion-TS con {N_REAL:,} trayectorias",
                flush=True,
            )
            generator = DiffusionTSGenerator(train_steps=TRAIN_STEPS).fit_paths(
                reconstructed.paths[selected],
                x_mu=std["x_mu"],
                x_sd=std["x_sd"],
                y_mu=std["y_mu"],
                y_sd=std["y_sd"],
                seed=seed,
            )
            generator.save(checkpoint)
            pd.DataFrame(generator.history_).to_csv(
                OUTPUT / f"training_history_seed{seed}.csv", index_label="step"
            )

        synthetic_file = OUTPUT / "synthetic" / f"seed{seed}_{digest}.npy"
        if synthetic_file.exists():
            synthetic = np.load(synthetic_file)
        else:
            print(f"[seed {seed}] muestreo {N_REAL * RATIO:,}", flush=True)
            synthetic = generator.sample(N_REAL * RATIO, seed=seed + 1000)
            np.save(synthetic_file, synthetic)

        existing = pd.read_csv(results_path)
        if not (
            (existing["brazo"] == "diffusion_ts_path81") & (existing["seed"] == seed)
        ).any():
            print(f"[seed {seed}] downstream mezcla", flush=True)
            metrics = evaluate(
                np.vstack([X_real, synthetic[:, :-1]]).astype(np.float32),
                np.concatenate([y_real, synthetic[:, -1]]).astype(np.float32),
                X_val=X_val_z,
                y_val=y_val_z,
                X_test=X_test_z,
                y_test_physical=data["y_test"],
                ref=ref,
                std=std,
                seed=seed,
                device=device,
            )
            append_result(
                results_path,
                {
                    "brazo": "diffusion_ts_path81",
                    "seed": seed,
                    "n_real": N_REAL,
                    "n_synth": N_REAL * RATIO,
                    **metrics,
                },
            )
        render_report(pd.read_csv(results_path), config)
        print(f"[seed {seed}] completa", flush=True)


if __name__ == "__main__":
    main()

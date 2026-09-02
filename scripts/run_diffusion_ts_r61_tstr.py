"""TSTR definitivo de Diffusion-TS con el contrato común R61 = [X60 | y]."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion_ts import DiffusionTSR61Generator
from src.gen_audit import audit_generator
from src.gen_utility import entrenar_y_evaluar
from src.training import get_device

SEEDS = (42, 43, 44)
TRAIN_STEPS = 3_000
SAMPLE_STEPS = 50
OUTPUT = REPO_ROOT / "reports/diffusion_ts_r61/tstr"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def upsert(path: Path, row: dict, key: str = "fit_seed") -> None:
    old = load_rows(path)
    if not old.empty:
        old = old.loc[old[key].astype(int) != int(row[key])]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, float) else str(value) for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def summarize(tstr: pd.DataFrame, audit: pd.DataFrame) -> dict:
    historical = pd.read_csv(REPO_ROOT / "data/processed/tstr_nb03.csv")
    diffusion = tstr.rename(columns={"fit_seed": "seed"})[["seed", "val_r2"]].rename(
        columns={"val_r2": "diffusion_r2"}
    )
    comparisons = {}
    for comparator in ("real", "wgan_gp", "vae", "realnvp", "jitter"):
        reference = historical.loc[
            historical["brazo"] == comparator, ["seed", "val_r2"]
        ].rename(columns={"val_r2": "reference_r2"})
        joined = diffusion.merge(reference, on="seed")
        delta = joined["diffusion_r2"] - joined["reference_r2"]
        mean, sd = float(delta.mean()), float(delta.std())
        half = float(stats.t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta)))
        comparisons[comparator] = {
            "delta_mean": mean,
            "delta_sd": sd,
            "ci95": [mean - half, mean + half],
            "wins": int((delta > 0).sum()),
            "pairs": len(delta),
        }
    real_mean = float(historical.loc[historical["brazo"] == "real", "val_r2"].mean())
    return {
        "representation": "R61 = [60 retornos estandarizados | target estandarizado]",
        "seeds": list(SEEDS),
        "complete": sorted(tstr["fit_seed"].astype(int).tolist()) == list(SEEDS),
        "tstr_r2_mean": float(tstr["val_r2"].mean()),
        "tstr_r2_sd": float(tstr["val_r2"].std()),
        "ratio_tstr_trtr": float(tstr["val_r2"].mean()) / real_mean,
        "paired_comparisons": comparisons,
        "audit_mean": audit.select_dtypes(include="number").mean().to_dict(),
        "audit_sd": audit.select_dtypes(include="number").std().to_dict(),
    }


def render(summary: dict, tstr: pd.DataFrame, audit: pd.DataFrame) -> None:
    comparisons = pd.DataFrame(
        [
            {
                "comparador": name,
                "delta_r2": values["delta_mean"],
                "ci95_inf": values["ci95"][0],
                "ci95_sup": values["ci95"][1],
                "victorias": f"{values['wins']}/{values['pairs']}",
            }
            for name, values in summary["paired_comparisons"].items()
        ]
    )
    fidelity_columns = [
        "fit_seed",
        "sd_x",
        "curtosis_x",
        "acf_abs_lag1",
        "err_corr_xy",
        "w1_x_col_media",
        "err_corr_xx_spearman",
        "discriminative_auc",
    ]
    report = [
        "# TSTR Diffusion-TS R61",
        "",
        (
            "Comparación estricta: Diffusion-TS recibe exactamente la misma matriz "
            "estandarizada `[X60 | y]` que VAE, WGAN-GP y RealNVP. El target es "
            "un token especial, no un retorno."
        ),
        "",
        markdown_table(
            tstr[["fit_seed", "val_r2", "val_mse", "epoca_mejor", "sampling_seconds"]]
        ),
        "",
        (
            f"R² medio: **{summary['tstr_r2_mean']:.4f}**; "
            f"ratio TSTR/TRTR: **{summary['ratio_tstr_trtr']:.3f}**."
        ),
        "",
        "## Comparaciones emparejadas",
        "",
        markdown_table(comparisons),
        "",
        "## Fidelidad",
        "",
        markdown_table(audit[fidelity_columns]),
        "",
    ]
    (OUTPUT / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT / "synthetic").mkdir(exist_ok=True)
    data = np.load(REPO_ROOT / "data/processed/windows_dataset.npz")
    std = read_json(REPO_ROOT / "data/processed/standardizer.json")
    ref = read_json(REPO_ROOT / "data/processed/downstream_reference.json")
    X_train = ((data["X_train"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    y_train = ((data["y_train"] - std["y_mu"]) / std["y_sd"]).astype(np.float32)
    X_val = ((data["X_val"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    XY_train = np.column_stack([X_train, y_train]).astype(np.float32)
    config = {
        "representation": "R61",
        "train_steps": TRAIN_STEPS,
        "sample_steps": SAMPLE_STEPS,
        "n_train": len(XY_train),
        "seeds": list(SEEDS),
        "posthoc_variance_scaling": False,
    }
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    (OUTPUT / "config.json").write_text(
        json.dumps({**config, "experiment_id": digest}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    device = get_device()
    tstr_path, audit_path = OUTPUT / "tstr.csv", OUTPUT / "audit.csv"

    for seed in SEEDS:
        checkpoint = OUTPUT / "checkpoints" / f"seed{seed}_{digest}.pt"
        if checkpoint.exists():
            generator = DiffusionTSR61Generator.load(checkpoint, device=device)
        else:
            print(f"[R61 seed {seed}] entrenamiento", flush=True)
            generator = DiffusionTSR61Generator(
                train_steps=TRAIN_STEPS, sample_steps=SAMPLE_STEPS
            ).fit(XY_train, seed=seed, verbose=True)
            generator.save(checkpoint)
            pd.DataFrame(generator.history_).to_csv(
                OUTPUT / f"training_history_seed{seed}.csv", index_label="step"
            )

        synthetic_file = OUTPUT / "synthetic" / f"seed{seed}_{digest}.npy"
        if synthetic_file.exists():
            synthetic = np.load(synthetic_file)
            sampling_seconds = 0.0
        else:
            print(f"[R61 seed {seed}] muestreo {len(XY_train):,}", flush=True)
            started = time.perf_counter()
            synthetic = generator.sample(len(XY_train), seed=seed + 1000)
            sampling_seconds = time.perf_counter() - started
            np.save(synthetic_file, synthetic)

        audit_done = load_rows(audit_path)
        if audit_done.empty or seed not in set(audit_done["fit_seed"].astype(int)):
            print(f"[R61 seed {seed}] auditoría", flush=True)
            upsert(
                audit_path,
                {
                    "fit_seed": seed,
                    "training_seconds": generator.training_seconds_,
                    "sampling_seconds": sampling_seconds,
                    **audit_generator(XY_train, synthetic, seed=seed),
                },
            )

        tstr_done = load_rows(tstr_path)
        if tstr_done.empty or seed not in set(tstr_done["fit_seed"].astype(int)):
            print(f"[R61 seed {seed}] downstream TSTR", flush=True)
            metrics = entrenar_y_evaluar(
                synthetic[:, :-1],
                synthetic[:, -1],
                X_val,
                data["y_val"],
                ref=ref,
                std=std,
                device=device,
                seed=seed,
            )
            upsert(
                tstr_path,
                {
                    "fit_seed": seed,
                    "sample_seed": seed + 1000,
                    "training_seconds": generator.training_seconds_,
                    "sampling_seconds": sampling_seconds,
                    **metrics,
                },
            )

        tstr, audit = load_rows(tstr_path), load_rows(audit_path)
        if len(tstr) == len(audit):
            summary = summarize(tstr, audit)
            (OUTPUT / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            render(summary, tstr, audit)
        print(f"[R61 seed {seed}] completa", flush=True)


if __name__ == "__main__":
    main()

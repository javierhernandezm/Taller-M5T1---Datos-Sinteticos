"""Refina la CNN ganadora con más épocas y regularización, sin usar test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import regression_metrics
from src.config import Config, set_global_seed
from src.models import build_model, count_params
from src.training import get_device, predict, train_model


EXPERIMENTS = {
    "wide_base": ({"dropout": 0.1}, {"weight_decay": 1e-5}),
    "wide_drop20": ({"dropout": 0.2}, {"weight_decay": 1e-5}),
    "wide_drop30": ({"dropout": 0.3}, {"weight_decay": 1e-5}),
    "wide_drop30_wd1e4": ({"dropout": 0.3}, {"weight_decay": 1e-4}),
}


def main() -> None:
    cfg = Config()
    device = get_device()
    d = np.load(cfg.out_dir / "windows_dataset.npz")
    std = json.load(open(cfg.out_dir / "standardizer.json", encoding="utf-8"))
    x_train = (d["X_train"] - std["x_mu"]) / std["x_sd"]
    y_train = (d["y_train"] - std["y_mu"]) / std["y_sd"]
    x_val = (d["X_val"] - std["x_mu"]) / std["x_sd"]
    y_val_std = (d["y_val"] - std["y_mu"]) / std["y_sd"]
    y_val = d["y_val"]

    rows, histories = [], {}
    for name, (model_extra, train_extra) in EXPERIMENTS.items():
        kwargs = {"channels": (64, 128, 256), "kernel": 5, "fc": 128,
                  **model_extra}
        set_global_seed(cfg.seed)
        model = build_model("cnn", **kwargs)
        result = train_model(
            model, x_train, y_train, x_val, y_val_std,
            epochs=80, batch_size=1024, lr=1e-3,
            patience=80, grad_clip=1.0, cosine=True,
            seed=cfg.seed, device=device, **train_extra,
        )
        pred = predict(model, x_val, device) * std["y_sd"] + std["y_mu"]
        metrics = regression_metrics(y_val, pred)
        row = {"name": name, "kwargs": kwargs, "train_kwargs": train_extra,
               "params": count_params(model), "best_epoch": result.best_epoch,
               "epochs_run": len(result.train_loss), "seconds": result.seconds,
               "val_loss_std": result.best_val, **metrics}
        rows.append(row)
        histories[name] = result
        print(f"{name:18s} epoch={row['best_epoch']:>2}/{row['epochs_run']:<2} "
              f"val_mse={row['mse']:.6f} val_r2={row['r2']:.6f} "
              f"time={row['seconds']:.0f}s", flush=True)

    rows.sort(key=lambda row: row["mse"])
    (cfg.out_dir / "cnn_refinement.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (name, result) in zip(axes.ravel(), histories.items()):
        ax.plot(result.train_loss, label="train", lw=1.2)
        ax.plot(result.val_loss, label="validación", lw=1.2)
        ax.axvline(result.best_epoch, color="grey", ls="--", lw=0.9)
        ax.set_title(name)
        ax.set_xlabel("época")
        ax.set_ylabel("MSE estandarizado")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("CNN wide: diagnóstico de regularización (solo train/validación)")
    fig.tight_layout()
    fig.savefig(cfg.fig_dir / "27_cnn_refinement.png", dpi=140, bbox_inches="tight")
    print(f"\nMejor: {rows[0]['name']}")


if __name__ == "__main__":
    main()

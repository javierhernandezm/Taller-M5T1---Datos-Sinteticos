"""Confirma la CNN candidata frente a cnn_l con varias semillas en validación."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import regression_metrics
from src.config import Config, set_global_seed
from src.models import build_model, count_params
from src.training import get_device, predict, train_model


MODELS = {
    "cnn_l": {"channels": (32, 64, 128), "kernel": 5, "fc": 128,
              "dropout": 0.1},
    "cnn_k3": {"channels": (32, 64, 128), "kernel": 3, "fc": 128,
               "dropout": 0.1},
    "cnn_deep_k3": {"channels": (32, 64, 128, 256), "kernel": 3,
                    "fc": 128, "dropout": 0.1},
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

    rows = []
    for name, kwargs in MODELS.items():
        for seed in (42, 43, 44):
            set_global_seed(seed)
            model = build_model("cnn", **kwargs)
            result = train_model(
                model, x_train, y_train, x_val, y_val_std,
                epochs=40, batch_size=1024, lr=1e-3, weight_decay=1e-5,
                patience=40, grad_clip=1.0, cosine=True,
                seed=seed, device=device,
            )
            pred = predict(model, x_val, device) * std["y_sd"] + std["y_mu"]
            metrics = regression_metrics(y_val, pred)
            rows.append({"name": name, "seed": seed, "kwargs": kwargs,
                         "params": count_params(model),
                         "best_epoch": result.best_epoch, **metrics})
            print(f"{name:8s} seed={seed} epoch={result.best_epoch:>2} "
                  f"val_mse={metrics['mse']:.6f} val_r2={metrics['r2']:.6f}",
                  flush=True)

    for name in MODELS:
        values = [row["mse"] for row in rows if row["name"] == name]
        print(f"{name:8s} MSE media={np.mean(values):.6f} sd={np.std(values, ddof=1):.6f}")
    (cfg.out_dir / "cnn_confirmation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

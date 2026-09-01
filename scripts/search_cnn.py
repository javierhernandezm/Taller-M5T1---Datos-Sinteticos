"""Búsqueda reproducible de CNN usando exclusivamente train y validación."""

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


CANDIDATES = {
    "cnn_l": ("cnn", {"channels": (32, 64, 128), "fc": 128}),
    "cnn_k3": ("cnn", {"channels": (32, 64, 128), "kernel": 3, "fc": 128}),
    "cnn_k7": ("cnn", {"channels": (32, 64, 128), "kernel": 7, "fc": 128}),
    "cnn_k9": ("cnn", {"channels": (32, 64, 128), "kernel": 9, "fc": 128}),
    "cnn_deep_k3": ("cnn", {"channels": (32, 64, 128, 256), "kernel": 3,
                             "fc": 128, "dropout": 0.1}),
    "cnn_wide": ("cnn", {"channels": (64, 128, 256), "kernel": 5,
                          "fc": 128, "dropout": 0.1}),
    "cnn_no_dropout": ("cnn", {"channels": (32, 64, 128), "fc": 128,
                                "dropout": 0.0}),
    "cnn_dropout_02": ("cnn", {"channels": (32, 64, 128), "fc": 128,
                                "dropout": 0.2}),
}


def main() -> None:
    cfg = Config()
    set_global_seed(cfg.seed)
    device = get_device()
    d = np.load(cfg.out_dir / "windows_dataset.npz")
    std = json.load(open(cfg.out_dir / "standardizer.json", encoding="utf-8"))
    x_train = (d["X_train"] - std["x_mu"]) / std["x_sd"]
    y_train = (d["y_train"] - std["y_mu"]) / std["y_sd"]
    x_val = (d["X_val"] - std["x_mu"]) / std["x_sd"]
    y_val_std = (d["y_val"] - std["y_mu"]) / std["y_sd"]
    y_val = d["y_val"]

    rows = []
    for name, (arch, kwargs) in CANDIDATES.items():
        # La semilla debe fijarse ANTES de construir el modelo: train_model la
        # fija de nuevo, pero para entonces los pesos ya están inicializados.
        set_global_seed(cfg.seed)
        model = build_model(arch, **kwargs)
        result = train_model(
            model, x_train, y_train, x_val, y_val_std,
            epochs=40, batch_size=1024, lr=1e-3, weight_decay=1e-5,
            patience=40, grad_clip=1.0, cosine=True,
            seed=cfg.seed, device=device,
        )
        pred = predict(model, x_val, device) * std["y_sd"] + std["y_mu"]
        metrics = regression_metrics(y_val, pred)
        row = {"name": name, "arch": arch, "kwargs": kwargs,
               "params": count_params(model), "best_epoch": result.best_epoch,
               "seconds": result.seconds, "val_loss_std": result.best_val, **metrics}
        rows.append(row)
        print(f"{name:12s} params={row['params']:>8,} epoch={row['best_epoch']:>2} "
              f"val_mse={row['mse']:.6f} val_r2={row['r2']:.6f} "
              f"time={row['seconds']:.0f}s", flush=True)

    rows.sort(key=lambda row: row["mse"])
    out = cfg.out_dir / "cnn_search.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nMejor: {rows[0]['name']} | resultados: {out}")


if __name__ == "__main__":
    main()

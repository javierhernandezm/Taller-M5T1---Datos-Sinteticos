"""Integra Diffusion-TS R61 en el protocolo TSTR oficial del notebook 03.

Los cinco brazos que no han cambiado y TRTR se conservan de la ejecución
histórica. Solo se calculan las tres filas de Diffusion-TS con el mismo ajuste
generativo seed 42 y las mismas semillas downstream 42/43/44. El resultado
activo vuelve a contener exactamente 21 filas.

La auditoría que escribe aquí es PARCIAL por el mismo motivo: solo la fila de
Diffusion-TS es nueva, las otras cinco se heredan. La tabla de fidelidad que
publica el informe la regenera el notebook 03, que ajusta los seis generadores
en una misma corrida — es el paso 5 de la reproducción
(`scripts/run_diffusion_ts_r61_notebook03.py`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.diffusion_ts import DiffusionTSR61Generator
from src.gen_audit import audit_generator
from src.gen_utility import COLUMNAS, entrenar_y_evaluar, resumir_tstr
from src.malla import firma_receta
from src.training import get_device

SEEDS = (42, 43, 44)
FIT_SEED = 42
TRAIN_STEPS = 3_000
SAMPLE_STEPS = 50
OUTPUT = REPO_ROOT / "reports/diffusion_ts_r61/nb03"
PROCESSED = REPO_ROOT / "data/processed"
UNCHANGED_ARMS = {"real", "jitter", "gaussiana", "block_bootstrap", "vae", "realnvp"}
ACTIVE_ARMS = {*UNCHANGED_ARMS, "diffusion_ts"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upsert(path: Path, row: dict) -> None:
    try:
        old = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        old = pd.DataFrame(columns=COLUMNAS)
    old = old.loc[
        ~((old.brazo == row["brazo"]) & (old.seed.astype(int) == int(row["seed"])))
    ]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[COLUMNAS].to_csv(
        path, index=False
    )


def validate_tstr(frame: pd.DataFrame) -> None:
    if len(frame) != 21 or set(frame.brazo) != ACTIVE_ARMS:
        raise RuntimeError(
            f"TSTR incompleto: {len(frame)} filas, brazos={sorted(set(frame.brazo))}"
        )
    counts = frame.groupby("brazo").seed.nunique()
    if not (counts == 3).all() or frame.duplicated(["brazo", "seed"]).any():
        raise RuntimeError(
            "Cada brazo TSTR debe tener exactamente tres semillas únicas"
        )
    if frame.select_dtypes(include="number").isna().any().any():
        raise RuntimeError("El TSTR contiene valores numéricos ausentes")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT / "synthetic").mkdir(exist_ok=True)
    data = np.load(PROCESSED / "windows_dataset.npz")
    std = read_json(PROCESSED / "standardizer.json")
    ref = read_json(PROCESSED / "downstream_reference.json")
    X_train = ((data["X_train"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    y_train = ((data["y_train"] - std["y_mu"]) / std["y_sd"]).astype(np.float32)
    X_val = ((data["X_val"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    XY_train = np.column_stack([X_train, y_train]).astype(np.float32)
    device = get_device()

    config = {
        "representation": "R61 = [X60 | y]",
        "fit_seed": FIT_SEED,
        "sample_and_downstream_seeds": list(SEEDS),
        "train_steps": TRAIN_STEPS,
        "sample_steps": SAMPLE_STEPS,
        "n_train": len(XY_train),
        "downstream_recipe": firma_receta(ref),
        "protocol": "one generator fit; three sample/downstream seeds",
    }
    experiment_id = hashlib.sha1(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("experiment_id") != experiment_id:
            raise RuntimeError(
                "El cache NB03 pertenece a otra configuración: "
                f"{previous.get('experiment_id')} != {experiment_id}"
            )

    checkpoint = OUTPUT / "checkpoints" / f"seed{FIT_SEED}_{experiment_id}.pt"
    sidecar = next(
        iter(
            (REPO_ROOT / "reports/diffusion_ts_r61/tstr/checkpoints").glob(
                "seed42_*.pt"
            )
        ),
        None,
    )
    if checkpoint.exists():
        generator = DiffusionTSR61Generator.load(checkpoint, device=device)
    elif sidecar is not None:
        generator = DiffusionTSR61Generator.load(sidecar, device=device)
        generator.save(checkpoint)
    else:
        print("Ajustando Diffusion-TS R61 seed 42", flush=True)
        generator = DiffusionTSR61Generator(
            train_steps=TRAIN_STEPS, sample_steps=SAMPLE_STEPS
        ).fit(XY_train, seed=FIT_SEED, verbose=True)
        generator.save(checkpoint)
    pd.DataFrame(generator.history_).to_csv(
        OUTPUT / "training_history.csv", index=False
    )

    active_path = OUTPUT / "tstr_active.csv"
    if not active_path.exists():
        historical = pd.read_csv(PROCESSED / "tstr_nb03.csv")
        base = historical.loc[historical.brazo.isin(UNCHANGED_ARMS), COLUMNAS]
        if len(base) != 18:
            raise RuntimeError(
                "El TSTR histórico no contiene los 18 controles esperados"
            )
        base.to_csv(active_path, index=False)

    for seed in SEEDS:
        current = pd.read_csv(active_path)
        if ((current.brazo == "diffusion_ts") & (current.seed == seed)).any():
            print(f"Diffusion-TS seed {seed}: ya calculada", flush=True)
            continue
        sample_path = OUTPUT / "synthetic" / f"seed{seed}_{experiment_id}.npy"
        if sample_path.exists():
            synthetic = np.load(sample_path)
        else:
            print(f"Diffusion-TS seed {seed}: muestreo {len(XY_train):,}", flush=True)
            synthetic = generator.sample(len(XY_train), seed=seed + 1_000)
            np.save(sample_path, synthetic)
        print(f"Diffusion-TS seed {seed}: TSTR downstream", flush=True)
        started = time.perf_counter()
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
            active_path,
            {"brazo": "diffusion_ts", "seed": seed, **metrics},
        )
        print(
            f"  R2={metrics['val_r2']:.4f}; {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    tstr = (
        pd.read_csv(active_path).sort_values(["brazo", "seed"]).reset_index(drop=True)
    )
    validate_tstr(tstr)
    summary = resumir_tstr(tstr)
    tstr.to_csv(OUTPUT / "tstr.csv", index=False)
    summary.to_csv(OUTPUT / "tstr_resumen.csv", index=False)

    rng = np.random.default_rng(42)
    real_ref = XY_train[rng.choice(len(XY_train), 8_000, replace=False)]
    audit_sample = generator.sample(8_000, seed=43)
    old_audit = pd.read_csv(PROCESSED / "auditoria_nb03.csv", index_col=0)
    active_audit = old_audit.loc[old_audit.index != "wgan_gp"].copy()
    active_audit.loc["diffusion_ts"] = audit_generator(real_ref, audit_sample, seed=42)
    active_audit = active_audit.loc[
        ["jitter", "gaussiana", "block_bootstrap", "vae", "diffusion_ts", "realnvp"]
    ]
    active_audit.to_csv(OUTPUT / "auditoria.csv")

    manifest = {
        **config,
        "experiment_id": experiment_id,
        "checkpoint_sha1": file_sha1(checkpoint),
        "complete": True,
        "rows": len(tstr),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    shutil.copy2(OUTPUT / "tstr.csv", PROCESSED / "tstr_nb03.csv")
    shutil.copy2(OUTPUT / "tstr_resumen.csv", PROCESSED / "tstr_nb03_resumen.csv")
    shutil.copy2(OUTPUT / "auditoria.csv", PROCESSED / "auditoria_nb03.csv")
    shutil.copy2(manifest_path, PROCESSED / "tstr_nb03_manifest.json")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

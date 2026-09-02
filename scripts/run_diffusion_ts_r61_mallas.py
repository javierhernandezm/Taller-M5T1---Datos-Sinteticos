"""Completa las dos mallas sustituyendo WGAN-GP por Diffusion-TS R61.

El runner parte de las celdas historicas ya calculadas, elimina WGAN-GP y
calcula solo las celdas de Diffusion-TS. Para un mismo ``(n_real, seed)`` el
submuestreo y el ajuste del generador son identicos en todos los presupuestos;
por eso el checkpoint y una muestra maxima se reutilizan entre ratios. Esto no
cambia el experimento: el muestreo DDIM es determinista y cada presupuesto usa
el mismo prefijo que obtendria una llamada independiente con la misma semilla.

Cada celda downstream se persiste al terminar. El proceso es reanudable.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines import regression_metrics
from src.diffusion_ts import DiffusionTSR61Generator
from src.malla import (
    COLUMNAS,
    GENERADORES_ACTIVOS,
    delta_pareado,
    firma_receta,
    plot_curvas_error_por_generador,
    plot_curvas_error_todos,
    resumir,
    submuestrear_por_fechas,
)
from src.models import build_model
from src.training import get_device, predict, train_model

GENERATOR = "diffusion_ts"
TRAIN_STEPS = 3_000
SAMPLE_STEPS = 50
N_VAL = 5_000
SEEDS = (0, 1, 2)
RATIO_N_REAL = (250, 1_000, 3_000)
RATIOS = (0.25, 0.5, 0.75, 1.0, 3.0, 5.0)
CURVE_N_REAL = (250, 1_000, 3_000, 10_000, 20_000)
N_SYNTHETIC = (500, 1_000, 3_000, 7_000, 20_000)
ACTIVE_GENERATORS = GENERADORES_ACTIVOS
OUTPUT = REPO_ROOT / "reports/diffusion_ts_r61/mallas"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_active_csv(source: Path, destination: Path, recipe: str) -> None:
    """Copia las celdas vigentes salvo WGAN-GP; nunca altera el historico."""
    if destination.exists():
        current = pd.read_csv(destination)
        allowed = {*ACTIVE_GENERATORS, "ninguno"}
        unexpected = set(current["generador"]) - allowed
        recipes = set(current["receta"].dropna().astype(str))
        if unexpected or recipes != {recipe}:
            raise RuntimeError(
                f"Cache incompatible en {destination}: "
                f"generadores inesperados={sorted(unexpected)}, recetas={sorted(recipes)}"
            )
        return
    old = pd.read_csv(source)
    active = old.loc[
        (old["generador"] != "wgan_gp") & (old["receta"] == recipe), COLUMNAS
    ].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    active.to_csv(destination, index=False)


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row])[COLUMNAS].to_csv(path, mode="a", header=False, index=False)


def completed_keys(path: Path) -> set[tuple[int, int, int]]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["generador"] == GENERATOR]
    return {
        (int(row.n_real), int(row.seed), int(row.n_synth)) for row in frame.itertuples()
    }


def required_cells() -> dict[tuple[int, int], dict[int, list[tuple[str, float]]]]:
    """Une ambos disenos para evaluar una sola vez los presupuestos repetidos."""
    cells: dict[tuple[int, int], dict[int, list[tuple[str, float]]]] = {}
    for n_real in RATIO_N_REAL:
        for seed in SEEDS:
            for ratio in RATIOS:
                n_synth = round(ratio * n_real)
                cells.setdefault((n_real, seed), {}).setdefault(n_synth, []).append(
                    ("ratios", ratio)
                )
    for n_real in CURVE_N_REAL:
        for seed in SEEDS:
            for n_synth in N_SYNTHETIC:
                cells.setdefault((n_real, seed), {}).setdefault(n_synth, []).append(
                    ("curves", n_synth / n_real)
                )
    return cells


def evaluate_downstream(
    synthetic: np.ndarray,
    *,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test_physical: np.ndarray,
    std: dict,
    ref: dict,
    seed: int,
    device: torch.device,
) -> tuple[dict, float]:
    started = time.perf_counter()
    X_mix = np.vstack([X_real, synthetic[:, :-1]]).astype(np.float32)
    y_mix = np.concatenate([y_real, synthetic[:, -1]]).astype(np.float32)
    kwargs = dict(ref["train_kwargs"])
    kwargs["patience"] = kwargs["epochs"]
    model = build_model(ref["arch"], **ref["arch_kwargs"])
    trained = train_model(
        model,
        X_mix,
        y_mix,
        X_val,
        y_val,
        seed=seed,
        device=device,
        **kwargs,
    )
    prediction = predict(model, X_test, device) * std["y_sd"] + std["y_mu"]
    metrics = regression_metrics(y_test_physical, prediction)
    elapsed = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "n_train": len(X_mix),
        "val_mse": trained.best_val,
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "epoca_mejor": trained.best_epoch,
    }, elapsed


def paired_generator_comparison(
    active: pd.DataFrame, historical: pd.DataFrame, design: str
) -> pd.DataFrame:
    """Compara Diffusion-TS con cada generador, pareando celda y semilla."""
    diffusion = active.loc[
        active.generador == GENERATOR,
        ["n_real", "n_synth", "seed", "test_r2"],
    ].rename(columns={"test_r2": "diffusion_r2"})
    rows = []
    comparators = [*ACTIVE_GENERATORS[:-2], "realnvp", "wgan_gp"]
    for comparator in comparators:
        source = historical if comparator == "wgan_gp" else active
        other = source.loc[
            source.generador == comparator,
            ["n_real", "n_synth", "seed", "test_r2"],
        ].rename(columns={"test_r2": "comparator_r2"})
        paired = diffusion.merge(other, on=["n_real", "n_synth", "seed"])
        delta = paired.diffusion_r2 - paired.comparator_r2
        rows.append(
            {
                "design": design,
                "comparator": comparator,
                "pairs": len(delta),
                "delta_r2_mean": delta.mean(),
                "delta_r2_sd": delta.std(),
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Renderiza una tabla pequena sin depender del paquete opcional tabulate."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, float) else str(value) for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_report(
    ratios: pd.DataFrame,
    curves: pd.DataFrame,
    comparisons: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    ratio_delta = delta_pareado(ratios, presupuesto="ratio")
    curve_delta = delta_pareado(curves, presupuesto="n_synth")
    ratio_diff = ratio_delta.loc[ratio_delta.generador == GENERATOR]
    curve_diff = curve_delta.loc[curve_delta.generador == GENERATOR]
    lines = [
        "# Mallas completas — Diffusion-TS R61",
        "",
        (
            "Comparacion estricta sobre `[X60 | y]`: WGAN-GP se retira de la "
            "lista activa y Diffusion-TS ocupa exactamente sus 129 celdas en "
            "las dos mallas."
        ),
        "",
        "## Cobertura",
        "",
        f"- Ratios finos: {len(ratios)} celdas; Diffusion-TS {sum(ratios.generador == GENERATOR)}.",
        f"- Curvas: {len(curves)} celdas; Diffusion-TS {sum(curves.generador == GENERATOR)}.",
        f"- Ajustes independientes de Diffusion-TS: {len(costs)}.",
        "",
        "## Efecto frente a solo real",
        "",
        (
            f"Ratios: {sum(ratio_diff.veredicto == 'mejora')} mejoras, "
            f"{sum(ratio_diff.veredicto == 'empeora')} empeoramientos y "
            f"{sum(ratio_diff.veredicto == 'no concluyente')} celdas no concluyentes."
        ),
        (
            f"Curvas: {sum(curve_diff.veredicto == 'mejora')} mejoras, "
            f"{sum(curve_diff.veredicto == 'empeora')} empeoramientos y "
            f"{sum(curve_diff.veredicto == 'no concluyente')} celdas no concluyentes."
        ),
        "",
        "## Comparacion pareada con generadores",
        "",
        markdown_table(comparisons),
        "",
        (
            "La conclusion definitiva debe leerse junto con `comparisons.csv`, "
            "los deltas pareados y el diagnostico TSTR; no se decide solo por "
            "una media global."
        ),
        "",
    ]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def persist_summaries() -> None:
    ratios_path = OUTPUT / "malla_ratios_finos.csv"
    curves_path = OUTPUT / "malla_curvas.csv"
    ratios, curves = pd.read_csv(ratios_path), pd.read_csv(curves_path)
    historical_ratios = pd.read_csv(REPO_ROOT / "data/processed/malla_ratios_finos.csv")
    historical_curves = pd.read_csv(REPO_ROOT / "data/processed/malla_curvas.csv")

    resumir(ratios).to_csv(OUTPUT / "malla_ratios_finos_resumen.csv", index=False)
    delta_pareado(ratios, presupuesto="ratio").to_csv(
        OUTPUT / "malla_ratios_finos_delta_pareado.csv", index=False
    )
    resumir(curves).to_csv(OUTPUT / "malla_curvas_resumen.csv", index=False)
    delta_pareado(curves, presupuesto="n_synth").to_csv(
        OUTPUT / "malla_curvas_delta_pareado.csv", index=False
    )
    comparisons = pd.concat(
        [
            paired_generator_comparison(ratios, historical_ratios, "ratios"),
            paired_generator_comparison(curves, historical_curves, "curves"),
        ],
        ignore_index=True,
    )
    comparisons.to_csv(OUTPUT / "comparisons.csv", index=False)

    costs_path = OUTPUT / "fit_costs.csv"
    costs = pd.read_csv(costs_path) if costs_path.exists() else pd.DataFrame()
    render_report(ratios, curves, comparisons, costs)

    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    fig = plot_curvas_error_por_generador(
        curves, CURVE_N_REAL, (0, *N_SYNTHETIC), ACTIVE_GENERATORS, metrica="val_mse"
    )
    fig.savefig(
        figures / "curvas_error_por_generador.png", dpi=140, bbox_inches="tight"
    )
    plt.close(fig)
    fig = plot_curvas_error_todos(
        curves, CURVE_N_REAL, (0, *N_SYNTHETIC), ACTIVE_GENERATORS, metrica="val_mse"
    )
    fig.savefig(figures / "curvas_error_todos.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUTPUT / "checkpoints"
    samples_dir = OUTPUT / "synthetic"
    checkpoints.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    data = np.load(REPO_ROOT / "data/processed/windows_dataset.npz")
    std = read_json(REPO_ROOT / "data/processed/standardizer.json")
    ref = read_json(REPO_ROOT / "data/processed/downstream_reference.json")
    recipe = firma_receta(ref)
    meta = pd.read_parquet(REPO_ROOT / "data/processed/windows_meta.parquet")
    meta_train = meta.loc[meta.split == "train"].reset_index(drop=True)
    standardize = lambda values, mean, sd: ((values - std[mean]) / std[sd]).astype(
        np.float32
    )
    X_train = standardize(data["X_train"], "x_mu", "x_sd")
    y_train = standardize(data["y_train"], "y_mu", "y_sd")
    X_val_all = standardize(data["X_val"], "x_mu", "x_sd")
    y_val_all = standardize(data["y_val"], "y_mu", "y_sd")
    rng = np.random.default_rng(42)
    val_idx = rng.choice(len(X_val_all), min(N_VAL, len(X_val_all)), replace=False)
    X_val, y_val = X_val_all[val_idx], y_val_all[val_idx]
    X_test = standardize(data["X_test"], "x_mu", "x_sd")
    y_test_physical = data["y_test"]
    device = get_device()

    cells = required_cells()
    config = {
        "representation": "R61 = [X60 | y]",
        "generator": GENERATOR,
        "train_steps": TRAIN_STEPS,
        "sample_steps": SAMPLE_STEPS,
        "recipe": recipe,
        "fit_keys": len(cells),
        "downstream_cells": sum(len(budgets) for budgets in cells.values()),
        "reuse_is_exact": True,
    }
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    manifest_path = OUTPUT / "config.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != digest:
            raise RuntimeError(
                "El cache de mallas pertenece a otra configuracion: "
                f"{manifest.get('experiment_id')} != {digest}"
            )
    else:
        manifest_path.write_text(
            json.dumps(
                {**config, "experiment_id": digest}, indent=2, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )

    ratio_csv = OUTPUT / "malla_ratios_finos.csv"
    curve_csv = OUTPUT / "malla_curvas.csv"
    initialize_active_csv(
        REPO_ROOT / "data/processed/malla_ratios_finos.csv", ratio_csv, recipe
    )
    initialize_active_csv(
        REPO_ROOT / "data/processed/malla_curvas.csv", curve_csv, recipe
    )

    started_all = time.perf_counter()
    for fit_number, ((n_real, seed), budgets) in enumerate(sorted(cells.items()), 1):
        destinations = {"ratios": ratio_csv, "curves": curve_csv}
        done = {name: completed_keys(path) for name, path in destinations.items()}
        pending = {
            n_synth: targets
            for n_synth, targets in budgets.items()
            if any((n_real, seed, n_synth) not in done[name] for name, _ in targets)
        }
        if not pending:
            print(
                f"[{fit_number:02d}/{len(cells)}] N={n_real} seed={seed}: completa",
                flush=True,
            )
            continue

        idx = submuestrear_por_fechas(meta_train, n_real, seed)
        X_real, y_real = X_train[idx], y_train[idx]
        XY_real = np.column_stack([X_real, y_real]).astype(np.float32)
        index_digest = hashlib.sha1(idx.astype(np.int64).tobytes()).hexdigest()[:10]
        checkpoint = checkpoints / f"n{n_real}_seed{seed}_{digest}_{index_digest}.pt"
        history_path = OUTPUT / f"history_n{n_real}_seed{seed}.csv"
        if checkpoint.exists():
            generator = DiffusionTSR61Generator.load(checkpoint, device=device)
            fit_seconds = generator.training_seconds_
        else:
            print(
                f"[{fit_number:02d}/{len(cells)}] N={n_real} seed={seed}: ajuste",
                flush=True,
            )
            generator = DiffusionTSR61Generator(
                train_steps=TRAIN_STEPS, sample_steps=SAMPLE_STEPS
            ).fit(XY_real, seed=seed, verbose=True)
            generator.save(checkpoint)
            pd.DataFrame(generator.history_).to_csv(history_path, index_label="step")
            fit_seconds = generator.training_seconds_

        max_samples = max(budgets)
        sample_path = (
            samples_dir / f"n{n_real}_seed{seed}_max{max_samples}_{digest}.npy"
        )
        if sample_path.exists():
            synthetic_max = np.load(sample_path)
            sample_seconds = 0.0
        else:
            print(f"  muestreo maximo: {max_samples:,}", flush=True)
            sample_started = time.perf_counter()
            synthetic_max = generator.sample(max_samples, seed=seed + 1_000)
            sample_seconds = time.perf_counter() - sample_started
            np.save(sample_path, synthetic_max)

        costs_path = OUTPUT / "fit_costs.csv"
        costs = pd.read_csv(costs_path) if costs_path.exists() else pd.DataFrame()
        cost_row = pd.DataFrame(
            [
                {
                    "n_real": n_real,
                    "seed": seed,
                    "fit_seconds": fit_seconds,
                    "max_n_synth": max_samples,
                    "sampling_seconds": sample_seconds,
                    "checkpoint": checkpoint.name,
                }
            ]
        )
        if not costs.empty:
            costs = costs.loc[
                ~(
                    (costs.n_real.astype(int) == n_real)
                    & (costs.seed.astype(int) == seed)
                )
            ]
        pd.concat([costs, cost_row], ignore_index=True).to_csv(costs_path, index=False)

        for n_synth, targets in sorted(pending.items()):
            metrics, elapsed = evaluate_downstream(
                synthetic_max[:n_synth],
                X_real=X_real,
                y_real=y_real,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test_physical=y_test_physical,
                std=std,
                ref=ref,
                seed=seed,
                device=device,
            )
            base_row = {
                "n_real": n_real,
                "generador": GENERATOR,
                "seed": seed,
                "n_synth": n_synth,
                "receta": recipe,
                "segundos": round(elapsed, 1),
                **metrics,
            }
            for design, ratio in targets:
                key = (n_real, seed, n_synth)
                if key not in done[design]:
                    append_row(destinations[design], {**base_row, "ratio": ratio})
                    done[design].add(key)
            elapsed_all = time.perf_counter() - started_all
            print(
                f"  synth={n_synth:>6,} -> R2={metrics['test_r2']:.4f} "
                f"({elapsed:.1f}s; total {elapsed_all / 60:.1f} min)",
                flush=True,
            )

        del generator, synthetic_max
        if device.type == "cuda":
            torch.cuda.empty_cache()

    persist_summaries()
    print(
        f"Mallas completas en {(time.perf_counter() - started_all) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

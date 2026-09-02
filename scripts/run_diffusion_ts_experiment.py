"""Ejecuta el piloto reproducible que decide si Diffusion-TS sustituye WGAN-GP.

Ejemplos
--------
Prueba de cableado (segundos, sin downstream)::

    python scripts/run_diffusion_ts_experiment.py --profile smoke

Piloto de decisión (tres ajustes completos, auditoría y TSTR)::

    python scripts/run_diffusion_ts_experiment.py --profile pilot

Los checkpoints y arrays sintéticos permiten reanudar una interrupción, pero
quedan ignorados por Git. Las métricas, configuración y conclusión sí se
versionan bajo ``reports/diffusion_ts_experiment/<perfil>/``.
"""

from __future__ import annotations

import argparse
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

from src.diffusion_ts import DiffusionTSGenerator, reconstruct_return_paths
from src.gen_audit import audit_generator
from src.gen_utility import entrenar_y_evaluar
from src.training import get_device

UPSTREAM_COMMIT = "007a829a7494133662693676133e059785e1ba3a"
PROFILES = {
    "smoke": {
        "train_steps": 20,
        "n_fit": 2_048,
        "n_synth": 512,
        "fit_seeds": [42],
        "run_downstream": False,
        "sample_steps": 5,
        "batch_size": 64,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "ff_mult": 2,
    },
    "pilot": {
        "train_steps": 3_000,
        "n_fit": None,
        "n_synth": None,
        "fit_seeds": [42, 43, 44],
        "run_downstream": True,
        "sample_steps": 50,
        "batch_size": 256,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "ff_mult": 4,
    },
    "full": {
        "train_steps": 10_000,
        "n_fit": None,
        "n_synth": None,
        "fit_seeds": [42, 43, 44],
        "run_downstream": True,
        "sample_steps": 50,
        "batch_size": 256,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "ff_mult": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--n-fit", type=int)
    parser.add_argument("--n-synth", type=int)
    parser.add_argument("--fit-seeds", type=int, nargs="+")
    parser.add_argument("--run-downstream", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignora checkpoints y sintéticos existentes (no borra los anteriores).",
    )
    return parser.parse_args()


def experiment_config(args: argparse.Namespace, n_train: int) -> dict:
    cfg = dict(PROFILES[args.profile])
    for name in (
        "train_steps",
        "n_fit",
        "n_synth",
        "fit_seeds",
        "run_downstream",
        "sample_steps",
    ):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    cfg["n_fit"] = min(cfg["n_fit"] or n_train, n_train)
    cfg["n_synth"] = cfg["n_synth"] or n_train
    cfg.update(
        {
            "profile": args.profile,
            "upstream_commit": UPSTREAM_COMMIT,
            "representation": "81 retornos físicos: X60 + futuro21; y se deriva, no se modela",
            "diffusion_steps": 500,
            "spectral_weight": 0.1,
            "ema_decay": 0.995,
            "annualization": 252,
        }
    )
    return cfg


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_inputs():
    data = np.load(REPO_ROOT / "data/processed/windows_dataset.npz")
    meta = pd.read_parquet(REPO_ROOT / "data/processed/windows_meta.parquet")
    std = json.loads(
        (REPO_ROOT / "data/processed/standardizer.json").read_text(encoding="utf-8")
    )
    ref = json.loads(
        (REPO_ROOT / "data/processed/downstream_reference.json").read_text(
            encoding="utf-8"
        )
    )
    meta_train = meta.loc[meta["split"] == "train"].reset_index(drop=True)
    if len(meta_train) != len(data["X_train"]):
        raise RuntimeError(
            "windows_dataset.npz y windows_meta.parquet no están alineados"
        )
    return data, meta_train, std, ref


def existing_rows(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def append_row(path: Path, row: dict) -> None:
    frame = existing_rows(path)
    key = "fit_seed"
    if (
        not frame.empty
        and key in frame
        and int(row[key]) in set(frame[key].astype(int))
    ):
        frame = frame.loc[frame[key].astype(int) != int(row[key])]
    pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    """Renderiza una tabla Markdown sin depender del paquete opcional tabulate."""

    def format_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def summarize(output_dir: Path, config: dict, reconstruction: dict) -> dict:
    audits = existing_rows(output_dir / "audit.csv")
    diffusion = existing_rows(output_dir / "tstr_diffusion.csv")
    historical = pd.read_csv(REPO_ROOT / "data/processed/tstr_nb03.csv")
    baseline_summary = (
        historical.groupby("brazo")["val_r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    baseline_means = dict(zip(baseline_summary["brazo"], baseline_summary["mean"]))

    summary: dict = {
        "profile": config["profile"],
        "complete_fit_seeds": sorted(
            diffusion.get("fit_seed", pd.Series(dtype=int)).astype(int).tolist()
        ),
        "expected_fit_seeds": config["fit_seeds"],
        "reconstruction": reconstruction,
        "historical_tstr": baseline_summary.to_dict(orient="records"),
    }
    if not audits.empty:
        summary["diffusion_audit_mean"] = (
            audits.select_dtypes(include="number").mean().to_dict()
        )
        summary["diffusion_audit_sd"] = (
            audits.select_dtypes(include="number").std().to_dict()
        )
    if not diffusion.empty:
        mean_r2 = float(diffusion["val_r2"].mean())
        sd_r2 = float(diffusion["val_r2"].std()) if len(diffusion) > 1 else None
        real = float(baseline_means["real"])
        wgan = float(baseline_means["wgan_gp"])
        vae = float(baseline_means["vae"])
        realnvp = float(baseline_means["realnvp"])
        paired = {}
        diffusion_by_seed = diffusion.rename(columns={"fit_seed": "seed"})[
            ["seed", "val_r2"]
        ].rename(columns={"val_r2": "diffusion_r2"})
        for comparator in ("real", "wgan_gp", "vae", "realnvp", "jitter"):
            reference = historical.loc[
                historical["brazo"] == comparator, ["seed", "val_r2"]
            ]
            joined = diffusion_by_seed.merge(reference, on="seed", how="inner")
            deltas = joined["diffusion_r2"] - joined["val_r2"]
            delta_mean = float(deltas.mean())
            delta_sd = float(deltas.std()) if len(deltas) > 1 else None
            if len(deltas) > 1 and delta_sd is not None:
                half_width = float(
                    stats.t.ppf(0.975, len(deltas) - 1)
                    * delta_sd
                    / np.sqrt(len(deltas))
                )
                interval = [delta_mean - half_width, delta_mean + half_width]
            else:
                interval = [None, None]
            paired[comparator] = {
                "delta_r2_mean": delta_mean,
                "delta_r2_sd": delta_sd,
                "ci95": interval,
                "wins": int((deltas > 0).sum()),
                "pairs": len(deltas),
            }

        if (
            paired["realnvp"]["ci95"][0] is not None
            and paired["realnvp"]["ci95"][0] > 0
        ):
            decision = "incorporar: supera incluso RealNVP en el piloto TSTR"
        elif paired["vae"]["ci95"][0] is not None and paired["vae"]["ci95"][0] > 0:
            decision = "incorporar en sustitución de WGAN-GP: supera también VAE"
        elif (
            paired["wgan_gp"]["ci95"][0] is not None
            and paired["wgan_gp"]["ci95"][0] > 0
        ):
            decision = "prometedor como sustituto de WGAN-GP, pero no mejora VAE; validar en malla"
        else:
            decision = "no incorporar: no supera WGAN-GP en utilidad TSTR"
        summary["diffusion_tstr"] = {
            "val_r2_mean": mean_r2,
            "val_r2_sd": sd_r2,
            "ratio_vs_trtr": mean_r2 / real,
            "delta_vs_wgan": mean_r2 - wgan,
            "delta_vs_vae": mean_r2 - vae,
            "delta_vs_realnvp": mean_r2 - realnvp,
            "paired_comparisons": paired,
            "decision": decision,
        }
    else:
        summary["diffusion_tstr"] = {
            "decision": "pendiente: el perfil no ejecutó downstream"
        }
    return summary


def render_report(output_dir: Path, config: dict, summary: dict) -> None:
    audit = existing_rows(output_dir / "audit.csv")
    tstr = existing_rows(output_dir / "tstr_diffusion.csv")
    hist = pd.DataFrame(summary["historical_tstr"])
    decision = summary["diffusion_tstr"]["decision"]
    lines = [
        "# Experimento Diffusion-TS",
        "",
        f"**Conclusión automática del protocolo:** {decision}.",
        "",
        "## Diseño",
        "",
        f"- Perfil: `{config['profile']}`; ajustes: {config['fit_seeds']}.",
        (
            f"- Entrenamiento: {config['train_steps']:,} actualizaciones por ajuste sobre "
            f"{config.get('effective_n_fit', config['n_fit']):,} trayectorias efectivas."
        ),
        (
            f"- Muestreo: {config['n_synth']:,} pares por ajuste, DDIM "
            f"{config['sample_steps']} pasos."
        ),
        "- Representación: 60 retornos observados + 21 futuros; el target se recalcula con su fórmula física.",
        f"- Fuente adaptada: `machine-learning-for-trading` en `{UPSTREAM_COMMIT}`.",
        "- Comparadores: resultados TSTR versionados de los mismos datos, predictor y semillas.",
        "",
        "## Reconstrucción",
        "",
        (
            f"Se recuperaron {summary['reconstruction']['n_paths']:,} trayectorias "
            f"({summary['reconstruction']['coverage']:.2%} de las ventanas de train). "
            f"El error máximo al reconstruir y fue "
            f"{summary['reconstruction']['target_max_abs_error']:.3g}."
        ),
        "",
        "## TSTR",
        "",
    ]
    if not tstr.empty:
        shown = tstr[
            ["fit_seed", "val_r2", "val_mse", "epoca_mejor", "sampling_seconds"]
        ]
        lines.extend([markdown_table(shown), ""])
        result = summary["diffusion_tstr"]
        lines.extend(
            [
                (
                    f"R² medio Diffusion-TS: **{result['val_r2_mean']:.4f}**; "
                    f"ratio TSTR/TRTR: **{result['ratio_vs_trtr']:.3f}**."
                ),
                "",
            ]
        )
        paired_rows = []
        for comparator, values in result["paired_comparisons"].items():
            paired_rows.append(
                {
                    "comparador": comparator,
                    "delta_r2": values["delta_r2_mean"],
                    "ci95_inf": values["ci95"][0],
                    "ci95_sup": values["ci95"][1],
                    "victorias": f"{values['wins']}/{values['pairs']}",
                }
            )
        lines.extend(
            [
                "### Comparaciones emparejadas por semilla",
                "",
                markdown_table(pd.DataFrame(paired_rows)),
                "",
            ]
        )
    else:
        lines.extend(["No ejecutado en este perfil.", ""])
    lines.extend(
        [
            "### Referencias históricas del notebook 03",
            "",
            markdown_table(hist),
            "",
        ]
    )
    if not audit.empty:
        selected = [
            "fit_seed",
            "curtosis_x",
            "acf_abs_lag1",
            "err_corr_xy",
            "w1_x_col_media",
            "err_corr_xx_spearman",
            "discriminative_auc",
        ]
        lines.extend(
            [
                "## Fidelidad",
                "",
                markdown_table(audit[selected]),
                "",
            ]
        )
    lines.extend(
        [
            "## Límite de inferencia",
            "",
            (
                "Este piloto compara sistemas: Diffusion-TS usa los 21 retornos futuros completos y "
                "deriva y, mientras los modelos originales reciben directamente el escalar y. Una "
                "mejora demuestra que la solución temporal merece avanzar, pero no separa el efecto "
                "de arquitectura del efecto de representación."
            ),
            "",
            (
                "La decisión definitiva de producción requiere la malla de escasez usando la "
                "trayectoria de 81 días como unidad de muestreo; este informe no reutiliza "
                "retornos fuera del presupuesto declarado."
            ),
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data, meta_train, std, ref = load_inputs()
    config = experiment_config(args, len(data["X_train"]))
    output_dir = (
        args.output_dir or REPO_ROOT / "reports/diffusion_ts_experiment" / args.profile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "synthetic").mkdir(exist_ok=True)

    config_text = json.dumps(json_ready(config), sort_keys=True)
    digest = hashlib.sha1(config_text.encode()).hexdigest()[:10]
    config["experiment_id"] = digest
    config_path = output_dir / "config.json"
    if config_path.exists() and not args.restart:
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("experiment_id") != digest:
            raise RuntimeError(
                f"{output_dir} contiene otra configuración. Usa otro --output-dir o --restart."
            )
    write_json(config_path, config)

    reconstruction_result = reconstruct_return_paths(
        data["X_train"], meta_train, y=data["y_train"], horizon=21
    )
    reconstruction = {
        "n_train_windows": len(data["X_train"]),
        "n_paths": len(reconstruction_result.paths),
        "coverage": len(reconstruction_result.paths) / len(data["X_train"]),
        "rejected_overlap": reconstruction_result.rejected_overlap,
        "target_max_abs_error": reconstruction_result.target_max_abs_error,
    }
    config["effective_n_fit"] = min(config["n_fit"], reconstruction["n_paths"])
    write_json(config_path, config)
    write_json(output_dir / "reconstruction.json", reconstruction)
    print(
        f"[paths] {reconstruction['n_paths']:,}/{reconstruction['n_train_windows']:,} "
        f"({reconstruction['coverage']:.2%}); error y máx="
        f"{reconstruction['target_max_abs_error']:.3g}",
        flush=True,
    )

    Xs_val = ((data["X_val"] - std["x_mu"]) / std["x_sd"]).astype(np.float32)
    y_real_standardized = ((data["y_train"] - std["y_mu"]) / std["y_sd"]).astype(
        np.float32
    )
    X_real_standardized = ((data["X_train"] - std["x_mu"]) / std["x_sd"]).astype(
        np.float32
    )
    XY_real = np.column_stack([X_real_standardized, y_real_standardized]).astype(
        np.float32
    )
    device = get_device()
    print(f"[device] {device}", flush=True)

    for fit_seed in config["fit_seeds"]:
        rng = np.random.default_rng(fit_seed)
        if config["n_fit"] < len(reconstruction_result.paths):
            fit_idx = rng.choice(
                len(reconstruction_result.paths), config["n_fit"], replace=False
            )
            fit_paths = reconstruction_result.paths[fit_idx]
        else:
            fit_paths = reconstruction_result.paths

        checkpoint = (
            output_dir / "checkpoints" / f"diffusion_ts_seed{fit_seed}_{digest}.pt"
        )
        if checkpoint.exists() and not args.restart:
            print(f"[seed {fit_seed}] cargando checkpoint", flush=True)
            generator = DiffusionTSGenerator.load(checkpoint, device=device)
        else:
            print(f"[seed {fit_seed}] entrenamiento", flush=True)
            generator = DiffusionTSGenerator(
                train_steps=config["train_steps"],
                sample_steps=config["sample_steps"],
                batch_size=config["batch_size"],
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                ff_mult=config["ff_mult"],
                diffusion_steps=config["diffusion_steps"],
                spectral_weight=config["spectral_weight"],
                ema_decay=config["ema_decay"],
            ).fit_paths(
                fit_paths,
                x_mu=std["x_mu"],
                x_sd=std["x_sd"],
                y_mu=std["y_mu"],
                y_sd=std["y_sd"],
                seed=fit_seed,
            )
            generator.save(checkpoint)
            pd.DataFrame(generator.history_).to_csv(
                output_dir / f"training_history_seed{fit_seed}.csv", index_label="step"
            )

        synthetic_path = output_dir / "synthetic" / f"xy_seed{fit_seed}_{digest}.npy"
        sample_started = time.perf_counter()
        if synthetic_path.exists() and not args.restart:
            XY_synth = np.load(synthetic_path)
            sampling_seconds = 0.0
            print(f"[seed {fit_seed}] sintético cargado: {len(XY_synth):,}", flush=True)
        else:
            print(f"[seed {fit_seed}] muestreando {config['n_synth']:,}", flush=True)
            XY_synth = generator.sample(config["n_synth"], seed=fit_seed + 1000)
            sampling_seconds = time.perf_counter() - sample_started
            np.save(synthetic_path, XY_synth)

        audit_path = output_dir / "audit.csv"
        audit_done = existing_rows(audit_path)
        if (
            args.restart
            or audit_done.empty
            or fit_seed not in set(audit_done["fit_seed"].astype(int))
        ):
            print(f"[seed {fit_seed}] auditoría de fidelidad", flush=True)
            audit = audit_generator(XY_real, XY_synth, seed=fit_seed)
            append_row(
                audit_path,
                {
                    "fit_seed": fit_seed,
                    "n_fit": len(fit_paths),
                    "n_synth": len(XY_synth),
                    "training_seconds": generator.training_seconds_,
                    "sampling_seconds": sampling_seconds,
                    **audit,
                },
            )

        if config["run_downstream"]:
            tstr_path = output_dir / "tstr_diffusion.csv"
            tstr_done = existing_rows(tstr_path)
            if (
                args.restart
                or tstr_done.empty
                or fit_seed not in set(tstr_done["fit_seed"].astype(int))
            ):
                print(f"[seed {fit_seed}] TSTR downstream", flush=True)
                result = entrenar_y_evaluar(
                    XY_synth[:, :-1],
                    XY_synth[:, -1],
                    Xs_val,
                    data["y_val"],
                    ref=ref,
                    std=std,
                    device=device,
                    seed=fit_seed,
                )
                append_row(
                    tstr_path,
                    {
                        "brazo": DiffusionTSGenerator.name,
                        "fit_seed": fit_seed,
                        "sample_seed": fit_seed + 1000,
                        "n_fit": len(fit_paths),
                        "n_synth": len(XY_synth),
                        "training_seconds": generator.training_seconds_,
                        "sampling_seconds": sampling_seconds,
                        **result,
                    },
                )

        summary = summarize(output_dir, config, reconstruction)
        write_json(output_dir / "summary.json", summary)
        render_report(output_dir, config, summary)
        print(f"[seed {fit_seed}] completada", flush=True)

    print(f"[ok] resultados en {output_dir}", flush=True)


if __name__ == "__main__":
    main()

"""
figuras_extension.py — Figuras 28 y 29 a partir de la malla extendida.

Genera las dos gráficas que sostienen la conclusión central de la práctica:

  28_u_invertida.png  La curva en U invertida: delta de R² pareado por semilla
                      frente al ratio sintético/real, por generador. Panel
                      derecho: las tres trayectorias individuales de RealNVP,
                      para mostrar que el giro no es un artefacto de promediar.

  29_mecanismo.png    Las dos regularidades que explican la forma de la curva:
                      la fidelidad (AUC discriminativo) fija cuánto se puede
                      ganar, y la utilidad TSTR fija hacia dónde cae la curva
                      cuando el ratio se dispara.

Requiere: malla_extendida.csv, auditoria_nb03.csv, tstr_nb03_resumen.csv.

Uso
---
    uv run python scripts/figuras_extension.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

GENS = ["jitter", "realnvp", "block_bootstrap", "gaussiana"]
RAT = [0, 1, 3, 10, 30]
COL = {"jitter": "#2E7D32", "realnvp": "#1565C0",
       "block_bootstrap": "#EF6C00", "gaussiana": "#B71C1C"}
LAB = {"jitter": "Jitter (ruido)", "realnvp": "RealNVP (flow)",
       "block_bootstrap": "Block bootstrap", "gaussiana": "Gaussiana"}
MRK = {"jitter": "o", "realnvp": "s", "block_bootstrap": "^", "gaussiana": "v"}


def deltas_pareados(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Delta de R² frente al escenario solo-real, restando semilla a semilla.

    El pareado es imprescindible: la varianza entre semillas del submuestreo
    (±0,04 de R²) es del mismo orden que los efectos que buscamos, y comparar
    medias sin parear los enmascara por completo.
    """
    base = d[d.generador == "ninguno"].set_index("seed")["test_r2"]
    out = {}
    for g in GENS:
        piv = d[d.generador == g].pivot_table(index="seed", columns="ratio", values="test_r2")
        out[g] = piv.sub(base, axis=0)
    return out


def figura_28(dl: dict[str, pd.DataFrame], destino: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for g in GENS:
        xs, mu, se = [0], [0.0], [0.0]
        for r in RAT[1:]:
            if r not in dl[g].columns:
                continue
            v = dl[g][r].dropna()
            if len(v) == 0:
                continue
            xs.append(r); mu.append(v.mean())
            se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        xp = np.arange(len(xs))
        ax.errorbar(xp, mu, yerr=se, color=COL[g], marker=MRK[g], lw=2.2, ms=8,
                    capsize=4, label=LAB[g], zorder=3)
        k = int(np.argmax(mu))
        if 0 < k < len(mu) - 1:  # marcar solo óptimos interiores (giro real)
            ax.scatter([xp[k]], [mu[k]], s=280, facecolors="none",
                       edgecolors=COL[g], lw=2.2, zorder=4)
    ax.axhline(0, color="#555", lw=1.4, ls="--")
    ax.set_xticks(range(len(RAT))); ax.set_xticklabels([f"{r}x" for r in RAT])
    ax.set_xlabel("Ratio sintético / real", fontsize=11.5)
    ax.set_ylabel("Δ R² test frente a solo-real (pareado por semilla)", fontsize=11.5)
    ax.set_title("La curva en U invertida: el dato sintético ayuda… hasta que sustituye a la señal",
                 fontsize=12.5, weight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.grid(alpha=.25, axis="y")
    ax.text(0.985, 0.965, "N real = 1.000 ventanas · 3 semillas · arquitectura congelada",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")

    ax = axes[1]
    for s in dl["realnvp"].index:
        row = dl["realnvp"].loc[s].dropna()
        xp = [RAT.index(r) for r in row.index]
        ax.plot([0] + xp, [0.0] + list(row.values), marker="o", ms=6, lw=1.8,
                alpha=.85, label=f"semilla {s}")
        k = int(np.argmax(row.values))
        ax.scatter([xp[k]], [row.values[k]], s=170, facecolors="none",
                   edgecolors="k", lw=1.6, zorder=5)
    ax.axhline(0, color="#555", lw=1.4, ls="--")
    ax.set_xticks(range(len(RAT))); ax.set_xticklabels([f"{r}x" for r in RAT])
    ax.set_xlabel("Ratio sintético / real", fontsize=11.5)
    ax.set_title("RealNVP: las 3 semillas giran\n(pico en 3x–10x, caída hasta 30x)",
                 fontsize=12, weight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9.5); ax.grid(alpha=.25, axis="y")

    plt.tight_layout(); plt.savefig(destino, dpi=190, bbox_inches="tight"); plt.close()


def figura_29(dl: dict[str, pd.DataFrame], aud: pd.DataFrame,
              tstr: pd.DataFrame, destino: Path) -> None:
    info = {}
    for g in GENS:
        m = dl[g].mean().dropna()
        info[g] = dict(pico=max(m.max(), 0.0),
                       rstar=(m.idxmax() if m.max() > 0 else 0.0),
                       alto=m.loc[m.index.max()], rmax=m.index.max(),
                       auc=aud.loc[g, "discriminative_auc"],
                       tstr=tstr.loc[g, "ratio_tstr_trtr"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    corto = {g: LAB[g].split(" (")[0] for g in GENS}

    ax = axes[0]
    for g, v in info.items():
        ax.scatter(v["auc"], v["pico"], s=330, color=COL[g], zorder=3,
                   edgecolors="white", lw=2)
        ax.annotate(f"{corto[g]}\nóptimo {v['rstar']:g}x", (v["auc"], v["pico"]),
                    textcoords="offset points", xytext=(0, -36 if g == "jitter" else 16),
                    ha="center", fontsize=10, weight="bold", color=COL[g])
    xs = np.array([info[g]["auc"] for g in GENS]); ys = np.array([info[g]["pico"] for g in GENS])
    o = np.argsort(xs); ax.plot(xs[o], ys[o], color="#999", ls="--", lw=1.6, zorder=1)
    ax.axhline(0, color="#555", lw=1.2)
    ax.axvline(0.5, color="#2E7D32", lw=1.2, ls=":", alpha=.7)
    ax.set_xlabel("AUC discriminativo   (0,5 = indistinguible del real)", fontsize=11)
    ax.set_ylabel("Mejor Δ R² alcanzable", fontsize=11)
    ax.set_title("1) La fidelidad fija cuánto se puede ganar", fontsize=12.5, weight="bold")
    ax.grid(alpha=.25)

    ax = axes[1]
    for g, v in info.items():
        ax.scatter(v["tstr"], v["alto"], s=330, color=COL[g], zorder=3,
                   edgecolors="white", lw=2)
        ax.annotate(f"{corto[g]}\n{v['rmax']:g}x", (v["tstr"], v["alto"]),
                    textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=10, weight="bold", color=COL[g])
    xs = np.array([info[g]["tstr"] for g in GENS]); ys = np.array([info[g]["alto"] for g in GENS])
    o = np.argsort(xs); ax.plot(xs[o], ys[o], color="#999", ls="--", lw=1.6, zorder=1)
    rho = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    ax.axhline(0, color="#555", lw=1.2)
    ax.set_xlabel("Utilidad TSTR: R² entrenando SOLO con sintético (÷ real)", fontsize=11)
    ax.set_ylabel("Δ R² en el ratio más alto probado", fontsize=11)
    ax.set_title("2) Y fija hacia dónde cae la curva", fontsize=12.5, weight="bold")
    ax.text(0.03, 0.93, f"Spearman = {rho:+.2f}\nCon ratio alto el modelo\nconverge a lo que el generador\nsabe enseñar por sí solo",
            transform=ax.transAxes, fontsize=9.5, va="top", color="#333")
    ax.grid(alpha=.25)

    plt.tight_layout(); plt.savefig(destino, dpi=190, bbox_inches="tight"); plt.close()


def tabla_resumen(dl: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Delta medio y t pareado por (generador, ratio). |t| >= 2.5 = significativo."""
    filas = []
    for g in GENS:
        for r in RAT[1:]:
            if r not in dl[g].columns:
                continue
            v = dl[g][r].dropna()
            if len(v) < 2:
                continue
            t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
            filas.append(dict(generador=g, ratio=r, n_seeds=len(v),
                              delta=v.mean(), sd=v.std(ddof=1), t=t,
                              significativo=abs(t) >= 2.5))
    return pd.DataFrame(filas)


def main() -> int:
    cfg = Config()
    proc, figs = cfg.out_dir, cfg.fig_dir
    figs.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(proc / "malla_extendida.csv")
    aud = pd.read_csv(proc / "auditoria_nb03.csv", index_col=0)
    tstr = pd.read_csv(proc / "tstr_nb03_resumen.csv").set_index("brazo")

    dl = deltas_pareados(d)
    figura_28(dl, figs / "28_u_invertida.png")
    figura_29(dl, aud, tstr, figs / "29_mecanismo.png")

    tab = tabla_resumen(dl)
    tab.to_csv(proc / "malla_extendida_resumen.csv", index=False)
    print(tab.round(4).to_string(index=False))
    print(f"\nFiguras en {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

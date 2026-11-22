"""Figures du projet (outputs/figures/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import LINES, N  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#1f5f8b", "#d9822b", "#3a8a5c", "#9aa5b1"
COLORS = {"chain_ladder": BLUE, "cape_cod": GREEN, "ml": ORANGE}
LABELS = {"chain_ladder": "Chain-Ladder", "cape_cod": "Cape Cod (BF)", "ml": "XGBoost (correction de CL)"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def example_triangle(tri: dict, title: str, path: Path):
    """Paiements cumulés / primes par année de survenance : connu fin 1997 (plein), réalisé ensuite (pointillé)."""
    C, prem = tri["paid"], tri["premium"]
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i in range(N):
        known = N - i
        color = cmap(i / (N - 1))
        ax.plot(range(1, known + 1), C[i, :known] / prem[i], color=color, lw=2, marker="o", ms=3,
                label=str(1988 + i) if i % 3 == 0 or i == N - 1 else None)
        if known < N:
            ax.plot(range(known, N + 1), C[i, known - 1:] / prem[i], color=color, lw=1.2, ls="--")
    ax.set_xlabel("Année de développement")
    ax.set_ylabel("Paiements cumulés / primes")
    ax.set_xticks(range(1, N + 1))
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Survenance", fontsize=9, ncol=2, loc="lower right")
    ax.text(0.01, 0.98, "trait plein : connu fin 1997 / pointillé : réalisé ensuite",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#5b6670")
    _save(fig, path)


def errors_by_line(df: pd.DataFrame, methods: list[str], path: Path):
    """Erreur de réserve en % des primes, par branche et par méthode."""
    lines = ["Toutes"] + [l for l in LINES if l in set(df["line"])]
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.26
    for k, m in enumerate(methods):
        data = [((df[m] - df["true"]) / df["premium"] * 100).to_numpy()] + \
               [((g[m] - g["true"]) / g["premium"] * 100).to_numpy() for l in lines[1:] for _, g in [(l, df[df["line"] == l])]]
        pos = np.arange(len(lines)) + (k - 1) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=COLORS[m], alpha=0.75)
        for med in bp["medians"]:
            med.set(color="black")
        ax.plot([], [], color=COLORS[m], lw=8, alpha=0.75, label=LABELS[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(lines)), ["Toutes"] + [LINES[l] for l in lines[1:]], rotation=15)
    ax.set_ylabel("(réserve estimée − réelle) / primes, en %")
    ax.set_title("Erreur de réserve par branche (backtest sur 343 triangles)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    _save(fig, path)


def predicted_vs_true(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lim = [df["true"].clip(lower=1).min() * 0.5, df["true"].max() * 2]
    for ax, m in zip(axes, ["chain_ladder", "ml"]):
        ax.scatter(df["true"].clip(lower=1), df[m].clip(lower=1), s=14, alpha=0.6, color=COLORS[m])
        ax.plot(lim, lim, color="black", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Réserve réelle (milliers de $)")
        ax.set_title(LABELS[m])
    axes[0].set_ylabel("Réserve estimée fin 1997 (milliers de $)")
    fig.suptitle("Réserves estimées contre réserves réelles (échelle log)", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def mack_coverage(summary: pd.DataFrame, path: Path):
    s = summary.set_index("branche")
    labels = ["Toutes"] + [LINES[b] for b in s.index[1:]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, s["couverture_mack_95_pct"], color=BLUE)
    ax.axhline(95, color=ORANGE, ls="--", lw=2, label="Couverture théorique : 95 %")
    for i, v in enumerate(s["couverture_mack_95_pct"]):
        ax.text(i, v + 1.5, f"{v:.0f} %", ha="center")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Réserves réelles dans l'intervalle (%)")
    ax.set_title("Intervalle de prédiction à 95 % de Mack : couverture réelle", fontweight="bold")
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, path)


def feature_importance(boosters: list, path: Path):
    imp = pd.DataFrame([b.get_score(importance_type="gain") for b in boosters]).fillna(0).mean()
    imp = (imp / imp.sum()).sort_values()
    names = {
        "case_to_paid": "Charge dossier / paiements", "cl_log_factor": "Facteur Chain-Ladder de la compagnie",
        "market_log_factor": "Facteur de développement du marché", "paid_to_premium": "Paiements / primes",
        "case_to_premium": "Charge dossier / primes", "last_growth": "Dernière croissance des paiements",
        "log_premium": "Taille (log des primes)", "lag": "Retard", "horizon": "Horizon restant", "line": "Branche",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([names.get(i, i) for i in imp.index], imp.values * 100, color=ORANGE)
    ax.set_xlabel("Part du gain total (%)")
    ax.set_title("Variables utilisées par XGBoost pour corriger Chain-Ladder", fontweight="bold")
    _save(fig, path)

"""Backtest du provisionnement sur la CAS Loss Reserving Database.

Pour chaque triangle (branche × assureur), on estime la réserve fin 1997 avec :
Chain-Ladder (et l'intervalle de Mack), Bornhuetter-Ferguson / Cape Cod, et XGBoost
(correction de Chain-Ladder apprise sur les autres compagnies). On compare ensuite
chaque estimation aux paiements réellement observés de 1998 à 2006.

Utilisation :
    python main.py
"""

import json
import sys
from pathlib import Path

from src import plots
from src.backtest import classical_reserves, summarize
from src.data import LINES, build_triangles, load_long
from src.ml import cross_fitted_reserves

OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
METHODS = ["chain_ladder", "cape_cod", "ml"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)

    triangles = build_triangles(load_long())
    print(f"Triangles retenus : {len(triangles)}")

    df = classical_reserves(triangles)
    print("XGBoost (validation croisée par compagnie)...")
    ml_reserves, boosters = cross_fitted_reserves(triangles)
    df["ml"] = [ml_reserves[(l, c)] for l, c in zip(df["line"], df["company"])]
    df.to_csv(OUT / "backtest_par_triangle.csv", index=False, float_format="%.1f")

    summary = summarize(df, METHODS)
    summary["branche"] = summary["branche"].map(lambda b: LINES.get(b, b))
    summary.to_csv(OUT / "resume_par_branche.csv", index=False, float_format="%.2f")

    # Exemple : un triangle d'auto particuliers de taille moyenne
    auto = df[df["line"] == "ppauto"].sort_values("premium")
    example_key = ("ppauto", int(auto.iloc[len(auto) // 2]["company"]))
    plots.example_triangle(triangles[example_key], "Exemple : un assureur auto (paiements cumulés / primes)",
                           FIG / "exemple_triangle.png")
    plots.errors_by_line(df, METHODS, FIG / "erreurs_par_branche.png")
    plots.predicted_vs_true(df, FIG / "reserves_estimees_vs_reelles.png")
    summary_codes = summarize(df, METHODS)
    plots.mack_coverage(summary_codes, FIG / "couverture_mack.png")
    plots.feature_importance(boosters, FIG / "variables_xgboost.png")

    metrics = summary.set_index("branche").round(2).to_dict(orient="index")
    with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(summary.round(1).T.to_string())


if __name__ == "__main__":
    main()

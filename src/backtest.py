"""Backtest : réserves estimées fin 1997 comparées aux paiements réellement observés ensuite."""

import numpy as np
import pandas as pd

from .chainladder import cape_cod, mack, mack_interval
from .data import N, upper_mask


def true_reserve(C: np.ndarray) -> float:
    """Réserve réelle = paiements cumulés au retard 10 − paiements connus fin 1997."""
    latest = np.array([C[i, N - 1 - i] for i in range(N)])
    return float((C[:, N - 1] - latest).sum())


def classical_reserves(triangles: dict) -> pd.DataFrame:
    rows = []
    mask = upper_mask()
    for key, tri in triangles.items():
        C_known = np.where(mask, tri["paid"], 0.0)
        mk = mack(C_known)
        cc = cape_cod(C_known, tri["premium"])
        cl_res = float(mk["reserve"].sum())
        lo, hi = mack_interval(cl_res, mk["total_se"])
        rows.append({
            "line": key[0], "company": key[1],
            "premium": float(tri["premium"].sum()),
            "true": true_reserve(tri["paid"]),
            "chain_ladder": cl_res,
            "cape_cod": float(cc["reserve"].sum()),
            "mack_se": mk["total_se"], "mack_lo": lo, "mack_hi": hi,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Erreurs de réserve par branche : en % de la réserve réelle et en % des primes."""
    out = []
    for line, g in [("Toutes branches", df)] + list(df.groupby("line")):
        row = {"branche": line, "triangles": len(g)}
        for m in methods:
            err = g[m] - g["true"]
            row[f"{m}_erreur_abs_mediane_pct_reserve"] = float(np.median(np.abs(err) / g["true"].clip(lower=1)) * 100)
            row[f"{m}_rmse_pct_prime"] = float(np.sqrt(np.mean((err / g["premium"]) ** 2)) * 100)
            row[f"{m}_biais_total_pct"] = float(err.sum() / g["true"].sum() * 100)
        row["ml_meilleur_que_cl_pct"] = float(np.mean(np.abs(g["ml"] - g["true"]) < np.abs(g["chain_ladder"] - g["true"])) * 100)
        row["couverture_mack_95_pct"] = float(np.mean((g["true"] >= g["mack_lo"]) & (g["true"] <= g["mack_hi"])) * 100)
        out.append(row)
    return pd.DataFrame(out)

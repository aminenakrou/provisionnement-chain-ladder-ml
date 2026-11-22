"""Provisionnement par machine learning (XGBoost), entraîné sur l'ensemble des triangles du marché.

Idée : Chain-Ladder n'utilise que les paiements d'une seule compagnie. Le modèle apprend, sur
toutes les compagnies, à corriger le développement Chain-Ladder à l'aide d'informations que
Chain-Ladder ignore : provisions dossier par dossier (charge déclarée / paiements), niveau de
sinistralité (paiements / primes), branche, retard.

Cible : la correction à apporter au développement Chain-Ladder de la compagnie,
    log(C[i, j+h] / C[i, j]) − log(facteur Chain-Ladder de j à j+h).
Le modèle part donc de Chain-Ladder et ne s'en écarte que si les données le justifient.
Entraînement : uniquement des couples de cellules connues fin 1997 (i + j + h ≤ 9).
Prédiction : pour chaque année de survenance, du dernier retard connu jusqu'au retard 10.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from .chainladder import development_factors
from .data import N

FEATURES = ["lag", "horizon", "log_premium", "paid_to_premium", "case_to_paid", "case_to_premium",
            "last_growth", "cl_log_factor", "market_log_factor", "line"]
SEED = 2026


def _loo_factors(C: np.ndarray, i: int) -> np.ndarray:
    """Facteurs Chain-Ladder de la compagnie, calculés sans l'année de survenance i (pas de fuite de la cible)."""
    f = np.full(N - 1, np.nan)
    for j in range(N - 1):
        rows = [r for r in range(N - 1 - j) if r != i]
        den = C[rows, j].sum()
        if rows and den > 0:
            f[j] = C[rows, j + 1].sum() / den
    return f


def _row(tri: dict, line: str, i: int, j: int, h: int, f: np.ndarray, market: dict) -> dict:
    C, case, prem = tri["paid"], tri["case"], tri["premium"]
    c = C[i, j]
    return {
        "line": line,
        "lag": j + 1,
        "horizon": h,
        "log_premium": np.log(prem[i]),
        "paid_to_premium": c / prem[i],
        "case_to_paid": np.clip(case[i, j] / c, 0, 20),
        "case_to_premium": case[i, j] / prem[i],
        "last_growth": np.log(c / C[i, j - 1]) if j >= 1 and C[i, j - 1] > 0 else np.nan,
        "cl_log_factor": np.log(np.prod(f[j:j + h])) if np.all(np.isfinite(f[j:j + h])) and np.all(f[j:j + h] > 0) else np.nan,
        "market_log_factor": float(np.sum(np.log(market[line][j:j + h]))),
    }


def market_factors(triangles: dict, keys) -> dict:
    """Facteurs de développement « marché » par branche (Chain-Ladder sur la somme des triangles)."""
    out = {}
    for line in {k[0] for k in keys}:
        total = sum(triangles[k]["paid"] for k in keys if k[0] == line)
        out[line] = development_factors(total)
    return out


def training_table(triangles: dict, keys, market: dict) -> pd.DataFrame:
    rows = []
    for key in keys:
        tri, line = triangles[key], key[0]
        C = tri["paid"]
        for i in range(N):
            f = _loo_factors(C, i)
            for j in range(N - 1 - i):
                for h in range(1, N - i - j):
                    r = _row(tri, line, i, j, h, f, market)
                    base = r["cl_log_factor"] if np.isfinite(r["cl_log_factor"]) else r["market_log_factor"]
                    r["base"] = base
                    r["target"] = np.log(C[i, j + h] / C[i, j]) - base
                    r["key"] = key
                    rows.append(r)
    df = pd.DataFrame(rows)
    df["line"] = df["line"].astype(pd.CategoricalDtype(sorted(market)))
    return df


def prediction_table(triangles: dict, keys, market: dict) -> pd.DataFrame:
    rows = []
    for key in keys:
        tri, line = triangles[key], key[0]
        f = development_factors(tri["paid"])
        for i in range(1, N):
            j = N - 1 - i
            r = _row(tri, line, i, j, N - 1 - j, f, market)
            r["key"], r["ay_index"], r["latest"] = key, i, tri["paid"][i, j]
            r["base"] = r["cl_log_factor"] if np.isfinite(r["cl_log_factor"]) else r["market_log_factor"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["line"] = df["line"].astype(pd.CategoricalDtype(sorted(market)))
    return df


PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "max_cat_to_onehot": 1,
    "seed": SEED,
}


def fit(train: pd.DataFrame):
    """XGBoost avec arrêt précoce sur 15 % des triangles d'entraînement (validation par triangle)."""
    keys = train["key"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED)
    val_keys = set(map(tuple, keys[rng.random(len(keys)) < 0.15]))
    is_val = train["key"].map(lambda k: k in val_keys).to_numpy()
    dtr = xgb.DMatrix(train.loc[~is_val, FEATURES], label=train.loc[~is_val, "target"], enable_categorical=True)
    dva = xgb.DMatrix(train.loc[is_val, FEATURES], label=train.loc[is_val, "target"], enable_categorical=True)
    booster = xgb.train(PARAMS, dtr, num_boost_round=3000, evals=[(dva, "val")],
                        early_stopping_rounds=100, verbose_eval=False)
    return booster


def predict_ultimate(booster, table: pd.DataFrame) -> np.ndarray:
    """Ultime = dernier paiement connu × facteur Chain-Ladder × correction apprise."""
    d = xgb.DMatrix(table[FEATURES], enable_categorical=True)
    correction = booster.predict(d, iteration_range=(0, booster.best_iteration + 1))
    # Contrainte : les paiements cumulés ne baissent pas, donc l'ultime est au moins le dernier paiement connu
    log_dev = np.maximum(table["base"].to_numpy() + correction, 0.0)
    return table["latest"].to_numpy() * np.exp(log_dev)


def cross_fitted_reserves(triangles: dict, n_folds: int = 5) -> tuple[dict, list]:
    """Réserves ML par triangle : chaque compagnie est provisionnée par un modèle qui ne l'a jamais vue."""
    keys = sorted(triangles)
    companies = sorted({k[1] for k in keys})
    rng = np.random.default_rng(SEED)
    fold_of = {c: f for c, f in zip(companies, rng.permutation(len(companies)) % n_folds)}

    reserves, boosters = {}, []
    for fold in range(n_folds):
        train_keys = [k for k in keys if fold_of[k[1]] != fold]
        test_keys = [k for k in keys if fold_of[k[1]] == fold]
        if not test_keys:
            continue
        market = market_factors(triangles, train_keys)
        booster = fit(training_table(triangles, train_keys, market))
        boosters.append(booster)
        table = prediction_table(triangles, test_keys, market)
        table["ultimate"] = predict_ultimate(booster, table)
        for key, g in table.groupby("key"):
            reserves[key] = float((g["ultimate"] - g["latest"]).sum())
    return reserves, boosters

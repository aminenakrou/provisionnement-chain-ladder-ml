"""Tests du modèle ML et du backtest : en particulier, aucune information postérieure à 1997."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import true_reserve
from src.data import N, upper_mask
from src.ml import market_factors, prediction_table, training_table


def _toy_triangles(seed=0, n_companies=4):
    rng = np.random.default_rng(seed)
    out = {}
    for c in range(n_companies):
        prem = rng.uniform(800, 1200, N)
        dev = np.cumprod(np.r_[0.4, rng.uniform(1.0, 1.6, N - 1)])
        paid = np.outer(prem * rng.uniform(0.6, 0.8, N), dev / dev[-1]) + 1
        case = paid * rng.uniform(1.0, 1.5, (N, N))
        out[("ppauto", c)] = {"paid": paid, "case": case, "premium": prem}
    return out


def test_upper_mask_counts_known_cells():
    assert upper_mask().sum() == N * (N + 1) // 2


def test_true_reserve_is_future_payments():
    C = np.tile(np.arange(1, N + 1, dtype=float), (N, 1))  # chaque ligne : 1, 2, ..., 10
    expected = sum(N - (N - i) for i in range(N))
    assert true_reserve(C) == pytest.approx(expected)


def test_no_leakage_from_lower_triangle():
    """Modifier la partie inférieure (le futur) ne doit changer ni l'entraînement ni les variables de prédiction."""
    tri = _toy_triangles()
    keys = sorted(tri)
    market = market_factors(tri, keys)
    train_a = training_table(tri, keys, market)
    pred_a = prediction_table(tri, keys, market)

    altered = {k: {**v, "paid": v["paid"].copy(), "case": v["case"].copy()} for k, v in tri.items()}
    lower = ~upper_mask()
    for v in altered.values():
        v["paid"][lower] *= 50
        v["case"][lower] *= 50
    market_b = market_factors(altered, keys)
    train_b = training_table(altered, keys, market_b)
    pred_b = prediction_table(altered, keys, market_b)

    cols = [c for c in train_a.columns if c not in ("key", "line")]
    pd.testing.assert_frame_equal(train_a[cols], train_b[cols])
    cols = [c for c in pred_a.columns if c not in ("key", "line")]
    pd.testing.assert_frame_equal(pred_a[cols], pred_b[cols])


def test_training_table_size():
    tri = _toy_triangles(n_companies=1)
    keys = sorted(tri)
    df = training_table(tri, keys, market_factors(tri, keys))
    # couples (i, j, h) avec h ≥ 1 et i + j + h ≤ N - 1
    expected = sum(1 for i in range(N) for j in range(N) for h in range(1, N) if i + j + h <= N - 1)
    assert len(df) == expected

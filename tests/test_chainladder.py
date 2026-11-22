"""Validation sur le triangle RAA (Mack, 1993), exemple de référence de la littérature actuarielle.

Valeurs attendues (package R ChainLadder, MackChainLadder(RAA)) :
réserve totale ≈ 52 135, erreur standard de Mack totale ≈ 26 909.
"""

import numpy as np
import pytest

from src.chainladder import cape_cod, chain_ladder, mack, mack_interval

RAA = [
    [5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834],
    [106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704],
    [3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466],
    [5655, 11555, 15766, 21266, 23425, 26083, 27067],
    [1092, 9565, 15836, 22169, 25955, 26180],
    [1513, 6445, 11702, 12935, 15852],
    [557, 4020, 10946, 12314],
    [1351, 6947, 13112],
    [3133, 5395],
    [2063],
]


def raa_square():
    C = np.full((10, 10), np.nan)
    for i, row in enumerate(RAA):
        C[i, : len(row)] = row
    return np.nan_to_num(C)


def test_chain_ladder_raa_total_reserve():
    res = chain_ladder(raa_square())
    assert res["reserve"].sum() == pytest.approx(52135, rel=1e-3)
    assert res["factors"][0] == pytest.approx(2.999, abs=1e-3)


def test_mack_raa_standard_error():
    res = mack(raa_square())
    assert res["total_se"] == pytest.approx(26909, rel=5e-3)
    assert res["se"][0] == 0


def test_mack_interval_contains_mean():
    lo, hi = mack_interval(52135, 26909)
    assert lo < 52135 < hi


def test_cape_cod_reserve_positive():
    C = raa_square()
    res = cape_cod(C, premium=np.full(10, 30000.0))
    assert res["elr"] > 0
    assert (res["reserve"][1:] > 0).all()

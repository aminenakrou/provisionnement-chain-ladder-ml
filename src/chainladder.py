"""Méthodes actuarielles classiques de provisionnement, implémentées sur un triangle de paiements cumulés.

Convention : C est un carré n × n (années de survenance en lignes, retards en colonnes) ;
seules les cellules i + j ≤ n - 1 sont supposées connues.
"""

import numpy as np
from scipy.stats import norm


def _known(n: int) -> np.ndarray:
    i, j = np.indices((n, n))
    return i + j <= n - 1


def development_factors(C: np.ndarray) -> np.ndarray:
    """Facteurs de développement Chain-Ladder pondérés par les volumes : f_j = Σ C[i, j+1] / Σ C[i, j]."""
    n = C.shape[0]
    f = np.ones(n - 1)
    for j in range(n - 1):
        rows = np.arange(n - 1 - j)  # années pour lesquelles le retard j+1 est connu
        f[j] = C[rows, j + 1].sum() / C[rows, j].sum()
    return f


def chain_ladder(C: np.ndarray) -> dict:
    """Projection Chain-Ladder : ultimes et réserves par année de survenance."""
    n = C.shape[0]
    f = development_factors(C)
    latest = np.array([C[i, n - 1 - i] for i in range(n)])
    cdf = np.array([np.prod(f[n - 1 - i:]) for i in range(n)])  # facteur cumulé jusqu'à l'ultime
    ultimate = latest * cdf
    return {"factors": f, "cdf": cdf, "latest": latest, "ultimate": ultimate, "reserve": ultimate - latest}


def mack(C: np.ndarray) -> dict:
    """Modèle de Mack (1993) : erreur quadratique de prédiction de la réserve, par année et au total."""
    n = C.shape[0]
    cl = chain_ladder(C)
    f = cl["factors"]

    sigma2 = np.zeros(n - 1)
    for j in range(n - 1):
        rows = np.arange(n - 1 - j)
        if len(rows) > 1:
            ratio = C[rows, j + 1] / C[rows, j]
            sigma2[j] = (C[rows, j] * (ratio - f[j]) ** 2).sum() / (len(rows) - 1)
    # Dernier sigma : extrapolation proposée par Mack
    if n >= 4:
        sigma2[n - 2] = min(sigma2[n - 3] ** 2 / sigma2[n - 4] if sigma2[n - 4] > 0 else 0, sigma2[n - 3], sigma2[n - 4])

    # Carré complété par Chain-Ladder
    full = C.astype(float).copy()
    for i in range(n):
        for j in range(n - i, n):
            full[i, j] = full[i, j - 1] * f[j - 1]

    col_sum = np.array([C[np.arange(n - 1 - j), j].sum() for j in range(n - 1)])
    mse = np.zeros(n)
    for i in range(1, n):
        terms = [sigma2[j] / f[j] ** 2 * (1 / full[i, j] + 1 / col_sum[j]) for j in range(n - 1 - i, n - 1)]
        mse[i] = full[i, n - 1] ** 2 * np.sum(terms)

    # Erreur totale : somme des erreurs par année + termes de covariance (paramètres communs)
    total = mse.sum()
    for i in range(1, n):
        for k in range(i + 1, n):
            terms = [2 * sigma2[j] / f[j] ** 2 / col_sum[j] for j in range(n - 1 - i, n - 1)]
            total += full[i, n - 1] * full[k, n - 1] * np.sum(terms)

    return {**cl, "sigma2": sigma2, "se": np.sqrt(mse), "total_se": float(np.sqrt(total))}


def mack_interval(reserve: float, se: float, level: float = 0.95) -> tuple[float, float]:
    """Intervalle de prédiction log-normal (même moyenne et même écart-type que la réserve de Mack)."""
    if reserve <= 0 or se <= 0:
        return reserve, reserve
    s2 = np.log(1 + (se / reserve) ** 2)
    mu = np.log(reserve) - s2 / 2
    z = norm.ppf(0.5 + level / 2)
    return float(np.exp(mu - z * np.sqrt(s2))), float(np.exp(mu + z * np.sqrt(s2)))


def cape_cod(C: np.ndarray, premium: np.ndarray) -> dict:
    """Bornhuetter-Ferguson avec un ratio sinistres / primes a priori estimé par la méthode Cape Cod.

    ELR = Σ paiements connus / Σ (prime × part développée) ; réserve_i = ELR × prime_i × (1 - 1 / CDF_i).
    """
    cl = chain_ladder(C)
    used_up = premium / cl["cdf"]
    elr = cl["latest"].sum() / used_up.sum()
    reserve = elr * premium * (1 - 1 / cl["cdf"])
    return {"elr": float(elr), "reserve": reserve, "ultimate": cl["latest"] + reserve}

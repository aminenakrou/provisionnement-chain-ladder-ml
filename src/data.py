"""CAS Loss Reserving Database (Meyers & Shi), données publiques de la Casualty Actuarial Society.

Pour chaque branche et chaque assureur américain (données réglementaires « Schedule P ») :
10 années de survenance (1988-1997) × 10 années de développement, soit le carré complet.
La partie supérieure du carré (années calendaires ≤ 1997) est ce que l'actuaire connaissait
fin 1997 ; la partie inférieure s'est réalisée ensuite et sert à vérifier les réserves.

Source : https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
URL = "https://www.casact.org/sites/default/files/2021-04/{}_pos.csv"
LINES = {
    "ppauto": "Auto particuliers",
    "wkcomp": "Accidents du travail",
    "comauto": "Auto commerciale",
    "medmal": "RC médicale",
    "prodliab": "RC produits",
    "othliab": "RC générale",
}
N = 10  # années de survenance et de développement


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for line in LINES:
        path = RAW / f"{line}_pos.csv"
        if not path.exists():
            urlretrieve(URL.format(line), path)


def load_long() -> pd.DataFrame:
    """Toutes les branches au format long, colonnes harmonisées (montants en milliers de dollars)."""
    download()
    frames = []
    for line in LINES:
        d = pd.read_csv(RAW / f"{line}_pos.csv")
        # Les colonnes portent un suffixe propre à chaque branche (ex. CumPaidLoss_B) : on le retire
        d.columns = [c.rsplit("_", 1)[0] if c.count("_") >= 1 and c.split("_")[0] in {
            "IncurLoss", "CumPaidLoss", "BulkLoss", "EarnedPremDIR", "EarnedPremCeded", "EarnedPremNet", "PostedReserve97"
        } else c for c in d.columns]
        d["line"] = line
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "GRCODE": "company", "AccidentYear": "ay", "DevelopmentLag": "lag",
        "CumPaidLoss": "paid", "IncurLoss": "incurred", "BulkLoss": "bulk", "EarnedPremNet": "premium",
    })
    # Charge « dossier par dossier » = sinistres déclarés hors provision IBNR forfaitaire
    df["case_incurred"] = df["incurred"] - df["bulk"]
    return df[["line", "company", "ay", "lag", "paid", "case_incurred", "premium"]]


def build_triangles(df: pd.DataFrame) -> dict:
    """Carrés 10 × 10 par (branche, assureur), après contrôle de qualité.

    On garde un triangle si : primes toutes positives, paiements cumulés tous positifs,
    et paiements cumulés finaux au moins égaux aux premiers (données cohérentes).
    """
    out = {}
    for (line, company), g in df.groupby(["line", "company"]):
        if len(g) != N * N:
            continue
        g = g.sort_values(["ay", "lag"])
        paid = g["paid"].to_numpy(float).reshape(N, N)
        case = g["case_incurred"].to_numpy(float).reshape(N, N)
        prem = g.groupby("ay")["premium"].first().to_numpy(float)
        if (prem <= 0).any() or (paid <= 0).any() or (paid[:, -1] < paid[:, 0]).any():
            continue
        out[(line, int(company))] = {"paid": paid, "case": case, "premium": prem}
    return out


def upper_mask(n: int = N) -> np.ndarray:
    """Cellules connues fin 1997 : année de survenance i + retard j ≤ n - 1 (indices à partir de 0)."""
    i, j = np.indices((n, n))
    return i + j <= n - 1


def latest_lag(i: int, n: int = N) -> int:
    """Dernier retard observé pour l'année de survenance i (indice 0 = 1988)."""
    return n - 1 - i

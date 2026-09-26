"""
Data loader.
Priority: custom path → data/creditcard.csv → synthetic generation.
No Kaggle account needed — synthetic data is generated automatically.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

_REAL = Path("data/creditcard.csv")


def _generate(n: int = 50_000, fraud_rate: float = 0.00172, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic credit-card fraud data mirroring the Kaggle ULB dataset structure.
    V1–V28 are PCA-compressed features (anonymized in the real dataset).
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(50, int(n * fraud_rate))
    n_legit = n - n_fraud

    def _block(size: int, fraud: bool) -> dict:
        d: dict[str, object] = {}
        d["Time"]   = rng.uniform(0, 172_800, size)
        mean_amt    = 4.3 if fraud else 3.1
        d["Amount"] = rng.lognormal(mean_amt, 1.8 if fraud else 1.5, size).clip(0.01, 8_000)

        for i in range(1, 29):
            if fraud and i in (1, 2, 3, 4, 14, 17):
                d[f"V{i}"] = rng.normal(-2.8, 1.6, size)
            elif fraud and i in (10, 11, 12):
                d[f"V{i}"] = rng.normal(2.2, 1.3, size)
            else:
                d[f"V{i}"] = rng.normal(0.0, 0.9 if not fraud else 1.4, size)

        d["Class"] = int(fraud)
        return d

    df = pd.concat(
        [pd.DataFrame(_block(n_legit, False)), pd.DataFrame(_block(n_fraud, True))],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def load_data(path: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns V1-V28, Amount, Time, Class."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(_REAL)

    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if required.issubset(df.columns):
                return df

    return _generate()

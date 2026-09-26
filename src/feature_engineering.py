"""
Feature engineering — adds 4 derived features on top of raw V1-V28.
FEATURE_COLS is the canonical list used for both training and inference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ── Canonical feature list ─────────────────────────────────────────────────
FEATURE_COLS: list[str] = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount_log", "Amount_zscore", "Hour", "Is_night"]
)

_AMT_MEAN = 88.35
_AMT_STD  = 250.12


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features. Input must contain: Amount, Time, V1–V28.
    Returns a new DataFrame preserving all original columns.
    """
    out = df.copy()
    out["Amount_log"]    = np.log1p(out["Amount"])
    out["Amount_zscore"] = (out["Amount"] - _AMT_MEAN) / (_AMT_STD + 1e-9)
    out["Hour"]          = (out["Time"] % 86_400 // 3_600).astype(int)
    out["Is_night"]      = out["Hour"].apply(lambda h: 1 if (h <= 6 or h >= 23) else 0)
    return out

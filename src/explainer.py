"""
SHAP explainability using TreeExplainer on the XGBoost model.
Returns a matplotlib waterfall figure + top-5 feature list.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.feature_engineering import FEATURE_COLS


def compute_shap(
    bundle: dict,
    fe_df: pd.DataFrame,
) -> tuple[plt.Figure, list[tuple[str, float, float]]]:
    """
    Compute SHAP values for a single transaction (first row of fe_df).

    Returns:
        fig   — matplotlib Figure (waterfall plot)
        top5  — list of (feature_name, raw_value, shap_value), sorted by |shap|
    """
    scaler    = bundle["scaler"]
    xgb_model = bundle["xgb_model"]

    X_raw    = fe_df[FEATURE_COLS].values          # shape (1, n_features)
    X_scaled = scaler.transform(X_raw)             # shape (1, n_features)

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)

    # Handle both array and list-of-arrays output from different XGB versions
    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]  # positive class, first row
    else:
        shap_row = shap_values[0]     # first row

    fvals = X_scaled[0]

    # Build Explanation object for waterfall plot
    explanation = shap.Explanation(
        values      = shap_row,
        base_values = float(explainer.expected_value
                            if not isinstance(explainer.expected_value, (list, np.ndarray))
                            else explainer.expected_value[1]),
        data        = fvals,
        feature_names = FEATURE_COLS,
    )

    # Waterfall plot
    try:
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig = plt.gcf()
        plt.tight_layout()
    except Exception:
        # Fallback: horizontal bar chart
        fig, ax = plt.subplots(figsize=(9, 5))
        sorted_idx = np.argsort(np.abs(shap_row))[-12:]
        colors = ["#dc3545" if v > 0 else "#28a745" for v in shap_row[sorted_idx]]
        ax.barh(
            [FEATURE_COLS[i] for i in sorted_idx],
            shap_row[sorted_idx],
            color=colors,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title("Feature attribution (|SHAP| ranked)")
        plt.tight_layout()

    # Top-5 by absolute SHAP magnitude
    top5 = sorted(
        zip(FEATURE_COLS, X_raw[0].tolist(), shap_row.tolist()),
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:5]

    return fig, list(top5)

"""
XGBoost + SMOTE training pipeline.
Returns a bundle dict consumed by app.py, explainer.py, and claude_analyst.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, average_precision_score, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from src.feature_engineering import FEATURE_COLS


def train_pipeline(df: pd.DataFrame) -> dict:
    """
    Train the full pipeline and return a bundle with all artifacts needed by the app.

    Bundle keys:
        pipeline          — fitted ImbPipeline (scaler + smote + xgb)
        scaler            — fitted StandardScaler (for SHAP)
        xgb_model         — fitted XGBClassifier (for SHAP)
        feature_names     — list[str]
        feature_importances — list[float]
        metrics           — dict of evaluation metrics + curve data
        X_test            — DataFrame[FEATURE_COLS] (unscaled, for pipeline.predict_proba)
        X_test_raw        — DataFrame[Amount, Time, V1..V28] (for form pre-filling)
        X_test_amounts    — np.ndarray of Amount values (for bulk display)
        y_test            — Series
        n_train, n_test   — int
        class_counts      — dict {0: n_legit, 1: n_fraud}
    """
    X = df[FEATURE_COLS].copy()
    y = df["Class"]

    # Preserve raw columns for display before splitting
    raw_cols = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
    X_raw = df[raw_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    _, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=42
    )

    X_train  = X_train.reset_index(drop=True)
    X_test   = X_test.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_train  = y_train.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)

    # ── Pipeline: Scaler → SMOTE (train-only) → XGBoost ──────────────────
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=42, k_neighbors=3)),
        ("model",  XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            tree_method="hist", device="cpu",
        )),
    ])
    pipeline.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "auc_roc":       float(roc_auc_score(y_test, y_prob)),
        "precision":     float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":        float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":            float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy":      float(accuracy_score(y_test, y_pred)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "fpr":           fpr.tolist(),
        "tpr":           tpr.tolist(),
        "pr_precision":  pr_prec.tolist(),
        "pr_recall":     pr_rec.tolist(),
        "confusion_matrix": cm,
    }

    return {
        "pipeline":           pipeline,
        "scaler":             pipeline.named_steps["scaler"],
        "xgb_model":          pipeline.named_steps["model"],
        "feature_names":      FEATURE_COLS,
        "feature_importances": pipeline.named_steps["model"].feature_importances_.tolist(),
        "metrics":            metrics,
        "X_test":             X_test,
        "X_test_raw":         X_test_raw,
        "X_test_amounts":     X_test_raw["Amount"].values,
        "y_test":             y_test,
        "n_train":            len(X_train),
        "n_test":             len(X_test),
        "class_counts":       y.value_counts().to_dict(),
    }


def predict_single(bundle: dict, fe_df: pd.DataFrame) -> tuple[float, float]:
    """Return (risk_score 0–100, fraud_probability 0–1) for one transaction."""
    X     = fe_df[FEATURE_COLS]
    proba = float(bundle["pipeline"].predict_proba(X)[0, 1])
    return round(proba * 100, 1), proba


def predict_batch(bundle: dict, X: pd.DataFrame) -> list[tuple[float, float]]:
    """Return [(risk_score, probability), ...] for all rows."""
    probas = bundle["pipeline"].predict_proba(X[FEATURE_COLS])[:, 1]
    return [(round(float(p) * 100, 1), float(p)) for p in probas]


def get_verdict(risk: float) -> tuple[str, str]:
    """Map risk score to (human label, CSS class suffix)."""
    if risk >= 86: return "CONFIRMED FRAUD", "fraud"
    if risk >= 61: return "HIGH RISK",        "high"
    if risk >= 31: return "NEEDS REVIEW",     "review"
    return "LEGITIMATE", "legit"

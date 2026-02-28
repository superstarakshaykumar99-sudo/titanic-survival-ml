"""
evaluation.py – compute metrics and generate reports.
"""

from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import REPORTS_DIR, get_logger

log = get_logger("evaluation")


def evaluate_model(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str = "best_model",
) -> dict:
    """
    Compute classification metrics and save them to reports/model_metrics.json.
    Returns a metrics dict.
    """
    y_pred = model.predict(X_test)

    # ROC-AUC – requires predict_proba
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = None

    metrics = {
        "model": model_name,
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":   round(roc_auc, 4) if roc_auc is not None else None,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    out_path = REPORTS_DIR / "model_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", out_path)
    log.info(
        "Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  AUC=%s",
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1_score"], metrics["roc_auc"],
    )
    return metrics


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """
    Generate and save a feature importance bar chart.
    Works for tree-based models; silently skips others.
    """
    importance_attr = getattr(model, "feature_importances_", None)
    if importance_attr is None:
        log.warning("Model has no feature_importances_; skipping importance plot.")
        return

    importances = pd.Series(importance_attr, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))  # type: ignore[attr-defined]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Feature importance plot saved to %s", out_path)

"""
model_training.py – train, compare, and persist the best classifier.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils import MODELS_DIR, get_logger

log = get_logger("model_training")

RANDOM_STATE = 42


def _build_candidates() -> dict:
    """Return a dict of {name: unfitted_estimator}."""
    candidates: dict = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
    }
    # XGBoost – optional dependency
    try:
        from xgboost import XGBClassifier  # type: ignore[import]
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        )
    except ImportError:
        log.warning("xgboost not installed – skipping XGBoost.")
    return candidates


def train_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    cv: int = 5,
) -> dict[str, dict]:
    """
    Fit every candidate on the full training set; evaluate with k-fold CV.
    Returns a results dict: {name: {model, cv_scores, mean_cv, std_cv}}.
    """
    candidates = _build_candidates()
    results: dict[str, dict] = {}

    for name, clf in candidates.items():
        log.info("Training %s …", name)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        clf.fit(X_train, y_train)
        results[name] = {
            "model": clf,
            "cv_scores": scores.tolist(),
            "mean_cv": float(scores.mean()),
            "std_cv": float(scores.std()),
        }
        log.info(
            "  %s – CV accuracy: %.4f ± %.4f", name, scores.mean(), scores.std()
        )

    return results


def select_best_model(results: dict[str, dict]) -> tuple[str, object]:
    """Return (best_name, best_estimator) by highest mean CV accuracy."""
    best_name = max(results, key=lambda n: results[n]["mean_cv"])
    log.info("Best model: %s (mean CV = %.4f)", best_name, results[best_name]["mean_cv"])
    return best_name, results[best_name]["model"]


def save_artifacts(model: object, scaler: StandardScaler) -> None:
    """Persist model and scaler to models/."""
    model_path  = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    log.info("Saved best_model.pkl and scaler.pkl to %s", MODELS_DIR)

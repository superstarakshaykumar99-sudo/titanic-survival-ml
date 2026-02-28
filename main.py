"""
main.py – end-to-end Titanic survival prediction pipeline.

Run:
    python main.py
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader       import load_raw_data
from src.feature_engineering import engineer_features
from src.preprocessing     import preprocess
from src.model_training    import train_models, select_best_model, save_artifacts
from src.evaluation        import evaluate_model, plot_feature_importance
from src.utils             import PROCESSED_DIR, get_logger

log = get_logger("main")

# ── Target and feature columns ────────────────────────────────────────────────
TARGET = "Survived"

# Columns to keep for modelling (after engineering + preprocessing)
FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone", "FarePerPerson",
]


def run_pipeline() -> None:
    log.info("=" * 60)
    log.info("  Titanic Survival ML – Pipeline Start")
    log.info("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    train_df, test_df = load_raw_data()

    # ── 2. Feature engineering (before Name is dropped) ───────────────────────
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)

    # ── 4. Persist processed data ─────────────────────────────────────────────
    processed_path = PROCESSED_DIR / "processed_data.csv"
    train_df.to_csv(processed_path, index=False)
    log.info("Processed dataset saved to %s", processed_path)

    # ── 5. Prepare X / y  ─────────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    log.info("Using features: %s", available_features)

    # Use pre-split test set if Survived is available; otherwise use train split
    if TARGET in test_df.columns:
        X_train = train_df[available_features]
        y_train = train_df[TARGET]
        X_test  = test_df[available_features]
        y_test  = test_df[TARGET]
    else:
        X = train_df[available_features]
        y = train_df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # ── 6. Scale ──────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=available_features)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=available_features)

    # ── 7. Train ──────────────────────────────────────────────────────────────
    results   = train_models(X_train, y_train)
    best_name, best_model = select_best_model(results)

    # ── 8. Save model + scaler ────────────────────────────────────────────────
    save_artifacts(best_model, scaler)

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate_model(best_model, X_test, y_test, model_name=best_name)
    plot_feature_importance(best_model, available_features)

    log.info("=" * 60)
    log.info("  Pipeline Complete!")
    log.info("  Best model  : %s", best_name)
    log.info("  Accuracy    : %.4f", metrics["accuracy"])
    log.info("  F1 Score    : %.4f", metrics["f1_score"])
    log.info("  ROC-AUC     : %s",  metrics["roc_auc"])
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()

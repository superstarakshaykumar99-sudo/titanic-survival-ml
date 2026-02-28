"""
preprocessing.py – clean and encode raw Titanic data.
"""

import pandas as pd
from src.utils import get_logger

log = get_logger("preprocessing")

# Columns to drop before modelling
COLS_TO_DROP = ["Cabin", "Ticket", "Name", "Boat", "Body", "HomeDest",
                "home.dest", "boat", "body"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a Titanic DataFrame:
      - Drop uninformative / high-cardinality columns
      - Impute missing values
      - Encode categoricals as integers

    Returns a new DataFrame (does not mutate input).
    """
    df = df.copy()

    # ── Drop unused columns ──────────────────────────────────────────────────
    drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    log.info("Dropped columns: %s", drop_cols)

    # ── Impute missing values ────────────────────────────────────────────────
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # ── Encode categoricals ──────────────────────────────────────────────────
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    if "Embarked" in df.columns:
        embarked_map = {"S": 0, "C": 1, "Q": 2}
        df["Embarked"] = df["Embarked"].map(embarked_map).fillna(0).astype(int)

    log.info("Preprocessing complete. Shape: %s", df.shape)
    return df

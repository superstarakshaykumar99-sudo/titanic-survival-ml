"""
feature_engineering.py – create domain-driven features for Titanic survival.
"""

import re
import pandas as pd
from src.utils import get_logger

log = get_logger("feature_engineering")

# Mapping from raw title string → grouped title
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Capt": "Rare",
    "Sir": "Rare", "Mme": "Mrs",
}

TITLE_ENCODE = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


def _extract_title(name_series: pd.Series) -> pd.Series:
    """Extract title from Name column, group rare ones, then label-encode."""
    titles = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    titles = titles.map(TITLE_MAP).fillna("Rare")
    return titles.map(TITLE_ENCODE).fillna(5).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features:
      - Title          : encoded passenger title extracted from Name
      - FamilySize     : SibSp + Parch + 1
      - IsAlone        : 1 if FamilySize == 1
      - FarePerPerson  : Fare / FamilySize
    """
    df = df.copy()

    # Title (must be extracted before Name is dropped in preprocessing)
    if "Name" in df.columns:
        df["Title"] = _extract_title(df["Name"])
        log.info("Extracted Title feature.")

    # Family features
    df["FamilySize"] = df.get("SibSp", 0) + df.get("Parch", 0) + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Fare per person (safe division)
    fare_col = df.get("Fare", pd.Series(0, index=df.index))
    df["FarePerPerson"] = (fare_col / df["FamilySize"]).round(4)

    log.info("Feature engineering complete. New columns: Title, FamilySize, IsAlone, FarePerPerson.")
    return df

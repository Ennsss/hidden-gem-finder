"""Data cleaning: position normalization, filtering, and outlier capping."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from src.storage import PlayerDatabase

logger = logging.getLogger(__name__)

# Position mapping from Transfermarkt strings to groups
TM_POSITION_MAP = {
    # Forwards
    "Centre-Forward": "FW",
    "Left Winger": "FW",
    "Right Winger": "FW",
    "Second Striker": "FW",
    "Striker": "FW",
    # Midfielders
    "Central Midfield": "MF",
    "Attacking Midfield": "MF",
    "Defensive Midfield": "MF",
    "Left Midfield": "MF",
    "Right Midfield": "MF",
    # Defenders
    "Centre-Back": "DF",
    "Left-Back": "DF",
    "Right-Back": "DF",
    # Goalkeeper
    "Goalkeeper": "GK",
}

# FBref position abbreviation mapping
FBREF_POSITION_MAP = {
    "FW": "FW",
    "MF": "MF",
    "DF": "DF",
    "GK": "GK",
}


@dataclass
class CleaningResult:
    """Result of the cleaning step."""

    total_raw: int = 0
    after_position_norm: int = 0
    gk_removed: int = 0
    after_age_filter: int = 0
    after_minutes_filter: int = 0
    outliers_capped: int = 0
    final_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Cleaning: {self.total_raw} raw → {self.final_count} final\n"
            f"  GK removed: {self.gk_removed}\n"
            f"  Age/minutes filtered: "
            f"{self.total_raw - self.gk_removed - self.final_count}\n"
            f"  Outliers capped: {self.outliers_capped}"
        )


def normalize_position(position_str: str | None, source: str = "fbref") -> str | None:
    """Map position string to FW/MF/DF/GK group.

    Args:
        position_str: Raw position string from source
        source: Data source ("fbref", "transfermarkt", "understat")

    Returns:
        Normalized position group or None if unmappable
    """
    if not position_str or pd.isna(position_str):
        return None

    position_str = str(position_str).strip()

    if source == "transfermarkt":
        return TM_POSITION_MAP.get(position_str)

    if source == "understat":
        # Understat already uses FW/MF/DF/GK
        if position_str in ("FW", "MF", "DF", "GK"):
            return position_str
        return None

    # FBref: can be comma-separated like "FW,MF" — take first
    if source == "fbref":
        first_pos = position_str.split(",")[0].strip()
        return FBREF_POSITION_MAP.get(first_pos)

    return None


def apply_filters(
    df: pd.DataFrame,
    min_minutes: int = 450,
    min_age: int = 17,
    max_age: int = 26,
) -> pd.DataFrame:
    """Filter by position (drop GK), age range, and minimum minutes.

    Args:
        df: DataFrame with position_group, age, minutes columns
        min_minutes: Minimum minutes played
        min_age: Minimum player age
        max_age: Maximum player age

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    # Drop goalkeepers
    filtered = filtered[filtered["position_group"] != "GK"]

    # Age filter
    filtered = filtered[
        (filtered["age"] >= min_age) & (filtered["age"] <= max_age)
    ]

    # Minutes filter
    filtered = filtered[filtered["minutes"] >= min_minutes]

    return filtered.reset_index(drop=True)


def cap_outliers(df: pd.DataFrame, percentile: int = 99) -> tuple[pd.DataFrame, int]:
    """Cap numeric columns at given percentile per position group.

    Args:
        df: DataFrame with position_group column
        percentile: Percentile to cap at (default 99)

    Returns:
        Tuple of (capped DataFrame, number of values capped)
    """
    capped = df.copy()
    numeric_cols = capped.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude identity/metadata columns
    exclude_cols = {"age", "birth_year", "games", "games_starts", "minutes",
                    "pens_made", "pens_att", "cards_yellow", "cards_red"}
    cols_to_cap = [c for c in numeric_cols if c not in exclude_cols]

    total_capped = 0
    for group in capped["position_group"].unique():
        mask = capped["position_group"] == group
        for col in cols_to_cap:
            values = capped.loc[mask, col].dropna()
            if len(values) == 0:
                continue
            cap_val = np.percentile(values, percentile)
            over_cap = capped.loc[mask, col] > cap_val
            count = over_cap.sum()
            if count > 0:
                capped.loc[mask & over_cap, col] = cap_val
                total_capped += count

    return capped, total_capped


def clean_fbref_data(
    db: PlayerDatabase,
    min_minutes: int = 450,
    min_age: int = 17,
    max_age: int = 26,
) -> tuple[pd.DataFrame, CleaningResult]:
    """Main entry point: clean FBref data from database.

    Args:
        db: PlayerDatabase instance
        min_minutes: Minimum minutes filter
        min_age: Minimum age filter
        max_age: Maximum age filter

    Returns:
        Tuple of (cleaned DataFrame, CleaningResult)
    """
    result = CleaningResult()

    # Read all FBref data
    df = db.get_fbref_players()
    result.total_raw = len(df)

    if df.empty:
        result.warnings.append("No FBref data found in database")
        return df, result

    # Normalize positions
    df["position_group"] = df["position"].apply(
        lambda x: normalize_position(x, source="fbref")
    )
    result.after_position_norm = df["position_group"].notna().sum()

    # Drop rows with unmapped positions
    df = df[df["position_group"].notna()].copy()

    # Count GKs before removal
    gk_count = (df["position_group"] == "GK").sum()
    result.gk_removed = gk_count

    # Apply filters
    df = apply_filters(df, min_minutes=min_minutes, min_age=min_age, max_age=max_age)
    result.after_age_filter = len(df)
    result.after_minutes_filter = len(df)

    # Cap outliers
    df, capped_count = cap_outliers(df)
    result.outliers_capped = capped_count

    result.final_count = len(df)

    logger.info(str(result))
    return df, result

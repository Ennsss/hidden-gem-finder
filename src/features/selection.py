"""Feature selection: remove highly correlated and low-variance features."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent.parent.parent / "config" / "features.yaml"


def _load_config() -> dict:
    with open(_config_path) as f:
        return yaml.safe_load(f)


@dataclass
class SelectionResult:
    """Result of the feature selection step."""

    input_features: int = 0
    correlated_removed: list[str] = field(default_factory=list)
    low_variance_removed: list[str] = field(default_factory=list)
    final_features: int = 0

    def __str__(self) -> str:
        return (
            f"Feature selection: {self.input_features} → {self.final_features}\n"
            f"  Correlated removed: {len(self.correlated_removed)}\n"
            f"  Low variance removed: {len(self.low_variance_removed)}"
        )


# Columns to never drop (identity, metadata, labels)
PROTECTED_COLUMNS = {
    "player_id", "name", "team", "league", "season", "position",
    "position_group", "nationality", "source", "scraped_at",
    "label", "breakout_league", "breakout_season",
    "age", "birth_year",
    "match_confidence_tm", "match_confidence_us",
}


def remove_correlated(
    df: pd.DataFrame, threshold: float = 0.90
) -> tuple[pd.DataFrame, list[str]]:
    """Remove highly correlated features, keeping the one with higher variance.

    Args:
        df: DataFrame with numeric features
        threshold: Correlation threshold (default 0.90)

    Returns:
        Tuple of (filtered DataFrame, list of removed column names)
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in PROTECTED_COLUMNS
    ]

    if len(numeric_cols) < 2:
        return df, []

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for corr_col in correlated:
            if corr_col in to_drop:
                continue
            # Drop the one with lower variance
            if df[col].var() >= df[corr_col].var():
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    removed = sorted(to_drop)
    result = df.drop(columns=removed)
    return result, removed


def remove_low_variance(
    df: pd.DataFrame, threshold: float = 0.01
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features with variance below threshold.

    Args:
        df: DataFrame with numeric features
        threshold: Minimum variance to keep (default 0.01)

    Returns:
        Tuple of (filtered DataFrame, list of removed column names)
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in PROTECTED_COLUMNS
    ]

    to_drop = []
    for col in numeric_cols:
        if df[col].var() < threshold:
            to_drop.append(col)

    removed = sorted(to_drop)
    result = df.drop(columns=removed)
    return result, removed


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, SelectionResult]:
    """Main entry point: run correlation and variance filtering.

    Args:
        df: Engineered feature DataFrame

    Returns:
        Tuple of (selected DataFrame, SelectionResult)
    """
    config = _load_config()
    corr_threshold = config["selection"]["correlation_threshold"]
    var_threshold = config["selection"]["variance_threshold"]

    result = SelectionResult()

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in PROTECTED_COLUMNS
    ]
    result.input_features = len(numeric_cols)

    # Remove low variance first
    df, low_var_removed = remove_low_variance(df, threshold=var_threshold)
    result.low_variance_removed = low_var_removed

    # Then remove correlated
    df, corr_removed = remove_correlated(df, threshold=corr_threshold)
    result.correlated_removed = corr_removed

    remaining_numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in PROTECTED_COLUMNS
    ]
    result.final_features = len(remaining_numeric)

    logger.info(str(result))
    return df, result

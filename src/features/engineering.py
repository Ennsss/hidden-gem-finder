"""Feature engineering: per-90 conversion, derived features, growth, league adjustments."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_config_dir = Path(__file__).parent.parent.parent / "config"


def _load_features_config() -> dict:
    with open(_config_dir / "features.yaml") as f:
        return yaml.safe_load(f)


def _load_leagues_config() -> dict:
    with open(_config_dir / "leagues.yaml") as f:
        return yaml.safe_load(f)


@dataclass
class FeatureResult:
    """Result of the feature engineering step."""

    input_rows: int = 0
    after_per90: int = 0
    derived_features_added: int = 0
    growth_features_added: int = 0
    league_adjustments_applied: int = 0
    final_features: int = 0
    final_rows: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Feature engineering: {self.input_rows} rows\n"
            f"  After per-90 filter: {self.after_per90}\n"
            f"  Derived features: {self.derived_features_added}\n"
            f"  Growth features: {self.growth_features_added}\n"
            f"  Final: {self.final_rows} rows x {self.final_features} features"
        )


# Columns to convert to per-90
PER90_COLUMNS = [
    "goals", "assists", "xg", "npxg", "xg_assist",
    "shots", "shots_on_target",
    "passes_completed", "passes", "progressive_passes",
    "progressive_carries",
    "tackles", "interceptions", "blocks", "clearances",
    "touches", "take_ons_won", "carries",
    "carries_into_final_third", "carries_into_penalty_area",
    "touches_att_pen_area", "assisted_shots",
]


def convert_to_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counting stats to per-90 minute rates.

    Players with minutes_90s < 0.5 are dropped.

    Args:
        df: DataFrame with raw stats and minutes_90s column

    Returns:
        DataFrame with per-90 columns added
    """
    result = df.copy()

    # Drop players with very few minutes
    result = result[result["minutes_90s"] >= 0.5].copy()

    for col in PER90_COLUMNS:
        if col in result.columns:
            per90_col = f"{col}_per90"
            result[per90_col] = result[col] / result["minutes_90s"]

    return result.reset_index(drop=True)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from per-90 stats.

    Args:
        df: DataFrame with per-90 columns

    Returns:
        DataFrame with derived features added
    """
    result = df.copy()
    config = _load_leagues_config()

    # Build league difficulty coefficient lookup
    league_coeff = {}
    for lg in config.get("source_leagues", []):
        slug = lg["name"].lower().replace(" ", "-")
        league_coeff[slug] = lg["difficulty_coefficient"]
        league_coeff[lg["name"]] = lg["difficulty_coefficient"]
    for lg in config.get("target_leagues", []):
        slug = lg["name"].lower().replace(" ", "-")
        league_coeff[slug] = lg.get("coefficient", 1.0)
        league_coeff[lg["name"]] = lg.get("coefficient", 1.0)

    # xG overperformance
    if "goals_per90" in result.columns and "xg_per90" in result.columns:
        result["xg_overperformance"] = result["goals_per90"] - result["xg_per90"]

    # xA overperformance
    if "assists_per90" in result.columns and "xg_assist_per90" in result.columns:
        result["xa_overperformance"] = result["assists_per90"] - result["xg_assist_per90"]

    # Goal contribution
    if "goals_per90" in result.columns and "assists_per90" in result.columns:
        result["goal_contribution_per90"] = result["goals_per90"] + result["assists_per90"]

    # npxg + xa
    if "npxg_per90" in result.columns and "xg_assist_per90" in result.columns:
        result["npxg_xa_per90"] = result["npxg_per90"] + result["xg_assist_per90"]

    # League-adjusted xG
    if "xg_per90" in result.columns:
        result["league_adjusted_xg"] = result.apply(
            lambda row: row["xg_per90"] * league_coeff.get(row["league"], 1.0),
            axis=1,
        )

    # Involvement score
    if "touches_per90" in result.columns and "passes_per90" in result.columns:
        result["involvement_score"] = (result["touches_per90"] + result["passes_per90"]) / 100

    # Age potential factor
    if "age" in result.columns:
        result["age_potential_factor"] = (27 - result["age"]) / 10

    # Minutes share (34 games * 90 min = 3060)
    if "minutes" in result.columns:
        result["minutes_share"] = result["minutes"] / 3060

    return result


def create_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create year-over-year growth features per player.

    For each player (by player_id), computes YoY change in key metrics.
    First season for each player gets NaN.

    Args:
        df: DataFrame with player_id, season, and per-90 features

    Returns:
        DataFrame with growth features added
    """
    config = _load_features_config()
    growth_metrics = config["growth_features"]["metrics_to_track"]
    epsilon = config["growth_features"]["epsilon"]

    result = df.copy()

    # Map metric names to actual column names
    col_map = {
        "xg_per90": "xg_per90",
        "xa_per90": "xg_assist_per90",  # FBref calls it xg_assist
        "progressive_passes_per90": "progressive_passes_per90",
        "progressive_carries_per90": "progressive_carries_per90",
        "minutes_played": "minutes",
    }

    # Sort by player and season for proper YoY
    result = result.sort_values(["player_id", "season"]).reset_index(drop=True)

    for metric in growth_metrics:
        actual_col = col_map.get(metric, metric)
        growth_col = f"{metric}_growth"

        if actual_col not in result.columns:
            result[growth_col] = np.nan
            continue

        # Compute growth per player
        growth_values = []
        for pid, group in result.groupby("player_id"):
            if len(group) < 2:
                growth_values.extend([np.nan] * len(group))
                continue

            sorted_group = group.sort_values("season")
            vals = sorted_group[actual_col].values

            # First season is NaN
            player_growth = [np.nan]
            for i in range(1, len(vals)):
                prev = vals[i - 1]
                curr = vals[i]
                if pd.notna(prev) and pd.notna(curr):
                    player_growth.append((curr - prev) / (abs(prev) + epsilon))
                else:
                    player_growth.append(np.nan)

            growth_values.extend(player_growth)

        result[growth_col] = growth_values

    return result


def apply_league_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Multiply attacking/passing per-90 stats by league difficulty coefficient.

    Args:
        df: DataFrame with league column and per-90 features

    Returns:
        DataFrame with league-adjusted stats
    """
    config = _load_leagues_config()

    # Build coefficient lookup
    league_coeff = {}
    for lg in config.get("source_leagues", []):
        slug = lg["name"].lower().replace(" ", "-")
        league_coeff[slug] = lg["difficulty_coefficient"]
        league_coeff[lg["name"]] = lg["difficulty_coefficient"]
    for lg in config.get("target_leagues", []):
        slug = lg["name"].lower().replace(" ", "-")
        league_coeff[slug] = lg.get("coefficient", 1.0)
        league_coeff[lg["name"]] = lg.get("coefficient", 1.0)

    result = df.copy()

    # Columns to adjust
    adjust_cols = [
        "goals_per90", "assists_per90", "xg_per90", "npxg_per90",
        "xg_assist_per90", "shots_per90", "shots_on_target_per90",
        "passes_completed_per90", "passes_per90", "progressive_passes_per90",
        "assisted_shots_per90", "carries_into_final_third_per90",
        "carries_into_penalty_area_per90", "touches_att_pen_area_per90",
    ]

    for col in adjust_cols:
        if col in result.columns:
            adj_col = f"{col}_league_adj"
            result[adj_col] = result.apply(
                lambda row: row[col] * league_coeff.get(row["league"], 1.0),
                axis=1,
            )

    return result


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureResult]:
    """Main entry point: run all feature engineering steps.

    Args:
        df: Labeled DataFrame from labeling step

    Returns:
        Tuple of (feature DataFrame, FeatureResult)
    """
    result = FeatureResult(input_rows=len(df))

    # Step 1: per-90 conversion
    df = convert_to_per90(df)
    result.after_per90 = len(df)

    # Step 2: derived features
    cols_before = len(df.columns)
    df = create_derived_features(df)
    result.derived_features_added = len(df.columns) - cols_before

    # Step 3: growth features
    cols_before = len(df.columns)
    df = create_growth_features(df)
    result.growth_features_added = len(df.columns) - cols_before

    # Step 4: league adjustments
    cols_before = len(df.columns)
    df = apply_league_adjustments(df)
    result.league_adjustments_applied = len(df.columns) - cols_before

    result.final_rows = len(df)
    result.final_features = len(df.columns)

    logger.info(str(result))
    return df, result

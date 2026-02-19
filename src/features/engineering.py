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
    multi_season_features_added: int = 0
    interaction_features_added: int = 0
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
            f"  Multi-season features: {self.multi_season_features_added}\n"
            f"  Interaction features: {self.interaction_features_added}\n"
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


def _build_league_coeff() -> dict:
    """Build league difficulty coefficient lookup from config."""
    config = _load_leagues_config()
    league_coeff = {}
    for lg in config.get("source_leagues", []):
        slug = lg["name"].lower().replace(" ", "-")
        league_coeff[slug] = lg["difficulty_coefficient"]
        league_coeff[lg["name"]] = lg["difficulty_coefficient"]
    for lg in config.get("target_leagues", []):
        slug = lg["name"].lower().replace(" ", "-")
        league_coeff[slug] = lg.get("coefficient", 1.0)
        league_coeff[lg["name"]] = lg.get("coefficient", 1.0)
    return league_coeff


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from per-90 stats.

    Args:
        df: DataFrame with per-90 columns

    Returns:
        DataFrame with derived features added
    """
    result = df.copy()
    league_coeff = _build_league_coeff()

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

    # Understat xG features (already per-90 from source)
    if "us_xg_per90" in result.columns and "goals_per90" in result.columns:
        result["us_xg_overperformance"] = result["goals_per90"] - result["us_xg_per90"]
    if "us_npxg_per90" in result.columns and "us_xa_per90" in result.columns:
        result["us_npxg_xa_per90"] = result["us_npxg_per90"] + result["us_xa_per90"]

    # --- Position one-hot encoding ---
    if "position_group" in result.columns:
        result["is_forward"] = (result["position_group"] == "FW").astype(int)
        result["is_midfielder"] = (result["position_group"] == "MF").astype(int)
        result["is_defender"] = (result["position_group"] == "DF").astype(int)

    # --- Tier 1: League-position baseline features ---
    result = _add_baseline_features(result, league_coeff)

    # --- Position-weighted interactions (after baseline features exist) ---
    if "is_forward" in result.columns and "goals_above_avg" in result.columns:
        result["fw_goals_above_avg"] = result["is_forward"] * result["goals_above_avg"]
    if "is_midfielder" in result.columns and "assists_per90" in result.columns:
        result["mf_assists_above_avg"] = result["is_midfielder"] * result["assists_per90"]
    if "is_defender" in result.columns and "defensive_actions_per90" in result.columns:
        result["df_defensive_actions"] = result["is_defender"] * result["defensive_actions_per90"]

    return result


def _add_baseline_features(df: pd.DataFrame, league_coeff: dict) -> pd.DataFrame:
    """Add league-position baseline features comparing players to peer averages.

    Computes per (league, position_group, season) averages and creates
    features measuring how each player deviates from their peer group.

    Args:
        df: DataFrame with per-90 stats and position_group/league/season columns
        league_coeff: League difficulty coefficient lookup

    Returns:
        DataFrame with baseline features added
    """
    result = df.copy()
    group_cols = ["league", "position_group", "season"]

    # Check we have the grouping columns
    if not all(c in result.columns for c in group_cols):
        logger.warning("Missing grouping columns for baseline features, skipping")
        return result

    # Non-penalty goals per 90 (independent of baselines)
    if all(c in result.columns for c in ["goals", "pens_made", "minutes_90s"]):
        result["non_penalty_goals_per90"] = (
            (result["goals"] - result["pens_made"].fillna(0)) / result["minutes_90s"]
        )
    elif "goals_per90" in result.columns:
        # Fallback: just use goals_per90 if pens_made not available
        result["non_penalty_goals_per90"] = result["goals_per90"]

    # Defensive actions per 90
    if all(c in result.columns for c in ["tackles_won", "interceptions_per90"]):
        result["defensive_actions_per90"] = (
            result["tackles_won"] / result["minutes_90s"] + result["interceptions_per90"]
        )
    elif all(c in result.columns for c in ["tackles_per90", "interceptions_per90"]):
        result["defensive_actions_per90"] = (
            result["tackles_per90"] + result["interceptions_per90"]
        )

    # Compute league-position-season averages for baseline comparisons
    avg_cols = {}
    if "goals_per90" in result.columns:
        avg_cols["goals_per90"] = "avg_goals_per90"
    if "assists_per90" in result.columns:
        avg_cols["assists_per90"] = "avg_assists_per90"
    if "shots_on_target_per90" in result.columns:
        avg_cols["shots_on_target_per90"] = "avg_sot_per90"
    if "goals_per90" in result.columns and "shots_per90" in result.columns:
        # We'll compute goals_per_shot average for naive xG
        avg_cols["goals_per90"] = "avg_goals_per90"  # already there

    if not avg_cols:
        return result

    # Compute group averages
    group_avgs = result.groupby(group_cols)[list(avg_cols.keys())].transform("mean")

    # Naive xG: shot volume * league-position average conversion rate
    if all(c in result.columns for c in ["shots_per90", "goals_per90"]):
        # Compute goals_per_shot at group level
        eps = 1e-6
        group_gps = result.groupby(group_cols).apply(
            lambda g: g["goals_per90"].mean() / (g["shots_per90"].mean() + eps),
            include_groups=False,
        )
        group_gps.name = "avg_goals_per_shot"
        result = result.merge(
            group_gps.reset_index(),
            on=group_cols,
            how="left",
        )
        result["naive_xg_per90"] = result["shots_per90"] * result["avg_goals_per_shot"]
        result.drop(columns=["avg_goals_per_shot"], inplace=True)

        # Finishing skill: actual goals - naive xG
        result["finishing_skill"] = result["goals_per90"] - result["naive_xg_per90"]

    # Goals above average
    if "goals_per90" in result.columns:
        result["goals_above_avg"] = result["goals_per90"] - group_avgs["goals_per90"]

    # Assists above average
    if "assists_per90" in result.columns:
        result["assists_above_avg"] = result["assists_per90"] - group_avgs["assists_per90"]

    # Shot volume above average
    if "shots_on_target_per90" in result.columns:
        result["shot_volume_above_avg"] = (
            result["shots_on_target_per90"] - group_avgs["shots_on_target_per90"]
        )

    return result


def create_multi_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create career-level features for players with multiple seasons.

    For each player-season row, computes stats using only data from
    seasons <= current season (no future leakage). Single-season
    players get NaN for all career features.

    New features:
        career_seasons: number of seasons observed up to current
        career_avg_goals_per90: mean goals_per90 across prior seasons
        career_avg_assists_per90: mean assists_per90 across prior seasons
        career_max_goals_per90: peak goals_per90 in any prior season
        career_avg_minutes: mean minutes across prior seasons
        goals_trend: linear slope of goals_per90 over seasons
        minutes_trend: linear slope of minutes over seasons

    Args:
        df: DataFrame with player_id, season, and per-90 features

    Returns:
        DataFrame with multi-season features added
    """
    result = df.copy()

    # Initialize new columns with NaN
    new_cols = [
        "career_seasons", "career_avg_goals_per90", "career_avg_assists_per90",
        "career_max_goals_per90", "career_avg_minutes",
        "goals_trend", "minutes_trend",
    ]
    for col in new_cols:
        result[col] = np.nan

    if "player_id" not in result.columns or "season" not in result.columns:
        return result

    # Sort by player and season
    result = result.sort_values(["player_id", "season"]).reset_index(drop=True)

    has_goals = "goals_per90" in result.columns
    has_assists = "assists_per90" in result.columns
    has_minutes = "minutes" in result.columns

    for pid, group in result.groupby("player_id"):
        if len(group) < 2:
            # Single-season: only set career_seasons = 1
            result.loc[group.index, "career_seasons"] = 1
            continue

        sorted_idx = group.sort_values("season").index

        # Expanding window stats (only use data up to current row)
        for i, idx in enumerate(sorted_idx):
            window = sorted_idx[: i + 1]
            n_seasons = i + 1
            result.loc[idx, "career_seasons"] = n_seasons

            if has_goals:
                vals = result.loc[window, "goals_per90"]
                result.loc[idx, "career_avg_goals_per90"] = vals.mean()
                result.loc[idx, "career_max_goals_per90"] = vals.max()

            if has_assists:
                vals = result.loc[window, "assists_per90"]
                result.loc[idx, "career_avg_assists_per90"] = vals.mean()

            if has_minutes:
                vals = result.loc[window, "minutes"]
                result.loc[idx, "career_avg_minutes"] = vals.mean()

            # Trend slopes need 2+ data points
            if n_seasons >= 2:
                x = np.arange(n_seasons, dtype=float)
                if has_goals:
                    y = result.loc[window, "goals_per90"].values.astype(float)
                    if not np.any(np.isnan(y)):
                        result.loc[idx, "goals_trend"] = np.polyfit(x, y, 1)[0]
                if has_minutes:
                    y = result.loc[window, "minutes"].values.astype(float)
                    if not np.any(np.isnan(y)):
                        result.loc[idx, "minutes_trend"] = np.polyfit(x, y, 1)[0]

    logger.info(
        f"Multi-season features: {(result['career_seasons'] > 1).sum()} "
        f"rows have 2+ seasons"
    )
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


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features by multiplying informative feature pairs.

    Each interaction is only created if both component columns exist.

    New features:
        age_x_minutes_growth: age_potential_factor * minutes_played_growth
        goals_above_avg_x_league_coeff: goals_above_avg * league_difficulty_coefficient
        finishing_x_volume: finishing_skill * shots_per90
        proxy_xg_x_age: proxy_xg_per90 * age_potential_factor
        defensive_x_progressive: defensive_actions_per90 * progressive_carries_per90

    Args:
        df: DataFrame with derived + growth features

    Returns:
        DataFrame with interaction features added
    """
    result = df.copy()
    league_coeff = _build_league_coeff()

    # age_potential_factor * minutes_played_growth
    if all(c in result.columns for c in ["age_potential_factor", "minutes_played_growth"]):
        result["age_x_minutes_growth"] = (
            result["age_potential_factor"] * result["minutes_played_growth"]
        )

    # goals_above_avg * league difficulty coefficient
    if "goals_above_avg" in result.columns and "league" in result.columns:
        result["goals_above_avg_x_league_coeff"] = result.apply(
            lambda row: row["goals_above_avg"] * league_coeff.get(row["league"], 1.0),
            axis=1,
        )

    # finishing_skill * shots_per90
    if all(c in result.columns for c in ["finishing_skill", "shots_per90"]):
        result["finishing_x_volume"] = result["finishing_skill"] * result["shots_per90"]

    # proxy_xg_per90 * age_potential_factor
    if all(c in result.columns for c in ["proxy_xg_per90", "age_potential_factor"]):
        result["proxy_xg_x_age"] = (
            result["proxy_xg_per90"] * result["age_potential_factor"]
        )

    # defensive_actions_per90 * progressive_carries_per90
    if all(c in result.columns for c in ["defensive_actions_per90", "progressive_carries_per90"]):
        result["defensive_x_progressive"] = (
            result["defensive_actions_per90"] * result["progressive_carries_per90"]
        )

    n_added = len(result.columns) - len(df.columns)
    logger.info(f"Interaction features: {n_added} added")
    return result


def apply_league_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Multiply attacking/passing per-90 stats by league difficulty coefficient.

    Args:
        df: DataFrame with league column and per-90 features

    Returns:
        DataFrame with league-adjusted stats
    """
    league_coeff = _build_league_coeff()

    result = df.copy()

    # Columns to adjust
    adjust_cols = [
        "goals_per90", "assists_per90", "xg_per90", "npxg_per90",
        "xg_assist_per90", "shots_per90", "shots_on_target_per90",
        "passes_completed_per90", "passes_per90", "progressive_passes_per90",
        "assisted_shots_per90", "carries_into_final_third_per90",
        "carries_into_penalty_area_per90", "touches_att_pen_area_per90",
        "us_xg_per90", "us_xa_per90", "us_npxg_per90",
        "naive_xg_per90", "non_penalty_goals_per90",
        "goal_contribution_per90",
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

    # Step 4: multi-season features (after growth, before interactions)
    cols_before = len(df.columns)
    df = create_multi_season_features(df)
    result.multi_season_features_added = len(df.columns) - cols_before

    # Step 5: interaction features
    cols_before = len(df.columns)
    df = create_interaction_features(df)
    result.interaction_features_added = len(df.columns) - cols_before

    # Step 6: league adjustments
    cols_before = len(df.columns)
    df = apply_league_adjustments(df)
    result.league_adjustments_applied = len(df.columns) - cols_before

    result.final_rows = len(df)
    result.final_features = len(df.columns)

    logger.info(str(result))
    return df, result

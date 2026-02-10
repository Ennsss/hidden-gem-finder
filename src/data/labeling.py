"""Breakout labeling and temporal split generation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.storage import PlayerDatabase

logger = logging.getLogger(__name__)

# Load config
_config_path = Path(__file__).parent.parent.parent / "config" / "leagues.yaml"


def _load_config() -> dict:
    with open(_config_path) as f:
        return yaml.safe_load(f)


def _load_model_config() -> dict:
    model_config_path = Path(__file__).parent.parent.parent / "config" / "model_params.yaml"
    with open(model_config_path) as f:
        return yaml.safe_load(f)


@dataclass
class LabelingResult:
    """Result of the labeling step."""

    total_observations: int = 0
    source_league_observations: int = 0
    positive_labels: int = 0
    negative_labels: int = 0
    excluded_lookforward: int = 0
    positive_rate: float = 0.0
    folds_created: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Labeling: {self.source_league_observations} source observations\n"
            f"  Positive: {self.positive_labels} ({self.positive_rate:.1%})\n"
            f"  Negative: {self.negative_labels}\n"
            f"  Excluded (lookforward): {self.excluded_lookforward}\n"
            f"  Folds created: {self.folds_created}"
        )


def _season_to_start_year(season: str) -> int:
    """Convert season string like '2019-20' or '2019-2020' to start year."""
    return int(season.split("-")[0])


def _seasons_within_window(
    season: str, lookforward: int, all_seasons: set[str]
) -> list[str]:
    """Get seasons within lookforward window from given season.

    Args:
        season: Base season string
        lookforward: Number of years to look forward
        all_seasons: Set of all available seasons

    Returns:
        List of future seasons within window
    """
    start_year = _season_to_start_year(season)
    future = []
    for s in all_seasons:
        s_year = _season_to_start_year(s)
        if 0 < (s_year - start_year) <= lookforward:
            future.append(s)
    return sorted(future)


def identify_breakouts(
    df: pd.DataFrame,
    target_leagues: list[str],
    source_leagues: list[str],
    lookforward_years: int = 3,
    min_breakout_minutes: int = 900,
) -> pd.DataFrame:
    """Label players who break out from source to target league.

    For each player-season in a source league, checks if the same player_id
    appears in a target league within the lookforward window with sufficient
    minutes.

    Args:
        df: DataFrame with fbref data (must have player_id, league, season, minutes)
        target_leagues: List of target league names
        source_leagues: List of source league names
        lookforward_years: Years to look forward for breakout
        min_breakout_minutes: Minimum minutes in target league

    Returns:
        DataFrame with label, breakout_league, breakout_season columns added.
        Only contains source-league observations.
    """
    all_seasons = set(df["season"].unique())

    if df.empty or not all_seasons:
        empty = df[df["league"].isin(source_leagues)].copy()
        empty["label"] = pd.Series(dtype=int)
        empty["breakout_league"] = pd.Series(dtype=str)
        empty["breakout_season"] = pd.Series(dtype=str)
        return empty

    # Build player history lookup: {player_id: [(league, season, minutes), ...]}
    player_history = {}
    for _, row in df.iterrows():
        pid = row["player_id"]
        if pid not in player_history:
            player_history[pid] = []
        player_history[pid].append((row["league"], row["season"], row["minutes"]))

    # Filter to source league observations only
    source_mask = df["league"].isin(source_leagues)
    source_df = df[source_mask].copy()

    # Determine max available season year
    max_season_year = max(_season_to_start_year(s) for s in all_seasons)

    # Label each source observation
    labels = []
    breakout_leagues = []
    breakout_seasons = []

    for _, row in source_df.iterrows():
        pid = row["player_id"]
        season = row["season"]
        start_year = _season_to_start_year(season)

        window_complete = (start_year + lookforward_years) <= max_season_year

        # Check if player appears in target league within window
        found_breakout = False
        history = player_history.get(pid, [])
        for h_league, h_season, h_minutes in history:
            if h_league not in target_leagues:
                continue
            h_year = _season_to_start_year(h_season)
            if 0 < (h_year - start_year) <= lookforward_years:
                if h_minutes >= min_breakout_minutes:
                    labels.append(1)
                    breakout_leagues.append(h_league)
                    breakout_seasons.append(h_season)
                    found_breakout = True
                    break

        if not found_breakout:
            if window_complete:
                # Full window available, confirmed no breakout
                labels.append(0)
                breakout_leagues.append(None)
                breakout_seasons.append(None)
            else:
                # Incomplete window, can't confirm — exclude
                labels.append(np.nan)
                breakout_leagues.append(None)
                breakout_seasons.append(None)

    source_df["label"] = labels
    source_df["breakout_league"] = breakout_leagues
    source_df["breakout_season"] = breakout_seasons

    # Remove excluded observations (incomplete lookforward)
    source_df = source_df[source_df["label"].notna()].copy()
    source_df["label"] = source_df["label"].astype(int)

    return source_df


def create_temporal_splits(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """Create walk-forward temporal splits from config.

    Args:
        df: Labeled DataFrame with season column

    Returns:
        Dict of fold_name -> {"train": df, "val": df, "test": df}
    """
    config = _load_model_config()
    folds_config = config["cross_validation"]["folds"]

    splits = {}
    for i, fold in enumerate(folds_config, 1):
        train_end_year = _season_to_start_year(fold["train_end"])
        val_year = _season_to_start_year(fold["val"])
        test_year = _season_to_start_year(fold["test"])

        train_mask = df["season"].apply(_season_to_start_year) <= train_end_year
        val_mask = df["season"].apply(_season_to_start_year) == val_year
        test_mask = df["season"].apply(_season_to_start_year) == test_year

        splits[f"fold_{i}"] = {
            "train": df[train_mask].copy(),
            "val": df[val_mask].copy(),
            "test": df[test_mask].copy(),
        }

    return splits


def validate_no_leakage(splits: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Validate no temporal leakage: training data must not contain future data.

    Args:
        splits: Dict from create_temporal_splits

    Returns:
        List of leakage warnings (empty = no leakage)
    """
    warnings = []

    for fold_name, fold_data in splits.items():
        train = fold_data["train"]
        val = fold_data["val"]
        test = fold_data["test"]

        if train.empty or val.empty:
            continue

        train_max = max(train["season"].apply(_season_to_start_year))
        val_min = min(val["season"].apply(_season_to_start_year))
        test_min = min(test["season"].apply(_season_to_start_year))

        if train_max >= val_min:
            warnings.append(
                f"{fold_name}: training data (max={train_max}) "
                f"overlaps validation (min={val_min})"
            )

        if train_max >= test_min:
            warnings.append(
                f"{fold_name}: training data (max={train_max}) "
                f"overlaps test (min={test_min})"
            )

        if not val.empty and not test.empty:
            val_max = max(val["season"].apply(_season_to_start_year))
            if val_max >= test_min:
                warnings.append(
                    f"{fold_name}: validation (max={val_max}) "
                    f"overlaps test (min={test_min})"
                )

        # Check no player in train also appears in val/test with same season
        if not train.empty and not val.empty:
            train_keys = set(zip(train["player_id"], train["season"]))
            val_keys = set(zip(val["player_id"], val["season"]))
            overlap = train_keys & val_keys
            if overlap:
                warnings.append(
                    f"{fold_name}: {len(overlap)} player-seasons in both train and val"
                )

    return warnings


def generate_labels(
    db: PlayerDatabase,
    enriched_df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelingResult]:
    """Main entry point: generate breakout labels for enriched data.

    Args:
        db: PlayerDatabase instance
        enriched_df: Enriched DataFrame from matching step

    Returns:
        Tuple of (labeled DataFrame, LabelingResult)
    """
    result = LabelingResult()
    config = _load_config()

    target_league_names = [lg["name"] for lg in config["target_leagues"]]
    source_league_names = [lg["name"] for lg in config["source_leagues"]]
    lookforward = config["training"]["lookforward_years"]
    min_breakout_min = config["training"]["min_breakout_minutes"]

    # We need ALL FBref data (including target leagues) for label checking
    all_fbref = db.get_fbref_players()
    result.total_observations = len(all_fbref)

    # Map league slug names from DB to display names for matching
    # The DB uses slugs like "premier-league", config uses "Premier League"
    # We need to work with whatever format is in the data
    # Use the enriched_df which has been cleaned - its league column format
    # should match the DB format

    # Get unique leagues in data
    data_leagues = set(all_fbref["league"].unique())

    # Build slug-to-name and name-to-slug mappings
    all_config_leagues = config["target_leagues"] + config["source_leagues"]
    slug_to_name = {}
    for lg in all_config_leagues:
        # Create slug from name: "Premier League" -> "premier-league"
        slug = lg["name"].lower().replace(" ", "-")
        slug_to_name[slug] = lg["name"]

    # Determine which format the data uses
    # Check if data leagues look like slugs or display names
    target_leagues_for_data = []
    source_leagues_for_data = []

    for lg_name in target_league_names:
        slug = lg_name.lower().replace(" ", "-")
        if slug in data_leagues:
            target_leagues_for_data.append(slug)
        elif lg_name in data_leagues:
            target_leagues_for_data.append(lg_name)

    for lg_name in source_league_names:
        slug = lg_name.lower().replace(" ", "-")
        if slug in data_leagues:
            source_leagues_for_data.append(slug)
        elif lg_name in data_leagues:
            source_leagues_for_data.append(lg_name)

    # Label using all FBref data for history lookup
    labeled = identify_breakouts(
        df=all_fbref,
        target_leagues=target_leagues_for_data,
        source_leagues=source_leagues_for_data,
        lookforward_years=lookforward,
        min_breakout_minutes=min_breakout_min,
    )

    result.source_league_observations = len(labeled)
    result.positive_labels = int((labeled["label"] == 1).sum())
    result.negative_labels = int((labeled["label"] == 0).sum())
    if result.source_league_observations > 0:
        result.positive_rate = result.positive_labels / result.source_league_observations

    # Merge labels back to enriched_df (which has the extra TM/Understat columns)
    # Match on player_id + league + season
    if not enriched_df.empty and not labeled.empty:
        label_cols = labeled[["player_id", "league", "season", "label",
                              "breakout_league", "breakout_season"]]
        merged = enriched_df.merge(
            label_cols,
            on=["player_id", "league", "season"],
            how="inner",
        )
    else:
        merged = labeled

    result.excluded_lookforward = (
        result.total_observations - result.source_league_observations - result.excluded_lookforward
    )

    logger.info(str(result))
    return merged, result

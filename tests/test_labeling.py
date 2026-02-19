"""Tests for breakout labeling and temporal splits."""

import numpy as np
import pandas as pd
import pytest

from src.data.labeling import (
    LabelingResult,
    create_temporal_splits,
    identify_breakouts,
    validate_no_leakage,
)


class TestIdentifyBreakouts:
    """Tests for breakout label generation."""

    @pytest.fixture
    def sample_data(self):
        """Multi-season data with known breakout paths."""
        return pd.DataFrame({
            "player_id": [
                # Player A: Eredivisie 2018-19, 2019-20, then PL 2021-22
                "pA", "pA", "pA",
                # Player B: stays in Eredivisie through 2022-23
                "pB", "pB", "pB", "pB",
                # Player C: moves to PL but low minutes
                "pC", "pC",
                # Player D: Eredivisie 2021-22 (edge of lookforward)
                "pD",
                # Player E: target league only
                "pE", "pE",
                # Filler to extend max season to 2022-23
                "pF",
            ],
            "name": [
                "Player A", "Player A", "Player A",
                "Player B", "Player B", "Player B", "Player B",
                "Player C", "Player C",
                "Player D",
                "Player E", "Player E",
                "Player F",
            ],
            "league": [
                "eredivisie", "eredivisie", "premier-league",
                "eredivisie", "eredivisie", "eredivisie", "eredivisie",
                "eredivisie", "premier-league",
                "eredivisie",
                "premier-league", "premier-league",
                "premier-league",
            ],
            "season": [
                "2018-19", "2019-20", "2021-22",
                "2018-19", "2019-20", "2020-21", "2022-23",
                "2019-20", "2021-22",
                "2021-22",
                "2020-21", "2021-22",
                "2022-23",
            ],
            "minutes": [
                2000, 2500, 1500,
                1800, 1900, 2000, 2100,
                1200, 500,
                1000,
                2700, 2800,
                1000,
            ],
        })

    @pytest.fixture
    def target_leagues(self):
        return ["premier-league", "la-liga", "bundesliga", "serie-a", "ligue-1"]

    @pytest.fixture
    def source_leagues(self):
        return ["eredivisie"]

    def test_player_breaks_out(self, sample_data, target_leagues, source_leagues):
        """Player A in Eredivisie 2019-20 → PL 2021-22 with 1500 min = label 1."""
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        player_a_2019 = labeled[
            (labeled["player_id"] == "pA") & (labeled["season"] == "2019-20")
        ]
        assert len(player_a_2019) == 1
        assert player_a_2019.iloc[0]["label"] == 1
        assert player_a_2019.iloc[0]["breakout_league"] == "premier-league"

    def test_player_a_2018_also_breaks_out(self, sample_data, target_leagues, source_leagues):
        """Player A in 2018-19 should also be labeled 1 (within 3yr window to 2021-22)."""
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        player_a_2018 = labeled[
            (labeled["player_id"] == "pA") & (labeled["season"] == "2018-19")
        ]
        assert len(player_a_2018) == 1
        assert player_a_2018.iloc[0]["label"] == 1

    def test_player_stays_in_source(self, sample_data, target_leagues, source_leagues):
        """Player B stays in Eredivisie → label 0."""
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        player_b = labeled[labeled["player_id"] == "pB"]
        assert all(player_b["label"] == 0)

    def test_low_minutes_breakout(self, sample_data, target_leagues, source_leagues):
        """Player C moves to PL but only 500 min → label 0."""
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        player_c = labeled[
            (labeled["player_id"] == "pC") & (labeled["season"] == "2019-20")
        ]
        assert len(player_c) == 1
        assert player_c.iloc[0]["label"] == 0

    def test_excluded_lookforward_window(self, sample_data, target_leagues, source_leagues):
        """Player D in 2021-22 with 3yr window → excluded (max data is 2021-22)."""
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        # Player D at 2021-22 should be excluded since lookforward goes to 2024-25
        # but max season is 2021-22
        player_d = labeled[labeled["player_id"] == "pD"]
        assert len(player_d) == 0  # Excluded from labeled data

    def test_only_source_league_observations(self, sample_data, target_leagues, source_leagues):
        """Target league observations should NOT appear in output."""
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        assert all(labeled["league"].isin(source_leagues))
        # Player E (PL only) should not appear
        assert "pE" not in labeled["player_id"].values

    def test_label_column_is_int(self, sample_data, target_leagues, source_leagues):
        labeled = identify_breakouts(
            sample_data, target_leagues, source_leagues,
            lookforward_years=3, min_breakout_minutes=900,
        )
        assert labeled["label"].dtype in [np.int64, np.int32, int]

    def test_empty_dataframe(self, target_leagues, source_leagues):
        empty = pd.DataFrame(columns=["player_id", "name", "league", "season", "minutes"])
        labeled = identify_breakouts(
            empty, target_leagues, source_leagues,
        )
        assert len(labeled) == 0


class TestCreateTemporalSplits:
    """Tests for walk-forward temporal splits."""

    @pytest.fixture
    def labeled_df(self):
        """Labeled data spanning multiple seasons."""
        seasons = ["2016-17", "2017-18", "2018-19", "2019-20",
                    "2020-21", "2021-22", "2022-23"]
        rows = []
        for i, season in enumerate(seasons):
            for j in range(10):
                rows.append({
                    "player_id": f"p{j}",
                    "name": f"Player {j}",
                    "league": "eredivisie",
                    "season": season,
                    "minutes": 1000 + j * 100,
                    "label": 1 if j == 0 else 0,
                })
        return pd.DataFrame(rows)

    def test_creates_three_folds(self, labeled_df):
        splits = create_temporal_splits(labeled_df)
        assert len(splits) == 3
        assert "fold_1" in splits
        assert "fold_2" in splits
        assert "fold_3" in splits

    def test_fold_has_train_val_test(self, labeled_df):
        splits = create_temporal_splits(labeled_df)
        for fold_name, fold in splits.items():
            assert "train" in fold
            assert "val" in fold
            assert "test" in fold

    def test_fold_1_train_end(self, labeled_df):
        """Fold 1: train ≤ 2017-18."""
        splits = create_temporal_splits(labeled_df)
        train = splits["fold_1"]["train"]
        max_train_year = max(int(s.split("-")[0]) for s in train["season"])
        assert max_train_year <= 2017

    def test_fold_1_val_season(self, labeled_df):
        """Fold 1: val = 2018-19."""
        splits = create_temporal_splits(labeled_df)
        val = splits["fold_1"]["val"]
        assert all(val["season"] == "2018-19")

    def test_fold_1_test_season(self, labeled_df):
        """Fold 1: test = 2019-20."""
        splits = create_temporal_splits(labeled_df)
        test = splits["fold_1"]["test"]
        assert all(test["season"] == "2019-20")

    def test_no_overlap_between_splits(self, labeled_df):
        """Train, val, test should not overlap."""
        splits = create_temporal_splits(labeled_df)
        for fold_name, fold in splits.items():
            train_seasons = set(fold["train"]["season"])
            val_seasons = set(fold["val"]["season"])
            test_seasons = set(fold["test"]["season"])
            assert train_seasons.isdisjoint(val_seasons), f"{fold_name} train/val overlap"
            assert train_seasons.isdisjoint(test_seasons), f"{fold_name} train/test overlap"
            assert val_seasons.isdisjoint(test_seasons), f"{fold_name} val/test overlap"


class TestValidateNoLeakage:
    """Tests for temporal leakage validation."""

    def test_no_leakage_clean_splits(self):
        train = pd.DataFrame({
            "player_id": ["p1"], "season": ["2017-18"], "label": [0],
        })
        val = pd.DataFrame({
            "player_id": ["p2"], "season": ["2019-20"], "label": [0],
        })
        test = pd.DataFrame({
            "player_id": ["p3"], "season": ["2020-21"], "label": [0],
        })
        splits = {"fold_1": {"train": train, "val": val, "test": test}}
        warnings = validate_no_leakage(splits)
        assert len(warnings) == 0

    def test_detects_train_val_overlap(self):
        train = pd.DataFrame({
            "player_id": ["p1"], "season": ["2020-21"], "label": [0],
        })
        val = pd.DataFrame({
            "player_id": ["p2"], "season": ["2019-20"], "label": [0],
        })
        test = pd.DataFrame({
            "player_id": ["p3"], "season": ["2021-22"], "label": [0],
        })
        splits = {"fold_1": {"train": train, "val": val, "test": test}}
        warnings = validate_no_leakage(splits)
        assert len(warnings) > 0
        assert "overlaps validation" in warnings[0]

    def test_detects_train_test_overlap(self):
        train = pd.DataFrame({
            "player_id": ["p1"], "season": ["2021-22"], "label": [0],
        })
        val = pd.DataFrame({
            "player_id": ["p2"], "season": ["2020-21"], "label": [0],
        })
        test = pd.DataFrame({
            "player_id": ["p3"], "season": ["2020-21"], "label": [0],
        })
        splits = {"fold_1": {"train": train, "val": val, "test": test}}
        warnings = validate_no_leakage(splits)
        assert any("overlaps test" in w for w in warnings)

    def test_detects_player_season_overlap(self):
        """Same player-season in train and val."""
        train = pd.DataFrame({
            "player_id": ["p1"], "season": ["2018-19"], "label": [0],
        })
        val = pd.DataFrame({
            "player_id": ["p1"], "season": ["2018-19"], "label": [0],
        })
        test = pd.DataFrame({
            "player_id": ["p2"], "season": ["2020-21"], "label": [0],
        })
        splits = {"fold_1": {"train": train, "val": val, "test": test}}
        warnings = validate_no_leakage(splits)
        assert any("player-seasons in both train and val" in w for w in warnings)

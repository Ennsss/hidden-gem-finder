"""Tests for data cleaning module."""

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    CleaningResult,
    apply_filters,
    cap_outliers,
    clean_fbref_data,
    normalize_position,
)


class TestNormalizePosition:
    """Tests for position normalization across sources."""

    def test_fbref_single_position_fw(self):
        assert normalize_position("FW", source="fbref") == "FW"

    def test_fbref_single_position_mf(self):
        assert normalize_position("MF", source="fbref") == "MF"

    def test_fbref_single_position_df(self):
        assert normalize_position("DF", source="fbref") == "DF"

    def test_fbref_single_position_gk(self):
        assert normalize_position("GK", source="fbref") == "GK"

    def test_fbref_comma_separated_takes_first(self):
        assert normalize_position("FW,MF", source="fbref") == "FW"

    def test_fbref_comma_separated_mf_fw(self):
        assert normalize_position("MF,FW", source="fbref") == "MF"

    def test_fbref_comma_separated_df_mf(self):
        assert normalize_position("DF,MF", source="fbref") == "DF"

    def test_transfermarkt_left_winger(self):
        assert normalize_position("Left Winger", source="transfermarkt") == "FW"

    def test_transfermarkt_centre_forward(self):
        assert normalize_position("Centre-Forward", source="transfermarkt") == "FW"

    def test_transfermarkt_second_striker(self):
        assert normalize_position("Second Striker", source="transfermarkt") == "FW"

    def test_transfermarkt_centre_back(self):
        assert normalize_position("Centre-Back", source="transfermarkt") == "DF"

    def test_transfermarkt_left_back(self):
        assert normalize_position("Left-Back", source="transfermarkt") == "DF"

    def test_transfermarkt_right_back(self):
        assert normalize_position("Right-Back", source="transfermarkt") == "DF"

    def test_transfermarkt_central_midfield(self):
        assert normalize_position("Central Midfield", source="transfermarkt") == "MF"

    def test_transfermarkt_defensive_midfield(self):
        assert normalize_position("Defensive Midfield", source="transfermarkt") == "MF"

    def test_transfermarkt_attacking_midfield(self):
        assert normalize_position("Attacking Midfield", source="transfermarkt") == "MF"

    def test_transfermarkt_goalkeeper(self):
        assert normalize_position("Goalkeeper", source="transfermarkt") == "GK"

    def test_understat_already_normalized(self):
        assert normalize_position("FW", source="understat") == "FW"
        assert normalize_position("MF", source="understat") == "MF"
        assert normalize_position("DF", source="understat") == "DF"
        assert normalize_position("GK", source="understat") == "GK"

    def test_none_input(self):
        assert normalize_position(None, source="fbref") is None

    def test_empty_string(self):
        assert normalize_position("", source="fbref") is None

    def test_nan_input(self):
        assert normalize_position(float("nan"), source="fbref") is None

    def test_unknown_position(self):
        assert normalize_position("XYZ", source="fbref") is None

    def test_unknown_tm_position(self):
        assert normalize_position("Unknown Position", source="transfermarkt") is None


class TestApplyFilters:
    """Tests for age, minutes, and position filtering."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "name": ["Player A", "Player B", "Player C", "Player D", "Player E"],
            "position_group": ["FW", "MF", "GK", "DF", "FW"],
            "age": [22, 19, 25, 28, 16],
            "minutes": [900, 500, 2000, 1500, 300],
        })

    def test_removes_goalkeepers(self, sample_df):
        result = apply_filters(sample_df)
        assert "GK" not in result["position_group"].values

    def test_removes_too_old(self, sample_df):
        result = apply_filters(sample_df)
        assert "Player D" not in result["name"].values  # age 28

    def test_removes_too_young(self, sample_df):
        result = apply_filters(sample_df)
        assert "Player E" not in result["name"].values  # age 16

    def test_removes_low_minutes(self, sample_df):
        result = apply_filters(sample_df, min_minutes=600)
        assert "Player B" not in result["name"].values  # 500 minutes

    def test_keeps_valid_players(self, sample_df):
        result = apply_filters(sample_df)
        assert "Player A" in result["name"].values  # FW, 22, 900 min

    def test_custom_age_range(self, sample_df):
        result = apply_filters(sample_df, min_age=18, max_age=30)
        assert "Player D" in result["name"].values  # age 28
        assert "Player E" not in result["name"].values  # age 16

    def test_custom_minutes(self, sample_df):
        result = apply_filters(sample_df, min_minutes=100)
        assert "Player E" not in result["name"].values  # age 16, filtered by age
        assert "Player B" in result["name"].values  # 500 >= 100

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["position_group", "age", "minutes"])
        result = apply_filters(df)
        assert len(result) == 0

    def test_resets_index(self, sample_df):
        result = apply_filters(sample_df)
        assert list(result.index) == list(range(len(result)))


class TestCapOutliers:
    """Tests for outlier capping."""

    def test_caps_at_99th_percentile(self):
        # Create 100 normal values + 1 extreme outlier per group
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "position_group": ["FW"] * n,
            "xg": np.concatenate([np.random.normal(0.5, 0.1, n - 1), [10.0]]),
            "minutes_90s": np.random.uniform(5, 30, n),
        })
        capped, count = cap_outliers(df)
        assert capped["xg"].max() < 10.0
        assert count >= 1

    def test_does_not_cap_protected_columns(self):
        df = pd.DataFrame({
            "position_group": ["FW"] * 3,
            "age": [17, 22, 99],  # Age 99 should NOT be capped
            "xg": [0.1, 0.2, 0.3],
        })
        capped, _ = cap_outliers(df)
        assert capped["age"].max() == 99  # Age is excluded from capping

    def test_handles_nan_values(self):
        df = pd.DataFrame({
            "position_group": ["FW", "FW", "FW"],
            "xg": [0.1, np.nan, 0.3],
        })
        capped, _ = cap_outliers(df)
        assert pd.isna(capped["xg"].iloc[1])

    def test_caps_per_position_group(self):
        df = pd.DataFrame({
            "position_group": ["FW"] * 50 + ["DF"] * 50,
            "xg": ([0.5] * 49 + [100.0]) + ([0.1] * 49 + [50.0]),
        })
        capped, count = cap_outliers(df)
        assert count >= 2  # At least one outlier per group

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["position_group", "xg"])
        capped, count = cap_outliers(df)
        assert count == 0


class TestCleanFbrefData:
    """Tests for the main clean_fbref_data function."""

    def test_returns_cleaning_result(self, mocker):
        mock_db = mocker.MagicMock()
        mock_db.get_fbref_players.return_value = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "name": ["Player A", "Player B"],
            "position": ["FW", "MF"],
            "age": [22, 20],
            "minutes": [900, 600],
            "minutes_90s": [10.0, 6.67],
            "goals": [5, 2],
            "league": ["eredivisie", "eredivisie"],
            "season": ["2020-21", "2020-21"],
        })

        df, result = clean_fbref_data(mock_db)
        assert isinstance(result, CleaningResult)
        assert result.total_raw == 2
        assert result.final_count == 2

    def test_empty_database(self, mocker):
        mock_db = mocker.MagicMock()
        mock_db.get_fbref_players.return_value = pd.DataFrame()

        df, result = clean_fbref_data(mock_db)
        assert len(df) == 0
        assert "No FBref data found" in result.warnings[0]

    def test_filters_goalkeepers(self, mocker):
        mock_db = mocker.MagicMock()
        mock_db.get_fbref_players.return_value = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "name": ["Keeper", "Forward"],
            "position": ["GK", "FW"],
            "age": [25, 22],
            "minutes": [2700, 900],
            "minutes_90s": [30.0, 10.0],
            "goals": [0, 5],
            "league": ["eredivisie", "eredivisie"],
            "season": ["2020-21", "2020-21"],
        })

        df, result = clean_fbref_data(mock_db)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Forward"
        assert result.gk_removed == 1

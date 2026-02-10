"""Tests for feature engineering and selection modules."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FeatureResult,
    apply_league_adjustments,
    convert_to_per90,
    create_derived_features,
    create_growth_features,
    engineer_features,
)
from src.features.selection import (
    SelectionResult,
    remove_correlated,
    remove_low_variance,
    select_features,
)


class TestConvertToPer90:
    """Tests for per-90 conversion."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "name": ["A", "B", "C"],
            "minutes_90s": [10.0, 20.0, 0.3],  # p3 has too few minutes
            "goals": [5, 10, 1],
            "assists": [3, 4, 0],
            "xg": [4.5, 9.0, 0.8],
            "npxg": [4.0, 8.0, 0.7],
            "xg_assist": [2.5, 3.5, 0.2],
            "shots": [30, 60, 5],
            "shots_on_target": [15, 30, 2],
            "passes_completed": [200, 500, 30],
            "passes": [250, 600, 40],
            "progressive_passes": [20, 50, 3],
            "progressive_carries": [15, 40, 2],
            "tackles": [10, 25, 1],
            "interceptions": [8, 20, 1],
            "blocks": [5, 12, 0],
            "clearances": [3, 8, 0],
            "touches": [300, 700, 50],
            "take_ons_won": [12, 30, 2],
            "carries": [150, 400, 25],
            "carries_into_final_third": [8, 20, 1],
            "carries_into_penalty_area": [3, 8, 0],
            "touches_att_pen_area": [20, 50, 4],
            "assisted_shots": [10, 25, 1],
            "league": ["eredivisie", "eredivisie", "eredivisie"],
            "season": ["2020-21", "2020-21", "2020-21"],
        })

    def test_goals_per90_calculation(self, sample_df):
        result = convert_to_per90(sample_df)
        p1 = result[result["player_id"] == "p1"].iloc[0]
        assert p1["goals_per90"] == pytest.approx(0.5)  # 5 / 10.0

    def test_assists_per90_calculation(self, sample_df):
        result = convert_to_per90(sample_df)
        p2 = result[result["player_id"] == "p2"].iloc[0]
        assert p2["assists_per90"] == pytest.approx(0.2)  # 4 / 20.0

    def test_drops_low_minutes(self, sample_df):
        result = convert_to_per90(sample_df)
        assert "p3" not in result["player_id"].values  # minutes_90s=0.3 < 0.5

    def test_preserves_original_columns(self, sample_df):
        result = convert_to_per90(sample_df)
        assert "goals" in result.columns  # Original kept
        assert "goals_per90" in result.columns  # New added

    def test_per90_columns_created(self, sample_df):
        result = convert_to_per90(sample_df)
        expected_per90 = [
            "goals_per90", "assists_per90", "xg_per90", "npxg_per90",
            "shots_per90", "shots_on_target_per90",
            "passes_completed_per90", "passes_per90",
            "progressive_passes_per90", "progressive_carries_per90",
            "tackles_per90", "interceptions_per90", "blocks_per90",
            "clearances_per90", "touches_per90", "take_ons_won_per90",
        ]
        for col in expected_per90:
            assert col in result.columns, f"Missing {col}"

    def test_resets_index(self, sample_df):
        result = convert_to_per90(sample_df)
        assert list(result.index) == list(range(len(result)))


class TestCreateDerivedFeatures:
    """Tests for derived feature computation."""

    @pytest.fixture
    def per90_df(self):
        return pd.DataFrame({
            "player_id": ["p1"],
            "goals_per90": [0.5],
            "assists_per90": [0.3],
            "xg_per90": [0.4],
            "npxg_per90": [0.35],
            "xg_assist_per90": [0.25],
            "touches_per90": [50.0],
            "passes_per90": [30.0],
            "age": [22],
            "minutes": [2000],
            "league": ["eredivisie"],
        })

    def test_xg_overperformance(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["xg_overperformance"] == pytest.approx(0.1)  # 0.5 - 0.4

    def test_xa_overperformance(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["xa_overperformance"] == pytest.approx(0.05)  # 0.3 - 0.25

    def test_goal_contribution(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["goal_contribution_per90"] == pytest.approx(0.8)

    def test_npxg_xa(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["npxg_xa_per90"] == pytest.approx(0.6)  # 0.35 + 0.25

    def test_league_adjusted_xg(self, per90_df):
        result = create_derived_features(per90_df)
        # Eredivisie coefficient = 0.85
        assert result.iloc[0]["league_adjusted_xg"] == pytest.approx(0.4 * 0.85)

    def test_involvement_score(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["involvement_score"] == pytest.approx(0.8)  # (50+30)/100

    def test_age_potential_factor(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["age_potential_factor"] == pytest.approx(0.5)  # (27-22)/10

    def test_minutes_share(self, per90_df):
        result = create_derived_features(per90_df)
        assert result.iloc[0]["minutes_share"] == pytest.approx(2000 / 3060)


class TestCreateGrowthFeatures:
    """Tests for year-over-year growth features."""

    @pytest.fixture
    def multi_season_df(self):
        return pd.DataFrame({
            "player_id": ["p1", "p1", "p2"],
            "season": ["2019-20", "2020-21", "2020-21"],
            "xg_per90": [0.3, 0.5, 0.4],
            "xg_assist_per90": [0.2, 0.3, 0.25],
            "progressive_passes_per90": [2.0, 3.0, 2.5],
            "progressive_carries_per90": [1.5, 2.0, 1.8],
            "minutes": [1500, 2000, 1800],
            "league": ["eredivisie", "eredivisie", "eredivisie"],
        })

    def test_growth_calculated(self, multi_season_df):
        result = create_growth_features(multi_season_df)
        # Player 1: xg went from 0.3 to 0.5 → (0.5 - 0.3) / (0.3 + 0.01)
        p1_second = result[
            (result["player_id"] == "p1") & (result["season"] == "2020-21")
        ].iloc[0]
        expected = (0.5 - 0.3) / (0.3 + 0.01)
        assert p1_second["xg_per90_growth"] == pytest.approx(expected, rel=0.01)

    def test_first_season_is_nan(self, multi_season_df):
        result = create_growth_features(multi_season_df)
        p1_first = result[
            (result["player_id"] == "p1") & (result["season"] == "2019-20")
        ].iloc[0]
        assert pd.isna(p1_first["xg_per90_growth"])

    def test_single_season_player_is_nan(self, multi_season_df):
        result = create_growth_features(multi_season_df)
        p2 = result[result["player_id"] == "p2"].iloc[0]
        assert pd.isna(p2["xg_per90_growth"])

    def test_growth_columns_created(self, multi_season_df):
        result = create_growth_features(multi_season_df)
        assert "xg_per90_growth" in result.columns
        assert "minutes_played_growth" in result.columns


class TestApplyLeagueAdjustments:
    """Tests for league difficulty adjustments."""

    def test_eredivisie_adjustment(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "xg_per90": [0.5],
            "goals_per90": [0.4],
            "assists_per90": [0.3],
            "league": ["eredivisie"],
        })
        result = apply_league_adjustments(df)
        # Eredivisie coefficient = 0.85
        assert result.iloc[0]["xg_per90_league_adj"] == pytest.approx(0.5 * 0.85)
        assert result.iloc[0]["goals_per90_league_adj"] == pytest.approx(0.4 * 0.85)

    def test_no_adjustment_for_unknown_league(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "xg_per90": [0.5],
            "league": ["unknown-league"],
        })
        result = apply_league_adjustments(df)
        # Default coefficient = 1.0
        assert result.iloc[0]["xg_per90_league_adj"] == pytest.approx(0.5)

    def test_creates_league_adj_columns(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "xg_per90": [0.5],
            "passes_per90": [30.0],
            "league": ["eredivisie"],
        })
        result = apply_league_adjustments(df)
        assert "xg_per90_league_adj" in result.columns
        assert "passes_per90_league_adj" in result.columns


class TestEngineerFeatures:
    """Tests for the full feature engineering pipeline."""

    @pytest.fixture
    def labeled_df(self):
        return pd.DataFrame({
            "player_id": ["p1", "p2"],
            "name": ["A", "B"],
            "position_group": ["FW", "MF"],
            "age": [22, 24],
            "minutes": [2000, 1800],
            "minutes_90s": [22.2, 20.0],
            "goals": [10, 4],
            "assists": [5, 8],
            "xg": [8.0, 3.5],
            "npxg": [7.0, 3.0],
            "xg_assist": [4.0, 7.0],
            "shots": [50, 30],
            "shots_on_target": [25, 15],
            "passes_completed": [400, 600],
            "passes": [500, 700],
            "progressive_passes": [30, 50],
            "progressive_carries": [20, 15],
            "tackles": [15, 30],
            "interceptions": [10, 20],
            "blocks": [5, 12],
            "clearances": [3, 15],
            "touches": [500, 700],
            "take_ons_won": [20, 10],
            "carries": [200, 300],
            "carries_into_final_third": [15, 10],
            "carries_into_penalty_area": [5, 3],
            "touches_att_pen_area": [30, 15],
            "assisted_shots": [15, 20],
            "league": ["eredivisie", "eredivisie"],
            "season": ["2020-21", "2020-21"],
            "label": [1, 0],
        })

    def test_returns_feature_result(self, labeled_df):
        result_df, result = engineer_features(labeled_df)
        assert isinstance(result, FeatureResult)
        assert result.input_rows == 2
        assert result.final_rows == 2

    def test_has_per90_features(self, labeled_df):
        result_df, _ = engineer_features(labeled_df)
        assert "goals_per90" in result_df.columns
        assert "xg_per90" in result_df.columns

    def test_has_derived_features(self, labeled_df):
        result_df, _ = engineer_features(labeled_df)
        assert "xg_overperformance" in result_df.columns
        assert "goal_contribution_per90" in result_df.columns

    def test_preserves_label(self, labeled_df):
        result_df, _ = engineer_features(labeled_df)
        assert "label" in result_df.columns
        assert set(result_df["label"]) == {0, 1}


class TestRemoveCorrelated:
    """Tests for correlated feature removal."""

    def test_removes_perfectly_correlated(self):
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect correlation with a
            "feat_c": [5.0, 3.0, 1.0, 4.0, 2.0],  # Uncorrelated
        })
        result, removed = remove_correlated(df, threshold=0.90)
        assert len(removed) == 1
        assert "feat_c" in result.columns  # Uncorrelated kept

    def test_keeps_lower_threshold(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
        })
        result, removed = remove_correlated(df, threshold=0.90)
        assert len(removed) == 0  # Random columns unlikely to be correlated

    def test_preserves_protected_columns(self):
        df = pd.DataFrame({
            "age": [22, 23, 24],
            "feat_a": [1.0, 2.0, 3.0],
            "label": [0, 1, 0],
        })
        result, removed = remove_correlated(df, threshold=0.50)
        assert "age" in result.columns
        assert "label" in result.columns


class TestRemoveLowVariance:
    """Tests for low-variance feature removal."""

    def test_removes_constant_feature(self):
        df = pd.DataFrame({
            "feat_a": [1.0, 1.0, 1.0, 1.0],
            "feat_b": [1.0, 2.0, 3.0, 4.0],
        })
        result, removed = remove_low_variance(df, threshold=0.01)
        assert "feat_a" in removed
        assert "feat_b" not in removed

    def test_keeps_varying_features(self):
        df = pd.DataFrame({
            "feat_a": [1.0, 10.0, 20.0, 30.0],
        })
        result, removed = remove_low_variance(df, threshold=0.01)
        assert len(removed) == 0

    def test_preserves_protected_columns(self):
        df = pd.DataFrame({
            "label": [0, 0, 0, 0],  # Constant but protected
            "feat_a": [0.0, 0.0, 0.0, 0.0],  # Constant, not protected
        })
        result, removed = remove_low_variance(df, threshold=0.01)
        assert "label" in result.columns
        assert "feat_a" in removed


class TestSelectFeatures:
    """Tests for the full feature selection pipeline."""

    def test_returns_selection_result(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4", "p5"],
            "feat_a": np.random.randn(5),
            "feat_b": np.random.randn(5),
            "feat_const": [1.0] * 5,
            "label": [0, 1, 0, 0, 1],
        })
        result_df, result = select_features(df)
        assert isinstance(result, SelectionResult)
        assert result.final_features <= result.input_features

    def test_removes_constant_and_correlated(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4", "p5"],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [2.0, 4.0, 6.0, 8.0, 10.0],  # Correlated
            "feat_const": [1.0] * 5,  # Constant
            "feat_c": [5.0, 3.0, 1.0, 4.0, 2.0],
            "label": [0, 1, 0, 0, 1],
        })
        result_df, result = select_features(df)
        assert len(result.low_variance_removed) >= 1  # feat_const
        assert len(result.correlated_removed) >= 1  # feat_a or feat_b

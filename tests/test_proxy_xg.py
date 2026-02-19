"""Tests for proxy xG model and Tier 1 baseline features."""

import numpy as np
import pandas as pd
import pytest

from src.features.proxy_xg import (
    PROXY_XG_FEATURES,
    _prepare_proxy_features,
    apply_proxy_xg,
    build_proxy_training_data,
    train_proxy_xg,
    run_proxy_xg_pipeline,
)
from src.features.engineering import create_derived_features, _add_baseline_features


# --- Proxy xG Tests ---


class TestPrepareProxyFeatures:
    """Tests for feature preparation."""

    def test_basic_feature_matrix(self):
        df = pd.DataFrame({
            "shots_per90": [2.0, 3.0, 1.5],
            "shots_on_target_per90": [1.0, 1.5, 0.8],
            "shots_on_target_pct": [50.0, 50.0, 53.3],
            "goals_per_shot_on_target": [0.3, 0.4, 0.2],
            "position_group": ["FW", "MF", "DF"],
        })
        X, names = _prepare_proxy_features(df)
        assert X.shape == (3, 7)  # 4 shooting + 3 position
        assert "pos_FW" in names
        assert "pos_MF" in names
        assert "pos_DF" in names

    def test_missing_position_column(self):
        df = pd.DataFrame({
            "shots_per90": [2.0, 3.0],
            "shots_on_target_per90": [1.0, 1.5],
            "shots_on_target_pct": [50.0, 50.0],
            "goals_per_shot_on_target": [0.3, 0.4],
        })
        X, names = _prepare_proxy_features(df)
        assert X.shape == (2, 4)  # Only shooting features

    def test_empty_when_insufficient_features(self):
        df = pd.DataFrame({
            "shots_per90": [2.0],
            "unrelated": [1.0],
        })
        X, names = _prepare_proxy_features(df)
        assert X.size == 0  # Only 1 of 4 proxy features present, need >=2

    def test_returns_empty_for_no_features(self):
        df = pd.DataFrame({"unrelated": [1.0, 2.0]})
        X, names = _prepare_proxy_features(df)
        assert X.size == 0


class TestBuildProxyTrainingData:
    """Tests for training data construction."""

    @pytest.fixture
    def xg_df(self):
        """DataFrame with Understat xG data for multiple seasons."""
        rng = np.random.RandomState(42)
        n = 200
        return pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n)],
            "season": (["2017-18"] * 50 + ["2018-19"] * 50 +
                       ["2020-21"] * 50 + ["2021-22"] * 50),
            "shots_per90": rng.uniform(0.5, 4.0, n),
            "shots_on_target_per90": rng.uniform(0.2, 2.0, n),
            "shots_on_target_pct": rng.uniform(20, 60, n),
            "goals_per_shot_on_target": rng.uniform(0.1, 0.5, n),
            "position_group": rng.choice(["FW", "MF", "DF"], n),
            "us_xg_per90": rng.uniform(0, 1.0, n),
            "league": ["premier-league"] * n,
        })

    def test_temporal_split(self, xg_df):
        X_train, y_train, X_val, y_val, names = build_proxy_training_data(xg_df)
        assert len(y_train) == 100  # 2017-18 + 2018-19
        assert len(y_val) == 100    # 2020-21 + 2021-22
        assert len(names) == 7       # 4 shooting + 3 position

    def test_no_xg_column(self):
        df = pd.DataFrame({
            "shots_per90": [1.0],
            "shots_on_target_per90": [0.5],
            "season": ["2020-21"],
        })
        X_train, y_train, X_val, y_val, names = build_proxy_training_data(df)
        assert X_train.size == 0

    def test_too_few_rows(self):
        df = pd.DataFrame({
            "shots_per90": [1.0, 2.0],
            "shots_on_target_per90": [0.5, 1.0],
            "shots_on_target_pct": [50.0, 50.0],
            "goals_per_shot_on_target": [0.3, 0.4],
            "position_group": ["FW", "MF"],
            "us_xg_per90": [0.3, 0.5],
            "season": ["2020-21", "2021-22"],
            "league": ["premier-league"] * 2,
        })
        X_train, y_train, X_val, y_val, names = build_proxy_training_data(df)
        assert X_train.size == 0  # <50 rows


class TestTrainProxyXg:
    """Tests for model training."""

    def test_trains_and_returns_metrics(self):
        rng = np.random.RandomState(42)
        n = 100
        # Create correlated features so R2 is positive
        shots = rng.uniform(0.5, 4.0, n)
        xg = shots * 0.25 + rng.normal(0, 0.05, n)  # xG ~= shots * 0.25

        X = np.column_stack([
            shots,
            shots * 0.5 + rng.normal(0, 0.1, n),
            rng.uniform(20, 60, n),
            rng.uniform(0.1, 0.5, n),
            (rng.rand(n) > 0.5).astype(float),
            (rng.rand(n) > 0.7).astype(float),
            (rng.rand(n) > 0.8).astype(float),
        ])

        model, metrics = train_proxy_xg(
            X[:80], xg[:80], X[80:], xg[80:]
        )

        assert "r2" in metrics
        assert "rmse" in metrics
        assert "rank_correlation" in metrics
        assert metrics["r2"] > 0  # Should be reasonably positive
        assert model is not None

    def test_returns_model_object(self):
        from sklearn.ensemble import GradientBoostingRegressor
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.rand(100)
        model, _ = train_proxy_xg(X[:80], y[:80], X[80:], y[80:])
        assert isinstance(model, GradientBoostingRegressor)


class TestApplyProxyXg:
    """Tests for applying proxy xG predictions."""

    @pytest.fixture
    def trained_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        rng = np.random.RandomState(42)
        n = 100
        shots = rng.uniform(0.5, 4.0, n)
        X = np.column_stack([
            shots,
            shots * 0.5,
            rng.uniform(20, 60, n),
            rng.uniform(0.1, 0.5, n),
            (rng.rand(n) > 0.5).astype(float),
            (rng.rand(n) > 0.7).astype(float),
            (rng.rand(n) > 0.8).astype(float),
        ])
        y = shots * 0.25
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        feature_names = [
            "shots_per90", "shots_on_target_per90", "shots_on_target_pct",
            "goals_per_shot_on_target", "pos_FW", "pos_MF", "pos_DF",
        ]
        return model, feature_names

    def test_adds_proxy_columns(self, trained_model):
        model, names = trained_model
        df = pd.DataFrame({
            "shots_per90": [2.0, 3.0],
            "shots_on_target_per90": [1.0, 1.5],
            "shots_on_target_pct": [50.0, 50.0],
            "goals_per_shot_on_target": [0.3, 0.4],
            "goals_per90": [0.4, 0.6],
            "position_group": ["FW", "MF"],
        })
        result = apply_proxy_xg(df, model, names)
        assert "proxy_xg_per90" in result.columns
        assert "proxy_xg_overperformance" in result.columns
        assert result["proxy_xg_per90"].notna().all()

    def test_nan_for_missing_features(self, trained_model):
        model, names = trained_model
        df = pd.DataFrame({
            "shots_per90": [2.0, np.nan],
            "shots_on_target_per90": [1.0, 1.5],
            "shots_on_target_pct": [50.0, 50.0],
            "goals_per_shot_on_target": [0.3, 0.4],
            "goals_per90": [0.4, 0.6],
            "position_group": ["FW", "MF"],
        })
        result = apply_proxy_xg(df, model, names)
        assert result["proxy_xg_per90"].notna().sum() >= 1
        # Row with NaN shots should have NaN proxy xG
        assert pd.isna(result.iloc[1]["proxy_xg_per90"])

    def test_predictions_clipped(self, trained_model):
        model, names = trained_model
        df = pd.DataFrame({
            "shots_per90": [2.0],
            "shots_on_target_per90": [1.0],
            "shots_on_target_pct": [50.0],
            "goals_per_shot_on_target": [0.3],
            "goals_per90": [0.4],
            "position_group": ["FW"],
        })
        result = apply_proxy_xg(df, model, names)
        assert result["proxy_xg_per90"].iloc[0] >= 0
        assert result["proxy_xg_per90"].iloc[0] <= 2.0


class TestRunProxyXgPipeline:
    """Tests for the full pipeline."""

    def test_pipeline_with_xg_data(self):
        rng = np.random.RandomState(42)
        n = 200
        shots = rng.uniform(0.5, 4.0, n)
        df = pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n)],
            "season": (["2017-18"] * 80 + ["2020-21"] * 120),
            "shots_per90": shots,
            "shots_on_target_per90": shots * 0.5 + rng.normal(0, 0.1, n),
            "shots_on_target_pct": rng.uniform(20, 60, n),
            "goals_per_shot_on_target": rng.uniform(0.1, 0.5, n),
            "goals_per90": shots * 0.2 + rng.normal(0, 0.05, n),
            "position_group": rng.choice(["FW", "MF", "DF"], n),
            "us_xg_per90": shots * 0.25 + rng.normal(0, 0.05, n),
            "league": ["premier-league"] * n,
        })
        result_df, metrics = run_proxy_xg_pipeline(df)
        assert "proxy_xg_per90" in result_df.columns
        assert "proxy_xg_overperformance" in result_df.columns
        assert "r2" in metrics
        assert metrics["r2"] > 0

    def test_pipeline_without_xg_data(self):
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "season": ["2020-21", "2021-22"],
            "shots_per90": [2.0, 3.0],
            "goals_per90": [0.4, 0.6],
            "position_group": ["FW", "MF"],
            "league": ["eredivisie", "eredivisie"],
        })
        result_df, metrics = run_proxy_xg_pipeline(df)
        assert "proxy_xg_per90" in result_df.columns
        assert "error" in metrics

    def test_pipeline_saves_and_loads_model(self, tmp_path):
        rng = np.random.RandomState(42)
        n = 200
        shots = rng.uniform(0.5, 4.0, n)
        df = pd.DataFrame({
            "player_id": [f"p{i}" for i in range(n)],
            "season": (["2017-18"] * 80 + ["2020-21"] * 120),
            "shots_per90": shots,
            "shots_on_target_per90": shots * 0.5 + rng.normal(0, 0.1, n),
            "shots_on_target_pct": rng.uniform(20, 60, n),
            "goals_per_shot_on_target": rng.uniform(0.1, 0.5, n),
            "goals_per90": shots * 0.2 + rng.normal(0, 0.05, n),
            "position_group": rng.choice(["FW", "MF", "DF"], n),
            "us_xg_per90": shots * 0.25 + rng.normal(0, 0.05, n),
            "league": ["premier-league"] * n,
        })

        model_path = tmp_path / "proxy_xg.joblib"

        # First run: trains and saves
        result1, metrics1 = run_proxy_xg_pipeline(df, model_path)
        assert model_path.exists()

        # Second run: loads from disk
        result2, metrics2 = run_proxy_xg_pipeline(df, model_path)
        assert "r2" in metrics2

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1["proxy_xg_per90"].values,
            result2["proxy_xg_per90"].values,
        )


# --- Tier 1: Baseline Feature Tests ---


class TestBaselineFeatures:
    """Tests for league-position baseline features."""

    @pytest.fixture
    def baseline_df(self):
        """DataFrame with per-90 stats for baseline feature testing."""
        return pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4"],
            "goals_per90": [0.5, 0.3, 0.2, 0.4],
            "assists_per90": [0.3, 0.2, 0.1, 0.25],
            "shots_per90": [3.0, 2.0, 1.5, 2.5],
            "shots_on_target_per90": [1.5, 1.0, 0.8, 1.2],
            "tackles_per90": [1.0, 2.0, 3.0, 1.5],
            "interceptions_per90": [0.5, 1.0, 1.5, 0.8],
            "minutes_90s": [20.0, 18.0, 22.0, 15.0],
            "league": ["eredivisie"] * 4,
            "position_group": ["FW", "FW", "MF", "MF"],
            "season": ["2020-21"] * 4,
        })

    def test_naive_xg_created(self, baseline_df):
        result = create_derived_features(baseline_df)
        assert "naive_xg_per90" in result.columns
        assert result["naive_xg_per90"].notna().all()

    def test_finishing_skill_created(self, baseline_df):
        result = create_derived_features(baseline_df)
        assert "finishing_skill" in result.columns
        # finishing_skill = goals_per90 - naive_xg_per90
        # A player scoring more than their shot volume predicts should be positive
        assert result["finishing_skill"].notna().all()

    def test_goals_above_avg(self, baseline_df):
        result = create_derived_features(baseline_df)
        assert "goals_above_avg" in result.columns
        # FW avg goals = (0.5 + 0.3) / 2 = 0.4
        # p1 goals_above_avg = 0.5 - 0.4 = 0.1
        fw_mask = result["position_group"] == "FW"
        fw_avg = baseline_df.loc[fw_mask, "goals_per90"].mean()
        p1 = result[result["player_id"] == "p1"].iloc[0]
        assert p1["goals_above_avg"] == pytest.approx(0.5 - fw_avg)

    def test_assists_above_avg(self, baseline_df):
        result = create_derived_features(baseline_df)
        assert "assists_above_avg" in result.columns

    def test_shot_volume_above_avg(self, baseline_df):
        result = create_derived_features(baseline_df)
        assert "shot_volume_above_avg" in result.columns

    def test_defensive_actions(self, baseline_df):
        result = create_derived_features(baseline_df)
        assert "defensive_actions_per90" in result.columns
        p1 = result[result["player_id"] == "p1"].iloc[0]
        # tackles_per90 + interceptions_per90 = 1.0 + 0.5 = 1.5
        assert p1["defensive_actions_per90"] == pytest.approx(1.5)

    def test_non_penalty_goals_fallback(self, baseline_df):
        """Without pens_made column, uses goals_per90 as fallback."""
        result = create_derived_features(baseline_df)
        assert "non_penalty_goals_per90" in result.columns
        p1 = result[result["player_id"] == "p1"].iloc[0]
        assert p1["non_penalty_goals_per90"] == pytest.approx(0.5)

    def test_non_penalty_goals_with_pens(self):
        df = pd.DataFrame({
            "player_id": ["p1"],
            "goals": [10],
            "pens_made": [2],
            "minutes_90s": [20.0],
            "goals_per90": [0.5],
            "league": ["eredivisie"],
            "position_group": ["FW"],
            "season": ["2020-21"],
            "shots_per90": [3.0],
        })
        result = create_derived_features(df)
        # (10 - 2) / 20.0 = 0.4
        assert result.iloc[0]["non_penalty_goals_per90"] == pytest.approx(0.4)

    def test_baseline_features_missing_grouping_cols(self):
        """Should gracefully handle missing league/position columns."""
        df = pd.DataFrame({
            "player_id": ["p1"],
            "goals_per90": [0.5],
        })
        result = _add_baseline_features(df, {})
        # Should not crash, just skip baseline features
        assert "goals_above_avg" not in result.columns


# --- Tier 3: Feature Hygiene Tests ---


class TestFeatureHygiene:
    """Tests for feature selection improvements."""

    def test_market_value_excluded(self):
        from src.features.selection import PROTECTED_COLUMNS
        assert "market_value_eur" in PROTECTED_COLUMNS

    def test_match_confidence_excluded(self):
        from src.features.selection import PROTECTED_COLUMNS
        assert "match_confidence_tm" in PROTECTED_COLUMNS
        assert "match_confidence_us" in PROTECTED_COLUMNS

    def test_trainer_market_value_excluded(self):
        from src.models.trainer import PROTECTED_COLUMNS
        assert "market_value_eur" in PROTECTED_COLUMNS

    def test_imputation(self):
        from src.features.selection import impute_with_league_position_median
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4"],
            "feat_a": [1.0, np.nan, 3.0, 4.0],
            "feat_b": [10.0, 20.0, np.nan, 40.0],
            "league": ["eredivisie"] * 4,
            "position_group": ["FW"] * 4,
            "season": ["2020-21"] * 4,
            "label": [0, 1, 0, 1],
        })
        result = impute_with_league_position_median(df)
        # feat_a NaN should be filled with median of [1, 3, 4] = 3.0
        assert result.iloc[1]["feat_a"] == pytest.approx(3.0)
        # feat_b NaN should be filled with median of [10, 20, 40] = 20.0
        assert result.iloc[2]["feat_b"] == pytest.approx(20.0)

    def test_imputation_cross_groups(self):
        from src.features.selection import impute_with_league_position_median
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3", "p4"],
            "feat_a": [1.0, np.nan, 10.0, np.nan],
            "league": ["eredivisie", "eredivisie", "championship", "championship"],
            "position_group": ["FW", "FW", "FW", "FW"],
            "season": ["2020-21"] * 4,
            "label": [0, 1, 0, 1],
        })
        result = impute_with_league_position_median(df)
        # p2 (eredivisie FW) should get median of eredivisie FW = 1.0
        assert result.iloc[1]["feat_a"] == pytest.approx(1.0)
        # p4 (championship FW) should get median of championship FW = 10.0
        assert result.iloc[3]["feat_a"] == pytest.approx(10.0)

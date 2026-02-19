"""Inference module: load saved models, run feature pipeline, score new players."""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.data.cleaning import clean_fbref_data
from src.data.matching import enrich_from_sources
from src.features.engineering import engineer_features
from src.features.proxy_xg import apply_proxy_xg
from src.features.selection import select_features
from src.models.trainer import PROTECTED_COLUMNS, get_feature_columns
from src.storage import PlayerDatabase

logger = logging.getLogger(__name__)


class BreakoutPredictor:
    """Score new players using trained fold-3 models (most training data).

    Pipeline: DB -> clean -> match -> engineer -> proxy_xg -> select -> predict -> SHAP
    """

    def __init__(
        self,
        model_dir: str | Path = "outputs/models",
        data_dir: str | Path = "data/processed",
        db_path: str | Path = "data/players.duckdb",
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)

        self.lgbm_model = None
        self.xgb_model = None
        self.calibrator = None
        self.training_features = None  # exact feature columns from training

    def load_models(self) -> None:
        """Load fold-3 LGBM, XGB, and calibrator from disk."""
        lgbm_path = self.model_dir / "lgbm_fold3.joblib"
        xgb_path = self.model_dir / "xgb_fold3.joblib"
        cal_path = self.model_dir / "calibrator_fold3.joblib"

        self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
        self.xgb_model = xgb.Booster(model_file=str(xgb_path))
        self.calibrator = joblib.load(cal_path)

        # Load exact training feature columns from fold_3_test parquet
        fold3_test = pd.read_parquet(self.data_dir / "fold_3_test.parquet")
        self.training_features = get_feature_columns(fold3_test)
        logger.info(
            f"Models loaded: LGBM ({self.lgbm_model.current_iteration()} trees), "
            f"XGB ({self.xgb_model.best_iteration} trees), "
            f"{len(self.training_features)} features"
        )

    def prepare_features(
        self,
        seasons: list[str] | None = None,
        leagues: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run feature engineering pipeline on DB data for given seasons.

        Args:
            seasons: Seasons to include (default: 2023-2024, 2024-2025)
            leagues: Leagues to include (default: priority-1 feeder leagues)

        Returns:
            DataFrame with engineered features and metadata columns
        """
        if seasons is None:
            seasons = ["2023-2024", "2024-2025"]
        if leagues is None:
            leagues = ["championship", "eredivisie", "primeira-liga", "belgian-pro-league"]

        db = PlayerDatabase(self.db_path)

        try:
            # Step 1: Clean (reads all FBref data, filters by age/minutes)
            logger.info("Step 1: Cleaning FBref data...")
            cleaned_df, clean_result = clean_fbref_data(db)
            logger.info(f"Cleaned: {clean_result.final_count} rows total")

            # Filter to requested seasons and leagues
            cleaned_df = cleaned_df[
                (cleaned_df["season"].isin(seasons)) & (cleaned_df["league"].isin(leagues))
            ].reset_index(drop=True)
            logger.info(f"Filtered to {len(cleaned_df)} rows for {leagues} in {seasons}")

            if cleaned_df.empty:
                logger.warning("No data after filtering. Check that feeder leagues are scraped.")
                return pd.DataFrame()

            # Step 2: Match and enrich with TM + Understat
            logger.info("Step 2: Matching cross-source data...")
            enriched_df, match_result = enrich_from_sources(db, cleaned_df)
            logger.info(f"Matched: TM={match_result.tm_matched}, US={match_result.understat_matched}")

            # Step 3: Engineer features (no labeling needed for inference)
            logger.info("Step 3: Engineering features...")
            features_df, feat_result = engineer_features(enriched_df)
            logger.info(f"Engineered: {feat_result.final_rows} rows x {feat_result.final_features} cols")

            # Step 4: Apply proxy xG
            proxy_model_path = self.data_dir / "proxy_xg_model.joblib"
            if proxy_model_path.exists():
                logger.info("Step 4: Applying proxy xG model...")
                saved = joblib.load(proxy_model_path)
                # Ensure required columns exist for proxy xG
                if "shots_on_target_pct" not in features_df.columns:
                    if "shots_on_target" in features_df.columns and "shots" in features_df.columns:
                        features_df["shots_on_target_pct"] = (
                            features_df["shots_on_target"]
                            / features_df["shots"].replace(0, float("nan"))
                            * 100
                        )
                if "goals_per_shot_on_target" not in features_df.columns:
                    if "goals" in features_df.columns and "shots_on_target" in features_df.columns:
                        features_df["goals_per_shot_on_target"] = (
                            features_df["goals"]
                            / features_df["shots_on_target"].replace(0, float("nan"))
                        )
                features_df = apply_proxy_xg(features_df, saved["model"], saved["feature_names"])
            else:
                logger.warning("Proxy xG model not found, skipping")
                features_df["proxy_xg_per90"] = np.nan
                features_df["proxy_xg_overperformance"] = np.nan

            # Step 5: Feature selection (impute + filter)
            logger.info("Step 5: Selecting features...")
            final_df, sel_result = select_features(features_df)
            logger.info(f"Selected: {sel_result.final_features} features")

            return final_df

        finally:
            db.close()

    def _align_features(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Align DataFrame columns to match training feature set.

        Adds missing columns (filled with 0), drops extra columns,
        and reorders to match training order.

        Returns:
            Tuple of (feature_matrix, metadata_df)
        """
        # Build feature matrix with exact training columns
        aligned = pd.DataFrame(index=df.index)
        for col in self.training_features:
            if col in df.columns:
                aligned[col] = df[col]
            else:
                aligned[col] = 0.0
                logger.debug(f"Feature '{col}' missing from new data, filled with 0")

        X = aligned.to_numpy(dtype=np.float64, na_value=np.nan)

        # Fill NaN with column medians (same as trainer.load_fold)
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_medians[i]

        # Build metadata (include age for display even though it's also a feature)
        meta_cols = [c for c in df.columns if c in PROTECTED_COLUMNS]
        if "age" not in meta_cols and "age" in df.columns:
            meta_cols.append("age")
        meta_df = df[meta_cols].reset_index(drop=True)

        return X, meta_df

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score all players: LGBM + XGB ensemble -> calibrate.

        Args:
            features_df: DataFrame from prepare_features()

        Returns:
            DataFrame with metadata + probability columns
        """
        X, meta = self._align_features(features_df)

        # Individual model predictions
        prob_lgbm = self.lgbm_model.predict(X)
        dmatrix = xgb.DMatrix(X)
        prob_xgb = self.xgb_model.predict(dmatrix)

        # Ensemble (equal weights, matching training config)
        prob_ensemble = 0.5 * prob_lgbm + 0.5 * prob_xgb

        # Calibrate
        prob_calibrated = self.calibrator.predict(prob_ensemble)
        prob_calibrated = np.clip(prob_calibrated, 0.0, 1.0)

        # Build results DataFrame
        results = meta.copy()
        results["prob_lgbm"] = prob_lgbm
        results["prob_xgb"] = prob_xgb
        results["prob_ensemble"] = prob_ensemble
        results["prob_calibrated"] = prob_calibrated

        return results.sort_values("prob_calibrated", ascending=False).reset_index(drop=True)

    def explain(self, features_df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for all players using LGBM TreeExplainer.

        Args:
            features_df: DataFrame from prepare_features()

        Returns:
            SHAP values array of shape (n_players, n_features)
        """
        X, _ = self._align_features(features_df)

        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_values = explainer.shap_values(X)

        # Handle 3D output (n_samples, n_features, n_outputs)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # positive class

        return shap_values

    def run(
        self,
        seasons: list[str] | None = None,
        leagues: list[str] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Full inference pipeline: features -> predict -> explain -> save.

        Args:
            seasons: Seasons to predict
            leagues: Leagues to predict

        Returns:
            Tuple of (predictions_df, shap_values)
        """
        if self.lgbm_model is None:
            self.load_models()

        # Prepare features
        features_df = self.prepare_features(seasons=seasons, leagues=leagues)
        if features_df.empty:
            logger.warning("No data to predict on")
            return pd.DataFrame(), np.array([])

        # Predict
        predictions = self.predict(features_df)
        logger.info(f"Predicted {len(predictions)} players")

        # SHAP explanations
        logger.info("Computing SHAP explanations...")
        shap_values = self.explain(features_df)
        logger.info(f"SHAP values shape: {shap_values.shape}")

        # Save outputs
        output_dir = self.model_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions.to_csv(output_dir / "predictions_current.csv", index=False)
        np.savez_compressed(
            output_dir / "shap_current.npz",
            shap_values=shap_values,
            feature_names=self.training_features,
        )

        # Save aligned feature matrix for dashboard profile page
        X, meta = self._align_features(features_df)
        feat_df = pd.DataFrame(X, columns=self.training_features)
        # Drop feature cols that overlap with meta to avoid duplicates
        overlap = set(meta.columns) & set(feat_df.columns)
        if overlap:
            feat_df = feat_df.drop(columns=list(overlap))
        feat_df = pd.concat([meta.reset_index(drop=True), feat_df], axis=1)
        feat_df.to_parquet(output_dir / "features_current.parquet", index=False)

        logger.info(f"Saved predictions_current.csv, shap_current.npz, features_current.parquet to {output_dir}")

        return predictions, shap_values

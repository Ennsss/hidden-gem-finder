"""Data pipeline for orchestrating scraping from all sources."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.scrapers import FBrefScraper, TransfermarktScraper, UnderstatScraper
from src.storage import PlayerDatabase

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result of a scraping operation for a single league-season."""

    league: str
    season: str
    fbref_count: int = 0
    transfermarkt_count: int = 0
    understat_count: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """True if no errors occurred."""
        return len(self.errors) == 0

    @property
    def total_records(self) -> int:
        """Total records inserted across all sources."""
        return self.fbref_count + self.transfermarkt_count + self.understat_count

    def __str__(self) -> str:
        status = "OK" if self.success else f"ERRORS: {len(self.errors)}"
        return (
            f"{self.league} {self.season}: "
            f"FBref={self.fbref_count}, TM={self.transfermarkt_count}, "
            f"US={self.understat_count} [{status}]"
        )


@dataclass
class ValidationReport:
    """Report on data quality for a league-season."""

    league: str
    season: str
    fbref_players: int = 0
    transfermarkt_players: int = 0
    understat_players: int = 0
    unified_players: int = 0
    match_rate: float = 0.0
    missing_market_values: int = 0
    missing_xg_data: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Validation: {self.league} {self.season}\n"
            f"  FBref: {self.fbref_players} | TM: {self.transfermarkt_players} | "
            f"US: {self.understat_players}\n"
            f"  Unified: {self.unified_players} | Match rate: {self.match_rate:.1%}\n"
            f"  Missing market values: {self.missing_market_values}\n"
            f"  Missing xG data: {self.missing_xg_data}\n"
            f"  Warnings: {len(self.warnings)}"
        )


class DataPipeline:
    """Orchestrates data collection from all sources into DuckDB.

    Handles:
    - Scraping from FBref, Transfermarkt, and Understat
    - Storing data in DuckDB
    - Validating data quality
    - Graceful handling of source limitations (e.g., Understat only has 6 leagues)
    """

    # Leagues supported by FBref and Transfermarkt
    ALL_LEAGUES = [
        # Feeder leagues (priority 1)
        "eredivisie",
        "primeira-liga",
        "belgian-pro-league",
        "championship",
        # Feeder leagues (priority 2)
        "serie-b",
        "ligue-2",
        "austrian-bundesliga",
        "scottish-premiership",
        # Target leagues (top 5)
        "premier-league",
        "la-liga",
        "bundesliga",
        "serie-a",
        "ligue-1",
    ]

    # Priority groupings for feeder leagues
    PRIORITY_1_LEAGUES = [
        "eredivisie",
        "primeira-liga",
        "belgian-pro-league",
        "championship",
    ]

    PRIORITY_2_LEAGUES = [
        "serie-b",
        "ligue-2",
        "austrian-bundesliga",
        "scottish-premiership",
    ]

    TARGET_LEAGUES = [
        "premier-league",
        "la-liga",
        "bundesliga",
        "serie-a",
        "ligue-1",
    ]

    # Understat only supports 5 leagues (Eredivisie was dropped)
    UNDERSTAT_LEAGUES = [
        "premier-league",
        "la-liga",
        "bundesliga",
        "serie-a",
        "ligue-1",
    ]

    def __init__(
        self,
        db_path: str | Path = "data/players.duckdb",
        cache_dir: str | Path = "data/raw",
        rate_limit: float = 3.0,
    ):
        """Initialize pipeline with database and scrapers.

        Args:
            db_path: Path to DuckDB database file
            cache_dir: Directory for caching scraped HTML
            rate_limit: Seconds between requests to each source
        """
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)
        self.rate_limit = rate_limit

        # Initialize database
        self.db = PlayerDatabase(db_path)

        # Initialize scrapers
        self.fbref = FBrefScraper(cache_dir=cache_dir, rate_limit=rate_limit)
        self.transfermarkt = TransfermarktScraper(
            cache_dir=cache_dir, rate_limit=rate_limit
        )
        self.understat = UnderstatScraper(cache_dir=cache_dir, rate_limit=rate_limit)

        logger.info(f"Pipeline initialized: db={db_path}, cache={cache_dir}")

    def scrape_league_season(
        self,
        league: str,
        season: str,
        sources: list[str] | None = None,
    ) -> ScrapeResult:
        """Scrape data for a single league-season from specified sources.

        Args:
            league: League name (e.g., "eredivisie")
            season: Season string (e.g., "2023-2024")
            sources: List of sources to scrape. Default: all sources.
                    Options: "fbref", "transfermarkt", "understat"

        Returns:
            ScrapeResult with counts and any errors
        """
        if sources is None:
            sources = ["fbref", "transfermarkt", "understat"]

        start_time = time.time()
        result = ScrapeResult(league=league, season=season)

        # Validate league
        if league not in self.ALL_LEAGUES:
            result.errors.append(f"Unknown league: {league}")
            return result

        # Scrape FBref
        if "fbref" in sources:
            try:
                logger.info(f"Scraping FBref: {league} {season}")
                players = self.fbref.scrape_league_season(league, season)
                result.fbref_count = self.db.insert_fbref_players(players)
                logger.info(f"FBref: inserted {result.fbref_count} players")
            except Exception as e:
                error_msg = f"FBref error: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Scrape Transfermarkt
        if "transfermarkt" in sources:
            try:
                logger.info(f"Scraping Transfermarkt: {league} {season}")
                players = self.transfermarkt.scrape_league_season(league, season)
                result.transfermarkt_count = self.db.insert_transfermarkt_players(
                    players
                )
                logger.info(f"Transfermarkt: inserted {result.transfermarkt_count} players")
            except Exception as e:
                error_msg = f"Transfermarkt error: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Scrape Understat (if available for this league)
        if "understat" in sources:
            if league in self.UNDERSTAT_LEAGUES:
                try:
                    logger.info(f"Scraping Understat: {league} {season}")
                    players = self.understat.scrape_league_season(league, season)
                    result.understat_count = self.db.insert_understat_players(players)
                    logger.info(f"Understat: inserted {result.understat_count} players")
                except Exception as e:
                    error_msg = f"Understat error: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            else:
                logger.info(f"Understat not available for {league}, skipping")

        result.duration_seconds = time.time() - start_time
        logger.info(f"Completed {league} {season} in {result.duration_seconds:.1f}s")

        return result

    def scrape_multiple_leagues(
        self,
        leagues: list[str],
        season: str,
        sources: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[ScrapeResult]:
        """Scrape multiple leagues for a single season.

        Args:
            leagues: List of league names
            season: Season string
            sources: Sources to scrape (default: all)
            show_progress: Show tqdm progress bar

        Returns:
            List of ScrapeResult objects
        """
        results = []
        iterator = tqdm(leagues, desc=f"Scraping {season}") if show_progress else leagues

        for league in iterator:
            result = self.scrape_league_season(league, season, sources)
            results.append(result)

            if show_progress:
                iterator.set_postfix(
                    {"total": result.total_records, "errors": len(result.errors)}
                )

        return results

    def scrape_multiple_seasons(
        self,
        league: str,
        seasons: list[str],
        sources: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[ScrapeResult]:
        """Scrape a single league across multiple seasons.

        Args:
            league: League name
            seasons: List of season strings
            sources: Sources to scrape (default: all)
            show_progress: Show tqdm progress bar

        Returns:
            List of ScrapeResult objects
        """
        results = []
        iterator = tqdm(seasons, desc=f"Scraping {league}") if show_progress else seasons

        for season in iterator:
            result = self.scrape_league_season(league, season, sources)
            results.append(result)

            if show_progress:
                iterator.set_postfix(
                    {"total": result.total_records, "errors": len(result.errors)}
                )

        return results

    def scrape_feeder_leagues(
        self,
        season: str,
        priority: int = 1,
        sources: list[str] | None = None,
        show_progress: bool = True,
    ) -> list[ScrapeResult]:
        """Scrape all feeder leagues (source leagues for talent scouting).

        Args:
            season: Season string
            priority: 1 for top feeder leagues, 2 for all feeder leagues
            sources: Sources to scrape (default: all)
            show_progress: Show progress bar

        Returns:
            List of ScrapeResult objects
        """
        if priority == 1:
            leagues = self.PRIORITY_1_LEAGUES
        else:
            leagues = self.PRIORITY_1_LEAGUES + self.PRIORITY_2_LEAGUES

        return self.scrape_multiple_leagues(leagues, season, sources, show_progress)

    def validate_data(self, league: str, season: str) -> ValidationReport:
        """Validate data quality for a league-season.

        Checks:
        - Record counts per source
        - Match rate in unified view
        - Missing market values
        - Missing xG data

        Args:
            league: League name
            season: Season string

        Returns:
            ValidationReport with quality metrics
        """
        report = ValidationReport(league=league, season=season)

        # Get counts per source
        fbref_df = self.db.get_fbref_players(league=league, season=season)
        tm_df = self.db.get_transfermarkt_players(league=league, season=season)
        us_df = self.db.get_understat_players(league=league, season=season)
        unified_df = self.db.get_unified_players(league=league, season=season)

        report.fbref_players = len(fbref_df)
        report.transfermarkt_players = len(tm_df)
        report.understat_players = len(us_df)
        report.unified_players = len(unified_df)

        # Calculate match rate (players found in multiple sources)
        if report.unified_players > 0:
            # Count players with IDs from multiple sources
            multi_source = unified_df[
                (unified_df["fbref_id"].notna().astype(int) +
                 unified_df["transfermarkt_id"].notna().astype(int) +
                 unified_df["understat_id"].notna().astype(int)) >= 2
            ]
            report.match_rate = len(multi_source) / report.unified_players

        # Check for missing data
        if report.unified_players > 0:
            report.missing_market_values = unified_df["market_value_eur"].isna().sum()
            report.missing_xg_data = unified_df["xg"].isna().sum()

        # Generate warnings
        if report.fbref_players == 0:
            report.warnings.append("No FBref data found")
        if report.transfermarkt_players == 0:
            report.warnings.append("No Transfermarkt data found")
        if league in self.UNDERSTAT_LEAGUES and report.understat_players == 0:
            report.warnings.append("No Understat data found (expected for this league)")
        if report.match_rate < 0.5:
            report.warnings.append(f"Low match rate: {report.match_rate:.1%}")
        if report.missing_market_values > report.unified_players * 0.5:
            report.warnings.append("More than 50% missing market values")

        return report

    def get_supported_leagues(self, source: str | None = None) -> list[str]:
        """Get list of supported leagues, optionally filtered by source.

        Args:
            source: Filter by source ("fbref", "transfermarkt", "understat")

        Returns:
            List of league names
        """
        if source is None:
            return self.ALL_LEAGUES
        elif source == "understat":
            return self.UNDERSTAT_LEAGUES
        else:
            return self.ALL_LEAGUES

    def get_scrape_summary(self) -> dict[str, Any]:
        """Get summary of all scraped data.

        Returns:
            Dictionary with database statistics
        """
        return self.db.get_stats()

    def run_phase2(self, output_dir: str | Path = "data/processed") -> dict:
        """Run full Phase 2 pipeline: clean → match → label → engineer → select.

        Args:
            output_dir: Directory to save output parquet files

        Returns:
            Dictionary with step results and output file paths
        """
        from src.data.cleaning import clean_fbref_data
        from src.data.labeling import create_temporal_splits, generate_labels, validate_no_leakage
        from src.data.matching import enrich_from_sources
        from src.features.engineering import engineer_features
        from src.features.selection import select_features

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Step 1: Clean
        logger.info("Phase 2 Step 1: Cleaning FBref data...")
        cleaned_df, clean_result = clean_fbref_data(self.db)
        results["cleaning"] = str(clean_result)
        logger.info(f"Cleaning complete: {clean_result.final_count} records")

        if cleaned_df.empty:
            logger.warning("No data after cleaning. Aborting Phase 2.")
            return results

        # Step 2: Match and enrich
        logger.info("Phase 2 Step 2: Matching cross-source data...")
        enriched_df, match_result = enrich_from_sources(self.db, cleaned_df)
        results["matching"] = str(match_result)
        enriched_df.to_parquet(output_path / "players_enriched.parquet", index=False)
        logger.info(f"Matching complete: {match_result.tm_matched} TM, {match_result.understat_matched} US matches")

        # Step 3: Label
        logger.info("Phase 2 Step 3: Generating breakout labels...")
        labeled_df, label_result = generate_labels(self.db, enriched_df)
        results["labeling"] = str(label_result)
        labeled_df.to_parquet(output_path / "players_labeled.parquet", index=False)
        logger.info(f"Labeling complete: {label_result.positive_labels} positives ({label_result.positive_rate:.1%})")

        if labeled_df.empty:
            logger.warning("No labeled data. Aborting remaining steps.")
            return results

        # Step 4: Engineer features
        logger.info("Phase 2 Step 4: Engineering features...")
        features_df, feat_result = engineer_features(labeled_df)
        results["engineering"] = str(feat_result)

        # Step 4.5: Proxy xG model
        # Train on FULL enriched data (has top-5 leagues with Understat xG),
        # then apply the trained model to the labeled feeder-league features.
        logger.info("Phase 2 Step 4.5: Training proxy xG on full enriched data...")
        from src.features.proxy_xg import run_proxy_xg_pipeline, apply_proxy_xg
        from src.features.engineering import convert_to_per90
        proxy_model_path = output_path / "proxy_xg_model.joblib"

        # Engineer the full enriched data just enough to get per-90 shooting stats
        enriched_per90 = convert_to_per90(enriched_df)
        # Need shots_on_target_pct and goals_per_shot_on_target which may exist as raw cols
        if "shots_on_target_pct" not in enriched_per90.columns:
            if "shots_on_target" in enriched_per90.columns and "shots" in enriched_per90.columns:
                enriched_per90["shots_on_target_pct"] = (
                    enriched_per90["shots_on_target"] / enriched_per90["shots"].replace(0, float("nan")) * 100
                )
        if "goals_per_shot_on_target" not in enriched_per90.columns:
            if "goals" in enriched_per90.columns and "shots_on_target" in enriched_per90.columns:
                enriched_per90["goals_per_shot_on_target"] = (
                    enriched_per90["goals"] / enriched_per90["shots_on_target"].replace(0, float("nan"))
                )

        # Train proxy xG model on full data (top-5 leagues have Understat xG)
        _, proxy_metrics = run_proxy_xg_pipeline(enriched_per90, proxy_model_path)
        results["proxy_xg"] = proxy_metrics

        if "error" not in proxy_metrics:
            logger.info(
                f"Proxy xG: R2={proxy_metrics.get('r2', 0):.3f}, "
                f"Rank corr={proxy_metrics.get('rank_correlation', 0):.3f}"
            )
            # Now apply the saved model to the labeled features_df
            import joblib
            saved = joblib.load(proxy_model_path)
            # Ensure features_df has the required columns
            if "shots_on_target_pct" not in features_df.columns:
                if "shots_on_target" in features_df.columns and "shots" in features_df.columns:
                    features_df["shots_on_target_pct"] = (
                        features_df["shots_on_target"] / features_df["shots"].replace(0, float("nan")) * 100
                    )
            if "goals_per_shot_on_target" not in features_df.columns:
                if "goals" in features_df.columns and "shots_on_target" in features_df.columns:
                    features_df["goals_per_shot_on_target"] = (
                        features_df["goals"] / features_df["shots_on_target"].replace(0, float("nan"))
                    )
            features_df = apply_proxy_xg(features_df, saved["model"], saved["feature_names"])
        else:
            logger.warning(f"Proxy xG skipped: {proxy_metrics.get('error')}")

        # Step 5: Select features
        logger.info("Phase 2 Step 5: Selecting features...")
        final_df, sel_result = select_features(features_df)
        results["selection"] = str(sel_result)
        final_df.to_parquet(output_path / "features_final.parquet", index=False)

        # Save feature metadata
        numeric_cols = [
            c for c in final_df.select_dtypes(include=["number"]).columns
            if c not in {"label", "age", "birth_year"}
        ]
        metadata = {
            "feature_names": sorted(numeric_cols),
            "n_features": len(numeric_cols),
            "n_rows": len(final_df),
            "positive_rate": float(label_result.positive_rate),
            "correlated_removed": sel_result.correlated_removed,
            "low_variance_removed": sel_result.low_variance_removed,
        }
        with open(output_path / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Step 6: Create temporal splits
        logger.info("Phase 2 Step 6: Creating temporal splits...")
        splits = create_temporal_splits(final_df)
        leakage_warnings = validate_no_leakage(splits)
        if leakage_warnings:
            for w in leakage_warnings:
                logger.warning(f"LEAKAGE: {w}")
            results["leakage_warnings"] = leakage_warnings

        for fold_name, fold_data in splits.items():
            for split_name, split_df in fold_data.items():
                fname = f"{fold_name}_{split_name}.parquet"
                split_df.to_parquet(output_path / fname, index=False)

        results["folds"] = len(splits)
        results["output_dir"] = str(output_path)

        logger.info(f"Phase 2 complete. Outputs saved to {output_path}")
        return results

    def run_phase3(
        self,
        input_dir: str | Path = "data/processed",
        output_dir: str | Path = "outputs/models",
        n_trials: int = 100,
        skip_tuning: bool = False,
    ) -> dict:
        """Run Phase 3 pipeline: tune → train → evaluate → explain → save.

        Args:
            input_dir: Directory with fold parquet files from Phase 2
            output_dir: Directory to save models, metrics, predictions
            n_trials: Number of Optuna trials per model
            skip_tuning: If True, use base params instead of tuning

        Returns:
            Dictionary with evaluation results and output paths
        """
        import json as json_mod
        import yaml
        import joblib
        import numpy as np

        from src.models.trainer import load_fold, train_baseline, train_lgbm, train_xgb
        from src.models.tuner import tune_lgbm, tune_xgb
        from src.models.evaluator import evaluate_fold, cross_fold_summary
        from src.models.explainer import generate_explanations

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load config
        config_path = Path("config/model_params.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        n_folds = config.get("cross_validation", {}).get("n_folds", 3)
        results = {"folds": {}}

        # Tune hyperparameters on fold 1 (or skip)
        logger.info("Phase 3: Loading fold 1 for tuning...")
        X_train, y_train, X_val, y_val, X_test, y_test, meta_test, feature_names = load_fold(
            input_path, 1
        )

        if skip_tuning:
            lgbm_best_params = config.get("lightgbm", {}).get("base_params", {})
            xgb_best_params = config.get("xgboost", {}).get("base_params", {})
            logger.info("Skipping tuning, using base params")
        else:
            logger.info(f"Tuning LightGBM ({n_trials} trials)...")
            lgbm_best_params, _ = tune_lgbm(X_train, y_train, X_val, y_val, config, n_trials)

            logger.info(f"Tuning XGBoost ({n_trials} trials)...")
            xgb_best_params, _ = tune_xgb(X_train, y_train, X_val, y_val, config, n_trials)

        # Save best params
        with open(output_path / "best_params_lgbm.json", "w") as f:
            json_mod.dump(lgbm_best_params, f, indent=2)
        with open(output_path / "best_params_xgb.json", "w") as f:
            json_mod.dump(xgb_best_params, f, indent=2)

        # Train baseline on fold 1
        logger.info("Training baseline logistic regression...")
        baseline_model, baseline_scaler = train_baseline(X_train, y_train, config)
        joblib.dump({"model": baseline_model, "scaler": baseline_scaler},
                    output_path / "baseline_logistic.joblib")

        # Train and evaluate on each fold
        fold_results = []
        all_predictions = []

        for fold_num in range(1, n_folds + 1):
            logger.info(f"Processing fold {fold_num}/{n_folds}...")
            X_train, y_train, X_val, y_val, X_test, y_test, meta_test, feature_names = load_fold(
                input_path, fold_num
            )

            # Train models
            lgbm_model = train_lgbm(X_train, y_train, X_val, y_val, lgbm_best_params, config)
            xgb_model = train_xgb(X_train, y_train, X_val, y_val, xgb_best_params, config)

            # Save models
            lgbm_model.save_model(str(output_path / f"lgbm_fold{fold_num}.joblib"))
            xgb_model.save_model(str(output_path / f"xgb_fold{fold_num}.joblib"))

            # Evaluate
            fold_result = evaluate_fold(
                lgbm_model, xgb_model, X_test, y_test, X_val, y_val, meta_test, config
            )

            # Save calibrator
            joblib.dump(fold_result["calibrator"],
                        output_path / f"calibrator_fold{fold_num}.joblib")

            # Add fold info to predictions
            preds = fold_result["predictions"]
            preds["fold"] = fold_num
            all_predictions.append(preds)

            # SHAP explanations
            logger.info(f"Computing SHAP explanations for fold {fold_num}...")
            shap_max = config.get("shap", {}).get("max_samples", 1000)
            explanations = generate_explanations(
                lgbm_model, xgb_model, X_test, feature_names, meta_test, shap_max
            )

            # Save SHAP values
            lgbm_shap_vals = explanations["lgbm_shap"].values
            xgb_shap_vals = explanations["xgb_shap"].values
            np.savez_compressed(
                output_path / f"shap_values_fold{fold_num}.npz",
                lgbm=lgbm_shap_vals,
                xgb=xgb_shap_vals,
            )

            fold_results.append(fold_result)
            results["folds"][f"fold_{fold_num}"] = {
                "lgbm_metrics": fold_result["lgbm_metrics"],
                "xgb_metrics": fold_result["xgb_metrics"],
                "ensemble_metrics": fold_result["ensemble_metrics"],
                "calibrated_metrics": fold_result["calibrated_metrics"],
            }

        # Cross-fold summary
        summary = cross_fold_summary(fold_results)
        results["summary"] = {
            model: {metric: info["mean"] for metric, info in metrics.items()}
            for model, metrics in summary.items()
        }

        # Save combined predictions
        import pandas as pd
        all_preds_df = pd.concat(all_predictions, ignore_index=True)
        all_preds_df.to_csv(output_path / "predictions_test.csv", index=False)

        # Save combined feature importance
        last_explanations = explanations  # from last fold
        last_explanations["combined_importance"].to_csv(
            output_path / "feature_importance.csv", index=False
        )

        # Save full evaluation results
        with open(output_path / "evaluation_results.json", "w") as f:
            json_mod.dump(results, f, indent=2, default=str)

        results["output_dir"] = str(output_path)
        logger.info(f"Phase 3 complete. Outputs saved to {output_path}")
        return results

    def close(self) -> None:
        """Close database connection and scraper browsers."""
        if hasattr(self.fbref, 'close'):
            self.fbref.close()
        if hasattr(self.understat, 'close'):
            self.understat.close()
        self.db.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

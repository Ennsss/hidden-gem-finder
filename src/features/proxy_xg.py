"""Proxy xG model: predict xG from basic shooting stats available in all leagues.

Trains a GradientBoostingRegressor on top-5 league player-seasons where we have
both FBref shooting stats and Understat xG. Then applies predictions to ALL
players (including feeder leagues) to provide xG-like features everywhere.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Features used to predict xG — all available in feeder leagues at 79%+ coverage
PROXY_XG_FEATURES = [
    "shots_per90",
    "shots_on_target_per90",
    "shots_on_target_pct",
    "goals_per_shot_on_target",
]

# Position groups for one-hot encoding
POSITION_GROUPS = ["FW", "MF", "DF"]


def _prepare_proxy_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix for proxy xG model from a DataFrame.

    Includes shooting stats + position one-hot encoding.
    Rows with any NaN in shooting features are dropped.

    Args:
        df: DataFrame with shooting stats and position_group column

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    feature_names = list(PROXY_XG_FEATURES)

    # Check which features exist
    available = [c for c in PROXY_XG_FEATURES if c in df.columns]
    if len(available) < 2:
        return np.array([]).reshape(0, 0), []

    # Build feature matrix
    X = df[available].copy()

    # Add position one-hot
    if "position_group" in df.columns:
        for pos in POSITION_GROUPS:
            col_name = f"pos_{pos}"
            X[col_name] = (df["position_group"] == pos).astype(float)
            feature_names.append(col_name)
    else:
        feature_names = list(available)

    feature_names = list(X.columns)
    # Convert to float64 to handle nullable integer/float types from DuckDB
    return X.to_numpy(dtype=np.float64, na_value=np.nan), feature_names


def build_proxy_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build train/validation sets from players with Understat xG data.

    Uses pre-2020 seasons for training, 2020+ for validation.

    Args:
        df: Full enriched DataFrame with us_xg_per90 and shooting stats

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names)
    """
    # Filter to rows with Understat xG (top-5 leagues only)
    if "us_xg_per90" not in df.columns:
        logger.warning("No us_xg_per90 column — cannot build proxy xG model")
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    has_xg = df["us_xg_per90"].notna()
    xg_df = df[has_xg].copy()

    if len(xg_df) < 50:
        logger.warning(f"Only {len(xg_df)} rows with Understat xG — too few for proxy model")
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    # Drop rows missing any proxy feature
    available_features = [c for c in PROXY_XG_FEATURES if c in xg_df.columns]
    xg_df = xg_df.dropna(subset=available_features)

    if len(xg_df) < 50:
        logger.warning(f"Only {len(xg_df)} complete rows — too few for proxy model")
        return np.array([]), np.array([]), np.array([]), np.array([]), []

    # Temporal split: train on pre-2020, validate on 2020+
    train_mask = xg_df["season"] < "2020"
    val_mask = ~train_mask

    train_df = xg_df[train_mask]
    val_df = xg_df[val_mask]

    if len(train_df) < 30 or len(val_df) < 10:
        # Fall back to 80/20 random split
        logger.info("Not enough temporal split data, using random 80/20")
        rng = np.random.RandomState(42)
        mask = rng.rand(len(xg_df)) < 0.8
        train_df = xg_df[mask]
        val_df = xg_df[~mask]

    X_train, feature_names = _prepare_proxy_features(train_df)
    X_val, _ = _prepare_proxy_features(val_df)

    y_train = train_df["us_xg_per90"].values
    y_val = val_df["us_xg_per90"].values

    # Align: only keep rows that had complete features
    # _prepare_proxy_features returns all rows (NaN handled before this call)
    logger.info(
        f"Proxy xG training data: {len(y_train)} train, {len(y_val)} val, "
        f"{len(feature_names)} features"
    )

    return X_train, y_train, X_val, y_val, feature_names


def train_proxy_xg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[GradientBoostingRegressor, dict]:
    """Train a GradientBoostingRegressor to predict xG from shooting stats.

    Args:
        X_train: Training features
        y_train: Training targets (us_xg_per90)
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Tuple of (fitted model, validation metrics dict)
    """
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Validation metrics
    y_pred_val = model.predict(X_val)
    r2 = r2_score(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(y_val, y_pred_val)

    metrics = {
        "r2": float(r2),
        "rmse": float(rmse),
        "rank_correlation": float(rank_corr),
        "train_size": len(y_train),
        "val_size": len(y_val),
    }

    logger.info(
        f"Proxy xG model: R2={r2:.3f}, RMSE={rmse:.4f}, "
        f"Rank corr={rank_corr:.3f}"
    )

    return model, metrics


def apply_proxy_xg(
    df: pd.DataFrame,
    model: GradientBoostingRegressor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply trained proxy xG model to all players.

    Adds proxy_xg_per90 and proxy_xg_overperformance columns.
    Players missing required shooting stats get NaN.

    Args:
        df: DataFrame with shooting stats
        model: Trained proxy xG model
        feature_names: Feature names the model expects

    Returns:
        DataFrame with proxy xG columns added
    """
    result = df.copy()

    # Build feature matrix for all players
    X_all, actual_names = _prepare_proxy_features(result)

    if X_all.size == 0 or actual_names != feature_names:
        logger.warning("Feature mismatch or empty features — skipping proxy xG")
        result["proxy_xg_per90"] = np.nan
        result["proxy_xg_overperformance"] = np.nan
        return result

    # Find rows with complete data
    complete_mask = ~np.isnan(X_all).any(axis=1)

    result["proxy_xg_per90"] = np.nan
    if complete_mask.any():
        predictions = model.predict(X_all[complete_mask])
        # Clip to reasonable range [0, max observed xG per 90]
        predictions = np.clip(predictions, 0, 2.0)
        result.loc[result.index[complete_mask], "proxy_xg_per90"] = predictions

    # Overperformance: actual goals - proxy xG
    if "goals_per90" in result.columns:
        result["proxy_xg_overperformance"] = (
            result["goals_per90"] - result["proxy_xg_per90"]
        )

    n_predicted = complete_mask.sum()
    logger.info(
        f"Proxy xG applied to {n_predicted}/{len(result)} players "
        f"({n_predicted/len(result)*100:.1f}%)"
    )

    return result


def run_proxy_xg_pipeline(
    df: pd.DataFrame,
    model_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Full proxy xG pipeline: train on Understat data, apply to all players.

    Args:
        df: Enriched DataFrame from matching step (has us_xg_per90 for top-5 leagues)
        model_path: Optional path to save/load model

    Returns:
        Tuple of (DataFrame with proxy xG features, metrics dict)
    """
    # Try to load existing model
    if model_path and model_path.exists():
        logger.info(f"Loading proxy xG model from {model_path}")
        saved = joblib.load(model_path)
        result = apply_proxy_xg(df, saved["model"], saved["feature_names"])
        return result, saved["metrics"]

    # Build training data
    X_train, y_train, X_val, y_val, feature_names = build_proxy_training_data(df)

    if X_train.size == 0:
        logger.warning("No training data for proxy xG — adding NaN columns")
        df = df.copy()
        df["proxy_xg_per90"] = np.nan
        df["proxy_xg_overperformance"] = np.nan
        return df, {"error": "no_training_data"}

    # Train
    model, metrics = train_proxy_xg(X_train, y_train, X_val, y_val)

    # Save model
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "feature_names": feature_names, "metrics": metrics},
            model_path,
        )
        logger.info(f"Proxy xG model saved to {model_path}")

    # Apply to all players
    result = apply_proxy_xg(df, model, feature_names)

    return result, metrics

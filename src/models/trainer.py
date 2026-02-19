"""Model training: data loading from fold parquets, LR/LGBM/XGB training."""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that are metadata, not features
PROTECTED_COLUMNS = {
    "player_id",
    "name",
    "team",
    "league",
    "season",
    "position",
    "position_group",
    "label",
    "breakout_league",
    "breakout_season",
    "source",
    "scraped_at",
    "match_confidence_tm",
    "match_confidence_us",
    "nationality",
    "birth_year",
    "market_value_eur",  # 99% null in training data — metadata, not a feature
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding protected metadata columns.

    Args:
        df: DataFrame with all columns

    Returns:
        Sorted list of feature column names
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in PROTECTED_COLUMNS]
    return sorted(feature_cols)


def load_fold(
    fold_dir: str | Path, fold_num: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Load a single walk-forward fold from parquet files.

    Args:
        fold_dir: Directory containing fold parquet files
        fold_num: Fold number (1, 2, or 3)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, meta_test, feature_names)
        where meta_test is a DataFrame with metadata columns for the test set
    """
    fold_dir = Path(fold_dir)

    train_df = pd.read_parquet(fold_dir / f"fold_{fold_num}_train.parquet")
    val_df = pd.read_parquet(fold_dir / f"fold_{fold_num}_val.parquet")
    test_df = pd.read_parquet(fold_dir / f"fold_{fold_num}_test.parquet")

    feature_cols = get_feature_columns(train_df)
    logger.info(f"Fold {fold_num}: {len(feature_cols)} features, "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Extract feature matrices (convert nullable dtypes to numpy-native first)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)

    # Extract labels
    y_train = train_df["label"].to_numpy(dtype=np.float64, na_value=np.nan)
    y_val = val_df["label"].to_numpy(dtype=np.float64, na_value=np.nan)
    y_test = test_df["label"].to_numpy(dtype=np.float64, na_value=np.nan)

    # Fill NaN with per-column median from training set
    train_medians = np.nanmedian(X_train, axis=0)
    # Handle all-NaN columns: fill with 0
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)

    for i in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, i])
        X_train[mask, i] = train_medians[i]

        mask = np.isnan(X_val[:, i])
        X_val[mask, i] = train_medians[i]

        mask = np.isnan(X_test[:, i])
        X_test[mask, i] = train_medians[i]

    # Build metadata for test set
    meta_cols = [c for c in test_df.columns if c in PROTECTED_COLUMNS]
    meta_test = test_df[meta_cols].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, meta_test, feature_cols


def train_baseline(
    X_train: np.ndarray, y_train: np.ndarray, config: dict
) -> tuple[LogisticRegression, StandardScaler]:
    """Train logistic regression baseline with class balancing.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Config dict (expects config["logistic"]["params"])

    Returns:
        Tuple of (fitted LogisticRegression, fitted StandardScaler)
    """
    lr_params = config.get("logistic", {}).get("params", {})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        l1_ratio=0,  # Equivalent to L2 penalty
        C=lr_params.get("C", 1.0),
        solver=lr_params.get("solver", "lbfgs"),
        max_iter=lr_params.get("max_iter", 1000),
        random_state=lr_params.get("random_state", 42),
    )
    model.fit(X_scaled, y_train)

    logger.info(f"Baseline LR trained: {X_train.shape[1]} features")
    return model, scaler


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    config: dict | None = None,
) -> lgb.Booster:
    """Train LightGBM with early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: LightGBM parameters (merged base + tuned)
        config: Optional full config for imbalance settings

    Returns:
        Trained LightGBM Booster
    """
    config = config or {}
    positive_weight = config.get("imbalance", {}).get("positive_weight", 10)
    early_stopping = config.get("lightgbm", {}).get("early_stopping_rounds", 50)

    full_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "scale_pos_weight": positive_weight,
    }
    full_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        full_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )

    logger.info(f"LightGBM trained: {model.current_iteration()} rounds")
    return model


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    config: dict | None = None,
) -> xgb.Booster:
    """Train XGBoost with early stopping.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: XGBoost parameters (merged base + tuned)
        config: Optional full config for imbalance settings

    Returns:
        Trained XGBoost Booster
    """
    config = config or {}
    positive_weight = config.get("imbalance", {}).get("positive_weight", 10)
    early_stopping = config.get("xgboost", {}).get("early_stopping_rounds", 50)

    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "scale_pos_weight": positive_weight,
    }
    full_params.update(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        full_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )

    logger.info(f"XGBoost trained: {model.best_iteration} rounds")
    return model

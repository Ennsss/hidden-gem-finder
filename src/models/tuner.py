"""Hyperparameter tuning with Optuna for LightGBM and XGBoost."""

import logging

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Suppress Optuna info logs by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _lgbm_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> float:
    """Optuna objective for LightGBM: maximize ROC-AUC on validation set."""
    from src.models.trainer import resolve_positive_weight

    tuning_space = config.get("lightgbm", {}).get("tuning_space", {})
    positive_weight = resolve_positive_weight(config, y_train)
    early_stopping = config.get("lightgbm", {}).get("early_stopping_rounds", 50)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "scale_pos_weight": positive_weight,
        "random_state": config.get("training", {}).get("random_seed", 42),
    }

    # Sample from tuning space
    for param_name, space in tuning_space.items():
        low = space["low"]
        high = space["high"]
        if isinstance(low, int) and isinstance(high, int) and not space.get("log", False):
            params[param_name] = trial.suggest_int(param_name, low, high)
        elif space.get("log", False):
            params[param_name] = trial.suggest_float(param_name, low, high, log=True)
        else:
            params[param_name] = trial.suggest_float(param_name, low, high)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )

    y_proba = model.predict(X_val)
    return float(roc_auc_score(y_val, y_proba))


def _xgb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> float:
    """Optuna objective for XGBoost: maximize ROC-AUC on validation set."""
    from src.models.trainer import resolve_positive_weight

    tuning_space = config.get("xgboost", {}).get("tuning_space", {})
    positive_weight = resolve_positive_weight(config, y_train)
    early_stopping = config.get("xgboost", {}).get("early_stopping_rounds", 50)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "scale_pos_weight": positive_weight,
        "random_state": config.get("training", {}).get("random_seed", 42),
    }

    for param_name, space in tuning_space.items():
        low = space["low"]
        high = space["high"]
        if isinstance(low, int) and isinstance(high, int) and not space.get("log", False):
            params[param_name] = trial.suggest_int(param_name, low, high)
        elif space.get("log", False):
            params[param_name] = trial.suggest_float(param_name, low, high, log=True)
        else:
            params[param_name] = trial.suggest_float(param_name, low, high)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )

    y_proba = model.predict(dval)
    return float(roc_auc_score(y_val, y_proba))


def tune_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    n_trials: int = 100,
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter search for LightGBM.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Full config dict
        n_trials: Number of Optuna trials

    Returns:
        Tuple of (best_params dict, Optuna Study)
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _lgbm_objective(trial, X_train, y_train, X_val, y_val, config),
        n_trials=n_trials,
    )

    best_params = study.best_params
    logger.info(f"LightGBM tuning complete: best ROC-AUC={study.best_value:.4f} "
                f"in {n_trials} trials")
    return best_params, study


def tune_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    n_trials: int = 100,
) -> tuple[dict, optuna.Study]:
    """Run Optuna hyperparameter search for XGBoost.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Full config dict
        n_trials: Number of Optuna trials

    Returns:
        Tuple of (best_params dict, Optuna Study)
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, X_val, y_val, config),
        n_trials=n_trials,
    )

    best_params = study.best_params
    logger.info(f"XGBoost tuning complete: best ROC-AUC={study.best_value:.4f} "
                f"in {n_trials} trials")
    return best_params, study

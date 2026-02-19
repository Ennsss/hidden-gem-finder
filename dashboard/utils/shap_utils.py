"""SHAP extraction and similar player search utilities."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .data_loader import (
    get_feature_columns,
    load_all_shap,
    load_fold_test,
    load_predictions,
)


def get_player_shap(player_idx: int) -> tuple[np.ndarray, float]:
    """Get averaged SHAP values for a player by their index in predictions_test.csv.

    The predictions CSV is ordered fold1 then fold2 then fold3.
    SHAP NPZ files match this ordering.

    Returns:
        Tuple of (averaged shap_values array, base_value estimate).
    """
    lgbm_shap, xgb_shap = load_all_shap()
    if lgbm_shap.size == 0:
        return np.array([]), 0.0

    # Average LGBM and XGB SHAP values
    avg_shap = (lgbm_shap[player_idx] + xgb_shap[player_idx]) / 2
    # Estimate base value from mean prediction minus mean SHAP contribution
    base_value = 0.5  # Approximate; exact base not stored
    return avg_shap, base_value


def get_player_features(player_idx: int, preds: pd.DataFrame) -> np.ndarray:
    """Get raw feature values for a player from fold test data."""
    row = preds.iloc[player_idx]
    fold = int(row["fold"])
    fold_df = load_fold_test(fold)
    feature_cols = get_feature_columns()

    # Find this player in the fold test data
    match = fold_df[
        (fold_df["player_id"] == row["player_id"]) & (fold_df["season"] == row["season"])
    ]
    if len(match) == 0:
        return np.full(len(feature_cols), np.nan)
    return match[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)[0]


def _build_feature_matrix(preds: pd.DataFrame) -> np.ndarray:
    """Build feature matrix for all predictions, ordered by preds index."""
    feature_cols = get_feature_columns()
    all_feats = np.full((len(preds), len(feature_cols)), np.nan)

    for fold_num in preds["fold"].unique():
        fold_df = load_fold_test(int(fold_num))
        fold_mask = preds["fold"] == fold_num
        fold_preds = preds[fold_mask]

        for local_idx, (global_idx, row) in enumerate(fold_preds.iterrows()):
            match = fold_df[
                (fold_df["player_id"] == row["player_id"]) & (fold_df["season"] == row["season"])
            ]
            if len(match) > 0:
                all_feats[global_idx] = match[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)[0]

    return all_feats


_feature_matrix_cache: np.ndarray | None = None


def _get_feature_matrix(preds: pd.DataFrame) -> np.ndarray:
    """Get or build cached feature matrix."""
    global _feature_matrix_cache
    if _feature_matrix_cache is None or len(_feature_matrix_cache) != len(preds):
        _feature_matrix_cache = _build_feature_matrix(preds)
    return _feature_matrix_cache


def find_similar_players(
    player_idx: int, preds: pd.DataFrame, top_n: int = 5, only_breakouts: bool = True
) -> pd.DataFrame:
    """Find similar players by Euclidean distance on standardized features.

    Args:
        player_idx: Index in predictions DataFrame
        preds: Predictions DataFrame
        top_n: Number of similar players to return
        only_breakouts: If True, only consider successful breakouts

    Returns:
        DataFrame with similar players and their distances
    """
    all_feats = _get_feature_matrix(preds)
    player_feats = all_feats[player_idx]
    if np.all(np.isnan(player_feats)):
        return pd.DataFrame()

    # Standardize
    means = np.nanmean(all_feats, axis=0)
    stds = np.nanstd(all_feats, axis=0)
    stds[stds == 0] = 1.0

    player_std = np.nan_to_num((player_feats - means) / stds, 0.0)
    all_std = np.nan_to_num((all_feats - means) / stds, 0.0)

    # Compute distances
    dists = cdist(player_std.reshape(1, -1), all_std, metric="euclidean")[0]

    # Filter candidates
    candidates = preds.copy()
    candidates["distance"] = dists
    candidates = candidates[candidates.index != player_idx]
    if only_breakouts:
        candidates = candidates[candidates["label"] == 1.0]

    return candidates.nsmallest(top_n, "distance")[
        ["name", "position", "team", "league", "season", "prob_calibrated", "distance"]
    ].reset_index(drop=True)


def compute_percentiles(player_idx: int, preds: pd.DataFrame,
                        pos_group: str) -> dict[str, float]:
    """Compute percentile rank of player features within their position group.

    Returns dict mapping feature_name -> percentile (0-100).
    """
    feature_cols = get_feature_columns()
    all_feats = _get_feature_matrix(preds)
    player_feats = all_feats[player_idx]

    pos_mask = (preds["position_group"] == pos_group).values
    pos_matrix = all_feats[pos_mask]

    if len(pos_matrix) == 0:
        return {f: 50.0 for f in feature_cols}

    percentiles = {}
    for i, col in enumerate(feature_cols):
        col_vals = pos_matrix[:, i]
        col_vals = col_vals[~np.isnan(col_vals)]
        if len(col_vals) == 0 or np.isnan(player_feats[i]):
            percentiles[col] = 50.0
        else:
            percentiles[col] = float(np.mean(col_vals <= player_feats[i]) * 100)
    return percentiles

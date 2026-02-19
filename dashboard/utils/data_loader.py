"""Cached data loaders for the scouting dashboard."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve project root (dashboard/ is one level below project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load predictions_test.csv with all model outputs."""
    df = pd.read_csv(OUTPUTS_DIR / "predictions_test.csv")
    # Compute age from birth_year and season
    df["age"] = df.apply(
        lambda r: int(str(r["season"]).split("-")[0]) - int(r["birth_year"])
        if pd.notna(r.get("birth_year")) else None,
        axis=1,
    )
    return df


@st.cache_data
def load_evaluation_results() -> dict:
    """Load evaluation_results.json."""
    with open(OUTPUTS_DIR / "evaluation_results.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load feature_importance.csv."""
    return pd.read_csv(OUTPUTS_DIR / "feature_importance.csv")


@st.cache_data
def load_features_final() -> pd.DataFrame:
    """Load features_final.parquet for player stats."""
    return pd.read_parquet(DATA_DIR / "features_final.parquet")


@st.cache_data
def load_fold_test(fold_num: int) -> pd.DataFrame:
    """Load a fold test parquet for feature columns."""
    return pd.read_parquet(DATA_DIR / f"fold_{fold_num}_test.parquet")


@st.cache_data
def get_feature_columns() -> list[str]:
    """Get feature column names from fold data."""
    from src.models.trainer import PROTECTED_COLUMNS
    df = load_fold_test(1)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return sorted([c for c in numeric_cols if c not in PROTECTED_COLUMNS])


@st.cache_data
def load_shap_values(fold_num: int) -> dict[str, np.ndarray]:
    """Load SHAP values for a specific fold."""
    path = OUTPUTS_DIR / f"shap_values_fold{fold_num}.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.keys()}


@st.cache_data
def load_all_shap() -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate SHAP values from all folds.

    Returns:
        Tuple of (lgbm_shap, xgb_shap) arrays concatenated across folds.
    """
    lgbm_all, xgb_all = [], []
    for fold in [1, 2, 3]:
        shap_data = load_shap_values(fold)
        if "lgbm" in shap_data:
            lgbm_all.append(shap_data["lgbm"])
        if "xgb" in shap_data:
            xgb_all.append(shap_data["xgb"].astype(np.float64))
    lgbm = np.concatenate(lgbm_all, axis=0) if lgbm_all else np.array([])
    xgb = np.concatenate(xgb_all, axis=0) if xgb_all else np.array([])
    return lgbm, xgb


@st.cache_data
def load_current_predictions() -> pd.DataFrame:
    """Load predictions_current.csv with current feeder-league predictions."""
    path = OUTPUTS_DIR / "predictions_current.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_current_features() -> pd.DataFrame:
    """Load features_current.parquet with aligned features for current predictions."""
    path = OUTPUTS_DIR / "features_current.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_current_shap() -> tuple[np.ndarray, list[str]]:
    """Load SHAP values for current predictions.

    Returns:
        Tuple of (shap_values array, feature_names list).
        Empty array and list if file not found.
    """
    path = OUTPUTS_DIR / "shap_current.npz"
    if not path.exists():
        return np.array([]), []
    data = np.load(path, allow_pickle=True)
    shap_values = data["shap_values"]
    feature_names = data["feature_names"].tolist()
    return shap_values, feature_names


def league_display(slug: str) -> str:
    """Convert league slug to display name."""
    mapping = {
        "championship": "Championship",
        "eredivisie": "Eredivisie",
        "primeira-liga": "Primeira Liga",
        "belgian-pro-league": "Belgian Pro League",
        "premier-league": "Premier League",
        "la-liga": "La Liga",
        "bundesliga": "Bundesliga",
        "serie-a": "Serie A",
        "ligue-1": "Ligue 1",
    }
    return mapping.get(slug, slug or "N/A")

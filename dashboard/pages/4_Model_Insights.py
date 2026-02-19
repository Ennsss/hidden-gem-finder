"""Model Insights page: ROC/PR curves, calibration, per-fold/per-league analysis."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from utils.charts import (
    calibration_plot,
    pr_curves,
    precision_at_k_chart,
    roc_curves,
)
from utils.data_loader import league_display, load_evaluation_results, load_predictions
from utils.styles import HIDE_STREAMLIT_STYLE

st.set_page_config(page_title="Model Insights - Hidden Gem Finder", layout="wide")
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("Model Insights")
st.markdown("---")

eval_results = load_evaluation_results()
preds = load_predictions()

# --- Per-Fold Performance Table ---
st.subheader("Per-Fold Performance")

fold_rows = []
for fold_num in [1, 2, 3]:
    fold_key = f"fold_{fold_num}"
    fold_data = eval_results.get("folds", {}).get(fold_key, {})
    fold_preds = preds[preds["fold"] == fold_num]
    n_test = len(fold_preds)
    n_pos = int(fold_preds["label"].sum())

    row = {"Fold": fold_num, "N_Test": n_test, "N_Positive": n_pos}
    for model in ["lgbm_metrics", "xgb_metrics", "ensemble_metrics", "calibrated_metrics"]:
        metrics = fold_data.get(model, {})
        prefix = model.replace("_metrics", "")
        row[f"{prefix}_roc_auc"] = metrics.get("roc_auc", None)
        row[f"{prefix}_p20"] = metrics.get("precision_at_20", None)
    fold_rows.append(row)

fold_df = pd.DataFrame(fold_rows)
st.dataframe(
    fold_df.style.format({
        col: "{:.4f}" for col in fold_df.columns if col not in ("Fold", "N_Test", "N_Positive")
    }),
    use_container_width=True,
)

st.markdown("---")

# --- ROC and PR Curves ---
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(roc_curves(preds), use_container_width=True)
with col2:
    st.plotly_chart(pr_curves(preds), use_container_width=True)

# --- Calibration and Precision@K ---
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(calibration_plot(preds), use_container_width=True)
with col4:
    st.plotly_chart(precision_at_k_chart(preds), use_container_width=True)

st.markdown("---")

# --- Per-League ROC-AUC ---
st.subheader("Per-League ROC-AUC")

league_rows = []
for league in sorted(preds["league"].unique()):
    lg = preds[preds["league"] == league]
    y_true = lg["label"].values
    y_score = lg["prob_calibrated"].values
    n = len(lg)
    n_pos = int(y_true.sum())
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = None
    league_rows.append({
        "League": league_display(league),
        "N_Players": n,
        "N_Breakouts": n_pos,
        "Breakout%": f"{100*n_pos/n:.1f}%",
        "ROC-AUC": auc,
    })

league_df = pd.DataFrame(league_rows)
st.dataframe(
    league_df.style.format({"ROC-AUC": "{:.4f}"}, na_rep="-"),
    use_container_width=True,
)

# --- Summary Stats ---
st.markdown("---")
st.subheader("Cross-Fold Summary")

summary = eval_results.get("summary", {})
summary_rows = []
for model in ["lgbm_metrics", "xgb_metrics", "ensemble_metrics", "calibrated_metrics"]:
    model_data = summary.get(model, {})
    row = {"Model": model.replace("_metrics", "").upper()}
    for metric in ["roc_auc", "average_precision", "brier_score",
                    "precision_at_10", "precision_at_20", "precision_at_50", "precision_at_100"]:
        val = model_data.get(metric, {})
        if isinstance(val, dict):
            row[metric] = val.get("mean", None)
        else:
            row[metric] = val
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df.style.format({
        col: "{:.4f}" for col in summary_df.columns if col != "Model"
    }),
    use_container_width=True,
)

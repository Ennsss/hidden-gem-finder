"""Hidden Gem Finder - Scouting Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Hidden Gem Finder",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.data_loader import load_evaluation_results, load_predictions
from utils.styles import HIDE_STREAMLIT_STYLE, metric_card_css, metric_card_html

st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
st.markdown(metric_card_css(), unsafe_allow_html=True)

# --- Header ---
st.title("Hidden Gem Finder")
st.markdown("**Scouting dashboard for identifying undervalued football talent**")
st.markdown("---")

# --- Summary Metrics ---
eval_results = load_evaluation_results()
preds = load_predictions()
cal = eval_results.get("summary", {}).get("calibrated_metrics", {})


def get_val(d):
    if isinstance(d, dict):
        return d.get("mean", 0)
    return d


roc_auc = get_val(cal.get("roc_auc", 0))
p20 = get_val(cal.get("precision_at_20", 0))
avg_prec = get_val(cal.get("average_precision", 0))
brier = get_val(cal.get("brier_score", 0))

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html("ROC-AUC", f"{roc_auc:.3f}"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Precision@20", f"{p20:.3f}"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Avg Precision", f"{avg_prec:.3f}"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("Brier Score", f"{brier:.3f}"), unsafe_allow_html=True)

st.markdown("")

# --- Project Overview ---
col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("""
    ### About This Project

    The Hidden Gem Finder uses machine learning to identify players in lower-tier European
    leagues (Championship, Eredivisie, Primeira Liga, Belgian Pro League) who are likely
    to break out into top-5 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1).

    **How it works:**
    - Scrapes player statistics from FBref, Transfermarkt, and Understat
    - Engineers 44 features including per-90 stats, growth trends, and league adjustments
    - Trains LightGBM + XGBoost ensemble with walk-forward validation
    - Calibrates probabilities and explains predictions with SHAP values

    **Navigate using the sidebar** to explore player rankings, individual profiles,
    and model performance details.
    """)

with col_right:
    n_total = len(preds)
    n_breakouts = int(preds["label"].sum())
    n_leagues = preds["league"].nunique()
    n_seasons = preds["season"].nunique()

    st.markdown("### Dataset")
    st.markdown(f"- **{n_total:,}** test predictions")
    st.markdown(f"- **{n_breakouts}** actual breakouts ({100*n_breakouts/n_total:.1f}%)")
    st.markdown(f"- **{n_leagues}** source leagues")
    st.markdown(f"- **{n_seasons}** seasons")
    st.markdown(f"- **3** walk-forward folds")

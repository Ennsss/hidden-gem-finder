"""Overview page: metrics, feature importance, probability distribution."""

import streamlit as st

from utils.charts import (
    breakout_destinations,
    feature_importance_chart,
    probability_distribution,
)
from utils.data_loader import (
    league_display,
    load_evaluation_results,
    load_feature_importance,
    load_predictions,
)
from utils.styles import HIDE_STREAMLIT_STYLE, metric_card_css, metric_card_html

st.set_page_config(page_title="Overview - Hidden Gem Finder", layout="wide")
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
st.markdown(metric_card_css(), unsafe_allow_html=True)

st.title("Overview")
st.markdown("---")

# Load data
eval_results = load_evaluation_results()
preds = load_predictions()
feat_imp = load_feature_importance()
cal = eval_results.get("summary", {}).get("calibrated_metrics", {})


def get_val(d):
    if isinstance(d, dict):
        return d.get("mean", 0)
    return d


# Metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card_html("ROC-AUC", f"{get_val(cal.get('roc_auc', 0)):.3f}"), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card_html("Precision@20", f"{get_val(cal.get('precision_at_20', 0)):.3f}"), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card_html("Avg Precision", f"{get_val(cal.get('average_precision', 0)):.3f}"), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card_html("Brier Score", f"{get_val(cal.get('brier_score', 0)):.3f}"), unsafe_allow_html=True)

st.markdown("")

# Feature importance + probability distribution
col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(feature_importance_chart(feat_imp, top_n=20), use_container_width=True)
with col_right:
    st.plotly_chart(probability_distribution(preds), use_container_width=True)

# Breakout destinations
st.plotly_chart(breakout_destinations(preds, league_display), use_container_width=True)

"""Scout Player page: search current feeder-league players, view breakout probability."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist

from utils.charts import radar_chart, shap_waterfall
from utils.data_loader import (
    get_feature_columns,
    league_display,
    load_current_features,
    load_current_predictions,
    load_current_shap,
    load_predictions,
)
from utils.styles import (
    ACCENT,
    HIDE_STREAMLIT_STYLE,
    friendly_name,
    metric_card_css,
    metric_card_html,
    position_badge,
)

st.set_page_config(page_title="Scout Player - Hidden Gem Finder", layout="wide")
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
st.markdown(metric_card_css(), unsafe_allow_html=True)

st.title("Scout a Player")
st.markdown("Search current feeder-league players and view breakout probability with explanations.")
st.markdown("---")

# Load data
current_preds = load_current_predictions()
shap_data, shap_feature_names = load_current_shap()
feature_cols = get_feature_columns()

# Load feature values from parquet (fixes stats-only-showing-age bug)
features_df = load_current_features()
if not features_df.empty and not current_preds.empty:
    # Merge feature columns into predictions by player_id + season
    feature_cols_to_add = [c for c in features_df.columns if c not in current_preds.columns]
    if "player_id" in features_df.columns and "season" in features_df.columns:
        current_preds = current_preds.merge(
            features_df[["player_id", "season"] + feature_cols_to_add],
            on=["player_id", "season"],
            how="left",
        )

if current_preds.empty:
    st.error(
        "No current predictions found. Run `python scripts/predict_current.py` first "
        "to generate predictions for feeder-league players."
    )
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

leagues = sorted(current_preds["league"].dropna().unique())
league_options = ["All"] + [league_display(l) for l in leagues]
selected_league = st.sidebar.selectbox("League", league_options)

positions = sorted(current_preds["position_group"].dropna().unique())
pos_options = ["All"] + positions
selected_pos = st.sidebar.selectbox("Position", pos_options)

seasons = sorted(current_preds["season"].dropna().unique())
season_options = ["All"] + list(seasons)
selected_season = st.sidebar.selectbox("Season", season_options)

age_min, age_max = st.sidebar.slider(
    "Age Range", min_value=15, max_value=40, value=(15, 40),
)

min_prob = st.sidebar.slider(
    "Min Probability", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
)

# Apply filters
df = current_preds.copy()
if selected_league != "All":
    slug = [l for l in leagues if league_display(l) == selected_league]
    if slug:
        df = df[df["league"] == slug[0]]

if selected_pos != "All":
    df = df[df["position_group"] == selected_pos]

if selected_season != "All":
    df = df[df["season"] == selected_season]

if df["age"].notna().any():
    df = df[(df["age"] >= age_min) & (df["age"] <= age_max) | df["age"].isna()]

df = df[df["prob_calibrated"] >= min_prob]

if df.empty:
    st.warning("No players match the selected filters.")
    st.stop()

# --- Player Search ---
name_search = st.text_input("Search by name", placeholder="Type a player name...")

if name_search:
    name_mask = df["name"].str.contains(name_search, case=False, na=False)
    df = df[name_mask]
    if df.empty:
        st.warning(f"No players found matching '{name_search}'.")
        st.stop()

# Build display strings for selectbox
df = df.sort_values("prob_calibrated", ascending=False).reset_index(drop=True)
df["display"] = df.apply(
    lambda r: f"{r.get('name', '?')} ({r.get('team', '?')}, "
              f"{league_display(r.get('league', ''))}, {r.get('season', '?')})",
    axis=1,
)

selected = st.selectbox(
    "Select a player",
    options=df["display"].tolist(),
    index=0,
    help="Players sorted by breakout probability (highest first)",
)

# Find the selected player
selected_idx = df["display"].tolist().index(selected)
player = df.iloc[selected_idx]

# Find original index in current_preds for SHAP lookup
original_mask = (
    (current_preds["name"] == player["name"]) &
    (current_preds["season"] == player["season"]) &
    (current_preds["team"] == player["team"])
)
original_idx = current_preds[original_mask].index[0] if original_mask.any() else 0

# --- Header Card ---
st.markdown("---")
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

pos_group = player.get("position_group", "")
badge = position_badge(pos_group) if pos_group else ""

with col1:
    st.markdown(f"### {player.get('name', 'Unknown')} {badge}", unsafe_allow_html=True)
    st.markdown(
        f"**{player.get('position', pos_group)}** | "
        f"{player.get('team', 'N/A')} | "
        f"{league_display(player.get('league', ''))} | "
        f"{player.get('season', 'N/A')}"
    )

with col2:
    prob = player.get("prob_calibrated", 0)
    prob_color = ACCENT if prob > 0.3 else "#FFC107" if prob > 0.15 else "#ADB5BD"
    st.markdown(
        f'<div style="text-align:center;padding:0.8rem;background:{prob_color};'
        f'color:white;border-radius:8px;">'
        f'<div style="font-size:2rem;font-weight:700;">{prob:.0%}</div>'
        f'<div style="font-size:0.85rem;">Breakout Prob</div></div>',
        unsafe_allow_html=True,
    )

with col3:
    age = player.get("age", None)
    age_str = str(int(age)) if pd.notna(age) else "N/A"
    st.markdown(metric_card_html("Age", age_str), unsafe_allow_html=True)

with col4:
    st.markdown(
        metric_card_html("LGBM", f"{player.get('prob_lgbm', 0):.0%}"),
        unsafe_allow_html=True,
    )
    st.markdown(
        metric_card_html("XGB", f"{player.get('prob_xgb', 0):.0%}"),
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- Two-column layout: SHAP + Radar ---
left_col, right_col = st.columns(2)

# Build feature values array from merged data
display_names = shap_feature_names if shap_feature_names else feature_cols
feature_values = np.zeros(len(display_names))
for i, fname in enumerate(display_names):
    if fname in player.index:
        val = player.get(fname)
        if pd.notna(val):
            feature_values[i] = float(val)

# --- SHAP Waterfall ---
with left_col:
    st.subheader("What Drives This Prediction?")

    if shap_data.size > 0 and original_idx < len(shap_data):
        player_shap = shap_data[original_idx]
        fig = shap_waterfall(
            player_shap, feature_values, display_names, base_value=0.5, top_n=12
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP values not available. Run predict_current.py to generate them.")

# --- Radar Chart ---
with right_col:
    st.subheader(f"Stats Radar ({pos_group})")

    radar_map = {
        "FW": [
            "goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90",
            "goal_contribution_per90", "minutes", "age_potential_factor", "naive_xg_per90",
        ],
        "MF": [
            "assists_per90", "goals_per90", "goal_contribution_per90",
            "minutes", "age_potential_factor", "tackles_won", "interceptions",
        ],
        "DF": [
            "tackles_won", "interceptions", "defensive_actions_per90", "minutes",
            "age_potential_factor", "assists_per90",
        ],
    }
    radar_features = radar_map.get(pos_group, radar_map["MF"])

    # Filter to features available in the data
    available_radar = [f for f in radar_features if f in current_preds.columns]

    if available_radar:
        pos_mask = current_preds["position_group"] == pos_group
        player_percentiles = {}
        avg_percentiles = {}

        for feat in available_radar:
            col_vals = current_preds.loc[pos_mask, feat].dropna()
            player_val = player.get(feat)
            if pd.notna(player_val) and len(col_vals) > 0:
                pct = float(np.mean(col_vals <= player_val) * 100)
                player_percentiles[feat] = pct
                avg_percentiles[feat] = 50.0
            else:
                player_percentiles[feat] = 50.0
                avg_percentiles[feat] = 50.0

        if player_percentiles:
            fig = radar_chart(
                player_percentiles, avg_percentiles,
                f"{player.get('name', 'Player')} vs {pos_group} Average",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for radar chart.")
    else:
        st.info("Radar features not available in current data.")

st.markdown("---")

# --- Player Stats Table ---
st.subheader("Player Statistics")

# Show feature columns with friendly names
excluded = {
    "display", "name", "team", "league", "season", "position", "position_group",
    "player_id", "source", "scraped_at", "nationality", "birth_year",
    "match_confidence_tm", "match_confidence_us", "market_value_eur",
    "breakout_league", "breakout_season", "label", "fold",
    "prob_lgbm", "prob_xgb", "prob_ensemble", "prob_calibrated",
}
stat_cols = [c for c in player.index if c not in excluded and pd.notna(player.get(c))]
numeric_stats = [c for c in stat_cols if isinstance(player.get(c), (int, float, np.integer, np.floating))]

if numeric_stats:
    stats_data = []
    for col in numeric_stats:
        val = player.get(col)
        if pd.notna(val):
            stats_data.append({
                "Stat": friendly_name(col),
                "Value": round(float(val), 2) if abs(float(val)) < 100 else int(float(val)),
            })
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, height=400, hide_index=True)
    else:
        st.info("No stats available for this player.")
else:
    st.info("No stats available for this player.")

st.markdown("---")

# --- Similar Historical Breakouts ---
st.subheader("Similar Historical Breakouts")
st.markdown("Top 5 confirmed breakout players from test data with most similar profiles.")

historical_preds = load_predictions()
breakouts = historical_preds[historical_preds["label"] == 1.0]

if not breakouts.empty and shap_data.size > 0 and shap_feature_names:
    player_shap_vec = shap_data[original_idx].reshape(1, -1)

    from utils.data_loader import load_all_shap
    hist_lgbm_shap, hist_xgb_shap = load_all_shap()

    if hist_lgbm_shap.size > 0:
        hist_shap = (hist_lgbm_shap + hist_xgb_shap) / 2
        breakout_indices = breakouts.index.tolist()
        valid_indices = [i for i in breakout_indices if i < len(hist_shap)]

        if valid_indices:
            breakout_shap = hist_shap[valid_indices]
            min_features = min(player_shap_vec.shape[1], breakout_shap.shape[1])
            dists = cdist(
                player_shap_vec[:, :min_features],
                breakout_shap[:, :min_features],
                metric="euclidean",
            )[0]

            breakout_subset = historical_preds.loc[valid_indices].copy()
            breakout_subset["distance"] = dists
            top_similar = breakout_subset.nsmallest(5, "distance")

            display_df = top_similar[[
                "name", "position_group", "team", "league", "season",
                "prob_calibrated", "distance",
            ]].copy()
            display_df["league"] = display_df["league"].map(league_display)
            display_df["prob_calibrated"] = (display_df["prob_calibrated"] * 100).round(1)
            display_df["distance"] = display_df["distance"].round(2)
            display_df.columns = ["Name", "Pos", "Team", "League", "Season", "Prob %", "Similarity"]
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("No matching historical breakout data for comparison.")
    else:
        st.info("Historical SHAP data not available.")
else:
    st.info("No historical breakout data available for comparison.")

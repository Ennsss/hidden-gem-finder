"""Player Profile page: SHAP waterfall, radar chart, similar players."""

import numpy as np
import pandas as pd
import streamlit as st

from utils.charts import radar_chart, shap_waterfall
from utils.data_loader import (
    get_feature_columns,
    league_display,
    load_current_features,
    load_current_predictions,
    load_current_shap,
    load_predictions,
)
from utils.shap_utils import (
    compute_percentiles,
    find_similar_players,
    get_player_features,
    get_player_shap,
)
from utils.styles import HIDE_STREAMLIT_STYLE, friendly_name, position_badge

st.set_page_config(page_title="Player Profile - Hidden Gem Finder", layout="wide")
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("Player Profile")
st.markdown("---")

# Load historical predictions (fold 1-3)
hist_preds = load_predictions()
n_historical = len(hist_preds)
feature_cols = get_feature_columns()

# Merge current feeder-league predictions (2023-25)
current = load_current_predictions()
if not current.empty:
    if "label" not in current.columns:
        current["label"] = float("nan")
    if "fold" not in current.columns:
        current["fold"] = 0
    if "age" not in current.columns and "birth_year" in current.columns:
        current["age"] = current.apply(
            lambda r: int(str(r["season"]).split("-")[0]) - int(r["birth_year"])
            if pd.notna(r.get("birth_year")) else None,
            axis=1,
        )
    shared_cols = [c for c in hist_preds.columns if c in current.columns]
    preds = pd.concat([hist_preds, current[shared_cols]], ignore_index=True)
else:
    preds = hist_preds

# --- Sidebar Filters ---
st.sidebar.header("Filters")

leagues = sorted(preds["league"].dropna().unique())
league_options = ["All"] + [league_display(l) for l in leagues]
selected_league = st.sidebar.selectbox("League", league_options)

positions = sorted(preds["position_group"].dropna().unique())
pos_options = ["All"] + positions
selected_pos = st.sidebar.selectbox("Position", pos_options)

seasons = sorted(preds["season"].dropna().unique())
season_options = ["All"] + list(seasons)
selected_season = st.sidebar.selectbox("Season", season_options)

age_min, age_max = st.sidebar.slider(
    "Age Range", min_value=15, max_value=40, value=(15, 40),
)

min_prob = st.sidebar.slider(
    "Min Probability", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
)

# Apply filters
filtered = preds.copy()
if selected_league != "All":
    slug = [l for l in leagues if league_display(l) == selected_league]
    if slug:
        filtered = filtered[filtered["league"] == slug[0]]

if selected_pos != "All":
    filtered = filtered[filtered["position_group"] == selected_pos]

if selected_season != "All":
    filtered = filtered[filtered["season"] == selected_season]

if filtered["age"].notna().any():
    filtered = filtered[
        (filtered["age"] >= age_min) & (filtered["age"] <= age_max) | filtered["age"].isna()
    ]

filtered = filtered[filtered["prob_calibrated"] >= min_prob]

if filtered.empty:
    st.warning("No players match the selected filters.")
    st.stop()

# --- Player Search ---
name_search = st.text_input("Search by name", placeholder="Type a player name...")

if name_search:
    name_mask = filtered["name"].str.contains(name_search, case=False, na=False)
    filtered = filtered[name_mask]
    if filtered.empty:
        st.warning(f"No players found matching '{name_search}'.")
        st.stop()

# Sort by probability and build display options
filtered_sorted = filtered.sort_values("prob_calibrated", ascending=False).reset_index()
display_options = [
    f"{row['name']} ({row['team']}, {row['season']})"
    for _, row in filtered_sorted.iterrows()
]

selected = st.selectbox("Select a player", options=display_options, index=0)
sorted_idx = display_options.index(selected)
player_row = filtered_sorted.iloc[sorted_idx]
player_idx = int(player_row["index"])  # original index in merged preds
player = preds.iloc[player_idx]
is_current = player_idx >= n_historical

# --- Header Card ---
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    pos_group = player.get("position_group", "")
    badge = position_badge(pos_group) if pos_group else ""
    if pd.isna(player["label"]):
        outcome = "TBD"
        outcome_color = "#6C757D"
    elif player["label"] == 1.0:
        outcome = "Breakout"
        outcome_color = "#52B788"
    else:
        outcome = "No breakout"
        outcome_color = "#ADB5BD"
    st.markdown(
        f"### {player['name']} {badge}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**{player['position']}** | {player['team']} | "
        f"{league_display(player['league'])} | {player['season']}"
    )
    if pd.notna(player.get("age")):
        st.markdown(f"Age: **{int(player['age'])}**")

with col2:
    prob = player["prob_calibrated"]
    st.metric("Breakout Probability", f"{prob:.1%}")

with col3:
    st.markdown(
        f'<div style="padding:1rem;text-align:center;background:{outcome_color};'
        f'color:white;border-radius:8px;font-weight:600;font-size:1.1rem;">'
        f'{outcome}</div>',
        unsafe_allow_html=True,
    )
    if player["label"] == 1.0 and pd.notna(player.get("breakout_league")):
        st.markdown(f"Moved to: **{league_display(player['breakout_league'])}**")

st.markdown("---")

# --- Load SHAP and Features based on source ---
if is_current:
    current_shap_all, current_feat_names = load_current_shap()
    current_idx = player_idx - n_historical
    if current_shap_all.size > 0 and current_idx < len(current_shap_all):
        shap_vals = current_shap_all[current_idx]
    else:
        shap_vals = np.array([])
    base_val = 0.5

    current_feats_df = load_current_features()
    if not current_feats_df.empty:
        feat_match = current_feats_df[
            (current_feats_df["player_id"] == player["player_id"]) &
            (current_feats_df["season"] == player["season"])
        ]
        if len(feat_match) == 0 and current_idx < len(current_feats_df):
            feat_match = current_feats_df.iloc[[current_idx]]
        if len(feat_match) > 0:
            active_feat_cols = current_feat_names if current_feat_names else feature_cols
            player_feats = np.array([
                float(feat_match.iloc[0].get(col, np.nan))
                for col in active_feat_cols
            ])
        else:
            active_feat_cols = feature_cols
            player_feats = np.full(len(feature_cols), np.nan)
    else:
        active_feat_cols = feature_cols
        player_feats = np.full(len(feature_cols), np.nan)
else:
    shap_vals, base_val = get_player_shap(player_idx)
    player_feats = get_player_features(player_idx, hist_preds)
    active_feat_cols = feature_cols

# --- SHAP Waterfall ---
st.subheader("Feature Contributions (SHAP)")

if shap_vals.size > 0:
    fig = shap_waterfall(shap_vals, player_feats, active_feat_cols, base_val, top_n=12)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("SHAP values not available for this player.")

# --- Radar Chart ---
st.subheader("Stats Radar (Percentile vs Position Group)")

radar_features_map = {
    "FW": ["goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90",
            "goal_contribution_per90", "minutes", "age_potential_factor", "naive_xg_per90"],
    "MF": ["assists_per90", "goals_per90", "goal_contribution_per90",
            "minutes", "age_potential_factor", "cards_yellow", "tackles_won", "interceptions"],
    "DF": ["tackles_won", "interceptions", "defensive_actions_per90", "minutes",
            "age_potential_factor", "cards_yellow", "cards_red", "assists_per90"],
}
radar_features = radar_features_map.get(pos_group, radar_features_map["MF"])
radar_features = [f for f in radar_features if f in active_feat_cols]

if radar_features and not np.all(np.isnan(player_feats)):
    if is_current:
        current_feats_df = load_current_features()
        pos_mask = (
            current_feats_df["position_group"] == pos_group
            if "position_group" in current_feats_df.columns
            else pd.Series([True] * len(current_feats_df))
        )
        pos_data = current_feats_df[pos_mask]
        player_percentiles = {}
        for feat in radar_features:
            feat_idx = active_feat_cols.index(feat) if feat in active_feat_cols else -1
            if feat_idx < 0 or np.isnan(player_feats[feat_idx]):
                player_percentiles[feat] = 50.0
                continue
            if feat in pos_data.columns:
                col_vals = pos_data[feat].dropna().values
                if len(col_vals) > 0:
                    player_percentiles[feat] = float(
                        np.mean(col_vals <= player_feats[feat_idx]) * 100
                    )
                else:
                    player_percentiles[feat] = 50.0
            else:
                player_percentiles[feat] = 50.0
    else:
        all_percentiles = compute_percentiles(player_idx, hist_preds, pos_group)
        player_percentiles = {f: all_percentiles.get(f, 50.0) for f in radar_features}

    avg_percentiles = {f: 50.0 for f in radar_features}
    fig = radar_chart(
        player_percentiles, avg_percentiles, f"{player['name']} vs {pos_group} Average"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Feature Details Table ---
st.subheader("Feature Details")
if not np.all(np.isnan(player_feats)):
    feat_data = []
    for i, feat in enumerate(active_feat_cols):
        shap_val = shap_vals[i] if shap_vals.size > 0 and i < len(shap_vals) else 0.0
        feat_data.append({
            "Feature": friendly_name(feat),
            "Value": round(float(player_feats[i]), 3) if not np.isnan(player_feats[i]) else None,
            "SHAP Impact": round(float(shap_val), 4),
        })
    feat_df = pd.DataFrame(feat_data)
    feat_df = feat_df.sort_values("SHAP Impact", ascending=False, key=abs).reset_index(drop=True)
    st.dataframe(feat_df, use_container_width=True, height=400, hide_index=True)

# --- Similar Players ---
st.subheader("Similar Successful Players")
st.markdown("Top 5 breakout players with most similar statistical profiles.")

if is_current:
    from scipy.spatial.distance import cdist
    from utils.shap_utils import _get_feature_matrix

    hist_feats = _get_feature_matrix(hist_preds)
    if not np.all(np.isnan(player_feats)) and hist_feats.size > 0:
        aligned_feats = np.array([
            player_feats[active_feat_cols.index(col)] if col in active_feat_cols else np.nan
            for col in feature_cols
        ])
        means = np.nanmean(hist_feats, axis=0)
        stds = np.nanstd(hist_feats, axis=0)
        stds[stds == 0] = 1.0
        player_std = np.nan_to_num((aligned_feats - means) / stds, 0.0)
        all_std = np.nan_to_num((hist_feats - means) / stds, 0.0)
        dists = cdist(player_std.reshape(1, -1), all_std, metric="euclidean")[0]

        candidates = hist_preds.copy()
        candidates["distance"] = dists
        candidates = candidates[candidates["label"] == 1.0]
        similar = candidates.nsmallest(5, "distance")[
            ["name", "position", "team", "league", "season", "prob_calibrated", "distance"]
        ].reset_index(drop=True)
    else:
        similar = pd.DataFrame()
else:
    similar = find_similar_players(player_idx, hist_preds, top_n=5, only_breakouts=True)

if not similar.empty:
    similar["league"] = similar["league"].map(league_display)
    similar["prob_calibrated"] = (similar["prob_calibrated"] * 100).round(1)
    similar["distance"] = similar["distance"].round(2)
    similar.columns = ["Name", "Pos", "Team", "League", "Season", "Prob %", "Similarity"]
    st.dataframe(similar, use_container_width=True, hide_index=True)
else:
    st.info("No similar breakout players found.")

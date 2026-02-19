"""Player Rankings page: filterable, sortable player table."""

import pandas as pd
import streamlit as st

from utils.data_loader import league_display, load_current_predictions, load_predictions
from utils.styles import HIDE_STREAMLIT_STYLE, POSITION_COLORS

st.set_page_config(page_title="Player Rankings - Hidden Gem Finder", layout="wide")
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

st.title("Player Rankings")
st.markdown("---")

preds = load_predictions()

# Merge current feeder-league predictions (2023-25)
current = load_current_predictions()
if not current.empty:
    # Add label column (unknown for current players) and align columns
    if "label" not in current.columns:
        current["label"] = float("nan")
    if "fold" not in current.columns:
        current["fold"] = 0
    # Compute age from birth_year if not present
    if "age" not in current.columns and "birth_year" in current.columns:
        current["age"] = current.apply(
            lambda r: int(str(r["season"]).split("-")[0]) - int(r["birth_year"])
            if pd.notna(r.get("birth_year")) else None,
            axis=1,
        )
    shared_cols = [c for c in preds.columns if c in current.columns]
    preds = pd.concat([preds, current[shared_cols]], ignore_index=True)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

leagues = sorted(preds["league"].unique())
league_options = ["All"] + [league_display(l) for l in leagues]
selected_league = st.sidebar.selectbox("League", league_options)

positions = sorted(preds["position_group"].dropna().unique())
pos_options = ["All"] + positions
selected_pos = st.sidebar.selectbox("Position", pos_options)

seasons = sorted(preds["season"].unique())
season_options = ["All"] + seasons
selected_season = st.sidebar.selectbox("Season", season_options)

age_min, age_max = st.sidebar.slider(
    "Age Range",
    min_value=15, max_value=40,
    value=(15, 40),
)

min_prob = st.sidebar.slider(
    "Min Probability",
    min_value=0.0, max_value=1.0,
    value=0.0, step=0.05,
)

# --- Apply Filters ---
df = preds.copy()
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

# --- Name Search ---
name_search = st.text_input("Search by name", placeholder="Type a player name...")
if name_search:
    df = df[df["name"].str.contains(name_search, case=False, na=False)]

# --- Summary bar ---
n_matching = len(df)
n_breakouts = int(df["label"].sum()) if df["label"].notna().any() else 0
st.markdown(
    f"**{n_matching}** players matching filters | "
    f"**{n_breakouts}** actual breakouts ({100*n_breakouts/n_matching:.1f}% rate)"
    if n_matching > 0 else "No players match the selected filters."
)

if n_matching > 0:
    # Build display table
    display = df.sort_values("prob_calibrated", ascending=False).reset_index(drop=True)
    display["rank"] = range(1, len(display) + 1)
    display["league_name"] = display["league"].map(league_display)
    display["outcome"] = display["label"].map({1.0: "Breakout", 0.0: "-"}).fillna("TBD")
    display["prob_pct"] = (display["prob_calibrated"] * 100).round(1)

    show_cols = {
        "rank": "Rank",
        "name": "Name",
        "position": "Pos",
        "team": "Team",
        "league_name": "League",
        "season": "Season",
        "age": "Age",
        "prob_pct": "Prob %",
        "outcome": "Outcome",
    }

    st.dataframe(
        display[list(show_cols.keys())].rename(columns=show_cols),
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "Prob %": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%",
            ),
            "Age": st.column_config.NumberColumn(format="%d"),
        },
    )

    # Tip for Player Profile
    st.info(
        "To see a detailed player profile with SHAP explanations, "
        "go to the **Player Profile** page and search for a player name."
    )

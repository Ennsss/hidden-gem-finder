"""CSS and styling constants for the scouting dashboard."""

# Color palette
PRIMARY = "#1B4332"
ACCENT = "#52B788"
ACCENT_LIGHT = "#95D5B2"
BG = "#F8F9FA"
BG_CARD = "#FFFFFF"
TEXT = "#1B4332"
TEXT_MUTED = "#6C757D"

# Position badge colors
POSITION_COLORS = {
    "FW": "#DC3545",
    "MF": "#0D6EFD",
    "DF": "#198754",
}

# Human-readable feature names for non-technical users
FEATURE_DISPLAY_NAMES = {
    # Basic stats
    "age": "Age",
    "minutes": "Minutes Played",
    "goals": "Goals",
    "assists": "Assists",
    "goals_assists": "Goals + Assists",
    "shots": "Total Shots",
    "shots_on_target": "Shots on Target",
    "tackles_won": "Tackles Won",
    "interceptions": "Interceptions",
    "cards_yellow": "Yellow Cards",
    "cards_red": "Red Cards",
    # Per-90 stats
    "goals_per90": "Goals per 90 min",
    "assists_per90": "Assists per 90 min",
    "shots_per90": "Shots per 90 min",
    "shots_on_target_per90": "Shots on Target per 90",
    "goal_contribution_per90": "Goal Contributions per 90",
    "defensive_actions_per90": "Defensive Actions per 90",
    # Derived features
    "shots_on_target_pct": "Shot Accuracy %",
    "goals_per_shot_on_target": "Conversion Rate",
    "naive_xg_per90": "Expected Goals per 90",
    "proxy_xg_per90": "Proxy xG per 90",
    "proxy_xg_overperformance": "xG Overperformance",
    # Career / trend features
    "career_avg_minutes": "Career Avg Minutes",
    "career_avg_minutes_per90": "Career Avg Min per 90",
    "career_avg_assists_per90": "Career Avg Assists per 90",
    "minutes_trend": "Minutes Trend",
    "goals_trend": "Goals Trend",
    # Comparative features
    "shot_volume_above_avg": "Shot Volume vs Avg",
    "goals_above_avg": "Goals vs Avg",
    "fw_goals_above_avg": "FW Goals vs Avg",
    # Age-related
    "age_potential_factor": "Youth Potential",
    "age_x_minutes_growth": "Age x Minutes Growth",
    # Position flag
    "is_defender": "Is Defender",
    # League-adjusted
    "goals_per90_league_adj": "Goals per 90 (League Adj)",
    "assists_per90_league_adj": "Assists per 90 (League Adj)",
    "shots_on_target_per90_league_adj": "Shots on Target per 90 (League Adj)",
    # Understat xG
    "us_xg_per90": "Understat xG per 90",
    "us_xa_per90": "Understat xA per 90",
    "us_npxg_per90": "Understat npxG per 90",
    "us_xg_overperformance": "Understat xG Overperformance",
    "us_npxg_xa_per90": "Understat npxG+xA per 90",
    "us_key_passes": "Key Passes",
    "us_shots": "Understat Shots",
}

# Plotly template overrides
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", color=TEXT),
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    colorway=[ACCENT, "#2D6A4F", "#40916C", "#74C69D", ACCENT_LIGHT, "#B7E4C7"],
    margin=dict(l=40, r=20, t=40, b=40),
)

HIDE_STREAMLIT_STYLE = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .block-container {padding-top: 1rem;}
</style>
"""


def friendly_name(feature_name: str) -> str:
    """Convert internal feature name to human-readable display name."""
    if feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature_name]
    # Fallback: replace underscores and title-case
    return feature_name.replace("_", " ").replace("per90", "per 90").title()


def position_badge(pos_group: str) -> str:
    """Return an HTML badge for a position group."""
    color = POSITION_COLORS.get(pos_group, TEXT_MUTED)
    return (
        f'<span style="background-color:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600;">{pos_group}</span>'
    )


def metric_card_css() -> str:
    """Return CSS for metric cards."""
    return """
    <style>
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #52B788;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1B4332;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #6C757D;
        margin-top: 0.2rem;
    }
    </style>
    """


def metric_card_html(label: str, value: str) -> str:
    """Return HTML for a single metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

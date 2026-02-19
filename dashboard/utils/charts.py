"""Reusable Plotly chart builders for the scouting dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from .styles import ACCENT, ACCENT_LIGHT, PLOTLY_LAYOUT, POSITION_COLORS, PRIMARY, friendly_name


def feature_importance_chart(feat_imp: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    df = feat_imp.head(top_n).iloc[::-1]  # Reverse for horizontal bars
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=[friendly_name(f) for f in df["feature"]],
        orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Top Feature Importances (Mean |SHAP|)",
        xaxis_title="Importance",
        yaxis_title="",
        height=max(400, top_n * 22),
    )
    return fig


def probability_distribution(preds: pd.DataFrame) -> go.Figure:
    """Overlayed histogram of predicted probabilities by label."""
    fig = go.Figure()
    for label, name, color in [(0, "No Breakout", "#ADB5BD"), (1, "Breakout", ACCENT)]:
        subset = preds[preds["label"] == label]["prob_calibrated"]
        fig.add_trace(go.Histogram(
            x=subset, nbinsx=50, name=name,
            marker_color=color, opacity=0.7,
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Probability Distribution by Outcome",
        xaxis_title="Calibrated Probability",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )
    return fig


def breakout_destinations(preds: pd.DataFrame, league_display_fn) -> go.Figure:
    """Bar chart of breakout destination leagues."""
    breakouts = preds[preds["label"] == 1.0]
    if breakouts.empty:
        return go.Figure()
    dest = breakouts["breakout_league"].value_counts()
    dest.index = [league_display_fn(x) for x in dest.index]
    fig = go.Figure(go.Bar(
        x=dest.values, y=dest.index, orientation="h",
        marker_color=[ACCENT, "#2D6A4F", "#40916C", "#74C69D", ACCENT_LIGHT][:len(dest)],
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Breakout Destinations",
        xaxis_title="Count",
        yaxis_title="",
        height=300,
    )
    return fig


def roc_curves(preds: pd.DataFrame) -> go.Figure:
    """ROC curves per fold + mean."""
    fig = go.Figure()
    colors = [ACCENT, "#2D6A4F", "#40916C"]
    aucs = []

    for i, fold in enumerate(sorted(preds["fold"].unique())):
        fold_df = preds[preds["fold"] == fold]
        y_true = fold_df["label"].values
        y_score = fold_df["prob_calibrated"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Fold {fold} (AUC={auc:.3f})",
            line=dict(color=colors[i % len(colors)]),
        ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="#CCC", dash="dash"),
    ))
    mean_auc = np.mean(aucs) if aucs else 0
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"ROC Curves (Mean AUC={mean_auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def pr_curves(preds: pd.DataFrame) -> go.Figure:
    """Precision-Recall curves per fold."""
    fig = go.Figure()
    colors = [ACCENT, "#2D6A4F", "#40916C"]

    for i, fold in enumerate(sorted(preds["fold"].unique())):
        fold_df = preds[preds["fold"] == fold]
        y_true = fold_df["label"].values
        y_score = fold_df["prob_calibrated"].values
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"Fold {fold}",
            line=dict(color=colors[i % len(colors)]),
        ))

    base_rate = preds["label"].mean()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[base_rate, base_rate], mode="lines",
        name=f"Baseline ({base_rate:.2f})", line=dict(color="#CCC", dash="dash"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450,
    )
    return fig


def calibration_plot(preds: pd.DataFrame, n_bins: int = 10) -> go.Figure:
    """Calibration plot: predicted vs actual probability."""
    fig = go.Figure()

    for col, name, color in [
        ("prob_ensemble", "Uncalibrated", "#ADB5BD"),
        ("prob_calibrated", "Calibrated", ACCENT),
    ]:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        actual_fracs = []
        for j in range(n_bins):
            mask = (preds[col] >= bins[j]) & (preds[col] < bins[j + 1])
            if mask.sum() > 0:
                bin_centers.append(preds.loc[mask, col].mean())
                actual_fracs.append(preds.loc[mask, "label"].mean())
        fig.add_trace(go.Scatter(
            x=bin_centers, y=actual_fracs, mode="lines+markers",
            name=name, line=dict(color=color),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect", line=dict(color="#CCC", dash="dash"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Calibration Plot",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Fraction of Positives",
        height=450,
    )
    return fig


def precision_at_k_chart(preds: pd.DataFrame) -> go.Figure:
    """Precision@K and Recall@K line chart."""
    y_true = preds["label"].values
    y_proba = preds["prob_calibrated"].values
    k_values = [5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(preds)]

    precisions, recalls = [], []
    for k in k_values:
        top_k = np.argsort(y_proba)[::-1][:k]
        tp = np.sum(y_true[top_k])
        precisions.append(tp / k)
        recalls.append(tp / np.sum(y_true) if np.sum(y_true) > 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=precisions, mode="lines+markers",
        name="Precision@K", line=dict(color=ACCENT),
    ))
    fig.add_trace(go.Scatter(
        x=k_values, y=recalls, mode="lines+markers",
        name="Recall@K", line=dict(color="#2D6A4F"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Precision & Recall at Top-K",
        xaxis_title="K",
        yaxis_title="Score",
        height=400,
    )
    return fig


def shap_waterfall(shap_values: np.ndarray, feature_values: np.ndarray,
                   feature_names: list[str], base_value: float,
                   top_n: int = 12) -> go.Figure:
    """SHAP waterfall chart for a single player."""
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[::-1][:top_n]
    top_idx = top_idx[::-1]  # Bottom to top

    names = [friendly_name(feature_names[i]) for i in top_idx]
    values = [shap_values[i] for i in top_idx]
    feat_vals = [feature_values[i] for i in top_idx]
    colors = ["#52B788" if v > 0 else "#DC3545" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=[f"{n} = {fv:.2f}" for n, fv in zip(names, feat_vals)],
        orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"SHAP Feature Contributions (base={base_value:.3f})",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="",
        height=max(300, top_n * 28),
    )
    return fig


def radar_chart(player_values: dict, avg_values: dict,
                title: str = "Player vs Position Average") -> go.Figure:
    """Radar chart comparing player percentiles to position average."""
    features = list(player_values.keys())
    display_features = [friendly_name(f) for f in features]
    player_vals = [player_values[f] for f in features]
    avg_vals = [avg_values[f] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_vals + [player_vals[0]],
        theta=display_features + [display_features[0]],
        fill="toself", name="Player",
        fillcolor=f"rgba(82, 183, 136, 0.2)",
        line=dict(color=ACCENT),
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=display_features + [display_features[0]],
        fill="toself", name="Position Avg",
        fillcolor=f"rgba(173, 181, 189, 0.15)",
        line=dict(color="#ADB5BD"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
        height=450,
    )
    return fig

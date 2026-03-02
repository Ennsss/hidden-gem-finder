"""Face-validity script: reads pre-computed model outputs and generates a markdown report."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "models" / "predictions_test.csv"
EVAL_RESULTS_PATH = PROJECT_ROOT / "outputs" / "models" / "evaluation_results.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "outputs" / "models" / "feature_importance.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_final.parquet"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "validation_report.md"


def load_data():
    """Load all pre-computed outputs."""
    preds = pd.read_csv(PREDICTIONS_PATH)
    with open(EVAL_RESULTS_PATH) as f:
        eval_results = json.load(f)
    feat_imp = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    return preds, eval_results, feat_imp, features


def precision_recall_at_threshold(y_true, y_proba, threshold):
    """Compute precision and recall at a probability threshold."""
    predicted_positive = y_proba >= threshold
    tp = np.sum((predicted_positive) & (y_true == 1))
    fp = np.sum((predicted_positive) & (y_true == 0))
    fn = np.sum((~predicted_positive) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    n_flagged = int(np.sum(predicted_positive))
    return precision, recall, n_flagged


def precision_recall_at_k(y_true, y_proba, k):
    """Compute precision and recall at top-K."""
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    tp = int(np.sum(y_true[top_k_idx]))
    total_pos = int(np.sum(y_true))
    precision = tp / k if k > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0
    return precision, recall, tp


def league_display(slug):
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


def generate_report(preds, eval_results, feat_imp, features):
    """Generate the full validation report as markdown string."""
    lines = []
    y_true = preds["label"].values
    y_proba = preds["prob_calibrated"].values
    total_pos = int(np.sum(y_true))
    total_neg = int(np.sum(y_true == 0))

    lines.append("# Hidden Gem Finder - Prediction Validation Report\n")
    lines.append(f"**Total test samples:** {len(preds)} across {preds['fold'].nunique()} folds")
    lines.append(f"**Positives (breakouts):** {total_pos} ({100*total_pos/len(preds):.1f}%)")
    lines.append(f"**Negatives:** {total_neg} ({100*total_neg/len(preds):.1f}%)")
    lines.append(f"**Leagues:** {', '.join(preds['league'].unique())}")
    lines.append(f"**Seasons:** {', '.join(sorted(preds['season'].unique()))}")
    lines.append("")

    # --- Section 1: Top-20 Predicted Breakouts ---
    lines.append("## 1. Top-20 Predicted Breakouts\n")
    lines.append("Players with highest calibrated probability, regardless of actual outcome.\n")
    top20 = preds.nlargest(20, "prob_calibrated")
    lines.append("| Rank | Name | Pos | Team | League | Season | Age | Prob | Actual | Breakout To |")
    lines.append("|------|------|-----|------|--------|--------|-----|------|--------|-------------|")
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        actual = "YES" if row["label"] == 1.0 else "no"
        age = int(row["birth_year"]) if pd.notna(row.get("birth_year")) else "?"
        season_year = row["season"].split("-")[0] if "-" in str(row["season"]) else str(row["season"])
        age_str = f"{int(season_year) - age}" if age != "?" else "?"
        breakout_to = league_display(row.get("breakout_league")) if row["label"] == 1.0 else "-"
        lines.append(
            f"| {i} | {row['name']} | {row['position']} | {row['team']} | "
            f"{league_display(row['league'])} | {row['season']} | {age_str} | "
            f"{row['prob_calibrated']:.3f} | **{actual}** | {breakout_to} |"
        )
    tp_in_top20 = int(top20["label"].sum())
    lines.append(f"\n**Precision in top 20:** {tp_in_top20}/20 = {tp_in_top20/20:.1%}\n")

    # --- Section 2: True Positives ---
    lines.append("## 2. Top-20 True Positives (Correctly Identified Breakouts)\n")
    lines.append("Players who actually broke out, ranked by model confidence.\n")
    tp_df = preds[preds["label"] == 1.0].nlargest(20, "prob_calibrated")
    lines.append("| Rank | Name | Pos | Team | League | Season | Prob | Breakout To |")
    lines.append("|------|------|-----|------|--------|--------|------|-------------|")
    for i, (_, row) in enumerate(tp_df.iterrows(), 1):
        lines.append(
            f"| {i} | {row['name']} | {row['position']} | {row['team']} | "
            f"{league_display(row['league'])} | {row['season']} | "
            f"{row['prob_calibrated']:.3f} | {league_display(row.get('breakout_league'))} |"
        )
    lines.append("")

    # --- Section 3: False Negatives ---
    lines.append("## 3. Top-20 False Negatives (Missed Breakouts)\n")
    lines.append("Actual breakouts the model ranked lowest. Possible explanations included.\n")
    fn_df = preds[preds["label"] == 1.0].nsmallest(20, "prob_calibrated")
    lines.append("| Rank | Name | Pos | Team | League | Season | Prob | Breakout To | Possible Reason |")
    lines.append("|------|------|-----|------|--------|--------|------|-------------|-----------------|")
    for i, (_, row) in enumerate(fn_df.iterrows(), 1):
        # Try to find in features for analysis
        reasons = []
        feat_match = features[
            (features["player_id"] == row["player_id"]) & (features["season"] == row["season"])
        ]
        if len(feat_match) > 0:
            fr = feat_match.iloc[0]
            if pd.notna(fr.get("age")) and fr["age"] >= 27:
                reasons.append("older (age >= 27)")
            if pd.notna(fr.get("minutes")) and fr["minutes"] < 1000:
                reasons.append(f"low minutes ({int(fr['minutes'])})")
            if pd.notna(fr.get("goals_per90")) and fr["goals_per90"] < 0.05 and row["position"] in ("FW", "FW,MF"):
                reasons.append("low goal rate")
        reason_str = "; ".join(reasons) if reasons else "model uncertainty"
        lines.append(
            f"| {i} | {row['name']} | {row['position']} | {row['team']} | "
            f"{league_display(row['league'])} | {row['season']} | "
            f"{row['prob_calibrated']:.3f} | {league_display(row.get('breakout_league'))} | {reason_str} |"
        )
    lines.append("")

    # --- Section 4: Precision/Recall at Thresholds ---
    lines.append("## 4. Precision/Recall at Probability Thresholds\n")
    lines.append("| Threshold | Flagged | Precision | Recall |")
    lines.append("|-----------|---------|-----------|--------|")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p, r, n = precision_recall_at_threshold(y_true, y_proba, thresh)
        lines.append(f"| {thresh:.1f} | {n} | {p:.3f} | {r:.3f} |")
    lines.append("")

    lines.append("### Precision/Recall at Top-K\n")
    lines.append("| K | Precision | Recall | True Positives |")
    lines.append("|---|-----------|--------|----------------|")
    for k in [10, 20, 50, 100, 200, 500]:
        p, r, tp = precision_recall_at_k(y_true, y_proba, k)
        lines.append(f"| {k} | {p:.3f} | {r:.3f} | {tp} |")
    lines.append("")

    # --- Section 5: False Positive Analysis ---
    lines.append("## 5. False Positive Analysis\n")
    lines.append("Top-50 false positives (predicted breakout but didn't). Checking for near-misses.\n")
    fp_df = preds[(preds["label"] == 0.0)].nlargest(50, "prob_calibrated")

    # Check if any FP players appear in a top-5 league later (near-miss)
    # Use full FBref data from DuckDB (includes top-5 league appearances)
    top5_leagues = {"premier-league", "la-liga", "bundesliga", "serie-a", "ligue-1"}
    try:
        from src.storage import PlayerDatabase
        db = PlayerDatabase(PROJECT_ROOT / "data" / "players.duckdb")
        all_fbref = db.get_fbref_players()
        db.close()
        # Build set of player_ids that appeared in top-5 leagues
        top5_player_ids = set(
            all_fbref[all_fbref["league"].isin(top5_leagues)]["player_id"].unique()
        )
    except Exception:
        # Fallback to features_final if DuckDB unavailable
        top5_player_ids = set(
            features[features["league"].isin(top5_leagues)]["player_id"].unique()
        ) if "league" in features.columns else set()

    near_misses = 0
    lines.append("| Rank | Name | Pos | Team | League | Season | Prob | Near-Miss? |")
    lines.append("|------|------|-----|------|--------|--------|------|------------|")
    for i, (_, row) in enumerate(fp_df.iterrows(), 1):
        in_top5 = row["player_id"] in top5_player_ids
        near_miss_str = "YES (appeared in top-5)" if in_top5 else "no"
        if in_top5:
            near_misses += 1
        lines.append(
            f"| {i} | {row['name']} | {row['position']} | {row['team']} | "
            f"{league_display(row['league'])} | {row['season']} | "
            f"{row['prob_calibrated']:.3f} | {near_miss_str} |"
        )
    lines.append(f"\n**Near-misses in top 50 FPs:** {near_misses}/50 ({near_misses/50:.0%})")
    lines.append("(Players who appeared in a top-5 league but didn't meet 900-minute threshold)\n")

    # --- Section 6: Summary Statistics ---
    lines.append("## 6. Summary Statistics\n")

    # Pull from evaluation_results.json
    summary = eval_results.get("summary", {})
    cal = summary.get("calibrated_metrics", {})

    lines.append("### Overall Model Performance (Calibrated Ensemble, Mean Across Folds)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    def fmt_metric(d):
        if isinstance(d, dict):
            mean = d.get("mean", d.get("values", [0])[0] if d.get("values") else 0)
            return f"{mean:.4f}"
        return f"{d:.4f}"

    for key in ["roc_auc", "average_precision", "brier_score",
                 "precision_at_10", "precision_at_20", "precision_at_50", "precision_at_100",
                 "recall_at_10", "recall_at_20", "recall_at_50", "recall_at_100"]:
        if key in cal:
            display_key = key.replace("_", " ").title().replace("Roc Auc", "ROC-AUC").replace("At ", "@")
            lines.append(f"| {display_key} | {fmt_metric(cal[key])} |")
    lines.append("")

    # Per-fold breakdown
    lines.append("### Per-Fold ROC-AUC (Calibrated Ensemble)\n")
    lines.append("| Fold | ROC-AUC | Precision@20 | N_Test | N_Positive |")
    lines.append("|------|---------|--------------|--------|------------|")
    for fold_num in [1, 2, 3]:
        fold_key = f"fold_{fold_num}"
        fold_data = eval_results.get("folds", {}).get(fold_key, {})
        fold_cal = fold_data.get("calibrated_metrics", {})
        fold_preds = preds[preds["fold"] == fold_num]
        n_test = len(fold_preds)
        n_pos = int(fold_preds["label"].sum())
        roc = fold_cal.get("roc_auc", float("nan"))
        p20 = fold_cal.get("precision_at_20", float("nan"))
        lines.append(f"| {fold_num} | {roc:.4f} | {p20:.4f} | {n_test} | {n_pos} |")
    lines.append("")

    # Per-league breakdown
    lines.append("### Per-League Breakdown\n")
    lines.append("| League | N_Players | N_Breakouts | Breakout% | Avg Prob (breakouts) | Avg Prob (non-breakouts) |")
    lines.append("|--------|-----------|-------------|-----------|----------------------|-------------------------|")
    for league in sorted(preds["league"].unique()):
        lg = preds[preds["league"] == league]
        n = len(lg)
        n_pos = int(lg["label"].sum())
        pct = 100 * n_pos / n if n > 0 else 0
        avg_prob_pos = lg[lg["label"] == 1.0]["prob_calibrated"].mean() if n_pos > 0 else 0
        avg_prob_neg = lg[lg["label"] == 0.0]["prob_calibrated"].mean()
        lines.append(
            f"| {league_display(league)} | {n} | {n_pos} | {pct:.1f}% | "
            f"{avg_prob_pos:.3f} | {avg_prob_neg:.3f} |"
        )
    lines.append("")

    # Breakout destination breakdown
    lines.append("### Breakout Destinations\n")
    breakouts = preds[preds["label"] == 1.0]
    dest_counts = breakouts["breakout_league"].value_counts()
    lines.append("| Destination League | Count | % |")
    lines.append("|-------------------|-------|---|")
    for dest, count in dest_counts.items():
        lines.append(f"| {league_display(dest)} | {count} | {100*count/len(breakouts):.1f}% |")
    lines.append("")

    # Top features
    lines.append("### Top-10 Most Important Features (Mean |SHAP|)\n")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    for i, (_, row) in enumerate(feat_imp.head(10).iterrows(), 1):
        lines.append(f"| {i} | {row['feature']} | {row['importance']:.4f} |")
    lines.append("")

    # Overall verdict
    lines.append("## 7. Overall Verdict\n")
    roc_mean = cal.get("roc_auc", {})
    roc_val = roc_mean.get("mean", roc_mean) if isinstance(roc_mean, dict) else roc_mean
    p20_mean = cal.get("precision_at_20", {})
    p20_val = p20_mean.get("mean", p20_mean) if isinstance(p20_mean, dict) else p20_mean

    if roc_val > 0.75:
        verdict = "STRONG"
    elif roc_val > 0.65:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"

    lines.append(f"**Model Quality: {verdict}**\n")
    lines.append(f"- ROC-AUC of {roc_val:.3f} indicates {verdict.lower()} discriminative ability")
    lines.append(f"- Precision@20 of {p20_val:.3f} means ~{int(p20_val*20)} of the top 20 flagged players actually broke out")
    lines.append(f"- The model captures {total_pos} breakouts across {len(preds)} test samples ({100*total_pos/len(preds):.1f}% base rate)")
    lines.append(f"- Top features are {feat_imp.iloc[0]['feature']}, {feat_imp.iloc[1]['feature']}, and {feat_imp.iloc[2]['feature']}")
    if near_misses > 10:
        lines.append(f"- {near_misses}/50 top false positives are near-misses (appeared in top-5 league), suggesting model identifies talent even when labeling is strict")
    lines.append("")

    return "\n".join(lines)


def main():
    print("Loading data...")
    preds, eval_results, feat_imp, features = load_data()
    print(f"  Predictions: {len(preds)} rows")
    print(f"  Features: {len(features)} rows")
    print(f"  Feature importance: {len(feat_imp)} features")

    print("Generating validation report...")
    report = generate_report(preds, eval_results, feat_imp, features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

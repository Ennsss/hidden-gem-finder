"""Run breakout predictions on current feeder-league players.

Loads saved fold-3 models, processes 2023-25 feeder-league data through
the feature pipeline, and outputs scored predictions + SHAP explanations.

Usage:
    python scripts/predict_current.py
    python scripts/predict_current.py --seasons 2024-2025
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predictor import BreakoutPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Score current feeder-league players")
    parser.add_argument("--seasons", nargs="+", default=["2023-2024", "2024-2025"])
    parser.add_argument("--leagues", nargs="+", default=None, help="Override leagues")
    args = parser.parse_args()

    predictor = BreakoutPredictor()
    predictions, shap_values = predictor.run(seasons=args.seasons, leagues=args.leagues)

    if predictions.empty:
        print("No predictions generated. Check that feeder-league data is in the database.")
        return

    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Players scored: {len(predictions)}")
    print(f"Seasons: {predictions['season'].unique().tolist()}")
    print(f"Leagues: {predictions['league'].unique().tolist()}")
    print(f"SHAP values: {shap_values.shape}")
    print(f"\nProbability distribution:")
    print(f"  Mean:   {predictions['prob_calibrated'].mean():.3f}")
    print(f"  Median: {predictions['prob_calibrated'].median():.3f}")
    print(f"  Max:    {predictions['prob_calibrated'].max():.3f}")
    print(f"  >50%:   {(predictions['prob_calibrated'] > 0.5).sum()} players")
    print(f"  >25%:   {(predictions['prob_calibrated'] > 0.25).sum()} players")

    print(f"\nTop 20 predicted breakouts:")
    top = predictions.head(20)
    for _, row in top.iterrows():
        name = row.get("name", "?")
        team = row.get("team", "?")
        league = row.get("league", "?")
        season = row.get("season", "?")
        prob = row["prob_calibrated"]
        pos = row.get("position_group", "?")
        print(f"  {prob:.1%}  {name:25s} {pos:3s}  {team:25s} {league} {season}")


if __name__ == "__main__":
    main()

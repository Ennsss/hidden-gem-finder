"""Scrape all leagues and seasons needed for model training.

Covers:
- 4 feeder leagues (priority 1): eredivisie, primeira-liga, belgian-pro-league, championship
- 5 target leagues: premier-league, la-liga, bundesliga, serie-a, ligue-1
- 6 seasons: 2017-2018 through 2022-2023

Walk-forward folds need:
  Fold 1: train <=2018-19, val 2019-20, test 2020-21
  Fold 2: train <=2019-20, val 2020-21, test 2021-22
  Fold 3: train <=2020-21, val 2021-22, test 2022-23
"""

import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.data.pipeline import DataPipeline

LEAGUES = [
    # Feeder leagues (priority 1)
    "eredivisie",
    "primeira-liga",
    "belgian-pro-league",
    "championship",
    # Target leagues (top 5)
    "premier-league",
    "la-liga",
    "bundesliga",
    "serie-a",
    "ligue-1",
]

SEASONS = [
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
]

# FBref needs a higher rate limit (Cloudflare protection)
FBREF_RATE_LIMIT = 5.0
DEFAULT_RATE_LIMIT = 3.0


def format_eta(elapsed: float, completed: int, total: int) -> str:
    """Format estimated time remaining."""
    if completed == 0:
        return "calculating..."
    avg_per_item = elapsed / completed
    remaining = (total - completed) * avg_per_item
    if remaining < 60:
        return f"{remaining:.0f}s"
    elif remaining < 3600:
        return f"{remaining / 60:.1f}m"
    else:
        return f"{remaining / 3600:.1f}h"


def main():
    total_combinations = len(LEAGUES) * len(SEASONS)
    print(f"Scraping {len(LEAGUES)} leagues x {len(SEASONS)} seasons = {total_combinations} combinations")
    print(f"Leagues: {', '.join(LEAGUES)}")
    print(f"Seasons: {', '.join(SEASONS)}")
    print(f"FBref rate limit: {FBREF_RATE_LIMIT}s | Others: {DEFAULT_RATE_LIMIT}s")
    print()

    pipeline = DataPipeline(
        db_path="data/players.duckdb",
        cache_dir="data/raw",
        rate_limit=DEFAULT_RATE_LIMIT,
    )
    # Override FBref rate limit to be higher (Cloudflare)
    pipeline.fbref.rate_limit = FBREF_RATE_LIMIT

    completed = 0
    errors = []
    source_counts = {"fbref": 0, "transfermarkt": 0, "understat": 0}
    start = time.time()

    for season in SEASONS:
        for league in LEAGUES:
            completed += 1
            elapsed = time.time() - start
            eta = format_eta(elapsed, completed - 1, total_combinations)
            print(f"\n[{completed}/{total_combinations}] {league} {season} (ETA: {eta})")

            try:
                result = pipeline.scrape_league_season(league, season)
                source_counts["fbref"] += result.fbref_count
                source_counts["transfermarkt"] += result.transfermarkt_count
                source_counts["understat"] += result.understat_count
                print(f"  -> {result}")
                if result.errors:
                    for e in result.errors:
                        errors.append(f"{league} {season}: {e}")
            except Exception as e:
                msg = f"{league} {season}: FATAL - {e}"
                print(f"  -> ERROR: {e}")
                errors.append(msg)

    elapsed = time.time() - start
    stats = pipeline.get_scrape_summary()
    pipeline.close()

    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Total records: {stats['total_records']:,}")
    print(f"  FBref: {stats['fbref_players']:,}")
    print(f"  Transfermarkt: {stats['transfermarkt_players']:,}")
    print(f"  Understat: {stats['understat_players']:,}")
    print(f"Leagues: {len(stats['leagues'])}")
    print(f"Seasons: {len(stats['seasons'])}")

    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nNo errors!")

    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())

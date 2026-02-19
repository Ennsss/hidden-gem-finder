"""Scrape feeder-league data for 2023-24 and 2024-25 seasons.

Targets 4 priority-1 feeder leagues:
  - Championship, Eredivisie, Primeira Liga, Belgian Pro League

Scrapes FBref (player stats) and Transfermarkt (market values).
Understat is skipped for feeder leagues (only covers top-5 leagues).

Usage:
    python scripts/scrape_recent_seasons.py
    python scripts/scrape_recent_seasons.py --seasons 2024-2025
    python scripts/scrape_recent_seasons.py --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scrape_log.txt", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

FEEDER_LEAGUES = [
    "championship",
    "eredivisie",
    "primeira-liga",
    "belgian-pro-league",
]

SEASONS = ["2023-2024", "2024-2025"]

# Feeder leagues are NOT on Understat — only scrape FBref + TM
SOURCES = ["fbref", "transfermarkt"]


def main():
    parser = argparse.ArgumentParser(description="Scrape feeder leagues for recent seasons")
    parser.add_argument("--seasons", nargs="+", default=SEASONS, help="Seasons to scrape")
    parser.add_argument("--leagues", nargs="+", default=FEEDER_LEAGUES, help="Leagues to scrape")
    parser.add_argument("--sources", nargs="+", default=SOURCES, help="Data sources")
    parser.add_argument("--rate-limit", type=float, default=4.0, help="Seconds between requests")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be scraped")
    args = parser.parse_args()

    total_jobs = len(args.leagues) * len(args.seasons)
    logger.info(f"Scraping plan: {len(args.leagues)} leagues x {len(args.seasons)} seasons = {total_jobs} jobs")
    logger.info(f"Leagues: {args.leagues}")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Sources: {args.sources}")

    if args.dry_run:
        for season in args.seasons:
            for league in args.leagues:
                print(f"  Would scrape: {league} {season} ({', '.join(args.sources)})")
        return

    pipeline = DataPipeline(rate_limit=args.rate_limit)

    results = []
    total_start = time.time()

    try:
        for season in args.seasons:
            for league in args.leagues:
                logger.info(f"--- Scraping {league} {season} ---")
                result = pipeline.scrape_league_season(league, season, sources=args.sources)
                results.append(result)
                logger.info(str(result))

        total_time = time.time() - total_start

        # Summary
        print("\n" + "=" * 60)
        print("SCRAPING SUMMARY")
        print("=" * 60)
        total_fbref = sum(r.fbref_count for r in results)
        total_tm = sum(r.transfermarkt_count for r in results)
        total_errors = sum(len(r.errors) for r in results)

        for r in results:
            status = "OK" if r.success else "ERRORS"
            print(f"  {r.league:25s} {r.season}: FBref={r.fbref_count:4d}, TM={r.transfermarkt_count:4d} [{status}]")

        print(f"\nTotals: FBref={total_fbref}, TM={total_tm}, Errors={total_errors}")
        print(f"Time: {total_time:.0f}s ({total_time/60:.1f}min)")

        if total_errors > 0:
            print("\nErrors:")
            for r in results:
                for err in r.errors:
                    print(f"  [{r.league} {r.season}] {err}")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()

"""Scrape real data from FBref, Transfermarkt, and Understat.

Orchestrates scraping for Priority 1 source leagues + target leagues
across 8 main seasons (2015-2016 to 2022-2023) plus 2 extra target-only
seasons (2023-2024, 2024-2025) for the labeling lookforward window.

Usage:
    # Smoke test (1 league-season)
    python scripts/scrape_real_data.py --test-only

    # Full scrape (fresh DB)
    python scripts/scrape_real_data.py

    # Resume from a specific league/season after interruption
    python scripts/scrape_real_data.py --start-league championship --start-season 2019-2020

    # Append to existing DB (don't delete)
    python scripts/scrape_real_data.py --append
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scrape_log.txt", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── League/season definitions ──────────────────────────────────────────

SOURCE_LEAGUES = [
    "eredivisie",
    "primeira-liga",
    "belgian-pro-league",
    "championship",
]

TARGET_LEAGUES = [
    "premier-league",
    "la-liga",
    "bundesliga",
    "serie-a",
    "ligue-1",
]

ALL_LEAGUES = SOURCE_LEAGUES + TARGET_LEAGUES

# 8 main seasons for all leagues
MAIN_SEASONS = [f"{y}-{y+1}" for y in range(2015, 2023)]

# 2 extra seasons for target leagues only (labeling lookforward window)
EXTRA_SEASONS = ["2023-2024", "2024-2025"]


def build_job_list():
    """Build ordered list of (league, season) tuples to scrape."""
    jobs = []

    # All 9 leagues x 8 main seasons
    for league in ALL_LEAGUES:
        for season in MAIN_SEASONS:
            jobs.append((league, season))

    # Target leagues x 2 extra seasons
    for league in TARGET_LEAGUES:
        for season in EXTRA_SEASONS:
            jobs.append((league, season))

    return jobs


def find_start_index(jobs, start_league, start_season):
    """Find the index to resume from."""
    for i, (league, season) in enumerate(jobs):
        if league == start_league and season == start_season:
            return i
    logger.warning(
        f"Could not find {start_league} {start_season} in job list. Starting from beginning."
    )
    return 0


def run_smoke_test(db_path, cache_dir, rate_limit):
    """Scrape 1 league-season as a quick test."""
    test_league = "eredivisie"
    test_season = "2022-2023"

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {test_league} {test_season}")
    print(f"{'='*60}\n")

    with DataPipeline(db_path=db_path, cache_dir=cache_dir, rate_limit=rate_limit) as pipeline:
        result = pipeline.scrape_league_season(test_league, test_season)

    print(f"\nResult: {result}")
    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  - {e}")
        return False

    if result.fbref_count == 0:
        print("\nWARNING: FBref returned 0 players!")
        return False
    if result.transfermarkt_count == 0:
        print("\nWARNING: Transfermarkt returned 0 players!")
        return False

    print("\nSmoke test PASSED")
    print(f"  FBref: {result.fbref_count} players")
    print(f"  Transfermarkt: {result.transfermarkt_count} players")
    print(f"  Understat: {result.understat_count} players (expected 0 for Eredivisie)")
    return True


def run_full_scrape(db_path, cache_dir, rate_limit, start_league=None, start_season=None, append=False):
    """Run the full scrape across all league-season combos."""
    jobs = build_job_list()
    total_jobs = len(jobs)

    # Find start index for resume
    start_idx = 0
    if start_league and start_season:
        start_idx = find_start_index(jobs, start_league, start_season)
        print(f"\nResuming from job {start_idx + 1}/{total_jobs}: {start_league} {start_season}")

    # Delete existing DB unless appending
    db_file = Path(db_path)
    if not append and start_idx == 0 and db_file.exists():
        print(f"\nDeleting existing database: {db_file}")
        db_file.unlink()

    print(f"\n{'='*60}")
    print(f"FULL SCRAPE: {total_jobs - start_idx} jobs remaining (of {total_jobs} total)")
    print(f"{'='*60}")
    print(f"  Leagues: {len(ALL_LEAGUES)} ({len(SOURCE_LEAGUES)} source + {len(TARGET_LEAGUES)} target)")
    print(f"  Seasons: {len(MAIN_SEASONS)} main + {len(EXTRA_SEASONS)} extra (target only)")
    print(f"  DB: {db_path}")
    print(f"  Cache: {cache_dir}")
    print(f"  Rate limit: {rate_limit}s")
    print()

    # Track results
    completed = 0
    errors_list = []
    cumulative_fbref = 0
    cumulative_tm = 0
    cumulative_us = 0
    start_time = time.time()

    with DataPipeline(db_path=db_path, cache_dir=cache_dir, rate_limit=rate_limit) as pipeline:
        for i in range(start_idx, total_jobs):
            league, season = jobs[i]
            job_num = i + 1

            print(f"\n[{job_num}/{total_jobs}] {league} {season} ...", flush=True)

            try:
                result = pipeline.scrape_league_season(league, season)

                cumulative_fbref += result.fbref_count
                cumulative_tm += result.transfermarkt_count
                cumulative_us += result.understat_count
                completed += 1

                status = "OK" if result.success else f"{len(result.errors)} error(s)"
                print(
                    f"  {status} | FBref={result.fbref_count} TM={result.transfermarkt_count} "
                    f"US={result.understat_count} | {result.duration_seconds:.0f}s"
                )

                if result.errors:
                    for e in result.errors:
                        errors_list.append(f"{league} {season}: {e}")
                        print(f"  ERROR: {e}")

            except KeyboardInterrupt:
                print(f"\n\nInterrupted at job {job_num}. Resume with:")
                print(f"  python scripts/scrape_real_data.py --append --start-league {league} --start-season {season}")
                break
            except Exception as e:
                errors_list.append(f"{league} {season}: FATAL: {e}")
                print(f"  FATAL ERROR: {e}")
                logger.exception(f"Fatal error on {league} {season}")

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("SCRAPE SUMMARY")
    print(f"{'='*60}")
    print(f"  Completed: {completed}/{total_jobs - start_idx} jobs")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Cumulative records:")
    print(f"    FBref:        {cumulative_fbref:,}")
    print(f"    Transfermarkt: {cumulative_tm:,}")
    print(f"    Understat:    {cumulative_us:,}")
    print(f"    Total:        {cumulative_fbref + cumulative_tm + cumulative_us:,}")

    if errors_list:
        print(f"\n  Errors ({len(errors_list)}):")
        for e in errors_list:
            print(f"    - {e}")
    else:
        print("\n  No errors!")

    return len(errors_list) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Scrape real football data from FBref, Transfermarkt, and Understat"
    )
    parser.add_argument(
        "--db", default="data/players.duckdb",
        help="Path to DuckDB database (default: data/players.duckdb)"
    )
    parser.add_argument(
        "--cache", default="data/raw",
        help="Cache directory for scraped HTML (default: data/raw)"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=4.0,
        help="Seconds between requests (default: 4.0)"
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Only run smoke test (1 league-season)"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing DB (don't delete)"
    )
    parser.add_argument(
        "--start-league",
        help="Resume from this league (requires --start-season)"
    )
    parser.add_argument(
        "--start-season",
        help="Resume from this season (requires --start-league)"
    )

    args = parser.parse_args()

    # Ensure data directory exists
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache).mkdir(parents=True, exist_ok=True)

    if args.test_only:
        success = run_smoke_test(args.db, args.cache, args.rate_limit)
        return 0 if success else 1

    # Validate resume args
    if bool(args.start_league) != bool(args.start_season):
        print("ERROR: --start-league and --start-season must be used together")
        return 1

    # If resuming, force append mode
    if args.start_league:
        args.append = True

    success = run_full_scrape(
        db_path=args.db,
        cache_dir=args.cache,
        rate_limit=args.rate_limit,
        start_league=args.start_league,
        start_season=args.start_season,
        append=args.append,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

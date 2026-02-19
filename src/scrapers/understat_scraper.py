"""Understat scraper for xG/xA statistics.

Uses seleniumbase UC mode to render the page and extract playersData
from the JavaScript context (data is no longer embedded in raw HTML).
"""

import json
import logging
import time
from typing import Any

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class UnderstatScraper(BaseScraper):
    """Scraper for Understat.com xG/xA statistics.

    Understat provides detailed expected goals data for 5 European leagues:
    - Premier League (EPL)
    - La Liga
    - Bundesliga
    - Serie A
    - Ligue 1

    Note: Eredivisie was previously available but has been dropped by Understat.
    Data is loaded via JavaScript — we use a browser to extract it.
    """

    BASE_URL = "https://understat.com"

    # Map our kebab-case league keys to Understat league names
    # None means the league is not available on Understat
    LEAGUE_IDS = {
        # Eredivisie was dropped by Understat (returns 404)
        "eredivisie": None,
        # Available on Understat
        "premier-league": "EPL",
        "la-liga": "La_liga",
        "bundesliga": "Bundesliga",
        "serie-a": "Serie_A",
        "ligue-1": "Ligue_1",
        # Not available on Understat
        "primeira-liga": None,
        "belgian-pro-league": None,
        "championship": None,
        "serie-b": None,
        "ligue-2": None,
        "austrian-bundesliga": None,
        "scottish-premiership": None,
    }

    # Position code normalization — Understat uses short codes like "F", "M S", "D"
    POSITION_MAP = {
        "F": "FW",
        "M": "MF",
        "D": "DF",
        "GK": "GK",
        "S": "FW",  # Sub/Striker -> Forward
    }

    def __init__(self, **kwargs):
        self._sb_driver = None
        super().__init__(**kwargs)

    @property
    def source_name(self) -> str:
        return "understat"

    def _get_sb_driver(self):
        """Get or create the seleniumbase driver (lazy init)."""
        if self._sb_driver is None:
            from seleniumbase import Driver
            logger.info("Starting seleniumbase UC browser for Understat...")
            self._sb_driver = Driver(uc=True, headless=True)
        return self._sb_driver

    def close(self):
        """Close the seleniumbase browser if open."""
        if self._sb_driver is not None:
            try:
                self._sb_driver.quit()
            except Exception:
                pass
            self._sb_driver = None

    def __del__(self):
        self.close()

    def _convert_season_format(self, season: str) -> str:
        """Convert '2023-2024' or '2023-24' to '2023' (start year)."""
        return season.split("-")[0]

    def _build_league_url(self, league: str, season: str) -> str:
        """Build URL for league season page."""
        league_key = league.lower().replace(" ", "-")
        league_name = self.LEAGUE_IDS.get(league_key)

        if league_name is None:
            available = [k for k, v in self.LEAGUE_IDS.items() if v is not None]
            raise ValueError(
                f"League '{league}' not available on Understat. "
                f"Available leagues: {available}"
            )

        season_year = self._convert_season_format(season)
        return f"{self.BASE_URL}/league/{league_name}/{season_year}"

    def _normalize_position(self, raw_position: str) -> str:
        """Normalize Understat position codes.

        Understat uses codes like "F", "M S", "D M", "GK".
        Take the first code and map it.
        """
        if not raw_position:
            return raw_position
        first_code = raw_position.strip().split()[0]
        return self.POSITION_MAP.get(first_code, raw_position)

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for Understat JSON data."""
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        clean_url = url.replace("https://", "").replace("/", "_")[:50]
        return f"{clean_url}_{url_hash}.json"

    def _get_cached_players(self, url: str) -> list[dict] | None:
        """Check for cached player data."""
        from pathlib import Path
        cache_path = self.cache_dir / self.source_name / self._get_cache_key(url)
        if cache_path.exists():
            logger.debug(f"Cache hit: {url}")
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return None

    def _cache_players(self, url: str, players: list[dict]) -> None:
        """Cache player data to disk."""
        from pathlib import Path
        cache_path = self.cache_dir / self.source_name / self._get_cache_key(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(players), encoding="utf-8")
        logger.debug(f"Cached: {url}")

    def _extract_players_via_browser(self, url: str) -> list[dict]:
        """Load page in browser and extract playersData JS variable."""
        # Check cache
        cached = self._get_cached_players(url)
        if cached is not None:
            return cached

        self._respect_rate_limit()

        logger.info(f"Fetching (browser): {url}")
        driver = self._get_sb_driver()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                driver.get(url)
                time.sleep(3)

                # Extract playersData from JS context
                result = driver.execute_script(
                    'return typeof playersData !== "undefined" ? JSON.stringify(playersData) : null'
                )

                if result is None:
                    if attempt < max_attempts:
                        logger.warning(f"playersData not available, retrying (attempt {attempt})...")
                        time.sleep(5)
                        continue
                    logger.error("playersData not found after all attempts")
                    return []

                self._last_request_time = time.time()
                self._request_count += 1

                players = json.loads(result)
                logger.info(f"Extracted {len(players)} players via browser JS")

                # Cache for future runs
                self._cache_players(url, players)
                return players

            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(5)
                else:
                    raise

        return []

    def _safe_float(self, val: Any, default: float | None = None) -> float | None:
        """Safely convert value to float."""
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _safe_int(self, val: Any, default: int | None = None) -> int | None:
        """Safely convert value to int."""
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _parse_player_record(
        self, raw: dict, league: str, season: str
    ) -> dict[str, Any]:
        """Parse raw Understat player record to normalized format."""
        minutes = self._safe_int(raw.get("time"), 0)
        minutes_90s = minutes / 90.0 if minutes > 0 else 0.0

        goals = self._safe_int(raw.get("goals"), 0)
        assists = self._safe_int(raw.get("assists"), 0)
        xg = self._safe_float(raw.get("xG"), 0.0)
        xa = self._safe_float(raw.get("xA"), 0.0)
        npxg = self._safe_float(raw.get("npxG"), 0.0)

        position = self._normalize_position(raw.get("position", ""))

        return {
            "player_id": str(raw.get("id", "")),
            "name": raw.get("player_name", ""),
            "position": position,
            "team": raw.get("team_title"),
            "games": self._safe_int(raw.get("games")),
            "minutes": minutes,
            "minutes_90s": round(minutes_90s, 2),
            "goals": goals,
            "assists": assists,
            "npg": self._safe_int(raw.get("npg")),
            "xg": xg,
            "xa": xa,
            "npxg": npxg,
            "xg_chain": self._safe_float(raw.get("xGChain")),
            "xg_buildup": self._safe_float(raw.get("xGBuildup")),
            "xg_per90": round(xg / minutes_90s, 2) if minutes_90s > 0 else None,
            "xa_per90": round(xa / minutes_90s, 2) if minutes_90s > 0 else None,
            "npxg_per90": round(npxg / minutes_90s, 2) if minutes_90s > 0 else None,
            "goals_per90": round(goals / minutes_90s, 2) if minutes_90s > 0 else None,
            "assists_per90": round(assists / minutes_90s, 2) if minutes_90s > 0 else None,
            "shots": self._safe_int(raw.get("shots")),
            "key_passes": self._safe_int(raw.get("key_passes")),
            "yellow_cards": self._safe_int(raw.get("yellow_cards")),
            "red_cards": self._safe_int(raw.get("red_cards")),
            "xg_overperformance": round(goals - xg, 2) if xg is not None else None,
            "xa_overperformance": round(assists - xa, 2) if xa is not None else None,
            "league": league,
            "season": season,
            "source": "understat",
        }

    def scrape_league_season(
        self,
        league: str,
        season: str,
    ) -> list[dict[str, Any]]:
        """Scrape all player xG/xA stats for a league season."""
        league_key = league.lower().replace(" ", "-")

        if league_key not in self.LEAGUE_IDS:
            available = [k for k, v in self.LEAGUE_IDS.items() if v is not None]
            raise ValueError(
                f"Unknown league: {league}. "
                f"Available on Understat: {available}"
            )

        if self.LEAGUE_IDS[league_key] is None:
            available = [k for k, v in self.LEAGUE_IDS.items() if v is not None]
            raise ValueError(
                f"League '{league}' not available on Understat. "
                f"Available leagues: {available}"
            )

        logger.info(f"Scraping Understat: {league} {season}")

        url = self._build_league_url(league_key, season)
        raw_players = self._extract_players_via_browser(url)
        logger.info(f"Found {len(raw_players)} players in raw data")

        players = []
        for raw_player in raw_players:
            try:
                player_data = self._parse_player_record(raw_player, league, season)
                players.append(player_data)
            except Exception as e:
                logger.warning(
                    f"Error parsing player {raw_player.get('player_name', 'unknown')}: {e}"
                )
                continue

        logger.info(f"Scraped {len(players)} players for {league} {season}")
        return players

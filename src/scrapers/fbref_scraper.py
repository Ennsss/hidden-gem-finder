"""FBref scraper for player statistics."""

import logging
import re
from datetime import date
from typing import Any

from bs4 import BeautifulSoup

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class FBrefScraper(BaseScraper):
    """Scraper for FBref.com player statistics.

    FBref provides comprehensive football statistics including:
    - Standard stats (goals, assists, minutes)
    - Advanced metrics (xG, xA, progressive passes/carries)
    - Defensive actions (tackles, interceptions, blocks)
    - Passing statistics (completion %, progressive passes)
    """

    BASE_URL = "https://fbref.com"

    # League IDs for FBref
    LEAGUE_IDS = {
        "eredivisie": "23",
        "primeira-liga": "32",
        "belgian-pro-league": "37",
        "championship": "10",
        "serie-b": "18",
        "ligue-2": "60",
        "austrian-bundesliga": "56",
        "scottish-premiership": "40",
        # Top 5 for validation
        "premier-league": "9",
        "la-liga": "12",
        "bundesliga": "20",
        "serie-a": "11",
        "ligue-1": "13",
    }

    @property
    def source_name(self) -> str:
        return "fbref"

    def _build_league_url(self, league_id: str, season: str) -> str:
        """Build URL for league season stats page."""
        # FBref URL format: /en/comps/{id}/{season}/stats/
        return f"{self.BASE_URL}/en/comps/{league_id}/{season}/stats/"

    def _build_shooting_url(self, league_id: str, season: str) -> str:
        """Build URL for league shooting stats."""
        return f"{self.BASE_URL}/en/comps/{league_id}/{season}/shooting/"

    def _build_passing_url(self, league_id: str, season: str) -> str:
        """Build URL for league passing stats."""
        return f"{self.BASE_URL}/en/comps/{league_id}/{season}/passing/"

    def _build_defense_url(self, league_id: str, season: str) -> str:
        """Build URL for league defensive stats."""
        return f"{self.BASE_URL}/en/comps/{league_id}/{season}/defense/"

    def _build_possession_url(self, league_id: str, season: str) -> str:
        """Build URL for league possession stats."""
        return f"{self.BASE_URL}/en/comps/{league_id}/{season}/possession/"

    def _parse_stat(self, value: str | None, as_int: bool = False) -> float | int | None:
        """Parse a stat value, handling empty strings and special characters."""
        if value is None or value.strip() == "" or value.strip() == "-":
            return None
        try:
            cleaned = value.strip().replace(",", "")
            return int(cleaned) if as_int else float(cleaned)
        except ValueError:
            return None

    def _extract_player_id(self, href: str | None) -> str | None:
        """Extract player ID from FBref player URL."""
        if href is None:
            return None
        # URL format: /en/players/{player_id}/{player_name}
        match = re.search(r"/players/([a-f0-9]+)/", href)
        return match.group(1) if match else None

    def _parse_standard_stats_table(self, soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
        """Parse the standard stats table.

        Returns:
            Dict mapping player_id to their stats
        """
        players = {}

        table = soup.find("table", {"id": "stats_standard"})
        if table is None:
            # Try alternate table ID
            table = soup.find("table", {"id": lambda x: x and "stats_standard" in x})

        if table is None:
            logger.warning("Could not find standard stats table")
            return players

        tbody = table.find("tbody")
        if tbody is None:
            return players

        for row in tbody.find_all("tr"):
            # Skip header rows within tbody
            if row.get("class") and "thead" in row.get("class", []):
                continue

            cells = row.find_all(["th", "td"])
            if len(cells) < 10:
                continue

            # Extract player link and ID
            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell is None:
                player_cell = row.find("td", {"data-stat": "player"})

            if player_cell is None:
                continue

            player_link = player_cell.find("a")
            if player_link is None:
                continue

            player_id = self._extract_player_id(player_link.get("href"))
            if player_id is None:
                continue

            player_name = player_link.get_text(strip=True)

            # Extract stats by data-stat attribute
            def get_stat(stat_name: str, as_int: bool = False) -> float | int | None:
                cell = row.find(["th", "td"], {"data-stat": stat_name})
                return self._parse_stat(cell.get_text() if cell else None, as_int)

            # Get nationality and position
            nation_cell = row.find(["td"], {"data-stat": "nationality"})
            nationality = None
            if nation_cell:
                nation_link = nation_cell.find("a")
                if nation_link:
                    nationality = nation_link.get_text(strip=True)

            pos_cell = row.find(["td"], {"data-stat": "position"})
            position = pos_cell.get_text(strip=True) if pos_cell else None

            # Get team
            team_cell = row.find(["td"], {"data-stat": "team"})
            team = None
            if team_cell:
                team_link = team_cell.find("a")
                team = team_link.get_text(strip=True) if team_link else team_cell.get_text(strip=True)

            # Get age and birth year
            age = get_stat("age", as_int=True)
            birth_year = get_stat("birth_year", as_int=True)

            players[player_id] = {
                "player_id": player_id,
                "name": player_name,
                "nationality": nationality,
                "position": position,
                "team": team,
                "age": age,
                "birth_year": birth_year,
                "games": get_stat("games", as_int=True),
                "games_starts": get_stat("games_starts", as_int=True),
                "minutes": get_stat("minutes", as_int=True),
                "minutes_90s": get_stat("minutes_90s"),
                "goals": get_stat("goals", as_int=True),
                "assists": get_stat("assists", as_int=True),
                "goals_assists": get_stat("goals_assists", as_int=True),
                "goals_pens": get_stat("goals_pens", as_int=True),
                "pens_made": get_stat("pens_made", as_int=True),
                "pens_att": get_stat("pens_att", as_int=True),
                "cards_yellow": get_stat("cards_yellow", as_int=True),
                "cards_red": get_stat("cards_red", as_int=True),
                "xg": get_stat("xg"),
                "npxg": get_stat("npxg"),
                "xg_assist": get_stat("xg_assist"),
                "npxg_xg_assist": get_stat("npxg_xg_assist"),
                "progressive_carries": get_stat("progressive_carries", as_int=True),
                "progressive_passes": get_stat("progressive_passes", as_int=True),
                "progressive_passes_received": get_stat("progressive_passes_received", as_int=True),
            }

        return players

    def _parse_shooting_table(self, soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
        """Parse shooting stats table."""
        stats = {}

        table = soup.find("table", {"id": lambda x: x and "shooting" in str(x).lower()})
        if table is None:
            logger.warning("Could not find shooting table")
            return stats

        tbody = table.find("tbody")
        if tbody is None:
            return stats

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue

            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell is None:
                player_cell = row.find("td", {"data-stat": "player"})
            if player_cell is None:
                continue

            player_link = player_cell.find("a")
            if player_link is None:
                continue

            player_id = self._extract_player_id(player_link.get("href"))
            if player_id is None:
                continue

            def get_stat(stat_name: str, as_int: bool = False) -> float | int | None:
                cell = row.find(["th", "td"], {"data-stat": stat_name})
                return self._parse_stat(cell.get_text() if cell else None, as_int)

            stats[player_id] = {
                "shots": get_stat("shots", as_int=True),
                "shots_on_target": get_stat("shots_on_target", as_int=True),
                "shots_on_target_pct": get_stat("shots_on_target_pct"),
                "shots_per90": get_stat("shots_per90"),
                "shots_on_target_per90": get_stat("shots_on_target_per90"),
                "goals_per_shot": get_stat("goals_per_shot"),
                "goals_per_shot_on_target": get_stat("goals_per_shot_on_target"),
                "average_shot_distance": get_stat("average_shot_distance"),
                "shots_free_kicks": get_stat("shots_free_kicks", as_int=True),
                "npxg_per_shot": get_stat("npxg_per_shot"),
                "xg_net": get_stat("xg_net"),
                "npxg_net": get_stat("npxg_net"),
            }

        return stats

    def _parse_passing_table(self, soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
        """Parse passing stats table."""
        stats = {}

        table = soup.find("table", {"id": lambda x: x and "passing" in str(x).lower()})
        if table is None:
            logger.warning("Could not find passing table")
            return stats

        tbody = table.find("tbody")
        if tbody is None:
            return stats

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue

            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell is None:
                player_cell = row.find("td", {"data-stat": "player"})
            if player_cell is None:
                continue

            player_link = player_cell.find("a")
            if player_link is None:
                continue

            player_id = self._extract_player_id(player_link.get("href"))
            if player_id is None:
                continue

            def get_stat(stat_name: str, as_int: bool = False) -> float | int | None:
                cell = row.find(["th", "td"], {"data-stat": stat_name})
                return self._parse_stat(cell.get_text() if cell else None, as_int)

            stats[player_id] = {
                "passes_completed": get_stat("passes_completed", as_int=True),
                "passes": get_stat("passes", as_int=True),
                "passes_pct": get_stat("passes_pct"),
                "passes_total_distance": get_stat("passes_total_distance", as_int=True),
                "passes_progressive_distance": get_stat("passes_progressive_distance", as_int=True),
                "passes_completed_short": get_stat("passes_completed_short", as_int=True),
                "passes_short": get_stat("passes_short", as_int=True),
                "passes_pct_short": get_stat("passes_pct_short"),
                "passes_completed_medium": get_stat("passes_completed_medium", as_int=True),
                "passes_medium": get_stat("passes_medium", as_int=True),
                "passes_pct_medium": get_stat("passes_pct_medium"),
                "passes_completed_long": get_stat("passes_completed_long", as_int=True),
                "passes_long": get_stat("passes_long", as_int=True),
                "passes_pct_long": get_stat("passes_pct_long"),
                "assisted_shots": get_stat("assisted_shots", as_int=True),
                "passes_into_final_third": get_stat("passes_into_final_third", as_int=True),
                "passes_into_penalty_area": get_stat("passes_into_penalty_area", as_int=True),
                "crosses_into_penalty_area": get_stat("crosses_into_penalty_area", as_int=True),
            }

        return stats

    def _parse_defense_table(self, soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
        """Parse defensive stats table."""
        stats = {}

        table = soup.find("table", {"id": lambda x: x and "defense" in str(x).lower()})
        if table is None:
            logger.warning("Could not find defense table")
            return stats

        tbody = table.find("tbody")
        if tbody is None:
            return stats

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue

            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell is None:
                player_cell = row.find("td", {"data-stat": "player"})
            if player_cell is None:
                continue

            player_link = player_cell.find("a")
            if player_link is None:
                continue

            player_id = self._extract_player_id(player_link.get("href"))
            if player_id is None:
                continue

            def get_stat(stat_name: str, as_int: bool = False) -> float | int | None:
                cell = row.find(["th", "td"], {"data-stat": stat_name})
                return self._parse_stat(cell.get_text() if cell else None, as_int)

            stats[player_id] = {
                "tackles": get_stat("tackles", as_int=True),
                "tackles_won": get_stat("tackles_won", as_int=True),
                "tackles_def_3rd": get_stat("tackles_def_3rd", as_int=True),
                "tackles_mid_3rd": get_stat("tackles_mid_3rd", as_int=True),
                "tackles_att_3rd": get_stat("tackles_att_3rd", as_int=True),
                "challenge_tackles": get_stat("challenge_tackles", as_int=True),
                "challenges": get_stat("challenges", as_int=True),
                "challenge_tackles_pct": get_stat("challenge_tackles_pct"),
                "challenges_lost": get_stat("challenges_lost", as_int=True),
                "blocks": get_stat("blocks", as_int=True),
                "blocked_shots": get_stat("blocked_shots", as_int=True),
                "blocked_passes": get_stat("blocked_passes", as_int=True),
                "interceptions": get_stat("interceptions", as_int=True),
                "tackles_interceptions": get_stat("tackles_interceptions", as_int=True),
                "clearances": get_stat("clearances", as_int=True),
                "errors": get_stat("errors", as_int=True),
            }

        return stats

    def _parse_possession_table(self, soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
        """Parse possession stats table."""
        stats = {}

        table = soup.find("table", {"id": lambda x: x and "possession" in str(x).lower()})
        if table is None:
            logger.warning("Could not find possession table")
            return stats

        tbody = table.find("tbody")
        if tbody is None:
            return stats

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue

            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell is None:
                player_cell = row.find("td", {"data-stat": "player"})
            if player_cell is None:
                continue

            player_link = player_cell.find("a")
            if player_link is None:
                continue

            player_id = self._extract_player_id(player_link.get("href"))
            if player_id is None:
                continue

            def get_stat(stat_name: str, as_int: bool = False) -> float | int | None:
                cell = row.find(["th", "td"], {"data-stat": stat_name})
                return self._parse_stat(cell.get_text() if cell else None, as_int)

            stats[player_id] = {
                "touches": get_stat("touches", as_int=True),
                "touches_def_pen_area": get_stat("touches_def_pen_area", as_int=True),
                "touches_def_3rd": get_stat("touches_def_3rd", as_int=True),
                "touches_mid_3rd": get_stat("touches_mid_3rd", as_int=True),
                "touches_att_3rd": get_stat("touches_att_3rd", as_int=True),
                "touches_att_pen_area": get_stat("touches_att_pen_area", as_int=True),
                "touches_live_ball": get_stat("touches_live_ball", as_int=True),
                "take_ons": get_stat("take_ons", as_int=True),
                "take_ons_won": get_stat("take_ons_won", as_int=True),
                "take_ons_won_pct": get_stat("take_ons_won_pct"),
                "take_ons_tackled": get_stat("take_ons_tackled", as_int=True),
                "take_ons_tackled_pct": get_stat("take_ons_tackled_pct"),
                "carries": get_stat("carries", as_int=True),
                "carries_distance": get_stat("carries_distance", as_int=True),
                "carries_progressive_distance": get_stat("carries_progressive_distance", as_int=True),
                "progressive_carries": get_stat("progressive_carries", as_int=True),
                "carries_into_final_third": get_stat("carries_into_final_third", as_int=True),
                "carries_into_penalty_area": get_stat("carries_into_penalty_area", as_int=True),
                "miscontrols": get_stat("miscontrols", as_int=True),
                "dispossessed": get_stat("dispossessed", as_int=True),
                "passes_received": get_stat("passes_received", as_int=True),
                "progressive_passes_received": get_stat("progressive_passes_received", as_int=True),
            }

        return stats

    def _merge_player_stats(
        self,
        standard: dict[str, dict],
        shooting: dict[str, dict],
        passing: dict[str, dict],
        defense: dict[str, dict],
        possession: dict[str, dict],
    ) -> list[dict[str, Any]]:
        """Merge all stat tables into a single list of player records."""
        all_players = []

        for player_id, base_stats in standard.items():
            player_data = base_stats.copy()

            # Merge additional stats
            if player_id in shooting:
                player_data.update(shooting[player_id])
            if player_id in passing:
                player_data.update(passing[player_id])
            if player_id in defense:
                player_data.update(defense[player_id])
            if player_id in possession:
                player_data.update(possession[player_id])

            all_players.append(player_data)

        return all_players

    def scrape_league_season(
        self,
        league: str,
        season: str,
        include_shooting: bool = True,
        include_passing: bool = True,
        include_defense: bool = True,
        include_possession: bool = True,
    ) -> list[dict[str, Any]]:
        """Scrape all player stats for a league season.

        Args:
            league: League name (e.g., "eredivisie", "championship")
            season: Season string (e.g., "2023-2024")
            include_shooting: Whether to fetch shooting stats
            include_passing: Whether to fetch passing stats
            include_defense: Whether to fetch defensive stats
            include_possession: Whether to fetch possession stats

        Returns:
            List of player stat dictionaries
        """
        league_id = self.LEAGUE_IDS.get(league.lower().replace(" ", "-"))
        if league_id is None:
            raise ValueError(f"Unknown league: {league}. Available: {list(self.LEAGUE_IDS.keys())}")

        logger.info(f"Scraping FBref: {league} {season}")

        # Fetch and parse standard stats (required)
        standard_url = self._build_league_url(league_id, season)
        standard_html = self.fetch(standard_url)
        standard_soup = BeautifulSoup(standard_html, "lxml")
        standard_stats = self._parse_standard_stats_table(standard_soup)

        logger.info(f"Found {len(standard_stats)} players in standard stats")

        # Fetch additional stat tables
        shooting_stats = {}
        passing_stats = {}
        defense_stats = {}
        possession_stats = {}

        if include_shooting:
            try:
                shooting_url = self._build_shooting_url(league_id, season)
                shooting_html = self.fetch(shooting_url)
                shooting_soup = BeautifulSoup(shooting_html, "lxml")
                shooting_stats = self._parse_shooting_table(shooting_soup)
                logger.info(f"Found {len(shooting_stats)} players in shooting stats")
            except Exception as e:
                logger.warning(f"Failed to fetch shooting stats: {e}")

        if include_passing:
            try:
                passing_url = self._build_passing_url(league_id, season)
                passing_html = self.fetch(passing_url)
                passing_soup = BeautifulSoup(passing_html, "lxml")
                passing_stats = self._parse_passing_table(passing_soup)
                logger.info(f"Found {len(passing_stats)} players in passing stats")
            except Exception as e:
                logger.warning(f"Failed to fetch passing stats: {e}")

        if include_defense:
            try:
                defense_url = self._build_defense_url(league_id, season)
                defense_html = self.fetch(defense_url)
                defense_soup = BeautifulSoup(defense_html, "lxml")
                defense_stats = self._parse_defense_table(defense_soup)
                logger.info(f"Found {len(defense_stats)} players in defense stats")
            except Exception as e:
                logger.warning(f"Failed to fetch defense stats: {e}")

        if include_possession:
            try:
                possession_url = self._build_possession_url(league_id, season)
                possession_html = self.fetch(possession_url)
                possession_soup = BeautifulSoup(possession_html, "lxml")
                possession_stats = self._parse_possession_table(possession_soup)
                logger.info(f"Found {len(possession_stats)} players in possession stats")
            except Exception as e:
                logger.warning(f"Failed to fetch possession stats: {e}")

        # Merge all stats
        all_players = self._merge_player_stats(
            standard_stats,
            shooting_stats,
            passing_stats,
            defense_stats,
            possession_stats,
        )

        # Add metadata
        for player in all_players:
            player["league"] = league
            player["season"] = season
            player["source"] = "fbref"

        logger.info(f"Scraped {len(all_players)} players for {league} {season}")
        return all_players

    def scrape_multiple_seasons(
        self,
        league: str,
        seasons: list[str],
    ) -> list[dict[str, Any]]:
        """Scrape multiple seasons for a league.

        Args:
            league: League name
            seasons: List of seasons to scrape

        Returns:
            Combined list of player stats across all seasons
        """
        all_data = []
        for season in seasons:
            try:
                season_data = self.scrape_league_season(league, season)
                all_data.extend(season_data)
            except Exception as e:
                logger.error(f"Failed to scrape {league} {season}: {e}")

        return all_data

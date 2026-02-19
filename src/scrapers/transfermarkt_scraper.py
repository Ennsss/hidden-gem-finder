"""Transfermarkt scraper for player valuations and transfer history."""

import logging
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class TransfermarktScraper(BaseScraper):
    """Scraper for Transfermarkt.com player data.

    Transfermarkt provides:
    - Market valuations
    - Transfer history
    - Contract information
    - Player demographics (age, nationality, position)
    """

    BASE_URL = "https://www.transfermarkt.com"

    # League IDs for Transfermarkt
    LEAGUE_IDS = {
        "eredivisie": "NL1",
        "primeira-liga": "PO1",
        "belgian-pro-league": "BE1",
        "championship": "GB2",
        "serie-b": "IT2",
        "ligue-2": "FR2",
        "austrian-bundesliga": "A1",
        "scottish-premiership": "SC1",
        # Top 5 for validation
        "premier-league": "GB1",
        "la-liga": "ES1",
        "bundesliga": "L1",
        "serie-a": "IT1",
        "ligue-1": "FR1",
    }

    # League names for URL construction
    LEAGUE_NAMES = {
        "eredivisie": "eredivisie",
        "primeira-liga": "liga-nos",
        "belgian-pro-league": "jupiler-pro-league",
        "championship": "championship",
        "serie-b": "serie-b",
        "ligue-2": "ligue-2",
        "austrian-bundesliga": "bundesliga",
        "scottish-premiership": "scottish-premiership",
        "premier-league": "premier-league",
        "la-liga": "laliga",
        "bundesliga": "1-bundesliga",
        "serie-a": "serie-a",
        "ligue-1": "ligue-1",
    }

    @property
    def source_name(self) -> str:
        return "transfermarkt"

    def _build_league_players_url(self, league: str, season: str, page: int = 1) -> str:
        """Build URL for league players page (market values listing).

        Uses /marktwerte/ which lists individual players with market values.
        Transfermarkt uses the start year for season, e.g., 2023-2024 -> 2023
        """
        league_name = self.LEAGUE_NAMES.get(league.lower().replace(" ", "-"))
        league_id = self.LEAGUE_IDS.get(league.lower().replace(" ", "-"))
        if league_name is None or league_id is None:
            raise ValueError(f"Unknown league: {league}")

        # Extract start year from season (e.g., "2023-2024" -> "2023")
        season_year = season.split("-")[0]

        return (
            f"{self.BASE_URL}/{league_name}/marktwerte/wettbewerb/{league_id}"
            f"/saison_id/{season_year}/page/{page}"
        )

    def _build_player_url(self, player_slug: str, player_id: str) -> str:
        """Build URL for individual player page."""
        return f"{self.BASE_URL}/{player_slug}/profil/spieler/{player_id}"

    def _build_transfers_url(self, player_slug: str, player_id: str) -> str:
        """Build URL for player transfer history."""
        return f"{self.BASE_URL}/{player_slug}/transfers/spieler/{player_id}"

    def _parse_market_value(self, value_str: str | None) -> int | None:
        """Parse market value string to integer (in EUR).

        Examples:
            "€25.00m" -> 25000000
            "€500k" -> 500000
            "€1.50m" -> 1500000
        """
        if value_str is None or value_str.strip() == "-":
            return None

        value_str = value_str.strip().replace("€", "").replace(",", ".")

        try:
            if "m" in value_str.lower():
                return int(float(value_str.lower().replace("m", "")) * 1_000_000)
            elif "k" in value_str.lower():
                return int(float(value_str.lower().replace("k", "")) * 1_000)
            else:
                return int(float(value_str))
        except ValueError:
            logger.warning(f"Could not parse market value: {value_str}")
            return None

    def _parse_date(self, date_str: str | None) -> str | None:
        """Parse date string to ISO format."""
        if date_str is None or date_str.strip() == "-":
            return None

        # Common formats: "Jan 1, 2000", "1.1.2000", "2000-01-01"
        formats = ["%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _extract_player_id(self, href: str | None) -> str | None:
        """Extract player ID from Transfermarkt URL."""
        if href is None:
            return None
        # URL format: /player-name/profil/spieler/123456
        match = re.search(r"/spieler/(\d+)", href)
        return match.group(1) if match else None

    def _extract_player_slug(self, href: str | None) -> str | None:
        """Extract player slug from Transfermarkt URL."""
        if href is None:
            return None
        # URL format: /player-name/profil/spieler/123456
        match = re.search(r"^/([^/]+)/", href)
        return match.group(1) if match else None

    def _parse_league_players_page(
        self, html: str, league: str, season: str
    ) -> list[dict[str, Any]]:
        """Parse a league players page.

        Returns list of player dictionaries with basic info and market values.
        """
        soup = BeautifulSoup(html, "lxml")
        players = []

        # Find the main players table
        table = soup.find("table", class_="items")
        if table is None:
            logger.warning("Could not find players table")
            return players

        tbody = table.find("tbody")
        if tbody is None:
            return players

        for row in tbody.find_all("tr", class_=["odd", "even"]):
            try:
                player_data = self._parse_player_row(row, league, season)
                if player_data:
                    players.append(player_data)
            except Exception as e:
                logger.warning(f"Error parsing player row: {e}")
                continue

        return players

    def _parse_player_row(
        self, row: BeautifulSoup, league: str, season: str
    ) -> dict[str, Any] | None:
        """Parse a single player row from the league table."""
        # Find player link
        player_cell = row.find("td", class_="hauptlink")
        if player_cell is None:
            return None

        player_link = player_cell.find("a")
        if player_link is None:
            return None

        href = player_link.get("href", "")
        player_id = self._extract_player_id(href)
        player_slug = self._extract_player_slug(href)

        if player_id is None:
            return None

        player_name = player_link.get_text(strip=True)

        # Find position
        position = None
        position_cells = row.find_all("td")
        for cell in position_cells:
            if cell.get_text(strip=True) in [
                "Centre-Forward", "Left Winger", "Right Winger",
                "Second Striker", "Attacking Midfield", "Central Midfield",
                "Defensive Midfield", "Left Midfield", "Right Midfield",
                "Centre-Back", "Left-Back", "Right-Back", "Goalkeeper"
            ]:
                position = cell.get_text(strip=True)
                break

        # Find market value (usually in the last column)
        market_value = None
        value_cell = row.find("td", class_="rechts hauptlink")
        if value_cell:
            market_value = self._parse_market_value(value_cell.get_text())

        # Find age
        age = None
        age_cell = row.find("td", class_="zentriert")
        if age_cell:
            age_text = age_cell.get_text(strip=True)
            if age_text.isdigit():
                age = int(age_text)

        # Find nationality
        nationality = None
        flag_img = row.find("img", class_="flaggenrahmen")
        if flag_img:
            nationality = flag_img.get("title", "")

        # Find team
        team = None
        team_link = row.find("a", class_="vereinprofil_tooltip")
        if team_link:
            team = team_link.get("title", team_link.get_text(strip=True))

        return {
            "player_id": player_id,
            "player_slug": player_slug,
            "name": player_name,
            "position": position,
            "age": age,
            "nationality": nationality,
            "team": team,
            "market_value_eur": market_value,
            "league": league,
            "season": season,
            "source": "transfermarkt",
        }

    def _parse_player_profile(self, html: str) -> dict[str, Any]:
        """Parse detailed player profile page."""
        soup = BeautifulSoup(html, "lxml")
        data = {}

        # Find info table
        info_table = soup.find("div", class_="info-table")
        if info_table:
            for row in info_table.find_all("span", class_="info-table__content"):
                label_elem = row.find_previous("span", class_="info-table__content--label")
                if label_elem:
                    label = label_elem.get_text(strip=True).lower()
                    value = row.get_text(strip=True)

                    if "date of birth" in label:
                        data["date_of_birth"] = self._parse_date(value)
                    elif "height" in label:
                        # Parse height like "1,85 m" -> 185
                        height_match = re.search(r"(\d)[,.](\d+)", value)
                        if height_match:
                            data["height_cm"] = int(height_match.group(1)) * 100 + int(
                                height_match.group(2)
                            )
                    elif "foot" in label:
                        data["foot"] = value.lower()
                    elif "citizenship" in label:
                        data["nationality"] = value
                    elif "position" in label:
                        data["position"] = value
                    elif "contract expires" in label:
                        data["contract_expires"] = self._parse_date(value)

        # Find current market value
        value_elem = soup.find("div", class_="tm-player-market-value-development__current-value")
        if value_elem:
            data["market_value_eur"] = self._parse_market_value(value_elem.get_text())

        return data

    def _parse_transfer_history(self, html: str) -> list[dict[str, Any]]:
        """Parse player transfer history page."""
        soup = BeautifulSoup(html, "lxml")
        transfers = []

        # Find transfer table
        table = soup.find("div", class_="grid tm-player-transfer-history-grid")
        if table is None:
            # Try alternate structure
            table = soup.find("table", class_="items")

        if table is None:
            logger.warning("Could not find transfer history table")
            return transfers

        rows = table.find_all("div", class_="grid__cell") or table.find_all("tr")

        current_transfer = {}
        for elem in rows:
            text = elem.get_text(strip=True)

            # Look for transfer fee
            if "€" in text or "free" in text.lower() or "loan" in text.lower():
                fee = None
                is_loan = "loan" in text.lower()

                if "€" in text:
                    fee = self._parse_market_value(text)
                elif "free" in text.lower():
                    fee = 0

                if current_transfer.get("from_team"):
                    current_transfer["transfer_fee_eur"] = fee
                    current_transfer["is_loan"] = is_loan
                    transfers.append(current_transfer)
                    current_transfer = {}

        return transfers

    def scrape_league_season(
        self, league: str, season: str, max_pages: int = 25
    ) -> list[dict[str, Any]]:
        """Scrape all players for a league season.

        Args:
            league: League name (e.g., "eredivisie", "championship")
            season: Season string (e.g., "2023-2024")
            max_pages: Maximum number of pages to scrape

        Returns:
            List of player dictionaries with market values
        """
        league_key = league.lower().replace(" ", "-")
        if league_key not in self.LEAGUE_IDS:
            raise ValueError(
                f"Unknown league: {league}. Available: {list(self.LEAGUE_IDS.keys())}"
            )

        logger.info(f"Scraping Transfermarkt: {league} {season}")

        all_players = []
        seen_ids = set()
        page = 1

        while page <= max_pages:
            try:
                url = self._build_league_players_url(league, season, page)
                html = self.fetch(url)
                players = self._parse_league_players_page(html, league, season)

                if not players:
                    logger.info(f"No players found on page {page}, stopping")
                    break

                # Detect pagination wrap-around (TM repeats pages after last)
                new_players = [p for p in players if p["player_id"] not in seen_ids]
                if not new_players:
                    logger.info(f"Page {page}: all duplicates, pagination wrapped. Stopping.")
                    break

                for p in new_players:
                    seen_ids.add(p["player_id"])
                all_players.extend(new_players)
                logger.info(f"Page {page}: found {len(new_players)} new players")
                page += 1

            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break

        logger.info(f"Scraped {len(all_players)} unique players for {league} {season}")
        return all_players

    def scrape_player_details(
        self, player_slug: str, player_id: str
    ) -> dict[str, Any]:
        """Scrape detailed info for a single player.

        Args:
            player_slug: Player URL slug (e.g., "cody-gakpo")
            player_id: Player ID (e.g., "363205")

        Returns:
            Dictionary with detailed player info
        """
        url = self._build_player_url(player_slug, player_id)
        html = self.fetch(url)
        return self._parse_player_profile(html)

    def scrape_player_transfers(
        self, player_slug: str, player_id: str
    ) -> list[dict[str, Any]]:
        """Scrape transfer history for a single player.

        Args:
            player_slug: Player URL slug
            player_id: Player ID

        Returns:
            List of transfer records
        """
        url = self._build_transfers_url(player_slug, player_id)
        html = self.fetch(url)
        return self._parse_transfer_history(html)

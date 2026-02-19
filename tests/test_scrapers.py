"""Tests for data scrapers."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scrapers import BaseScraper, FBrefScraper, TransfermarktScraper, UnderstatScraper


class TestBaseScraper:
    """Tests for BaseScraper base class."""

    def test_cache_path_generation(self):
        """Test that cache paths are generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a concrete implementation for testing
            class TestScraper(BaseScraper):
                @property
                def source_name(self) -> str:
                    return "test"

                def scrape_league_season(self, league_id, season):
                    return []

            scraper = TestScraper(cache_dir=tmpdir)
            url = "https://example.com/test/page"
            cache_path = scraper._get_cache_path(url)

            assert cache_path.parent.name == "test"
            assert cache_path.suffix == ".html"
            assert len(cache_path.stem) > 0

    def test_rate_limiting(self):
        """Test that rate limiting works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class TestScraper(BaseScraper):
                @property
                def source_name(self) -> str:
                    return "test"

                def scrape_league_season(self, league_id, season):
                    return []

            scraper = TestScraper(cache_dir=tmpdir, rate_limit=0.1)

            # Simulate two requests
            import time

            scraper._last_request_time = time.time()
            start = time.time()
            scraper._respect_rate_limit()
            elapsed = time.time() - start

            # Should have slept for approximately rate_limit seconds
            assert elapsed >= 0.05  # Allow some tolerance

    def test_caching(self):
        """Test that responses are cached correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class TestScraper(BaseScraper):
                @property
                def source_name(self) -> str:
                    return "test"

                def scrape_league_season(self, league_id, season):
                    return []

            scraper = TestScraper(cache_dir=tmpdir)
            url = "https://example.com/test"
            content = "<html>Test content</html>"

            # Cache a response
            scraper._cache_response(url, content)

            # Verify it can be retrieved
            cached = scraper._get_cached_response(url)
            assert cached == content

    def test_stats_tracking(self):
        """Test that request stats are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class TestScraper(BaseScraper):
                @property
                def source_name(self) -> str:
                    return "test"

                def scrape_league_season(self, league_id, season):
                    return []

            scraper = TestScraper(cache_dir=tmpdir)
            stats = scraper.get_stats()

            assert stats["source"] == "test"
            assert stats["requests_made"] == 0
            assert "rate_limit" in stats


class TestFBrefScraper:
    """Tests for FBref scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance for testing."""
        return FBrefScraper(cache_dir=tmp_path, rate_limit=0)

    @pytest.fixture
    def standard_stats_html(self):
        """Load the standard stats fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "fbref_standard_stats.html"
        return fixture_path.read_text(encoding="utf-8")

    @pytest.fixture
    def shooting_stats_html(self):
        """Load the shooting stats fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "fbref_shooting_stats.html"
        return fixture_path.read_text(encoding="utf-8")

    def test_source_name(self, scraper):
        """Test that source name is correct."""
        assert scraper.source_name == "fbref"

    def test_league_ids_defined(self, scraper):
        """Test that league IDs are defined."""
        assert "eredivisie" in scraper.LEAGUE_IDS
        assert "championship" in scraper.LEAGUE_IDS
        assert "premier-league" in scraper.LEAGUE_IDS

    def test_extract_player_id(self, scraper):
        """Test player ID extraction from URLs."""
        url = "/en/players/abc12345/Some-Player-Name"
        player_id = scraper._extract_player_id(url)
        assert player_id == "abc12345"

        # Test with None
        assert scraper._extract_player_id(None) is None

        # Test with invalid URL
        assert scraper._extract_player_id("/invalid/url") is None

    def test_parse_stat(self, scraper):
        """Test stat parsing."""
        assert scraper._parse_stat("10") == 10.0
        assert scraper._parse_stat("10.5") == 10.5
        assert scraper._parse_stat("1,234") == 1234.0
        assert scraper._parse_stat("1,234", as_int=True) == 1234
        assert scraper._parse_stat("") is None
        assert scraper._parse_stat("-") is None
        assert scraper._parse_stat(None) is None

    def test_parse_standard_stats_table(self, scraper, standard_stats_html):
        """Test parsing of standard stats table."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(standard_stats_html, "lxml")
        stats = scraper._parse_standard_stats_table(soup)

        assert len(stats) == 3

        # Check Cody Gakpo's stats
        gakpo = stats.get("abc12345")
        assert gakpo is not None
        assert gakpo["name"] == "Cody Gakpo"
        assert gakpo["position"] == "FW,MF"
        assert gakpo["team"] == "PSV Eindhoven"
        assert gakpo["age"] == 23
        assert gakpo["goals"] == 21
        assert gakpo["assists"] == 15
        assert gakpo["xg"] == 15.8
        assert gakpo["minutes"] == 2856

        # Check Test Player
        test_player = stats.get("def67890")
        assert test_player is not None
        assert test_player["name"] == "Test Player"
        assert test_player["position"] == "MF"
        assert test_player["goals"] == 5

        # Check Young Talent
        young = stats.get("cba11111")
        assert young is not None
        assert young["name"] == "Young Talent"
        assert young["age"] == 19
        assert young["position"] == "DF"

    def test_parse_shooting_table(self, scraper, shooting_stats_html):
        """Test parsing of shooting stats table."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(shooting_stats_html, "lxml")
        stats = scraper._parse_shooting_table(soup)

        assert len(stats) == 3

        gakpo = stats.get("abc12345")
        assert gakpo is not None
        assert gakpo["shots"] == 98
        assert gakpo["shots_on_target"] == 45
        assert gakpo["shots_on_target_pct"] == 45.9
        assert gakpo["shots_per90"] == 3.09

    def test_merge_player_stats(self, scraper, standard_stats_html, shooting_stats_html):
        """Test merging of multiple stat tables."""
        from bs4 import BeautifulSoup

        standard_soup = BeautifulSoup(standard_stats_html, "lxml")
        shooting_soup = BeautifulSoup(shooting_stats_html, "lxml")

        standard = scraper._parse_standard_stats_table(standard_soup)
        shooting = scraper._parse_shooting_table(shooting_soup)

        merged = scraper._merge_player_stats(
            standard, shooting, {}, {}, {}
        )

        assert len(merged) == 3

        # Find Gakpo in merged results
        gakpo = next(p for p in merged if p["player_id"] == "abc12345")

        # Check standard stats are present
        assert gakpo["name"] == "Cody Gakpo"
        assert gakpo["goals"] == 21

        # Check shooting stats are merged
        assert gakpo["shots"] == 98
        assert gakpo["shots_on_target"] == 45

    @patch.object(FBrefScraper, "fetch")
    def test_scrape_league_season(
        self, mock_fetch, scraper, standard_stats_html, shooting_stats_html
    ):
        """Test full league season scraping with mocked requests."""
        # Set up mock to return different HTML for different URLs
        def mock_fetch_impl(url, use_cache=True):
            if "shooting" in url:
                return shooting_stats_html
            return standard_stats_html

        mock_fetch.side_effect = mock_fetch_impl

        # Scrape with only standard and shooting (simpler test)
        players = scraper.scrape_league_season(
            "eredivisie",
            "2023-2024",
            include_passing=False,
            include_defense=False,
            include_possession=False,
        )

        assert len(players) == 3

        # Check metadata is added
        for player in players:
            assert player["league"] == "eredivisie"
            assert player["season"] == "2023-2024"
            assert player["source"] == "fbref"

        # Check Gakpo has merged stats
        gakpo = next(p for p in players if p["name"] == "Cody Gakpo")
        assert gakpo["goals"] == 21
        assert gakpo["shots"] == 98

    def test_unknown_league_raises_error(self, scraper):
        """Test that unknown league raises ValueError."""
        with pytest.raises(ValueError, match="Unknown league"):
            scraper.scrape_league_season("fake-league", "2023-2024")

    def test_uncomment_tables(self, scraper):
        """Test that HTML comment tables are extracted."""
        html = '''<html><body>
        <div>visible content</div>
        <!-- <div class="table_container" id="div_stats_shooting">
        <table id="stats_shooting"><tbody>
        <tr><td data-stat="xg">1.5</td></tr>
        </tbody></table></div> -->
        </body></html>'''
        result = scraper._uncomment_tables(html)
        assert "<!--" not in result or "stats_shooting" not in result.split("<!--")[1] if "<!--" in result else True
        assert '<table id="stats_shooting">' in result

    def test_uncomment_tables_passthrough(self, scraper):
        """Test that non-commented tables are unaffected."""
        html = '<html><body><table id="stats_standard"><tr><td>data</td></tr></table></body></html>'
        assert scraper._uncomment_tables(html) == html

    def test_build_urls(self, scraper):
        """Test URL building methods."""
        league_id = "23"  # Eredivisie
        season = "2023-2024"

        standard_url = scraper._build_league_url(league_id, season)
        assert "fbref.com" in standard_url
        assert league_id in standard_url
        assert season in standard_url

        shooting_url = scraper._build_shooting_url(league_id, season)
        assert "shooting" in shooting_url

        passing_url = scraper._build_passing_url(league_id, season)
        assert "passing" in passing_url

        defense_url = scraper._build_defense_url(league_id, season)
        assert "defense" in defense_url

        possession_url = scraper._build_possession_url(league_id, season)
        assert "possession" in possession_url


class TestFBrefScraperIntegration:
    """Integration tests that would hit the real API.

    These are marked as skip by default - run with:
    pytest -m integration tests/test_scrapers.py
    """

    @pytest.mark.skip(reason="Integration test - hits real API")
    def test_real_eredivisie_scrape(self, tmp_path):
        """Test scraping real Eredivisie data."""
        scraper = FBrefScraper(cache_dir=tmp_path, rate_limit=5)
        players = scraper.scrape_league_season(
            "eredivisie",
            "2023-2024",
            include_shooting=False,
            include_passing=False,
            include_defense=False,
            include_possession=False,
        )

        assert len(players) > 0
        assert all("name" in p for p in players)
        assert all("player_id" in p for p in players)


class TestTransfermarktScraper:
    """Tests for Transfermarkt scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance for testing."""
        return TransfermarktScraper(cache_dir=tmp_path, rate_limit=0)

    @pytest.fixture
    def league_html(self):
        """Load the league players fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "transfermarkt_league.html"
        return fixture_path.read_text(encoding="utf-8")

    def test_source_name(self, scraper):
        """Test that source name is correct."""
        assert scraper.source_name == "transfermarkt"

    def test_league_ids_defined(self, scraper):
        """Test that league IDs are defined."""
        assert "eredivisie" in scraper.LEAGUE_IDS
        assert "championship" in scraper.LEAGUE_IDS
        assert "premier-league" in scraper.LEAGUE_IDS

    def test_parse_market_value(self, scraper):
        """Test market value parsing."""
        assert scraper._parse_market_value("€25.00m") == 25_000_000
        assert scraper._parse_market_value("€500k") == 500_000
        assert scraper._parse_market_value("€1.50m") == 1_500_000
        assert scraper._parse_market_value("€800k") == 800_000
        assert scraper._parse_market_value("-") is None
        assert scraper._parse_market_value(None) is None

    def test_extract_player_id(self, scraper):
        """Test player ID extraction."""
        url = "/cody-gakpo/profil/spieler/363205"
        assert scraper._extract_player_id(url) == "363205"
        assert scraper._extract_player_id(None) is None
        assert scraper._extract_player_id("/invalid/url") is None

    def test_extract_player_slug(self, scraper):
        """Test player slug extraction."""
        url = "/cody-gakpo/profil/spieler/363205"
        assert scraper._extract_player_slug(url) == "cody-gakpo"
        assert scraper._extract_player_slug(None) is None

    def test_parse_date(self, scraper):
        """Test date parsing."""
        assert scraper._parse_date("Jan 1, 2000") == "2000-01-01"
        assert scraper._parse_date("1.1.2000") == "2000-01-01"
        assert scraper._parse_date("2000-01-01") == "2000-01-01"
        assert scraper._parse_date("-") is None
        assert scraper._parse_date(None) is None

    def test_parse_league_players_page(self, scraper, league_html):
        """Test parsing of league players page."""
        players = scraper._parse_league_players_page(
            league_html, "eredivisie", "2023-2024"
        )

        assert len(players) == 5

        # Check Cody Gakpo
        gakpo = next(p for p in players if p["name"] == "Cody Gakpo")
        assert gakpo["player_id"] == "363205"
        assert gakpo["player_slug"] == "cody-gakpo"
        assert gakpo["position"] == "Left Winger"
        assert gakpo["age"] == 23
        assert gakpo["market_value_eur"] == 45_000_000
        assert gakpo["team"] == "PSV Eindhoven"
        assert gakpo["league"] == "eredivisie"
        assert gakpo["season"] == "2023-2024"

        # Check Xavi Simons
        xavi = next(p for p in players if p["name"] == "Xavi Simons")
        assert xavi["player_id"] == "533869"
        assert xavi["market_value_eur"] == 50_000_000
        assert xavi["position"] == "Attacking Midfield"

        # Check budget player (€800k)
        budget = next(p for p in players if p["name"] == "Budget Player")
        assert budget["market_value_eur"] == 800_000

    def test_build_league_url(self, scraper):
        """Test URL building."""
        url = scraper._build_league_players_url("eredivisie", "2023-2024")
        assert "transfermarkt.com" in url
        assert "NL1" in url
        assert "2023" in url

    def test_unknown_league_raises_error(self, scraper):
        """Test that unknown league raises ValueError."""
        with pytest.raises(ValueError, match="Unknown league"):
            scraper.scrape_league_season("fake-league", "2023-2024")

    @patch.object(TransfermarktScraper, "fetch")
    def test_scrape_league_season(self, mock_fetch, scraper, league_html):
        """Test full league season scraping with mocked requests."""
        # First call returns data, second returns empty (no more pages)
        mock_fetch.side_effect = [league_html, "<html><body></body></html>"]

        players = scraper.scrape_league_season("eredivisie", "2023-2024", max_pages=2)

        assert len(players) == 5
        assert all(p["source"] == "transfermarkt" for p in players)
        assert all(p["league"] == "eredivisie" for p in players)


class TestTransfermarktIntegration:
    """Integration tests for Transfermarkt scraper."""

    @pytest.mark.skip(reason="Integration test - hits real API")
    def test_real_eredivisie_scrape(self, tmp_path):
        """Test scraping real Eredivisie data."""
        scraper = TransfermarktScraper(cache_dir=tmp_path, rate_limit=5)
        players = scraper.scrape_league_season("eredivisie", "2023-2024", max_pages=1)

        assert len(players) > 0
        assert all("name" in p for p in players)
        assert all("player_id" in p for p in players)


class TestUnderstatScraper:
    """Tests for Understat scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance for testing."""
        return UnderstatScraper(cache_dir=tmp_path, rate_limit=0)

    @pytest.fixture
    def league_html(self):
        """Load the league fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "understat_league.html"
        return fixture_path.read_text(encoding="utf-8")

    def test_source_name(self, scraper):
        """Test that source name is correct."""
        assert scraper.source_name == "understat"

    def test_league_ids_defined(self, scraper):
        """Test that league IDs are defined."""
        assert "eredivisie" in scraper.LEAGUE_IDS
        assert "premier-league" in scraper.LEAGUE_IDS
        assert scraper.LEAGUE_IDS["premier-league"] == "EPL"

    def test_unavailable_leagues(self, scraper):
        """Test that unavailable leagues are marked as None."""
        assert scraper.LEAGUE_IDS.get("eredivisie") is None  # Dropped by Understat
        assert scraper.LEAGUE_IDS.get("championship") is None
        assert scraper.LEAGUE_IDS.get("primeira-liga") is None
        assert scraper.LEAGUE_IDS.get("belgian-pro-league") is None

    def test_convert_season_format(self, scraper):
        """Test season format conversion."""
        assert scraper._convert_season_format("2023-2024") == "2023"
        assert scraper._convert_season_format("2023-24") == "2023"
        assert scraper._convert_season_format("2022-2023") == "2022"

    def test_build_league_url(self, scraper):
        """Test URL building."""
        url_epl = scraper._build_league_url("premier-league", "2023-2024")
        assert "understat.com" in url_epl
        assert "EPL" in url_epl
        assert "2023" in url_epl

        url_liga = scraper._build_league_url("la-liga", "2023-2024")
        assert "La_liga" in url_liga

    def test_position_mapping(self, scraper):
        """Test position code mapping."""
        assert scraper.POSITION_MAP["F"] == "FW"
        assert scraper.POSITION_MAP["M"] == "MF"
        assert scraper.POSITION_MAP["D"] == "DF"
        assert scraper.POSITION_MAP["GK"] == "GK"
        assert scraper.POSITION_MAP["S"] == "FW"

    def test_decode_json_string(self, scraper):
        """Test JSON string decoding."""
        # Normal string should pass through
        normal = '{"key": "value"}'
        assert scraper._decode_json_string(normal) == normal

    def test_extract_players_data(self, scraper, league_html):
        """Test player data extraction from HTML."""
        players = scraper._extract_players_data(league_html)

        assert len(players) == 3
        assert players[0]["player_name"] == "Cody Gakpo"
        assert players[1]["player_name"] == "Test Midfielder"
        assert players[2]["player_name"] == "Young Defender"

    def test_extract_players_data_missing(self, scraper):
        """Test extraction with missing playersData."""
        html = "<html><body>No data here</body></html>"
        players = scraper._extract_players_data(html)
        assert players == []

    def test_parse_player_record(self, scraper):
        """Test player record parsing."""
        raw = {
            "id": "1234",
            "player_name": "Cody Gakpo",
            "games": "34",
            "time": "2856",
            "goals": "21",
            "assists": "15",
            "xG": "15.8",
            "xA": "10.2",
            "npg": "18",
            "npxG": "12.4",
            "xGChain": "22.5",
            "xGBuildup": "8.3",
            "shots": "98",
            "key_passes": "67",
            "yellow_cards": "3",
            "red_cards": "0",
            "position": "F",
            "team_title": "PSV Eindhoven",
        }

        player = scraper._parse_player_record(raw, "eredivisie", "2023-2024")

        assert player["player_id"] == "1234"
        assert player["name"] == "Cody Gakpo"
        assert player["position"] == "FW"  # F -> FW
        assert player["team"] == "PSV Eindhoven"
        assert player["games"] == 34
        assert player["minutes"] == 2856
        assert player["goals"] == 21
        assert player["assists"] == 15
        assert player["xg"] == 15.8
        assert player["xa"] == 10.2
        assert player["npg"] == 18
        assert player["npxg"] == 12.4
        assert player["shots"] == 98
        assert player["key_passes"] == 67
        assert player["league"] == "eredivisie"
        assert player["season"] == "2023-2024"
        assert player["source"] == "understat"

    def test_per90_calculations(self, scraper):
        """Test per-90 stat calculations."""
        raw = {
            "id": "1234",
            "player_name": "Test Player",
            "time": "900",  # Exactly 10 90s
            "goals": "10",
            "assists": "5",
            "xG": "8.0",
            "xA": "4.0",
            "npxG": "7.0",
            "position": "F",
        }

        player = scraper._parse_player_record(raw, "eredivisie", "2023-2024")

        assert player["minutes_90s"] == 10.0
        assert player["goals_per90"] == 1.0
        assert player["assists_per90"] == 0.5
        assert player["xg_per90"] == 0.8
        assert player["xa_per90"] == 0.4
        assert player["npxg_per90"] == 0.7

    def test_zero_minutes_handling(self, scraper):
        """Test handling of players with zero minutes."""
        raw = {
            "id": "1234",
            "player_name": "Bench Warmer",
            "time": "0",
            "goals": "0",
            "assists": "0",
            "xG": "0",
            "xA": "0",
            "npxG": "0",
            "position": "F",
        }

        player = scraper._parse_player_record(raw, "eredivisie", "2023-2024")

        assert player["minutes"] == 0
        assert player["minutes_90s"] == 0.0
        assert player["goals_per90"] is None
        assert player["assists_per90"] is None
        assert player["xg_per90"] is None

    def test_overperformance_calculations(self, scraper):
        """Test xG/xA overperformance calculations."""
        raw = {
            "id": "1234",
            "player_name": "Clinical Striker",
            "time": "900",
            "goals": "15",  # 5 goals above xG
            "assists": "8",  # 3 assists above xA
            "xG": "10.0",
            "xA": "5.0",
            "npxG": "9.0",
            "position": "F",
        }

        player = scraper._parse_player_record(raw, "eredivisie", "2023-2024")

        assert player["xg_overperformance"] == 5.0
        assert player["xa_overperformance"] == 3.0

    @patch.object(UnderstatScraper, "fetch")
    def test_scrape_league_season(self, mock_fetch, scraper, league_html):
        """Test full league season scraping with mocked requests."""
        mock_fetch.return_value = league_html

        players = scraper.scrape_league_season("premier-league", "2023-2024")

        assert len(players) == 3

        # Check metadata is added
        for player in players:
            assert player["league"] == "premier-league"
            assert player["season"] == "2023-2024"
            assert player["source"] == "understat"

        # Check Gakpo
        gakpo = next(p for p in players if p["name"] == "Cody Gakpo")
        assert gakpo["goals"] == 21
        assert gakpo["position"] == "FW"
        assert gakpo["xg"] == 15.8

        # Check midfielder
        midfielder = next(p for p in players if p["name"] == "Test Midfielder")
        assert midfielder["position"] == "MF"
        assert midfielder["assists"] == 8

    def test_unknown_league_raises_error(self, scraper):
        """Test that unknown league raises ValueError."""
        with pytest.raises(ValueError, match="Unknown league"):
            scraper.scrape_league_season("fake-league", "2023-2024")

    def test_unavailable_league_raises_error(self, scraper):
        """Test that unavailable league raises descriptive error."""
        with pytest.raises(ValueError, match="not available on Understat"):
            scraper.scrape_league_season("championship", "2023-2024")

    def test_eredivisie_unavailable(self, scraper):
        """Test that Eredivisie (dropped by Understat) raises error."""
        with pytest.raises(ValueError, match="not available on Understat"):
            scraper.scrape_league_season("eredivisie", "2023-2024")


class TestUnderstatIntegration:
    """Integration tests for Understat scraper."""

    @pytest.mark.skip(reason="Integration test - hits real API")
    def test_real_epl_scrape(self, tmp_path):
        """Test scraping real EPL data."""
        scraper = UnderstatScraper(cache_dir=tmp_path, rate_limit=5)
        players = scraper.scrape_league_season("premier-league", "2023-2024")

        assert len(players) > 0
        assert all("name" in p for p in players)
        assert all("player_id" in p for p in players)
        assert all("xg" in p for p in players)

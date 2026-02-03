"""Tests for data scrapers."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scrapers import BaseScraper, FBrefScraper


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

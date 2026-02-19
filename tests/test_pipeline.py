"""Tests for data pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data import DataPipeline, ScrapeResult, ValidationReport


class TestScrapeResult:
    """Tests for ScrapeResult dataclass."""

    def test_success_when_no_errors(self):
        """Test that success is True when no errors."""
        result = ScrapeResult(league="eredivisie", season="2023-2024")
        assert result.success is True

    def test_success_false_with_errors(self):
        """Test that success is False when there are errors."""
        result = ScrapeResult(
            league="eredivisie",
            season="2023-2024",
            errors=["Some error"],
        )
        assert result.success is False

    def test_total_records(self):
        """Test total_records calculation."""
        result = ScrapeResult(
            league="eredivisie",
            season="2023-2024",
            fbref_count=100,
            transfermarkt_count=120,
            understat_count=80,
        )
        assert result.total_records == 300

    def test_str_representation(self):
        """Test string representation."""
        result = ScrapeResult(
            league="eredivisie",
            season="2023-2024",
            fbref_count=100,
            transfermarkt_count=120,
            understat_count=80,
        )
        str_repr = str(result)
        assert "eredivisie" in str_repr
        assert "2023-2024" in str_repr
        assert "OK" in str_repr


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        report = ValidationReport(
            league="eredivisie",
            season="2023-2024",
            fbref_players=100,
            transfermarkt_players=120,
            understat_players=80,
            unified_players=130,
            match_rate=0.75,
        )
        str_repr = str(report)
        assert "eredivisie" in str_repr
        assert "75.0%" in str_repr


class TestDataPipeline:
    """Tests for DataPipeline class."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline with temp database."""
        db_path = tmp_path / "test.duckdb"
        cache_dir = tmp_path / "cache"
        pipeline = DataPipeline(
            db_path=db_path,
            cache_dir=cache_dir,
            rate_limit=0,
        )
        yield pipeline
        pipeline.close()

    def test_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline.db is not None
        assert pipeline.fbref is not None
        assert pipeline.transfermarkt is not None
        assert pipeline.understat is not None

    def test_league_constants(self, pipeline):
        """Test league constant definitions."""
        assert "eredivisie" in pipeline.ALL_LEAGUES
        assert "premier-league" in pipeline.ALL_LEAGUES
        assert len(pipeline.UNDERSTAT_LEAGUES) == 5  # Eredivisie dropped

        # Understat should be subset of all leagues
        for league in pipeline.UNDERSTAT_LEAGUES:
            assert league in pipeline.ALL_LEAGUES

    def test_priority_leagues(self, pipeline):
        """Test priority league groupings."""
        assert len(pipeline.PRIORITY_1_LEAGUES) == 4
        assert len(pipeline.PRIORITY_2_LEAGUES) == 4
        assert "eredivisie" in pipeline.PRIORITY_1_LEAGUES
        assert "serie-b" in pipeline.PRIORITY_2_LEAGUES

    def test_unknown_league_returns_error(self, pipeline):
        """Test that unknown league returns error in result."""
        result = pipeline.scrape_league_season("fake-league", "2023-2024")
        assert not result.success
        assert "Unknown league" in result.errors[0]

    def test_get_supported_leagues(self, pipeline):
        """Test getting supported leagues."""
        all_leagues = pipeline.get_supported_leagues()
        assert len(all_leagues) == 13

        understat_leagues = pipeline.get_supported_leagues("understat")
        assert len(understat_leagues) == 5  # Eredivisie dropped

        fbref_leagues = pipeline.get_supported_leagues("fbref")
        assert len(fbref_leagues) == 13

    def test_context_manager(self, tmp_path):
        """Test pipeline as context manager."""
        db_path = tmp_path / "test.duckdb"
        with DataPipeline(db_path=db_path, rate_limit=0) as pipeline:
            stats = pipeline.get_scrape_summary()
            assert stats["total_records"] == 0

    @patch.object(DataPipeline, "scrape_league_season")
    def test_scrape_multiple_leagues(self, mock_scrape, pipeline):
        """Test scraping multiple leagues."""
        mock_scrape.return_value = ScrapeResult(
            league="test",
            season="2023-2024",
            fbref_count=100,
        )

        results = pipeline.scrape_multiple_leagues(
            ["eredivisie", "championship"],
            "2023-2024",
            show_progress=False,
        )

        assert len(results) == 2
        assert mock_scrape.call_count == 2

    @patch.object(DataPipeline, "scrape_league_season")
    def test_scrape_multiple_seasons(self, mock_scrape, pipeline):
        """Test scraping multiple seasons."""
        mock_scrape.return_value = ScrapeResult(
            league="eredivisie",
            season="test",
            fbref_count=100,
        )

        results = pipeline.scrape_multiple_seasons(
            "eredivisie",
            ["2022-2023", "2023-2024"],
            show_progress=False,
        )

        assert len(results) == 2
        assert mock_scrape.call_count == 2

    @patch.object(DataPipeline, "scrape_league_season")
    def test_scrape_feeder_leagues_priority_1(self, mock_scrape, pipeline):
        """Test scraping priority 1 feeder leagues."""
        mock_scrape.return_value = ScrapeResult(
            league="test",
            season="2023-2024",
            fbref_count=100,
        )

        results = pipeline.scrape_feeder_leagues(
            "2023-2024",
            priority=1,
            show_progress=False,
        )

        assert len(results) == 4  # Priority 1 has 4 leagues
        assert mock_scrape.call_count == 4

    @patch.object(DataPipeline, "scrape_league_season")
    def test_scrape_feeder_leagues_priority_2(self, mock_scrape, pipeline):
        """Test scraping priority 1+2 feeder leagues."""
        mock_scrape.return_value = ScrapeResult(
            league="test",
            season="2023-2024",
            fbref_count=100,
        )

        results = pipeline.scrape_feeder_leagues(
            "2023-2024",
            priority=2,
            show_progress=False,
        )

        assert len(results) == 8  # Priority 1 + 2 has 8 leagues
        assert mock_scrape.call_count == 8


class TestDataPipelineWithMockedScrapers:
    """Tests for DataPipeline with mocked scrapers."""

    @pytest.fixture
    def pipeline_with_mocks(self, tmp_path):
        """Create pipeline with mocked scrapers."""
        db_path = tmp_path / "test.duckdb"
        cache_dir = tmp_path / "cache"

        with patch("src.data.pipeline.FBrefScraper") as mock_fbref, \
             patch("src.data.pipeline.TransfermarktScraper") as mock_tm, \
             patch("src.data.pipeline.UnderstatScraper") as mock_us:

            # Configure mocks
            mock_fbref_instance = MagicMock()
            mock_fbref_instance.scrape_league_season.return_value = [
                {"player_id": "fb1", "name": "Test Player", "league": "eredivisie", "season": "2023-2024"}
            ]
            mock_fbref.return_value = mock_fbref_instance

            mock_tm_instance = MagicMock()
            mock_tm_instance.scrape_league_season.return_value = [
                {"player_id": "tm1", "name": "Test Player", "league": "eredivisie", "season": "2023-2024"}
            ]
            mock_tm.return_value = mock_tm_instance

            mock_us_instance = MagicMock()
            mock_us_instance.scrape_league_season.return_value = [
                {"player_id": "us1", "name": "Test Player", "league": "eredivisie", "season": "2023-2024"}
            ]
            mock_us.return_value = mock_us_instance

            pipeline = DataPipeline(
                db_path=db_path,
                cache_dir=cache_dir,
                rate_limit=0,
            )

            yield pipeline, mock_fbref_instance, mock_tm_instance, mock_us_instance

            pipeline.close()

    def test_scrape_all_sources(self, pipeline_with_mocks):
        """Test scraping from all sources."""
        pipeline, mock_fbref, mock_tm, mock_us = pipeline_with_mocks

        # Use premier-league since it has all 3 sources (Eredivisie dropped from Understat)
        result = pipeline.scrape_league_season("premier-league", "2023-2024")

        assert result.success
        assert result.fbref_count == 1
        assert result.transfermarkt_count == 1
        assert result.understat_count == 1
        assert result.total_records == 3

        mock_fbref.scrape_league_season.assert_called_once()
        mock_tm.scrape_league_season.assert_called_once()
        mock_us.scrape_league_season.assert_called_once()

    def test_scrape_only_fbref(self, pipeline_with_mocks):
        """Test scraping only from FBref."""
        pipeline, mock_fbref, mock_tm, mock_us = pipeline_with_mocks

        result = pipeline.scrape_league_season(
            "eredivisie", "2023-2024", sources=["fbref"]
        )

        assert result.success
        assert result.fbref_count == 1
        assert result.transfermarkt_count == 0
        assert result.understat_count == 0

        mock_fbref.scrape_league_season.assert_called_once()
        mock_tm.scrape_league_season.assert_not_called()
        mock_us.scrape_league_season.assert_not_called()

    def test_understat_skipped_for_unsupported_league(self, pipeline_with_mocks):
        """Test that Understat is skipped for unsupported leagues."""
        pipeline, mock_fbref, mock_tm, mock_us = pipeline_with_mocks

        # Championship is not supported by Understat
        result = pipeline.scrape_league_season("championship", "2023-2024")

        assert result.success
        assert result.fbref_count == 1
        assert result.transfermarkt_count == 1
        assert result.understat_count == 0  # Skipped

        mock_us.scrape_league_season.assert_not_called()

    def test_handles_scraper_error(self, pipeline_with_mocks):
        """Test that scraper errors are captured."""
        pipeline, mock_fbref, mock_tm, mock_us = pipeline_with_mocks

        # Make FBref raise an error
        mock_fbref.scrape_league_season.side_effect = Exception("FBref is down")

        # Use premier-league since it has all 3 sources
        result = pipeline.scrape_league_season("premier-league", "2023-2024")

        assert not result.success
        assert "FBref error" in result.errors[0]
        # Other scrapers should still work
        assert result.transfermarkt_count == 1
        assert result.understat_count == 1


class TestValidation:
    """Tests for data validation."""

    @pytest.fixture
    def pipeline_with_data(self, tmp_path):
        """Create pipeline with some test data."""
        db_path = tmp_path / "test.duckdb"
        pipeline = DataPipeline(db_path=db_path, rate_limit=0)

        # Insert test data directly
        pipeline.db.insert_fbref_players([
            {"player_id": "fb1", "name": "Player 1", "team": "Team A", "league": "eredivisie", "season": "2023-2024"},
            {"player_id": "fb2", "name": "Player 2", "team": "Team B", "league": "eredivisie", "season": "2023-2024"},
        ])
        pipeline.db.insert_transfermarkt_players([
            {"player_id": "tm1", "name": "Player 1", "team": "Team A", "league": "eredivisie", "season": "2023-2024", "market_value_eur": 1000000},
        ])
        pipeline.db.insert_understat_players([
            {"player_id": "us1", "name": "Player 1", "team": "Team A", "league": "eredivisie", "season": "2023-2024", "xg": 5.0},
        ])

        yield pipeline
        pipeline.close()

    def test_validate_data(self, pipeline_with_data):
        """Test data validation report."""
        report = pipeline_with_data.validate_data("eredivisie", "2023-2024")

        assert report.fbref_players == 2
        assert report.transfermarkt_players == 1
        assert report.understat_players == 1
        assert report.unified_players > 0

    def test_validate_empty_data(self, tmp_path):
        """Test validation with no data."""
        db_path = tmp_path / "test.duckdb"
        with DataPipeline(db_path=db_path, rate_limit=0) as pipeline:
            report = pipeline.validate_data("eredivisie", "2023-2024")

            assert report.fbref_players == 0
            assert report.transfermarkt_players == 0
            assert report.understat_players == 0
            assert "No FBref data found" in report.warnings
            assert "No Transfermarkt data found" in report.warnings


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_scrape_args(self):
        """Test scrape command argument parsing."""
        from src.data.cli import main
        import sys

        # Just test that the module imports correctly
        assert callable(main)

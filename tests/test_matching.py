"""Tests for cross-source fuzzy matching module."""

import numpy as np
import pandas as pd
import pytest

from src.data.matching import (
    MatchResult,
    match_score,
    match_sources,
    normalize_name,
)


class TestNormalizeName:
    """Tests for name normalization."""

    def test_basic_lowercase(self):
        assert normalize_name("Cody Gakpo") == "cody gakpo"

    def test_unicode_transliteration(self):
        assert normalize_name("Dušan Vlahović") == "dusan vlahovic"

    def test_accented_characters(self):
        assert normalize_name("André Silva") == "andre silva"

    def test_strips_whitespace(self):
        assert normalize_name("  Player Name  ") == "player name"

    def test_removes_punctuation(self):
        assert normalize_name("O'Brien Jr.") == "obrien jr"

    def test_preserves_hyphens(self):
        assert normalize_name("Pierre-Emerick") == "pierre-emerick"

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_none_input(self):
        assert normalize_name(None) == ""

    def test_nan_input(self):
        assert normalize_name(float("nan")) == ""


class TestMatchScore:
    """Tests for match score computation."""

    def test_exact_match(self):
        score = match_score("Cody Gakpo", "Cody Gakpo", "PSV", "PSV")
        assert score == 100.0

    def test_close_name_variation(self):
        score = match_score("Cody Gakpo", "C. Gakpo", "PSV", "PSV")
        assert score > 70  # Should be reasonably high

    def test_different_players(self):
        score = match_score("Cody Gakpo", "Memphis Depay", "PSV", "Barcelona")
        assert score < 50

    def test_same_name_different_team(self):
        score = match_score("Player Name", "Player Name", "Team A", "Team B")
        # Name match is 100 * 0.8 = 80, team mismatch reduces it
        assert score >= 80

    def test_unicode_matching(self):
        score = match_score("Dušan Vlahović", "Dusan Vlahovic", "Fiorentina", "Fiorentina")
        assert score == 100.0

    def test_empty_names(self):
        assert match_score("", "Player", "", "") == 0.0

    def test_none_names(self):
        assert match_score(None, "Player", "", "") == 0.0

    def test_name_weight_dominates(self):
        # Name weight is 0.8, team weight is 0.2
        score1 = match_score("Perfect Match", "Perfect Match", "Team A", "Team Z")
        score2 = match_score("Completely Different", "No Match At All", "Team A", "Team A")
        assert score1 > score2


class TestMatchSources:
    """Tests for source matching and enrichment."""

    @pytest.fixture
    def fbref_df(self):
        return pd.DataFrame({
            "player_id": ["fb1", "fb2", "fb3"],
            "name": ["Cody Gakpo", "Xavi Simons", "Jurrien Timber"],
            "team": ["PSV", "PSV", "Ajax"],
            "league": ["eredivisie", "eredivisie", "eredivisie"],
            "season": ["2021-22", "2021-22", "2021-22"],
            "position_group": ["FW", "MF", "DF"],
            "age": [22, 19, 20],
            "minutes": [2500, 1200, 2800],
            "xg": [np.nan, np.nan, np.nan],
            "npxg": [np.nan, np.nan, np.nan],
            "xg_assist": [np.nan, np.nan, np.nan],
        })

    @pytest.fixture
    def tm_df(self):
        return pd.DataFrame({
            "player_id": ["tm1", "tm2", "tm3"],
            "name": ["Cody Gakpo", "Xavi Simons", "Jurriën Timber"],
            "team": ["PSV", "PSV", "Ajax"],
            "league": ["eredivisie", "eredivisie", "eredivisie"],
            "season": ["2021-22", "2021-22", "2021-22"],
            "market_value_eur": [25000000, 8000000, 20000000],
        })

    @pytest.fixture
    def understat_df(self):
        return pd.DataFrame({
            "player_id": ["us1", "us2"],
            "name": ["Cody Gakpo", "Xavi Simons"],
            "team": ["PSV", "PSV"],
            "league": ["eredivisie", "eredivisie"],
            "season": ["2021-22", "2021-22"],
            "xg": [15.8, 5.0],
            "xa": [10.2, 4.0],
            "npxg": [12.4, 4.5],
            "xg_chain": [8.5, 3.2],
            "xg_buildup": [5.1, 2.8],
            "xg_overperformance": [2.1, 0.5],
            "xa_overperformance": [1.0, 0.3],
            "xg_per90": [0.50, 0.42],
            "xa_per90": [0.32, 0.33],
            "npxg_per90": [0.39, 0.38],
            "key_passes": [67, 30],
            "shots": [98, 40],
        })

    def test_enriches_with_market_value(self, fbref_df, tm_df, understat_df):
        result_df, result = match_sources(fbref_df, tm_df, understat_df)
        assert "market_value_eur" in result_df.columns
        # Gakpo should have market value
        gakpo = result_df[result_df["name"] == "Cody Gakpo"].iloc[0]
        assert gakpo["market_value_eur"] == 25000000

    def test_enriches_with_xg_chain(self, fbref_df, tm_df, understat_df):
        result_df, result = match_sources(fbref_df, tm_df, understat_df)
        gakpo = result_df[result_df["name"] == "Cody Gakpo"].iloc[0]
        assert gakpo["xg_chain"] == 8.5

    def test_match_result_counts(self, fbref_df, tm_df, understat_df):
        _, result = match_sources(fbref_df, tm_df, understat_df)
        assert result.fbref_records == 3
        assert result.tm_matched == 3  # All should match
        assert result.understat_matched == 2  # Only 2 in understat

    def test_missing_understat_graceful(self, fbref_df, tm_df):
        """Understat not available for some leagues → NaN."""
        empty_us = pd.DataFrame(columns=[
            "player_id", "name", "team", "league", "season",
            "xg", "xa", "npxg",
            "xg_chain", "xg_buildup", "xg_overperformance", "xa_overperformance",
            "xg_per90", "xa_per90", "npxg_per90", "key_passes", "shots",
        ])
        result_df, result = match_sources(fbref_df, tm_df, empty_us)
        timber = result_df[result_df["name"] == "Jurrien Timber"].iloc[0]
        assert pd.isna(timber["xg_chain"])
        assert result.understat_matched == 0

    def test_empty_tm_graceful(self, fbref_df, understat_df):
        empty_tm = pd.DataFrame(columns=[
            "player_id", "name", "team", "league", "season", "market_value_eur",
        ])
        result_df, result = match_sources(fbref_df, empty_tm, understat_df)
        assert result.tm_matched == 0
        assert pd.isna(result_df.iloc[0]["market_value_eur"])

    def test_preserves_fbref_columns(self, fbref_df, tm_df, understat_df):
        result_df, _ = match_sources(fbref_df, tm_df, understat_df)
        assert "player_id" in result_df.columns
        assert "position_group" in result_df.columns
        assert len(result_df) == 3  # Same count as input

    def test_match_confidence_stored(self, fbref_df, tm_df, understat_df):
        result_df, _ = match_sources(fbref_df, tm_df, understat_df)
        gakpo = result_df[result_df["name"] == "Cody Gakpo"].iloc[0]
        assert gakpo["match_confidence_tm"] == 100.0  # Exact match
        assert gakpo["match_confidence_us"] == 100.0

    def test_threshold_filtering(self, fbref_df, tm_df, understat_df):
        """Very high threshold should reject some matches."""
        # Modify TM names to be slightly different
        tm_df_modified = tm_df.copy()
        tm_df_modified.loc[0, "name"] = "C. Gakpo"  # Abbreviated

        result_df, result = match_sources(fbref_df, tm_df_modified, understat_df, threshold=99)
        # The abbreviated name shouldn't match at threshold=99
        assert result.tm_matched < 3

    def test_cross_league_no_match(self, fbref_df, tm_df, understat_df):
        """Players in different leagues shouldn't match."""
        tm_df_diff = tm_df.copy()
        tm_df_diff["league"] = "premier-league"  # Different league

        result_df, result = match_sources(fbref_df, tm_df_diff, understat_df)
        assert result.tm_matched == 0  # No matches across leagues

    def test_backfills_xg_from_understat(self, fbref_df, tm_df, understat_df):
        """FBref NULL xG should be backfilled from Understat."""
        result_df, _ = match_sources(fbref_df, tm_df, understat_df)
        gakpo = result_df[result_df["name"] == "Cody Gakpo"].iloc[0]
        # FBref xg was NaN, should be backfilled from Understat
        assert gakpo["xg"] == 15.8
        assert gakpo["npxg"] == 12.4
        assert gakpo["xg_assist"] == 10.2

    def test_preserves_existing_fbref_xg(self, fbref_df, tm_df, understat_df):
        """If FBref already has xG, don't overwrite with Understat."""
        fbref_df.loc[0, "xg"] = 16.0  # Gakpo has FBref xG
        result_df, _ = match_sources(fbref_df, tm_df, understat_df)
        gakpo = result_df[result_df["name"] == "Cody Gakpo"].iloc[0]
        assert gakpo["xg"] == 16.0  # Preserved FBref value

    def test_understat_per90_columns(self, fbref_df, tm_df, understat_df):
        """Understat per-90 features should be populated."""
        result_df, _ = match_sources(fbref_df, tm_df, understat_df)
        gakpo = result_df[result_df["name"] == "Cody Gakpo"].iloc[0]
        assert gakpo["us_xg_per90"] == 0.50
        assert gakpo["us_xa_per90"] == 0.32
        assert gakpo["us_npxg_per90"] == 0.39
        assert gakpo["us_key_passes"] == 67
        assert gakpo["us_shots"] == 98

    def test_unmatched_players_have_nan_understat(self, fbref_df, tm_df, understat_df):
        """Players without Understat match should have NaN for us_* columns."""
        result_df, _ = match_sources(fbref_df, tm_df, understat_df)
        timber = result_df[result_df["name"] == "Jurrien Timber"].iloc[0]
        assert pd.isna(timber["us_xg_per90"])
        assert pd.isna(timber["us_shots"])
        # xg should stay NaN too (no Understat match)
        assert pd.isna(timber["xg"])

"""Cross-source fuzzy matching to enrich FBref data with TM and Understat."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from src.storage import PlayerDatabase

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of the matching step."""

    fbref_records: int = 0
    tm_records: int = 0
    understat_records: int = 0
    tm_matched: int = 0
    understat_matched: int = 0
    avg_tm_confidence: float = 0.0
    avg_us_confidence: float = 0.0
    final_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Matching: FBref={self.fbref_records}, TM={self.tm_records}, "
            f"US={self.understat_records}\n"
            f"  TM matched: {self.tm_matched} "
            f"(avg confidence: {self.avg_tm_confidence:.1f})\n"
            f"  US matched: {self.understat_matched} "
            f"(avg confidence: {self.avg_us_confidence:.1f})"
        )


def normalize_name(name: str) -> str:
    """Normalize player name for matching.

    Transliterates unicode, lowercases, strips whitespace and punctuation.

    Args:
        name: Raw player name

    Returns:
        Normalized name string
    """
    if not name or pd.isna(name):
        return ""
    normalized = unidecode(str(name)).lower().strip()
    # Remove punctuation except spaces and hyphens
    normalized = "".join(
        c for c in normalized if c.isalnum() or c in " -"
    )
    return normalized


def match_score(name1: str, name2: str, team1: str = "", team2: str = "") -> float:
    """Compute match score between two player records.

    Weighted combination of name similarity (0.8) and team similarity (0.2).

    Args:
        name1: First player name (raw)
        name2: Second player name (raw)
        team1: First player team (raw)
        team2: Second player team (raw)

    Returns:
        Match score 0-100
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    if not norm1 or not norm2:
        return 0.0

    name_sim = fuzz.ratio(norm1, norm2)

    team_sim = 0.0
    if team1 and team2:
        t1 = normalize_name(team1)
        t2 = normalize_name(team2)
        if t1 and t2:
            team_sim = fuzz.ratio(t1, t2)

    return name_sim * 0.8 + team_sim * 0.2


def _match_single_source(
    fbref_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_name: str,
    threshold: float = 80,
) -> dict[int, tuple[int, float]]:
    """Match FBref rows to a source DataFrame within same league+season groups.

    Args:
        fbref_df: FBref records
        source_df: Source records (TM or Understat)
        source_name: Name for logging
        threshold: Minimum match score

    Returns:
        Dict mapping fbref row index -> (source row index, confidence score)
    """
    matches = {}

    if source_df.empty:
        return matches

    # Group by league+season
    for (league, season), fb_group in fbref_df.groupby(["league", "season"]):
        src_mask = (source_df["league"] == league) & (source_df["season"] == season)
        src_group = source_df[src_mask]

        if src_group.empty:
            continue

        # Pre-normalize source names for efficiency
        src_names = {
            idx: normalize_name(row["name"])
            for idx, row in src_group.iterrows()
        }
        src_teams = {
            idx: normalize_name(str(row.get("team", "")))
            for idx, row in src_group.iterrows()
        }

        for fb_idx, fb_row in fb_group.iterrows():
            fb_name = normalize_name(fb_row["name"])
            fb_team = normalize_name(str(fb_row.get("team", "")))

            best_score = 0.0
            best_idx = None

            for src_idx in src_group.index:
                name_sim = fuzz.ratio(fb_name, src_names[src_idx])
                team_sim = fuzz.ratio(fb_team, src_teams[src_idx]) if fb_team and src_teams[src_idx] else 0.0
                score = name_sim * 0.8 + team_sim * 0.2

                if score > best_score:
                    best_score = score
                    best_idx = src_idx

            if best_score >= threshold and best_idx is not None:
                matches[fb_idx] = (best_idx, best_score)

    return matches


def match_sources(
    fbref_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    understat_df: pd.DataFrame,
    threshold: float = 80,
) -> tuple[pd.DataFrame, MatchResult]:
    """Match FBref records with TM and Understat, enrich with extra columns.

    Args:
        fbref_df: Cleaned FBref DataFrame
        tm_df: Transfermarkt DataFrame
        understat_df: Understat DataFrame
        threshold: Minimum match score (0-100)

    Returns:
        Tuple of (enriched DataFrame, MatchResult)
    """
    result = MatchResult(
        fbref_records=len(fbref_df),
        tm_records=len(tm_df),
        understat_records=len(understat_df),
    )

    enriched = fbref_df.copy()

    # Initialize enrichment columns
    enriched["market_value_eur"] = np.nan
    enriched["match_confidence_tm"] = np.nan
    enriched["xg_chain"] = np.nan
    enriched["xg_buildup"] = np.nan
    enriched["understat_xg_overperformance"] = np.nan
    enriched["understat_xa_overperformance"] = np.nan
    enriched["match_confidence_us"] = np.nan
    # Understat-specific per-90 features
    enriched["us_xg_per90"] = np.nan
    enriched["us_xa_per90"] = np.nan
    enriched["us_npxg_per90"] = np.nan
    enriched["us_key_passes"] = np.nan
    enriched["us_shots"] = np.nan

    # Match Transfermarkt
    if not tm_df.empty:
        tm_matches = _match_single_source(fbref_df, tm_df, "Transfermarkt", threshold)
        for fb_idx, (tm_idx, confidence) in tm_matches.items():
            enriched.loc[fb_idx, "market_value_eur"] = tm_df.loc[tm_idx, "market_value_eur"]
            enriched.loc[fb_idx, "match_confidence_tm"] = confidence
        result.tm_matched = len(tm_matches)
        if tm_matches:
            result.avg_tm_confidence = np.mean([c for _, c in tm_matches.values()])

    # Match Understat
    if not understat_df.empty:
        us_matches = _match_single_source(fbref_df, understat_df, "Understat", threshold)
        for fb_idx, (us_idx, confidence) in us_matches.items():
            enriched.loc[fb_idx, "xg_chain"] = understat_df.loc[us_idx, "xg_chain"]
            enriched.loc[fb_idx, "xg_buildup"] = understat_df.loc[us_idx, "xg_buildup"]
            enriched.loc[fb_idx, "understat_xg_overperformance"] = understat_df.loc[us_idx, "xg_overperformance"]
            enriched.loc[fb_idx, "understat_xa_overperformance"] = understat_df.loc[us_idx, "xa_overperformance"]
            enriched.loc[fb_idx, "match_confidence_us"] = confidence

            # Backfill FBref xG from Understat when FBref has NULL
            for col_fb, col_us in [("xg", "xg"), ("npxg", "npxg"), ("xg_assist", "xa")]:
                if col_fb in enriched.columns and col_us in understat_df.columns:
                    if pd.isna(enriched.loc[fb_idx, col_fb]):
                        enriched.loc[fb_idx, col_fb] = understat_df.loc[us_idx, col_us]

            # Populate Understat-specific per-90 features
            for col_enr, col_us in [
                ("us_xg_per90", "xg_per90"),
                ("us_xa_per90", "xa_per90"),
                ("us_npxg_per90", "npxg_per90"),
                ("us_key_passes", "key_passes"),
                ("us_shots", "shots"),
            ]:
                if col_us in understat_df.columns:
                    enriched.loc[fb_idx, col_enr] = understat_df.loc[us_idx, col_us]
        result.understat_matched = len(us_matches)
        if us_matches:
            result.avg_us_confidence = np.mean([c for _, c in us_matches.values()])

    result.final_count = len(enriched)
    logger.info(str(result))
    return enriched, result


def enrich_from_sources(
    db: PlayerDatabase,
    fbref_df: pd.DataFrame,
    threshold: float = 80,
) -> tuple[pd.DataFrame, MatchResult]:
    """Main entry point: reads TM and Understat from DB, matches to FBref.

    Args:
        db: PlayerDatabase instance
        fbref_df: Cleaned FBref DataFrame (from cleaning step)
        threshold: Match confidence threshold

    Returns:
        Tuple of (enriched DataFrame, MatchResult)
    """
    tm_df = db.get_transfermarkt_players()
    understat_df = db.get_understat_players()

    return match_sources(fbref_df, tm_df, understat_df, threshold)

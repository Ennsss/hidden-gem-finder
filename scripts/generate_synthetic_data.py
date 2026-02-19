"""Generate synthetic player data for end-to-end pipeline validation.

Populates all 3 DuckDB tables (fbref_players, transfermarkt_players, understat_players)
with realistic, internally-consistent data that exercises every pipeline stage.

Usage:
    python scripts/generate_synthetic_data.py --db data/players.duckdb --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage import PlayerDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Constants ----------

SOURCE_LEAGUES = [
    "eredivisie", "primeira-liga", "belgian-pro-league", "championship",
    "serie-b", "ligue-2", "austrian-bundesliga", "scottish-premiership",
]

TARGET_LEAGUES = [
    "premier-league", "la-liga", "bundesliga", "serie-a", "ligue-1",
]

# Understat only covers target leagues
UNDERSTAT_LEAGUES = set(TARGET_LEAGUES)

# Main seasons for source+target data
MAIN_SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23",
]

# Extra target-league seasons to extend max_season_year for label window
EXTRA_TARGET_SEASONS = ["2023-24", "2024-25", "2025-26"]

# Name pools
FIRST_NAMES = [
    "Lucas", "Marco", "Davi", "Santiago", "Mohamed", "Liam", "Noah",
    "Mateo", "Hugo", "Arthur", "Jules", "Luca", "Finn", "Jan", "Adam",
    "Nils", "Erik", "Joao", "Pedro", "Rafael", "Sander", "Thomas",
    "Florian", "Kevin", "Milan", "Stefan", "Patrick", "Daniel",
    "Viktor", "Andreas", "Tobias", "Alexander", "Christian", "Lukas",
    "Maximilian", "Felix", "Leon", "Paul", "Simon", "David", "Oliver",
    "William", "James", "Benjamin", "Sebastian", "Gabriel", "Adrian",
    "Carlos", "Diego", "Felipe", "Andres", "Ivan", "Nikola", "Marko",
    "Luka", "Filip", "Mikael", "Oscar", "Emil", "Axel", "Isak",
]

LAST_NAMES = [
    "Silva", "Santos", "De Jong", "Jansen", "Van Dijk", "Mueller",
    "Schmidt", "Martin", "Garcia", "Lopez", "Rodriguez", "Martinez",
    "Fernandez", "Gonzalez", "Hernandez", "Dubois", "Leroy", "Moreau",
    "Rossi", "Bianchi", "Romano", "Eriksen", "Andersen", "Nielsen",
    "Berg", "Lindqvist", "Svensson", "Kowalski", "Novak", "Horvat",
    "Petrov", "Ivanov", "Weber", "Fischer", "Wagner", "Becker",
    "Richter", "Krause", "Hofmann", "Scholz", "Moller", "Larsen",
    "Poulsen", "Christensen", "Jensen", "Pedersen", "Olsen", "Bakker",
    "Visser", "Smit", "Meijer", "Vos", "Peters", "Hendriks",
    "Almeida", "Costa", "Pereira", "Oliveira", "Carvalho", "Sousa",
]

NATIONALITIES = [
    "Netherlands", "Portugal", "Belgium", "England", "Italy", "France",
    "Austria", "Scotland", "Germany", "Spain", "Brazil", "Argentina",
    "Colombia", "Serbia", "Croatia", "Sweden", "Denmark", "Norway",
]

POSITIONS = ["FW", "MF", "DF"]
DUAL_POSITIONS = ["FW,MF", "MF,DF"]  # some players get dual format

# Team name pools per league
TEAMS = {
    "eredivisie": ["Ajax", "PSV", "Feyenoord", "AZ Alkmaar", "FC Utrecht", "Vitesse", "FC Twente", "SC Heerenveen"],
    "primeira-liga": ["Benfica", "Porto", "Sporting CP", "Braga", "Vitoria Guimaraes", "Rio Ave", "Gil Vicente", "Famalicao"],
    "belgian-pro-league": ["Club Brugge", "Anderlecht", "Genk", "Standard Liege", "Gent", "Antwerp", "Mechelen", "Charleroi"],
    "championship": ["Leeds United", "Norwich City", "Brentford", "Watford", "Swansea City", "Barnsley", "Reading", "Bournemouth"],
    "serie-b": ["Lecce", "Cremonese", "Monza", "Pisa", "Brescia", "Benevento", "Frosinone", "Parma"],
    "ligue-2": ["Toulouse", "Ajaccio", "Auxerre", "Sochaux", "Paris FC", "Guingamp", "Caen", "Le Havre"],
    "austrian-bundesliga": ["Red Bull Salzburg", "Rapid Wien", "Sturm Graz", "LASK", "Wolfsberger AC", "Austria Wien", "Hartberg", "Altach"],
    "scottish-premiership": ["Celtic", "Rangers", "Aberdeen", "Hibernian", "Hearts", "Dundee United", "St Johnstone", "Motherwell"],
    "premier-league": ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham", "Newcastle", "Aston Villa"],
    "la-liga": ["Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla", "Villarreal", "Real Sociedad", "Real Betis", "Valencia"],
    "bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Wolfsburg", "Frankfurt", "Freiburg", "Union Berlin"],
    "serie-a": ["Juventus", "Inter Milan", "AC Milan", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"],
    "ligue-1": ["PSG", "Marseille", "Lyon", "Monaco", "Lille", "Rennes", "Nice", "Lens"],
}


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


class SyntheticDataGenerator:
    """Generates synthetic player data across all 3 sources."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.player_counter = 0
        self.fbref_rows = []
        self.tm_rows = []
        self.understat_rows = []

    def _next_id(self, prefix: str) -> str:
        self.player_counter += 1
        return f"{prefix}_{self.player_counter:05d}"

    def _pick_name(self) -> str:
        first = self.rng.choice(FIRST_NAMES)
        last = self.rng.choice(LAST_NAMES)
        return f"{first} {last}"

    def _pick_position(self) -> str:
        r = self.rng.random()
        if r < 0.05:
            return self.rng.choice(DUAL_POSITIONS)
        return self.rng.choice(POSITIONS)

    def _clamp(self, val, lo=0, hi=None):
        val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    # ---------- Stat generators ----------

    def _gen_fbref_stats(self, talent: float, position: str, minutes: int) -> dict:
        """Generate correlated FBref stats based on latent talent."""
        rng = self.rng
        pos = position.split(",")[0]
        m90 = minutes / 90.0
        games = max(1, int(minutes / rng.uniform(60, 90)))
        games_starts = max(1, int(games * rng.uniform(0.7, 1.0)))

        # Position-specific multipliers
        atk_mult = {"FW": 1.5, "MF": 0.8, "DF": 0.2}.get(pos, 0.5)
        def_mult = {"FW": 0.3, "MF": 0.7, "DF": 1.5}.get(pos, 0.5)
        pass_mult = {"FW": 0.7, "MF": 1.3, "DF": 0.9}.get(pos, 0.8)

        # Core attacking
        goals = self._clamp(int(rng.poisson(talent * 8 * atk_mult * m90 / 34)), 0)
        assists = self._clamp(int(rng.poisson(talent * 5 * atk_mult * m90 / 34)), 0)
        pens_made = self._clamp(int(rng.poisson(0.3 * atk_mult)), 0)
        pens_att = pens_made + self._clamp(int(rng.poisson(0.1)), 0)
        goals_pens = max(0, goals - pens_made)

        xg = max(0.0, goals + rng.normal(0, 1.0))
        npxg = max(0.0, xg - pens_made * 0.76)
        xg_assist = max(0.0, assists + rng.normal(0, 0.8))
        npxg_xg_assist = npxg + xg_assist

        # Shooting
        shots = max(goals, int(rng.poisson(max(1, talent * 30 * atk_mult * m90 / 34))))
        shots_on_target = self._clamp(int(shots * rng.uniform(0.25, 0.50)), 0, shots)
        shots_on_target_pct = (shots_on_target / shots * 100) if shots > 0 else 0.0
        shots_per90 = shots / m90 if m90 > 0 else 0.0
        shots_on_target_per90 = shots_on_target / m90 if m90 > 0 else 0.0
        goals_per_shot = goals / shots if shots > 0 else 0.0
        goals_per_shot_on_target = goals / shots_on_target if shots_on_target > 0 else 0.0
        average_shot_distance = rng.uniform(14, 22)
        shots_free_kicks = self._clamp(int(rng.poisson(0.5)), 0)
        npxg_per_shot = npxg / shots if shots > 0 else 0.0
        xg_net = xg - goals
        npxg_net = npxg - goals_pens

        # Passing
        passes = max(10, int(rng.poisson(max(5, talent * 600 * pass_mult * m90 / 34))))
        passes_completed = self._clamp(int(passes * rng.uniform(0.70, 0.92)), 0, passes)
        passes_pct = (passes_completed / passes * 100) if passes > 0 else 0.0
        passes_total_distance = int(passes * rng.uniform(10, 20))
        passes_progressive_distance = int(passes * rng.uniform(2, 6))

        short_frac = rng.uniform(0.3, 0.5)
        med_frac = rng.uniform(0.3, 0.4)
        long_frac = 1.0 - short_frac - med_frac

        passes_short = max(1, int(passes * short_frac))
        passes_completed_short = self._clamp(int(passes_short * rng.uniform(0.80, 0.95)), 0, passes_short)
        passes_pct_short = (passes_completed_short / passes_short * 100) if passes_short > 0 else 0.0

        passes_medium = max(1, int(passes * med_frac))
        passes_completed_medium = self._clamp(int(passes_medium * rng.uniform(0.75, 0.90)), 0, passes_medium)
        passes_pct_medium = (passes_completed_medium / passes_medium * 100) if passes_medium > 0 else 0.0

        passes_long = max(1, int(passes * long_frac))
        passes_completed_long = self._clamp(int(passes_long * rng.uniform(0.50, 0.75)), 0, passes_long)
        passes_pct_long = (passes_completed_long / passes_long * 100) if passes_long > 0 else 0.0

        assisted_shots = self._clamp(int(rng.poisson(max(1, talent * 15 * pass_mult * m90 / 34))), 0)
        passes_into_final_third = self._clamp(int(rng.poisson(max(1, talent * 20 * pass_mult * m90 / 34))), 0)
        passes_into_penalty_area = self._clamp(int(rng.poisson(max(1, talent * 8 * atk_mult * m90 / 34))), 0)
        crosses_into_penalty_area = self._clamp(int(rng.poisson(max(1, talent * 4 * m90 / 34))), 0)

        # Progressive actions
        progressive_passes = self._clamp(int(rng.poisson(max(1, talent * 40 * pass_mult * m90 / 34))), 0)
        progressive_carries = self._clamp(int(rng.poisson(max(1, talent * 25 * m90 / 34))), 0)
        progressive_passes_received = self._clamp(int(rng.poisson(max(1, talent * 30 * atk_mult * m90 / 34))), 0)

        # Defense
        tackles = self._clamp(int(rng.poisson(max(1, talent * 30 * def_mult * m90 / 34))), 0)
        tackles_won = self._clamp(int(tackles * rng.uniform(0.4, 0.7)), 0, tackles)
        tackles_def_3rd = self._clamp(int(tackles * rng.uniform(0.3, 0.5) * def_mult), 0)
        tackles_mid_3rd = self._clamp(int(tackles * rng.uniform(0.3, 0.4)), 0)
        tackles_att_3rd = self._clamp(int(tackles * rng.uniform(0.1, 0.3) * atk_mult), 0)
        challenges = max(tackles, tackles + self._clamp(int(rng.poisson(5)), 0))
        challenge_tackles = tackles_won
        challenge_tackles_pct = (challenge_tackles / challenges * 100) if challenges > 0 else 0.0
        challenges_lost = challenges - challenge_tackles
        blocks_val = self._clamp(int(rng.poisson(max(1, talent * 15 * def_mult * m90 / 34))), 0)
        blocked_shots = self._clamp(int(blocks_val * rng.uniform(0.3, 0.6)), 0, blocks_val)
        blocked_passes = blocks_val - blocked_shots
        interceptions = self._clamp(int(rng.poisson(max(1, talent * 15 * def_mult * m90 / 34))), 0)
        tackles_interceptions = tackles + interceptions
        clearances = self._clamp(int(rng.poisson(max(1, talent * 20 * def_mult * m90 / 34))), 0)
        errors = self._clamp(int(rng.poisson(0.5)), 0)

        # Possession
        touches = max(100, int(rng.poisson(max(50, talent * 800 * m90 / 34))))
        touches_def_pen_area = self._clamp(int(touches * rng.uniform(0.02, 0.08) * def_mult), 0)
        touches_def_3rd = self._clamp(int(touches * rng.uniform(0.10, 0.30) * def_mult), 0)
        touches_mid_3rd = self._clamp(int(touches * rng.uniform(0.30, 0.50)), 0)
        touches_att_3rd = self._clamp(int(touches * rng.uniform(0.10, 0.30) * atk_mult), 0)
        touches_att_pen_area = self._clamp(int(touches * rng.uniform(0.02, 0.08) * atk_mult), 0)
        touches_live_ball = self._clamp(int(touches * rng.uniform(0.85, 0.95)), 0, touches)

        take_ons = self._clamp(int(rng.poisson(max(1, talent * 20 * m90 / 34))), 0)
        take_ons_won = self._clamp(int(take_ons * rng.uniform(0.35, 0.65)), 0, take_ons)
        take_ons_won_pct = (take_ons_won / take_ons * 100) if take_ons > 0 else 0.0
        take_ons_tackled = take_ons - take_ons_won
        take_ons_tackled_pct = (take_ons_tackled / take_ons * 100) if take_ons > 0 else 0.0

        carries = max(50, int(rng.poisson(max(30, talent * 500 * m90 / 34))))
        carries_distance = int(carries * rng.uniform(3, 8))
        carries_progressive_distance = int(carries * rng.uniform(1, 3))
        carries_into_final_third = self._clamp(int(rng.poisson(max(1, talent * 15 * m90 / 34))), 0)
        carries_into_penalty_area = self._clamp(int(rng.poisson(max(1, talent * 5 * atk_mult * m90 / 34))), 0)
        miscontrols = self._clamp(int(rng.poisson(max(1, 20 * m90 / 34))), 0)
        dispossessed = self._clamp(int(rng.poisson(max(1, 10 * m90 / 34))), 0)
        passes_received = self._clamp(int(rng.poisson(max(10, talent * 300 * m90 / 34))), 0)

        # Cards
        cards_yellow = self._clamp(int(rng.poisson(2.0)), 0, 15)
        cards_red = 1 if rng.random() < 0.03 else 0

        return {
            "games": games,
            "games_starts": games_starts,
            "minutes": minutes,
            "minutes_90s": round(m90, 2),
            "goals": goals,
            "assists": assists,
            "goals_assists": goals + assists,
            "goals_pens": goals_pens,
            "pens_made": pens_made,
            "pens_att": pens_att,
            "xg": round(xg, 2),
            "npxg": round(npxg, 2),
            "xg_assist": round(xg_assist, 2),
            "npxg_xg_assist": round(npxg_xg_assist, 2),
            "cards_yellow": cards_yellow,
            "cards_red": cards_red,
            "progressive_carries": progressive_carries,
            "progressive_passes": progressive_passes,
            "progressive_passes_received": progressive_passes_received,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "shots_on_target_pct": round(shots_on_target_pct, 1),
            "shots_per90": round(shots_per90, 2),
            "shots_on_target_per90": round(shots_on_target_per90, 2),
            "goals_per_shot": round(goals_per_shot, 3),
            "goals_per_shot_on_target": round(goals_per_shot_on_target, 3),
            "average_shot_distance": round(average_shot_distance, 1),
            "shots_free_kicks": shots_free_kicks,
            "npxg_per_shot": round(npxg_per_shot, 3),
            "xg_net": round(xg_net, 2),
            "npxg_net": round(npxg_net, 2),
            "passes_completed": passes_completed,
            "passes": passes,
            "passes_pct": round(passes_pct, 1),
            "passes_total_distance": passes_total_distance,
            "passes_progressive_distance": passes_progressive_distance,
            "passes_completed_short": passes_completed_short,
            "passes_short": passes_short,
            "passes_pct_short": round(passes_pct_short, 1),
            "passes_completed_medium": passes_completed_medium,
            "passes_medium": passes_medium,
            "passes_pct_medium": round(passes_pct_medium, 1),
            "passes_completed_long": passes_completed_long,
            "passes_long": passes_long,
            "passes_pct_long": round(passes_pct_long, 1),
            "assisted_shots": assisted_shots,
            "passes_into_final_third": passes_into_final_third,
            "passes_into_penalty_area": passes_into_penalty_area,
            "crosses_into_penalty_area": crosses_into_penalty_area,
            "tackles": tackles,
            "tackles_won": tackles_won,
            "tackles_def_3rd": tackles_def_3rd,
            "tackles_mid_3rd": tackles_mid_3rd,
            "tackles_att_3rd": tackles_att_3rd,
            "challenge_tackles": challenge_tackles,
            "challenges": challenges,
            "challenge_tackles_pct": round(challenge_tackles_pct, 1),
            "challenges_lost": challenges_lost,
            "blocks": blocks_val,
            "blocked_shots": blocked_shots,
            "blocked_passes": blocked_passes,
            "interceptions": interceptions,
            "tackles_interceptions": tackles_interceptions,
            "clearances": clearances,
            "errors": errors,
            "touches": touches,
            "touches_def_pen_area": touches_def_pen_area,
            "touches_def_3rd": touches_def_3rd,
            "touches_mid_3rd": touches_mid_3rd,
            "touches_att_3rd": touches_att_3rd,
            "touches_att_pen_area": touches_att_pen_area,
            "touches_live_ball": touches_live_ball,
            "take_ons": take_ons,
            "take_ons_won": take_ons_won,
            "take_ons_won_pct": round(take_ons_won_pct, 1),
            "take_ons_tackled": take_ons_tackled,
            "take_ons_tackled_pct": round(take_ons_tackled_pct, 1),
            "carries": carries,
            "carries_distance": carries_distance,
            "carries_progressive_distance": carries_progressive_distance,
            "carries_into_final_third": carries_into_final_third,
            "carries_into_penalty_area": carries_into_penalty_area,
            "miscontrols": miscontrols,
            "dispossessed": dispossessed,
            "passes_received": passes_received,
        }

    def _gen_understat_stats(self, talent: float, position: str, minutes: int, fbref_stats: dict) -> dict:
        """Generate Understat stats consistent with FBref values."""
        rng = self.rng
        m90 = minutes / 90.0
        pos = position.split(",")[0]
        atk_mult = {"FW": 1.5, "MF": 0.8, "DF": 0.2}.get(pos, 0.5)

        goals = fbref_stats["goals"]
        assists = fbref_stats["assists"]
        npg = fbref_stats["goals_pens"]
        xg = max(0.0, fbref_stats["xg"] + rng.normal(0, 0.3))
        xa = max(0.0, fbref_stats["xg_assist"] + rng.normal(0, 0.2))
        npxg = max(0.0, fbref_stats["npxg"] + rng.normal(0, 0.2))
        shots = fbref_stats["shots"]
        key_passes = self._clamp(int(rng.poisson(max(1, talent * 15 * atk_mult * m90 / 34))), 0)

        xg_chain = max(0.0, xg + rng.uniform(0.5, 3.0))
        xg_buildup = max(0.0, xg_chain - xg + rng.normal(0, 0.5))

        return {
            "games": fbref_stats["games"],
            "minutes": minutes,
            "minutes_90s": round(m90, 2),
            "goals": goals,
            "assists": assists,
            "npg": npg,
            "xg": round(xg, 2),
            "xa": round(xa, 2),
            "npxg": round(npxg, 2),
            "xg_chain": round(xg_chain, 2),
            "xg_buildup": round(xg_buildup, 2),
            "xg_per90": round(xg / m90, 3) if m90 > 0 else 0.0,
            "xa_per90": round(xa / m90, 3) if m90 > 0 else 0.0,
            "npxg_per90": round(npxg / m90, 3) if m90 > 0 else 0.0,
            "goals_per90": round(goals / m90, 3) if m90 > 0 else 0.0,
            "assists_per90": round(assists / m90, 3) if m90 > 0 else 0.0,
            "shots": shots,
            "key_passes": key_passes,
            "yellow_cards": fbref_stats["cards_yellow"],
            "red_cards": fbref_stats["cards_red"],
            "xg_overperformance": round(goals - xg, 2),
            "xa_overperformance": round(assists - xa, 2),
        }

    def _gen_market_value(self, talent: float, league: str, age: int) -> int:
        """Generate market value correlated with talent and league tier."""
        rng = self.rng
        league_mult = {
            "premier-league": 5.0, "la-liga": 4.0, "bundesliga": 3.5,
            "serie-a": 3.0, "ligue-1": 2.5,
            "eredivisie": 1.5, "primeira-liga": 1.3, "championship": 1.4,
            "belgian-pro-league": 1.0, "serie-b": 0.8, "ligue-2": 0.7,
            "austrian-bundesliga": 0.6, "scottish-premiership": 0.5,
        }.get(league, 1.0)
        age_mult = max(0.3, 1.0 - (age - 22) * 0.05)
        base = talent * league_mult * age_mult * 2_000_000
        noise = rng.uniform(0.7, 1.3)
        return max(50_000, int(base * noise))

    # ---------- Player pool generation ----------

    def generate(self):
        """Generate all synthetic data."""
        logger.info("Generating synthetic player data...")

        # --- Source league players ---
        source_players = []
        for league in SOURCE_LEAGUES:
            for season in MAIN_SEASONS:
                n_players = self.rng.randint(55, 65)
                for _ in range(n_players):
                    player = self._create_player_identity(league, is_breakout=False)
                    source_players.append((player, league, season))

        logger.info(f"Generated {len(source_players)} source-league player-season slots")

        # Assign multi-season appearances (~40% of unique players appear 2+ times)
        # Group by player identity for tracking
        # For simplicity: some players reappear in consecutive seasons
        # We'll track by name+league to create continuity
        seen_players = {}  # (name, league) -> player_info
        deduped_source = []

        for player, league, season in source_players:
            key = (player["name"], league)
            if key in seen_players and self.rng.random() < 0.40:
                # Reuse existing player (multi-season)
                existing = seen_players[key]
                player["player_id_fb"] = existing["player_id_fb"]
                player["player_id_tm"] = existing["player_id_tm"]
                # Age increments
                sy = season_start_year(season)
                player["age"] = sy - existing["birth_year"]
                if player["age"] < 17 or player["age"] > 26:
                    continue
            else:
                seen_players[key] = player
            deduped_source.append((player, league, season))

        source_players = deduped_source
        logger.info(f"After multi-season dedup: {len(source_players)} source player-seasons")

        # --- Select breakout players (~6%) ---
        n_breakouts = max(1, int(len(source_players) * 0.06))
        # Pick unique players for breakout (by fbref_id)
        unique_source_ids = list({p["player_id_fb"] for p, _, _ in source_players})
        self.rng.shuffle(unique_source_ids)
        breakout_ids = set(unique_source_ids[:n_breakouts])
        logger.info(f"Selected {len(breakout_ids)} breakout players")

        # Build map: fbref_id -> (last_source_season, player_info, league)
        breakout_info = {}
        for player, league, season in source_players:
            fb_id = player["player_id_fb"]
            if fb_id in breakout_ids:
                sy = season_start_year(season)
                if fb_id not in breakout_info or sy > season_start_year(breakout_info[fb_id][1]):
                    breakout_info[fb_id] = (player, season, league)

        # --- Target league players (regular + breakout arrivals) ---
        target_all_seasons = MAIN_SEASONS + EXTRA_TARGET_SEASONS

        for league in TARGET_LEAGUES:
            for season in target_all_seasons:
                n_players = self.rng.randint(110, 130)
                for _ in range(n_players):
                    player = self._create_player_identity(league, is_breakout=False)
                    # Target league players: wider age range (no upper filter needed for label window)
                    player["age"] = self.rng.randint(18, 35)
                    player["birth_year"] = season_start_year(season) - player["age"]
                    talent = self.rng.uniform(0.4, 0.8)
                    minutes = self.rng.randint(500, 3000)
                    self._emit_player_row(player, league, season, talent, minutes)

        # --- Emit breakout target-league rows ---
        for fb_id, (player, last_season, src_league) in breakout_info.items():
            last_year = season_start_year(last_season)
            # Appear in target league 1-3 seasons later with 900+ minutes
            offset = self.rng.randint(1, 4)  # 1 to 3
            target_year = last_year + offset
            target_season = f"{target_year}-{str(target_year + 1)[-2:]}"

            # Only emit if target_season is in our range
            if target_season not in set(target_all_seasons):
                continue

            target_league = self.rng.choice(TARGET_LEAGUES)
            team = self.rng.choice(TEAMS[target_league])
            talent = self.rng.uniform(0.6, 0.95)
            minutes = self.rng.randint(900, 2800)
            age = target_year - player["birth_year"]

            fbref_stats = self._gen_fbref_stats(talent, player["position"], minutes)
            fb_row = {
                "player_id": player["player_id_fb"],
                "name": player["name"],
                "position": player["position"],
                "team": team,
                "nationality": player["nationality"],
                "age": age,
                "birth_year": player["birth_year"],
                "league": target_league,
                "season": target_season,
                "source": "fbref",
            }
            fb_row.update(fbref_stats)
            self.fbref_rows.append(fb_row)

            # TM row
            tm_row = {
                "player_id": player["player_id_tm"],
                "name": player["name"],
                "position": player["position"],
                "team": team,
                "age": age,
                "nationality": player["nationality"],
                "market_value_eur": self._gen_market_value(talent, target_league, age),
                "league": target_league,
                "season": target_season,
                "source": "transfermarkt",
            }
            self.tm_rows.append(tm_row)

            # Understat row (target leagues always have understat)
            if target_league in UNDERSTAT_LEAGUES:
                us_stats = self._gen_understat_stats(talent, player["position"], minutes, fbref_stats)
                us_row = {
                    "player_id": self._next_id("us"),
                    "name": player["name"],
                    "position": player["position"].split(",")[0],
                    "team": team,
                    "league": target_league,
                    "season": target_season,
                    "source": "understat",
                }
                us_row.update(us_stats)
                self.understat_rows.append(us_row)

        # --- Emit source league player rows ---
        for player, league, season in source_players:
            is_bo = player["player_id_fb"] in breakout_ids
            talent = self.rng.uniform(0.6, 0.95) if is_bo else self.rng.uniform(0.3, 0.7)
            minutes = self.rng.randint(500, 2800)
            self._emit_player_row(player, league, season, talent, minutes)

        logger.info(f"Generated: fbref={len(self.fbref_rows)}, tm={len(self.tm_rows)}, "
                     f"understat={len(self.understat_rows)}")

    def _create_player_identity(self, league: str, is_breakout: bool) -> dict:
        """Create a new player with stable IDs."""
        fb_id = self._next_id("fb")
        tm_id = self._next_id("tm")
        name = self._pick_name()
        position = self._pick_position()
        birth_year = self.rng.randint(1992, 2004)
        nationality = self.rng.choice(NATIONALITIES)
        return {
            "player_id_fb": fb_id,
            "player_id_tm": tm_id,
            "name": name,
            "position": position,
            "birth_year": birth_year,
            "nationality": nationality,
            "age": 2020 - birth_year,  # placeholder, recalculated per season
        }

    def _emit_player_row(self, player: dict, league: str, season: str, talent: float, minutes: int):
        """Emit fbref + tm (+ understat if applicable) rows for one player-season."""
        team = self.rng.choice(TEAMS[league])
        age = season_start_year(season) - player["birth_year"]
        position = player["position"]

        fbref_stats = self._gen_fbref_stats(talent, position, minutes)

        fb_row = {
            "player_id": player["player_id_fb"],
            "name": player["name"],
            "position": position,
            "team": team,
            "nationality": player["nationality"],
            "age": age,
            "birth_year": player["birth_year"],
            "league": league,
            "season": season,
            "source": "fbref",
        }
        fb_row.update(fbref_stats)
        self.fbref_rows.append(fb_row)

        # TM row (all leagues)
        tm_row = {
            "player_id": player["player_id_tm"],
            "name": player["name"],
            "position": position,
            "team": team,
            "age": age,
            "nationality": player["nationality"],
            "market_value_eur": self._gen_market_value(talent, league, age),
            "league": league,
            "season": season,
            "source": "transfermarkt",
        }
        self.tm_rows.append(tm_row)

        # Understat row (target leagues only)
        if league in UNDERSTAT_LEAGUES:
            us_stats = self._gen_understat_stats(talent, position, minutes, fbref_stats)
            us_row = {
                "player_id": self._next_id("us"),
                "name": player["name"],
                "position": position.split(",")[0],
                "team": team,
                "league": league,
                "season": season,
                "source": "understat",
            }
            us_row.update(us_stats)
            self.understat_rows.append(us_row)

    def insert_into_db(self, db: PlayerDatabase):
        """Insert all generated rows into the database."""
        # Deduplicate by primary key (player_id, league, season)
        def dedup(rows, key_fn):
            seen = set()
            result = []
            for row in rows:
                k = key_fn(row)
                if k not in seen:
                    seen.add(k)
                    result.append(row)
            return result

        fb_key = lambda r: (r["player_id"], r["league"], r["season"])
        tm_key = lambda r: (r["player_id"], r["league"], r["season"])
        us_key = lambda r: (r["player_id"], r["league"], r["season"])

        self.fbref_rows = dedup(self.fbref_rows, fb_key)
        self.tm_rows = dedup(self.tm_rows, tm_key)
        self.understat_rows = dedup(self.understat_rows, us_key)

        logger.info(f"Inserting (after dedup): fbref={len(self.fbref_rows)}, "
                     f"tm={len(self.tm_rows)}, understat={len(self.understat_rows)}")

        # Insert in batches
        batch = 2000
        for i in range(0, len(self.fbref_rows), batch):
            db.insert_fbref_players(self.fbref_rows[i:i + batch])
        for i in range(0, len(self.tm_rows), batch):
            db.insert_transfermarkt_players(self.tm_rows[i:i + batch])
        for i in range(0, len(self.understat_rows), batch):
            db.insert_understat_players(self.understat_rows[i:i + batch])

        stats = db.get_stats()
        logger.info(f"DB stats: fbref={stats['fbref_players']}, "
                     f"tm={stats['transfermarkt_players']}, us={stats['understat_players']}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic player data")
    parser.add_argument("--db", default="data/players.duckdb", help="DuckDB path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB to start fresh
    if db_path.exists():
        db_path.unlink()
        logger.info(f"Removed existing DB: {db_path}")

    db = PlayerDatabase(db_path)

    gen = SyntheticDataGenerator(seed=args.seed)
    gen.generate()
    gen.insert_into_db(db)

    db.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()

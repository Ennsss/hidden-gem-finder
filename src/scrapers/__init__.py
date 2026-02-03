"""Data scrapers for football statistics sources."""

from .base_scraper import BaseScraper
from .fbref_scraper import FBrefScraper
from .transfermarkt_scraper import TransfermarktScraper

__all__ = ["BaseScraper", "FBrefScraper", "TransfermarktScraper"]

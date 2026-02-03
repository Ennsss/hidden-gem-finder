"""Data scrapers for football statistics sources."""

from .base_scraper import BaseScraper
from .fbref_scraper import FBrefScraper

__all__ = ["BaseScraper", "FBrefScraper"]

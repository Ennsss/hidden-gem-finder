"""Base scraper class with rate limiting, caching, and error handling."""

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Abstract base class for all scrapers.

    Provides:
    - Rate limiting between requests
    - Response caching to disk
    - Automatic retries with exponential backoff
    - User-agent rotation
    """

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]

    def __init__(
        self,
        cache_dir: Path | str = "data/raw",
        rate_limit: float = 3.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """Initialize the scraper.

        Args:
            cache_dir: Directory to store cached responses
            rate_limit: Minimum seconds between requests
            max_retries: Number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0
        self._request_count = 0
        self._user_agent_index = 0

        # Set up session with retries
        self.session = self._create_session()
        self._setup_retries(max_retries)

    # Subclasses can set this to add a Referer header
    BASE_URL: str = ""

    def _create_session(self) -> requests.Session:
        """Create the HTTP session. Override in subclasses for custom sessions."""
        return requests.Session()

    def _setup_retries(self, max_retries: int) -> None:
        """Mount retry adapter on the session. Override to skip for custom sessions."""
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[403, 429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source (e.g., 'fbref', 'transfermarkt')."""
        pass

    def _get_cache_path(self, url: str) -> Path:
        """Generate a cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        # Create a readable filename from the URL
        clean_url = url.replace("https://", "").replace("http://", "")
        clean_url = clean_url.replace("/", "_").replace("?", "_")[:50]
        filename = f"{clean_url}_{url_hash}.html"
        return self.cache_dir / self.source_name / filename

    def _get_cached_response(self, url: str) -> str | None:
        """Retrieve cached response if it exists."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            logger.debug(f"Cache hit: {url}")
            return cache_path.read_text(encoding="utf-8")
        return None

    def _cache_response(self, url: str, content: str) -> None:
        """Cache a response to disk."""
        cache_path = self._get_cache_path(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
        logger.debug(f"Cached: {url}")

    def _respect_rate_limit(self) -> None:
        """Sleep if necessary to respect rate limit, with random jitter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            jitter = random.uniform(0, 2.0)
            sleep_time = self.rate_limit - elapsed + jitter
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with rotating user agent and Referer."""
        ua = self.USER_AGENTS[self._user_agent_index % len(self.USER_AGENTS)]
        self._user_agent_index += 1
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        if self.BASE_URL:
            headers["Referer"] = self.BASE_URL + "/"
        return headers

    def fetch(self, url: str, use_cache: bool = True) -> str:
        """Fetch a URL with caching and rate limiting.

        Args:
            url: The URL to fetch
            use_cache: Whether to use cached response if available

        Returns:
            The response content as a string

        Raises:
            requests.RequestException: If the request fails after retries
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached_response(url)
            if cached is not None:
                return cached

        # Respect rate limit
        self._respect_rate_limit()

        # Make request
        logger.info(f"Fetching: {url}")
        response = self.session.get(
            url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        self._last_request_time = time.time()
        self._request_count += 1

        # Cache and return
        content = response.text
        if use_cache:
            self._cache_response(url, content)

        return content

    def fetch_json(self, url: str, use_cache: bool = True) -> dict[str, Any]:
        """Fetch a URL and parse as JSON."""
        content = self.fetch(url, use_cache)
        return json.loads(content)

    @abstractmethod
    def scrape_league_season(self, league_id: str, season: str) -> list[dict[str, Any]]:
        """Scrape all player stats for a league season.

        Args:
            league_id: The league identifier for this source
            season: The season string (e.g., "2023-24")

        Returns:
            List of player stat dictionaries
        """
        pass

    def clear_cache(self) -> int:
        """Clear all cached files for this source.

        Returns:
            Number of files deleted
        """
        cache_path = self.cache_dir / self.source_name
        if not cache_path.exists():
            return 0

        count = 0
        for file in cache_path.glob("*.html"):
            file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached files for {self.source_name}")
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get scraper statistics."""
        return {
            "source": self.source_name,
            "requests_made": self._request_count,
            "rate_limit": self.rate_limit,
        }

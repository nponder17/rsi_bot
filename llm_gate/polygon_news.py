"""Polygon.io news API client.

Fetches recent news articles for a given ticker symbol.
Polygon free tier: 5 API calls/minute, last 2 years of news.
Polygon paid tier: unlimited, real-time.

Usage:
    from llm_agent.fetchers.polygon_news import fetch_news
    articles = fetch_news("AAPL", days_back=7)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io/v2/reference/news"
REQUEST_TIMEOUT = 10   # seconds
RATE_LIMIT_SLEEP = 12  # seconds between calls on free tier (5/min limit)


class PolygonNewsError(Exception):
    pass


def fetch_news(
    symbol: str,
    days_back: int = 7,
    limit: int = 20,
    respect_rate_limit: bool = False,
    as_of_date: Optional[str] = None,
) -> list[dict]:
    """Fetch recent news articles for a symbol from Polygon.io.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL"
    days_back : int
        How many calendar days of news to fetch (default 7)
    limit : int
        Max number of articles to return (max 1000 per Polygon docs)
    respect_rate_limit : bool
        If True, sleep 12s between calls to respect free tier limits.
        Set False if you have a paid Polygon subscription.

    Returns
    -------
    list[dict]
        Each dict contains:
          symbol        : str
          title         : str
          description   : str  (article summary/teaser)
          published_utc : str  (ISO 8601)
          publisher     : str
          article_url   : str
          tickers       : list[str]
          keywords      : list[str]
    """
    if not POLYGON_API_KEY:
        raise PolygonNewsError("POLYGON_API_KEY not set in .env")

    # as_of_date lets us cap news to a historical date (for backtesting)
    anchor = (
        datetime.fromisoformat(as_of_date).replace(tzinfo=timezone.utc)
        if as_of_date
        else datetime.now(timezone.utc)
    )
    published_after = (anchor - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    published_before = anchor.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "ticker"            : symbol.upper(),
        "published_utc.gte" : published_after,
        "published_utc.lte" : published_before,
        "limit"             : min(limit, 1000),
        "sort"              : "published_utc",
        "order"             : "desc",
        "apiKey"            : POLYGON_API_KEY,
    }

    if respect_rate_limit:
        time.sleep(RATE_LIMIT_SLEEP)

    try:
        resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise PolygonNewsError(f"Polygon API request failed for {symbol}: {e}") from e

    data = resp.json()
    if data.get("status") not in ("OK", "DELAYED"):
        raise PolygonNewsError(
            f"Polygon API error for {symbol}: {data.get('status')} — {data.get('error', '')}"
        )

    articles = []
    for item in data.get("results", []):
        articles.append({
            "symbol"        : symbol.upper(),
            "title"         : item.get("title", ""),
            "description"   : item.get("description", ""),
            "published_utc" : item.get("published_utc", ""),
            "publisher"     : item.get("publisher", {}).get("name", ""),
            "article_url"   : item.get("article_url", ""),
            "tickers"       : item.get("tickers", []),
            "keywords"      : item.get("keywords", []),
        })

    return articles


def format_for_prompt(articles: list[dict], max_articles: int = 10) -> str:
    """Format news articles into a clean string for Claude prompts.

    Trims to most recent max_articles and formats as numbered list
    of headline + description.
    """
    if not articles:
        return "No recent news found."

    lines = []
    for i, a in enumerate(articles[:max_articles], 1):
        pub = a["published_utc"][:10] if a["published_utc"] else "unknown date"
        title = a["title"] or "(no title)"
        desc  = a["description"] or ""
        # trim long descriptions
        if len(desc) > 300:
            desc = desc[:297] + "..."
        lines.append(f"{i}. [{pub}] {title}")
        if desc:
            lines.append(f"   {desc}")

    return "\n".join(lines)


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching news for {symbol} …")
    try:
        arts = fetch_news(symbol, days_back=7, limit=10)
        print(f"Found {len(arts)} articles\n")
        print(format_for_prompt(arts))
    except PolygonNewsError as e:
        print(f"Error: {e}")

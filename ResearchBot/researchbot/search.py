"""Tavily web search."""

import os
from functools import lru_cache

from tavily import TavilyClient

from researchbot.config import settings


@lru_cache
def _client() -> TavilyClient:
    return TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


def web_search(query: str) -> list[dict]:
    """Search the web and return results with the page text included.

    Errors are reported and treated as "no results" so one bad query
    doesn't fail the whole run.
    """
    try:
        res = _client().search(
            query,
            max_results=settings.results_per_query,
            search_depth="basic",
            include_raw_content="markdown",
        )
    except Exception as exc:  # network errors, quota, bad key ...
        print(f"[search] '{query}' failed: {exc}")
        return []
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "raw_content": r.get("raw_content") or "",
            "score": r.get("score", 0.0),
        }
        for r in res.get("results", [])
        if r.get("url")
    ]

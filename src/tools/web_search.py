"""Brave Web Search tool."""
from __future__ import annotations

from typing import Any

import requests

from .. import config


_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def web_search(query: str, count: int = 5) -> dict[str, Any]:
    if not query or not query.strip():
        return {"error": "empty query"}
    if not config.BRAVE_API_KEY:
        return {"error": "BRAVE_API_KEY is not configured."}

    count = max(1, min(10, int(count)))
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": config.BRAVE_API_KEY,
    }
    params = {"q": query, "count": count}
    try:
        resp = requests.get(_ENDPOINT, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"brave request failed: {exc!s}"}

    data = resp.json() or {}
    results: list[dict[str, str]] = []
    for item in (data.get("web") or {}).get("results", [])[:count]:
        results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "snippet": (item.get("description") or "").strip(),
            }
        )
    return {"query": query, "results": results}


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web via the Brave Search API. "
            "Use for recent events, current docs, or facts not in local memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "count": {
                    "type": "integer",
                    "description": "How many results to return (1-10).",
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

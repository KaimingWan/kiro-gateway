# -*- coding: utf-8 -*-

"""
Gateway-side web search implementation.

Anthropic's web_search_20250305 is a server tool — the API executes searches
server-side and returns results inline. Kiro API doesn't support this, so
the gateway implements it:

1. Convert web_search_20250305 → regular tool for Kiro model
2. Intercept web_search tool_use in stream → execute search → follow-up request
3. Emit Anthropic-compatible server_tool_use + web_search_tool_result blocks

Supports multiple search backends:
- Brave Search API (BRAVE_SEARCH_API_KEY) — recommended, free tier
- Tavily API (TAVILY_API_KEY) — alternative
- DuckDuckGo (no key needed) — fallback
"""

import os
import re
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import httpx
from loguru import logger

# ── Configuration ──────────────────────────────────────────────────────────

BRAVE_SEARCH_API_KEY: str = os.getenv("BRAVE_SEARCH_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes")

# Tool definition injected into Kiro request when web_search is requested
WEB_SEARCH_TOOL_DEFINITION = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Use this when you need up-to-date "
        "information that may not be in your training data, such as current events, "
        "recent developments, live data, or anything that changes over time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web"
            }
        },
        "required": ["query"]
    }
}


def is_web_search_server_tool(tool: Any) -> bool:
    """Check if a tool definition is Anthropic's web_search server tool."""
    if isinstance(tool, dict):
        return tool.get("type", "").startswith("web_search")
    tool_type = getattr(tool, "type", None) or ""
    return tool_type.startswith("web_search")


def extract_web_search_config(tools: list) -> Optional[Dict[str, Any]]:
    """Extract web_search server tool config from tools list."""
    for tool in tools:
        if isinstance(tool, dict):
            if tool.get("type", "").startswith("web_search"):
                return {
                    "type": tool.get("type", "web_search_20250305"),
                    "name": tool.get("name", "web_search"),
                    "max_uses": tool.get("max_uses", 5),
                }
        else:
            tool_type = getattr(tool, "type", None) or ""
            if tool_type.startswith("web_search"):
                return {
                    "type": tool_type,
                    "name": getattr(tool, "name", "web_search") or "web_search",
                    "max_uses": getattr(tool, "max_uses", 5) or 5,
                }
    return None


# ── Search Backends ────────────────────────────────────────────────────────

async def _search_brave(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search using Brave Search API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append({
            "type": "web_search_result",
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "encrypted_content": item.get("description", ""),
            "page_age": item.get("page_age", ""),
        })
    return results


async def _search_tavily(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search using Tavily API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "type": "web_search_result",
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "encrypted_content": item.get("content", ""),
            "page_age": "",
        })
    return results


async def _search_duckduckgo(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo HTML (no API key needed)."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; KiroGateway/1.0)"},
        )
        resp.raise_for_status()
        html = resp.text

    results = []
    links = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )

    for i, (url, title) in enumerate(links[:max_results]):
        clean_title = re.sub(r'<[^>]+>', '', title).strip()
        clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
        if "uddg=" in url:
            match = re.search(r'uddg=([^&]+)', url)
            if match:
                url = unquote(match.group(1))
        results.append({
            "type": "web_search_result",
            "url": url,
            "title": clean_title,
            "encrypted_content": clean_snippet,
            "page_age": "",
        })

    return results


async def execute_web_search(query: str, max_results: int = 0) -> List[Dict[str, Any]]:
    """
    Execute web search using the best available backend.
    Priority: Brave > Tavily > DuckDuckGo
    """
    if max_results <= 0:
        max_results = WEB_SEARCH_MAX_RESULTS

    backend = "none"
    try:
        if BRAVE_SEARCH_API_KEY:
            backend = "brave"
            results = await _search_brave(query, max_results)
        elif TAVILY_API_KEY:
            backend = "tavily"
            results = await _search_tavily(query, max_results)
        else:
            backend = "duckduckgo"
            results = await _search_duckduckgo(query, max_results)

        logger.info(f"Web search [{backend}] query={query!r} -> {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Web search [{backend}] failed: {e}")
        return [{
            "type": "web_search_result",
            "url": "",
            "title": "Search Error",
            "encrypted_content": f"Web search failed ({backend}): {str(e)}",
            "page_age": "",
        }]


# ── Anthropic SSE Block Formatters ─────────────────────────────────────────

def format_server_tool_use_events(
    index: int, query: str, tool_use_id: str
) -> List[Tuple[str, Dict]]:
    """Generate Anthropic server_tool_use SSE events."""
    return [
        ("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": {},
            }
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps({"query": query}, ensure_ascii=False),
            }
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }),
    ]


def format_web_search_result_events(
    index: int, tool_use_id: str, results: List[Dict[str, Any]]
) -> List[Tuple[str, Dict]]:
    """Generate Anthropic web_search_tool_result SSE events."""
    return [
        ("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": results,
            }
        }),
        ("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }),
    ]


def format_search_results_as_text(results: List[Dict[str, Any]]) -> str:
    """Format search results as text for injecting into Kiro follow-up request."""
    lines = ["Web search results:\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("encrypted_content", "")
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    URL: {url}")
        if content:
            lines.append(f"    {content}")
        lines.append("")
    return "\n".join(lines)

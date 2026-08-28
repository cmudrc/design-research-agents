"""Web search and instant-answer lookup tools.

``web.instant_answer`` uses DuckDuckGo's no-key-required Instant Answer API.
It returns encyclopedic and infobox-style hits for well-known entities and
frequently returns no results for open-ended research or discovery queries.

``web.search`` uses the Tavily Search API (https://tavily.com) to provide
genuine ranked web-result discovery. It requires a ``TAVILY_API_KEY``
environment variable and is only registered when that key is present, so a
missing key results in the tool simply not being offered rather than a
registered tool that always fails.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._policy import ToolPolicy
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str

_INSTANT_ANSWER_ENDPOINT = "https://api.duckduckgo.com/"
_TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
_TAVILY_API_KEY_ENV_VAR = "TAVILY_API_KEY"


def register_web_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register network-gated web tools, gated behind the policy's network allowance.

    Tools are only registered when ``policy.config.allow_network`` is already
    ``True``. Registering a network tool unconditionally would make
    ``CoreToolSource.list_tools()`` raise for any caller running with the
    default (network-disabled) policy, since every listed spec is validated
    against current policy settings.

    ``web.search`` additionally requires a ``TAVILY_API_KEY`` environment
    variable and is silently omitted when that key is absent, so a caller
    without a key sees a shorter tool list rather than a tool that always
    fails at invocation time.

    Args:
        source: In-process tool source to register web tools on.
        policy: Runtime tool policy used to decide whether network tools are exposed.
    """
    if not policy.config.allow_network:
        return

    source.register_tool(
        spec=ToolSpec(
            name="web.instant_answer",
            description=(
                "Look up a quick factual or encyclopedic answer using a no-key-required "
                "instant-answer provider. Returns a topic summary and related-topic titles, "
                "URLs, and snippets. This is not general web search: open-ended research or "
                "discovery queries often return no results. Use this for well-known entities, "
                "definitions, and quick reference lookups only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=ToolMetadata(
                source="core",
                side_effects=ToolSideEffects(network=True),
                timeout_s=15,
                max_output_bytes=65_536,
                risky=True,
            ),
        ),
        handler=lambda i, r, d: _instant_answer(i),
    )

    if not os.environ.get(_TAVILY_API_KEY_ENV_VAR, "").strip():
        return

    source.register_tool(
        spec=ToolSpec(
            name="web.search",
            description=(
                "Search the live web and return ranked, relevance-scored results with titles, "
                "URLs, and content snippets. Suitable for open-ended research and discovery "
                "queries, unlike web.instant_answer. Requires a TAVILY_API_KEY environment "
                "variable to be configured; this tool is unavailable otherwise."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=ToolMetadata(
                source="core",
                side_effects=ToolSideEffects(network=True),
                timeout_s=15,
                max_output_bytes=65_536,
                risky=True,
            ),
        ),
        handler=lambda i, r, d: _tavily_search(i),
    )


def _instant_answer(input_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Query the instant-answer endpoint and normalize the response into result records.

    Args:
        input_dict: Structured input payload containing ``query`` and optional ``max_results``.

    Returns:
        Result payload with the query, engine identifier, and normalized results.

    Raises:
        ValueError: If ``query`` is empty.
        RuntimeError: If the request fails or returns an unparseable response.
    """
    query = get_str(input_dict, "query").strip()
    if not query:
        raise ValueError("query must be a non-empty string.")
    max_results = get_int(input_dict, "max_results", default=10)

    payload = _fetch_instant_answer(query)
    results = _normalize_results(payload, max_results=max_results)

    return {
        "engine": "duckduckgo_instant_answer",
        "query": query,
        "count": len(results),
        "results": results,
    }


def _fetch_instant_answer(query: str) -> Mapping[str, object]:
    """Fetch and parse the instant-answer JSON response for one query.

    Args:
        query: Search text to send to the instant-answer endpoint.

    Returns:
        Parsed JSON response body as a mapping.

    Raises:
        RuntimeError: If the request fails or the response body is not valid JSON.
    """
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
    )
    url = f"{_INSTANT_ANSWER_ENDPOINT}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "design-research-agents/web.instant_answer"})

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_body = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Instant-answer request failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Instant-answer response was not valid JSON.") from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError("Instant-answer response had an unexpected shape.")
    return parsed


def _normalize_results(payload: Mapping[str, object], *, max_results: int) -> list[dict[str, object]]:
    """Normalize an instant-answer payload into a flat list of search results.

    Args:
        payload: Parsed instant-answer JSON response.
        max_results: Maximum number of results to return.

    Returns:
        Normalized result records, each with ``title``, ``url``, and ``snippet``.
    """
    results: list[dict[str, object]] = []

    heading = payload.get("Heading")
    abstract = payload.get("AbstractText")
    abstract_url = payload.get("AbstractURL")
    if isinstance(abstract, str) and abstract.strip() and isinstance(abstract_url, str) and abstract_url.strip():
        results.append(
            {
                "title": str(heading) if isinstance(heading, str) and heading.strip() else abstract_url,
                "url": abstract_url,
                "snippet": abstract,
            }
        )

    related_topics = payload.get("RelatedTopics")
    if isinstance(related_topics, list):
        for topic in related_topics:
            if len(results) >= max_results:
                break
            if not isinstance(topic, Mapping):
                continue
            topic_text = topic.get("Text")
            topic_url = topic.get("FirstURL")
            if not isinstance(topic_text, str) or not topic_text.strip():
                continue
            if not isinstance(topic_url, str) or not topic_url.strip():
                continue
            title = topic_text.split(" - ", 1)[0]
            results.append({"title": title, "url": topic_url, "snippet": topic_text})

    return results[:max_results]


def _tavily_search(input_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Query the Tavily Search API and normalize the response into result records.

    Args:
        input_dict: Structured input payload containing ``query`` and optional ``max_results``.

    Returns:
        Result payload with the query, engine identifier, and normalized results.

    Raises:
        ValueError: If ``query`` is empty.
        RuntimeError: If the ``TAVILY_API_KEY`` environment variable is unset, the request
            fails, or the response is unparseable.
    """
    query = get_str(input_dict, "query").strip()
    if not query:
        raise ValueError("query must be a non-empty string.")
    max_results = get_int(input_dict, "max_results", default=10)

    api_key = os.environ.get(_TAVILY_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise RuntimeError(f"{_TAVILY_API_KEY_ENV_VAR} is not set; web.search is unavailable.")

    payload = _fetch_tavily_results(query, api_key=api_key, max_results=max_results)
    results = _normalize_tavily_results(payload)

    return {
        "engine": "tavily",
        "query": query,
        "count": len(results),
        "results": results,
    }


def _fetch_tavily_results(query: str, *, api_key: str, max_results: int) -> Mapping[str, object]:
    """Fetch and parse the Tavily Search API JSON response for one query.

    Args:
        query: Search text to send to the Tavily Search API.
        api_key: Tavily API key used for bearer authentication.
        max_results: Maximum number of results Tavily should return, clamped to its
            documented 0-20 range.

    Returns:
        Parsed JSON response body as a mapping.

    Raises:
        RuntimeError: If the request fails or the response body is not valid JSON.
    """
    body = json.dumps(
        {
            "query": query,
            "max_results": max(0, min(max_results, 20)),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _TAVILY_SEARCH_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "design-research-agents/web.search",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw_body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Tavily search request failed with status {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tavily search request failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tavily search response was not valid JSON.") from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError("Tavily search response had an unexpected shape.")
    return parsed


def _normalize_tavily_results(payload: Mapping[str, object]) -> list[dict[str, object]]:
    """Normalize a Tavily Search API payload into a flat list of result records.

    Args:
        payload: Parsed Tavily Search API JSON response.

    Returns:
        Normalized result records, each with ``title``, ``url``, ``snippet``, and ``score``.
    """
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    results: list[dict[str, object]] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        title = item.get("title")
        content = item.get("content")
        score = item.get("score")
        results.append(
            {
                "title": title if isinstance(title, str) and title.strip() else url,
                "url": url,
                "snippet": content if isinstance(content, str) else "",
                "score": score if isinstance(score, (int, float)) else None,
            }
        )
    return results


__all__ = ["register_web_tools"]

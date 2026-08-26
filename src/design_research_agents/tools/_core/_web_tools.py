"""Live web search tool backed by a no-key-required search provider."""

from __future__ import annotations

import json
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


def register_web_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register live web search tools, gated behind the policy's network allowance.

    The tool is only registered when ``policy.config.allow_network`` is already
    ``True``. Registering a network tool unconditionally would make
    ``CoreToolSource.list_tools()`` raise for any caller running with the
    default (network-disabled) policy, since every listed spec is validated
    against current policy settings.

    Args:
        source: In-process tool source to register the web search tool on.
        policy: Runtime tool policy used to decide whether network tools are exposed.
    """
    if not policy.config.allow_network:
        return

    source.register_tool(
        spec=ToolSpec(
            name="web.search",
            description=(
                "Search the live web using a no-key-required instant-answer provider and return "
                "matching topic titles, URLs, and snippets. Best for quick factual lookups and "
                "reference discovery rather than exhaustive result pages."
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
        handler=lambda i, r, d: _web_search(i),
    )


def _web_search(input_dict: Mapping[str, object]) -> Mapping[str, object]:
    """Query the instant-answer endpoint and normalize the response into search results.

    Args:
        input_dict: Structured input payload containing ``query`` and optional ``max_results``.

    Returns:
        Result payload with the query, engine identifier, and normalized results.

    Raises:
        ValueError: If ``query`` is empty.
        RuntimeError: If the search request fails or returns an unparseable response.
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
    request = urllib.request.Request(url, headers={"User-Agent": "design-research-agents/web.search"})

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw_body = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Web search request failed: {exc}") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Web search response was not valid JSON.") from exc

    if not isinstance(parsed, Mapping):
        raise RuntimeError("Web search response had an unexpected shape.")
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


__all__ = ["register_web_tools"]

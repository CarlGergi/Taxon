"""Stand-ins for the shopping agent's model and catalog, for tests and runs that spend nothing.

Replies are real SDK Message objects, so the agent loop is exercised exactly as against the API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence

from anthropic.types import Message

from taxon.bench.agent import SUBMIT
from taxon.bench.search import TOOL

_CHARS_PER_TOKEN = 4
_WORD = re.compile(r"[a-z0-9]+")

FAKE_TOOL = {
    "name": TOOL,
    "description": "Search the product catalog.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "max_price": {"type": "number"},
            "limit": {"type": "integer"},
        },
    },
}


def message(
    content: Sequence[dict],
    stop_reason: str,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached: int = 0,
    written: int = 0,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": written,
            },
        }
    )


def tool_use(name: str, arguments: Mapping, use_id: str = "tu_1") -> dict:
    return {"type": "tool_use", "id": use_id, "name": name, "input": dict(arguments)}


class ScriptedModel:
    """Answers each call with the next reply from `script(params, call_number)`."""

    def __init__(self, script: Callable[[dict, int], Message]):
        self.script = script
        self.calls: list[dict] = []

    async def create(self, **params) -> Message:
        # A snapshot: the agent keeps appending to the same message list after the call.
        sent = {**params, "messages": list(params["messages"])}
        self.calls.append(sent)
        return self.script(sent, len(self.calls) - 1)


def _tokens(params: Mapping) -> int:
    sent = json.dumps([params["system"], params["tools"], params["messages"]], ensure_ascii=False)
    return max(1, len(sent) // _CHARS_PER_TOKEN)


def demo_reply(params: dict, call: int) -> Message:
    """Search the request's own words, then submit the top three ids found.

    It reads the conversation, not the call count, so it works with many runs at once.

    Token counts follow the request's size, so cost reporting is exercised too.
    """
    tokens = _tokens(params)
    messages = params["messages"]
    if len(messages) == 1:
        query = messages[0]["content"]
        content = [tool_use(TOOL, {"query": query}, f"tu_{call}")]
        return message(content, "tool_use", input_tokens=tokens, output_tokens=30)
    last = messages[-1]["content"]
    found: list[str] = []
    if isinstance(last, list) and last and last[0].get("type") == "tool_result":
        reply = json.loads(last[0]["content"])
        found = [hit["id"] for hit in reply.get("results", [])[:3]]
    content = [tool_use(SUBMIT, {"product_ids": found}, f"tu_{call}")]
    return message(content, "tool_use", input_tokens=tokens, output_tokens=30)


class FakeCatalog:
    """Keyword search over listings in memory, with the enriched catalogs' category filter."""

    def __init__(self, listings: Sequence[Mapping], tool: dict = FAKE_TOOL):
        self.listings = list(listings)
        self.tool = tool
        self.requests: list[dict] = []

    async def search(self, arguments: dict) -> dict:
        self.requests.append(arguments)
        if "query" in arguments and not isinstance(arguments["query"], str):
            return {"error": "query must be text"}
        words = set(_WORD.findall(str(arguments.get("query", "")).lower()))
        ignored = []
        rows = self.listings
        category = arguments.get("category")
        if category is not None:
            if any("category" in row for row in rows):
                rows = [r for r in rows if str(r.get("category", "")).startswith(category)]
            else:
                ignored.append("category")
        if "max_price" in arguments:
            rows = [
                r
                for r in rows
                if r.get("price") is not None and r["price"] <= arguments["max_price"]
            ]

        def score(row: Mapping) -> int:
            return len(words & set(_WORD.findall(row["title"].lower())))

        ranked = sorted(rows, key=lambda r: (-score(r), r["id"]))
        if words:
            ranked = [r for r in ranked if score(r)]
        limit = int(arguments.get("limit", 10))
        results = [{k: r[k] for k in ("id", "title", "price") if k in r} for r in ranked[:limit]]
        return {"results": results, "total": len(ranked), "ignored_filters": ignored}

"""The catalogs an agent shops: raw listings, and the same listings with enrichment added.

Enrichment only adds `category` and `attributes`, so the data is the one thing that differs
between benchmark conditions. Amazon's own category fields stay out of every catalog: a raw
listing must not carry labels, and the engine refuses a catalog with fields it does not know.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from taxon.bench.tasks import numeric_price
from taxon.products import read_jsonl
from taxon.taxonomy import Taxonomy


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def raw_listing(product: Mapping) -> dict:
    return {
        "id": product["parent_asin"],
        "title": product["title"],
        "store": product.get("store") or None,
        "features": list(product.get("features") or []),
        "description": product.get("description") or "",
        "details": {str(k): _text(v) for k, v in (product.get("details") or {}).items()},
        "price": numeric_price(product.get("price")),
    }


def raw_catalog(products: Iterable[Mapping]) -> list[dict]:
    return sorted((raw_listing(p) for p in products), key=lambda row: row["id"])


def enriched_catalog(
    raw: Iterable[Mapping], enrichments: Mapping[str, Mapping], taxonomy: Taxonomy
) -> tuple[list[dict], dict]:
    """Add enrichment to raw listings, keeping only what the taxonomy allows.

    A model's invalid output leaves its listing as it was, or loses just that value, and is counted
    in the report. An agent should never be able to filter on a value that cannot exist.
    """
    counts = Counter({"enriched": 0, "not_enriched": 0, "invalid_path": 0, "dropped_values": 0})
    rows = []
    for listing in raw:
        row = dict(listing)
        rows.append(row)
        enrichment = enrichments.get(listing["id"])
        if enrichment is None:
            counts["not_enriched"] += 1
            continue
        path = enrichment.get("path")
        if path not in taxonomy:
            counts["invalid_path"] += 1
            continue
        allowed = taxonomy.allowed_values(path)
        attributes = {}
        for handle, given in sorted((enrichment.get("attributes") or {}).items()):
            values = [given] if isinstance(given, str) else list(given)
            kept = [v for v in values if v in allowed.get(handle, ())]
            counts["dropped_values"] += len(values) - len(kept)
            if kept:
                attributes[handle] = kept
        row["category"] = path
        row["attributes"] = attributes
        counts["enriched"] += 1
    return rows, dict(counts)


def build_catalog(
    products: Iterable[Mapping], enrichments_path: Path | None, taxonomy: Taxonomy | None
) -> tuple[list[dict], dict]:
    raw = raw_catalog(products)
    if enrichments_path is None:
        return raw, {}
    enrichments = {row["parent_asin"]: row for row in read_jsonl(enrichments_path)}
    return enriched_catalog(raw, enrichments, taxonomy)

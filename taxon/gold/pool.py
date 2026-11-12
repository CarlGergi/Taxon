"""The gold pool: products set aside before any training, so they can never leak into it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence

from taxon.products import seeded_hash

GOLD_POOL_SIZE = 500
FIRST_TARGET = 50


class GoldOverlap(ValueError):
    """Gold products reached something meant for training."""


def stratified_order(products: Sequence[dict], seed: int, salt: str) -> list[dict]:
    """Seeded order that cycles through domains, so any prefix stays balanced across them."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        by_domain[product["domain"]].append(product)
    for rows in by_domain.values():
        rows.sort(key=lambda p: seeded_hash(seed, f"{salt}|{p['parent_asin']}"))
    ordered = []
    depth = 0
    while len(ordered) < len(products):
        for domain in sorted(by_domain):
            if depth < len(by_domain[domain]):
                ordered.append(by_domain[domain][depth])
        depth += 1
    return ordered


def gold_queue(products: Sequence[dict], seed: int, size: int = GOLD_POOL_SIZE) -> list[str]:
    """The ids reserved for gold, in labeling order. Growing the set just walks further along."""
    return [p["parent_asin"] for p in stratified_order(products, seed, "gold")[:size]]


def training_products(products: Iterable[dict], queue: Collection[str]) -> list[dict]:
    pool = set(queue)
    return [p for p in products if p["parent_asin"] not in pool]


def assert_disjoint(training_ids: Iterable[str], gold_ids: Iterable[str]) -> None:
    overlap = set(training_ids) & set(gold_ids)
    if overlap:
        sample = sorted(overlap)[:5]
        raise GoldOverlap(f"{len(overlap)} gold products in training data, such as {sample}")

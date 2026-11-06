"""What the labeling page can see and do, independent of HTTP.

It is built only from product listings, the taxonomy, and the baseline's suggestions. It has no
access to annotator or arbitrator output, so the gold set stays independent of them.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache

from taxon.gold.store import GoldStore
from taxon.silver import singular_words, stated_attributes
from taxon.taxonomy import SEPARATOR, Taxonomy, split_path

SEARCH_LIMIT = 15
VALUE_LIMIT = 10
_PRODUCT_FIELDS = (
    "parent_asin",
    "title",
    "description",
    "features",
    "details",
    "price",
    "store",
    "image_url",
)


def _words(text: str) -> list[str]:
    return singular_words(text)


class PathSearch:
    """Word-prefix search over category paths, ranked to favor matches in a node's own name."""

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self._name_words = [frozenset(_words(split_path(p)[-1])) for p in self.paths]
        postings: dict[str, set[int]] = defaultdict(set)
        for i, path in enumerate(self.paths):
            for word in _words(path):
                postings[word].add(i)
        self._postings = dict(postings)
        self._vocabulary = sorted(postings)

    def _matching(self, prefix: str) -> set[int]:
        start = bisect.bisect_left(self._vocabulary, prefix)
        found: set[int] = set()
        for word in self._vocabulary[start:]:
            if not word.startswith(prefix):
                break
            found |= self._postings[word]
        return found

    def search(self, query: str, limit: int = SEARCH_LIMIT) -> list[str]:
        words = _words(query)
        if not words:
            return []
        hits: set[int] | None = None
        for word in words:
            found = self._matching(word)
            hits = found if hits is None else hits & found
            if not hits:
                return []

        def rank(i: int) -> tuple:
            in_name = sum(w in self._name_words[i] for w in words)
            return (-in_name, self.paths[i].count(SEPARATOR), len(self.paths[i]), self.paths[i])

        return [self.paths[i] for i in sorted(hits, key=rank)[:limit]]


class GoldApp:
    def __init__(
        self,
        products: Mapping[str, dict],
        suggestions: Mapping[str, list[str]],
        taxonomy: Taxonomy,
        store: GoldStore,
        target: int,
    ):
        self.products = products
        self.suggestions = suggestions
        self.taxonomy = taxonomy
        self.store = store
        self.target = target
        self.search_index = PathSearch(taxonomy.paths())
        self._sorted_values = lru_cache(maxsize=256)(self._build_sorted_values)

    def _item(self, asin: str) -> dict:
        product = self.products[asin]
        item = {name: product.get(name) for name in _PRODUCT_FIELDS}
        item["suggestions"] = list(self.suggestions.get(asin, []))
        return item

    def state(self) -> dict:
        done = self.store.n_labeled >= self.target
        asin = None if done else self.store.next_asin()
        return {
            "target": self.target,
            "labeled": self.store.n_labeled,
            "skipped": self.store.n_skipped,
            "item": self._item(asin) if asin else None,
        }

    def search(self, query: str) -> list[str]:
        return self.search_index.search(query)

    def _build_sorted_values(self, attribute_id: str) -> tuple[str, ...]:
        return tuple(sorted(self.taxonomy.attributes[attribute_id].values, key=str.casefold))

    def node(self, path: str, asin: str | None = None) -> dict:
        category = self.taxonomy.category(path)
        parts = split_path(path)
        details = self.products[asin]["details"] if asin in self.products else {}
        prefilled = stated_attributes(details, path, self.taxonomy)
        return {
            "path": path,
            "parent": SEPARATOR.join(parts[:-1]) or None,
            "children": self.taxonomy.children(path),
            "attributes": [
                {
                    "handle": handle,
                    "name": handle.replace("-", " ").capitalize(),
                    "prefill": prefilled.get(handle, ""),
                }
                for handle, _ in sorted(category.attribute_ids)
            ],
        }

    def values(self, path: str, handle: str, query: str) -> list[str]:
        attribute_ids = dict(self.taxonomy.category(path).attribute_ids)
        needle = query.casefold()
        values = self._sorted_values(attribute_ids[handle])
        starts = [v for v in values if v.casefold().startswith(needle)]
        contains = [v for v in values if needle in v.casefold() and v not in starts[:VALUE_LIMIT]]
        return (starts + contains)[:VALUE_LIMIT]

    def label(self, asin: str, path: str, attributes: Mapping[str, str]) -> dict:
        self.store.label(asin, path, {k: v for k, v in attributes.items() if v})
        return self.state()

    def skip(self, asin: str) -> dict:
        self.store.skip(asin)
        return self.state()

    def undo(self) -> dict:
        self.store.undo()
        return self.state()

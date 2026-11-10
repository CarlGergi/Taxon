"""Shortlist an attribute's allowed values for one listing.

Some attributes allow tens of thousands of values (color alone has over 22,000), too many to send.
Character n-grams find spellings close to what the listing says, so "navy blue" in a listing still
reaches "Navy", which an exact match would have thrown away.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from taxon.taxonomy import Taxonomy

SHORTLIST_SIZE = 20
_QUERY_CHARS = 3_000


def listing_text(product: Mapping) -> str:
    """What the listing says, details first because they state attributes most directly."""
    details = " ".join(f"{k} {v}" for k, v in sorted((product.get("details") or {}).items()))
    parts = [product["title"], details, *(product.get("features") or [])]
    parts.append(product.get("description") or "")
    return " ".join(str(p) for p in parts)[:_QUERY_CHARS]


class ValueShortlister:
    def __init__(self, taxonomy: Taxonomy, size: int = SHORTLIST_SIZE):
        self.taxonomy = taxonomy
        self.size = size
        self._indexes: dict[str, tuple[TfidfVectorizer, object, list[str]]] = {}

    def _index(self, attribute_id: str):
        if attribute_id not in self._indexes:
            values = sorted(self.taxonomy.attributes[attribute_id].values)
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
            self._indexes[attribute_id] = (vectorizer, vectorizer.fit_transform(values), values)
        return self._indexes[attribute_id]

    def shortlist(self, path: str, product: Mapping) -> dict[str, list[str]]:
        """Up to `size` values per attribute of `path`, best first, ties broken alphabetically.

        A value scores by how much of its n-gram weight appears anywhere in the listing, so a long
        listing does not dilute a short value.
        """
        text = listing_text(product)
        options = {}
        for handle, attribute_id in sorted(self.taxonomy.category(path).attribute_ids):
            vectorizer, matrix, values = self._index(attribute_id)
            present = (vectorizer.transform([text]) > 0).astype(np.float64).T
            scores = np.asarray((matrix @ present).todense()).ravel()
            order = sorted(range(len(values)), key=lambda i: (-scores[i], values[i]))
            options[handle] = [values[i] for i in order[: self.size]]
        return options

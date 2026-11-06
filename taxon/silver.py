"""Provisional labels from Amazon's own breadcrumbs.

Milestone 1 needs labels before the annotator pipeline exists, so each breadcrumb is
mapped onto the taxonomy by token overlap. Only the breadcrumb is used, never the title
or description, so a classifier trained on product text cannot simply recover the rule.
A score against these labels measures agreement with this mapping, not correctness.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping

from taxon.taxonomy import SEPARATOR, Taxonomy, split_path

MIN_NAME_OVERLAP = 0.6
CONTEXT_WEIGHT = 0.5
# At this depth and below, a breadcrumb entry naming several things cannot justify a node that
# covers only one of them: "Plants, Seeds & Bulbs" says nothing that makes a product a bulb.
SPECIFIC_DEPTH = 3

_STOPWORDS = frozenset(
    {"a", "an", "and", "by", "for", "in", "of", "on", "other", "the", "to", "with"}
)

# Verticals a domain's products can belong to. A breadcrumb match outside them is not trusted,
# since a generic name such as "Parts & Accessories" occurs across the whole tree.
DOMAIN_VERTICALS = {
    "All_Beauty": ("Health & Beauty",),
    "Amazon_Fashion": ("Apparel & Accessories", "Luggage & Bags"),
    "Appliances": ("Home & Garden", "Hardware"),
    "Arts_Crafts_and_Sewing": ("Arts & Entertainment",),
    "Automotive": ("Vehicles & Parts",),
    "Baby_Products": ("Baby & Toddler",),
    "Beauty_and_Personal_Care": ("Health & Beauty",),
    "Books": ("Media",),
    "CDs_and_Vinyl": ("Media",),
    "Cell_Phones_and_Accessories": ("Electronics",),
    "Clothing_Shoes_and_Jewelry": ("Apparel & Accessories", "Luggage & Bags"),
    "Electronics": ("Electronics", "Cameras & Optics"),
    "Grocery_and_Gourmet_Food": ("Food & Beverages",),
    "Handmade_Products": (
        "Apparel & Accessories",
        "Arts & Entertainment",
        "Health & Beauty",
        "Home & Garden",
    ),
    "Health_and_Household": ("Health & Beauty", "Home & Garden"),
    "Health_and_Personal_Care": ("Health & Beauty",),
    "Home_and_Kitchen": ("Home & Garden", "Furniture"),
    "Industrial_and_Scientific": ("Business & Industrial", "Hardware"),
    "Movies_and_TV": ("Media",),
    "Musical_Instruments": ("Arts & Entertainment",),
    "Office_Products": ("Office Supplies",),
    "Patio_Lawn_and_Garden": ("Home & Garden", "Furniture"),
    "Pet_Supplies": ("Animals & Pet Supplies",),
    "Software": ("Software",),
    "Sports_and_Outdoors": ("Sporting Goods",),
    "Tools_and_Home_Improvement": ("Hardware",),
    "Toys_and_Games": ("Toys & Games",),
    "Video_Games": ("Electronics", "Software"),
}


_IRREGULAR = {
    "calves": "calf",
    "children": "child",
    "feet": "foot",
    "geese": "goose",
    "halves": "half",
    "hooves": "hoof",
    "knives": "knife",
    "leaves": "leaf",
    "lives": "life",
    "loaves": "loaf",
    "men": "man",
    "mice": "mouse",
    "scarves": "scarf",
    "shelves": "shelf",
    "teeth": "tooth",
    "wolves": "wolf",
    "women": "woman",
}


def _singular(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    # "hoodie" and "hoodies" must land on the same form, whatever that form looks like.
    if len(word) > 3 and word.endswith("ie"):
        return word[:-2] + "y"
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def singular_words(text: str) -> list[str]:
    """Lowercase ASCII words in order, singular, with stopwords removed."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", ascii_text.lower())
    return [_singular(w) for w in words if w not in _STOPWORDS]


def tokens(text: str) -> frozenset[str]:
    return frozenset(singular_words(text))


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


class SilverLabeler:
    def __init__(self, taxonomy: Taxonomy):
        self._name_tokens: dict[str, frozenset[str]] = {}
        self._ancestor_tokens: dict[str, frozenset[str]] = {}
        self._index: dict[str, list[str]] = defaultdict(list)
        for path in taxonomy.paths():
            *ancestors, name = split_path(path)
            self._name_tokens[path] = tokens(name)
            self._ancestor_tokens[path] = tokens(" ".join(ancestors))
            for token in self._name_tokens[path]:
                self._index[token].append(path)
        self._cache: dict[tuple, str | None] = {}

    def _best_match(
        self, crumb: frozenset[str], context: frozenset[str], verticals: tuple[str, ...]
    ) -> str | None:
        candidates = {path for token in crumb for path in self._index.get(token, ())}
        ranked = []
        for path in candidates:
            if path.split(SEPARATOR, 1)[0] not in verticals:
                continue
            name = self._name_tokens[path]
            if _overlap(crumb, name) < MIN_NAME_OVERLAP:
                continue
            # Every word of the node's name must come from the breadcrumb, otherwise a generic
            # "Parts & Accessories" would match the very specific "Dishwasher Parts & Accessories".
            if not name <= context:
                continue
            ancestors = self._ancestor_tokens[path]
            depth = path.count(SEPARATOR) + 1
            if depth >= SPECIFIC_DEPTH and not crumb - name <= ancestors:
                continue
            agreement = len(context & ancestors) / len(ancestors) if ancestors else 0.0
            score = _overlap(crumb, name) + CONTEXT_WEIGHT * agreement
            ranked.append((-score, depth, path))
        return min(ranked)[2] if ranked else None

    def label(self, domain: str, breadcrumb: list[str]) -> str | None:
        """The taxonomy path for the deepest breadcrumb entry that matches, or None."""
        key = (domain, tuple(breadcrumb))
        if key not in self._cache:
            context = tokens(" ".join(breadcrumb))
            verticals = DOMAIN_VERTICALS[domain]
            result = None
            for crumb in reversed(breadcrumb):
                crumb_tokens = tokens(crumb)
                if crumb_tokens:
                    result = self._best_match(crumb_tokens, context, verticals)
                if result is not None:
                    break
            self._cache[key] = result
        return self._cache[key]


def label_products(products: list[dict], taxonomy: Taxonomy) -> list[dict]:
    labeler = SilverLabeler(taxonomy)
    return [
        {
            "parent_asin": p["parent_asin"],
            "path": labeler.label(p["domain"], p["categories"]),
        }
        for p in products
    ]


def coverage(products: list[dict], labels: list[dict]) -> dict:
    per_domain: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for product, label in zip(products, labels, strict=True):
        per_domain[product["domain"]][0] += label["path"] is not None
        per_domain[product["domain"]][1] += 1
    labeled = sum(hit for hit, _ in per_domain.values())
    return {
        "labeled": labeled,
        "abstained": len(products) - labeled,
        "coverage": labeled / len(products),
        "distinct_paths": len({row["path"] for row in labels if row["path"]}),
        "per_domain_coverage": {d: h / n for d, (h, n) in sorted(per_domain.items())},
    }


def stated_attributes(details: Mapping, path: str, taxonomy: Taxonomy) -> dict[str, str]:
    """Attribute values a listing's details state exactly, spelled as the taxonomy spells them."""
    stated = {str(k).strip().casefold(): str(v).strip().casefold() for k, v in details.items()}
    found = {}
    for handle, values in taxonomy.allowed_values(path).items():
        wanted = stated.get(handle.replace("-", " "))
        if wanted is None:
            continue
        matches = [value for value in values if value.casefold() == wanted]
        if matches:
            found[handle] = min(matches, key=lambda value: (value.casefold(), value))
    return found

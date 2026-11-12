"""Build the shortlist of category paths an annotator chooses between.

Walking a 64,000-node tree top-down costs several calls per product per annotator, so each
product first gets a shortlist. TF-IDF similarity covers the whole tree, a classifier trained on
provisional labels adds what similarity misses, and ancestors let an annotator stay broad.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from eval.metrics import wilson
from taxon.baseline import TfidfLogisticRegression
from taxon.products import seeded_hash
from taxon.silver import singular_words
from taxon.taxonomy import Taxonomy, path_prefixes, split_path

TOP_K = 50
CORE_K = 30
SIMILARITY_K = 100
CLASSIFIER_K = 30
CROSS_FIT_FOLDS = 5
_FUSION_CONSTANT = 30
_NAME_WEIGHT = 3
_TITLE_WEIGHT = 3
_MAX_FEATURES = 5
_MAX_DESCRIPTION_CHARS = 500


def _tokenize(text: str) -> list[str]:
    words = singular_words(text)
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def path_document(path: str) -> str:
    """The text a path is matched on, with its own name outweighing its ancestors."""
    *ancestors, name = split_path(path)
    return " . ".join([*ancestors, *[name] * _NAME_WEIGHT])


def product_query(product: dict) -> str:
    """The text a product is matched on, led by its title."""
    features = product["features"][:_MAX_FEATURES]
    description = product["description"][:_MAX_DESCRIPTION_CHARS]
    return " . ".join([*[product["title"]] * _TITLE_WEIGHT, *features, description])


class CandidateRetriever:
    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, token_pattern=None, sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(path_document(p) for p in self.paths)

    def top_k(
        self, products: Sequence[dict], k: int = TOP_K, batch_size: int = 256
    ) -> list[list[str]]:
        """The `k` most similar paths for each product, best first, ties broken by path."""
        k = min(k, len(self.paths))
        order = np.arange(len(self.paths))
        results: list[list[str]] = []
        for start in range(0, len(products), batch_size):
            queries = self._vectorizer.transform(
                product_query(p) for p in products[start : start + batch_size]
            )
            scores = (queries @ self._matrix.T).toarray()
            for row in scores:
                # Sorting all paths by (-score, position) keeps ties deterministic.
                ranked = np.lexsort((order, -row))[:k]
                results.append([self.paths[i] for i in ranked])
        return results


def fuse(*rankings: Sequence[str]) -> list[str]:
    """Reciprocal rank fusion: a path high in several rankings beats one high in just one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, path in enumerate(ranking):
            scores[path] = scores.get(path, 0.0) + 1 / (_FUSION_CONSTANT + rank)
    return sorted(scores, key=lambda path: (-scores[path], path))


def with_ancestors(ranked: Sequence[str], k: int = TOP_K, core: int = CORE_K) -> list[str]:
    """The top `core` paths, then their ancestors below the vertical, then the rest up to `k`."""
    shortlist = list(dict.fromkeys(ranked[:core]))
    seen = set(shortlist)
    for path in ranked[:core]:
        for ancestor in path_prefixes(path)[1:-1]:
            if len(shortlist) < k and ancestor not in seen:
                shortlist.append(ancestor)
                seen.add(ancestor)
    for path in ranked[core:]:
        if len(shortlist) >= k:
            break
        if path not in seen:
            shortlist.append(path)
            seen.add(path)
    return shortlist[:k]


def classifier_rankings(
    products: Sequence[dict],
    labels: Mapping[str, str | None],
    held_out: Collection[str],
    seed: int,
    c: float,
    folds: int = CROSS_FIT_FOLDS,
    k: int = CLASSIFIER_K,
    n_jobs: int = -1,
) -> dict[str, list[str]]:
    """Rank labels per product without ever using a model trained on that product.

    Labeled products outside `held_out` are ranked by a model fitted on the other folds. Everything
    else is ranked by a model fitted on all of them. Held-out products never train any model.
    """
    trainable = [
        p for p in products if labels.get(p["parent_asin"]) and p["parent_asin"] not in held_out
    ]
    if not trainable:
        return {p["parent_asin"]: [] for p in products}
    fold_of = {
        p["parent_asin"]: int(seeded_hash(seed, f"fold|{p['parent_asin']}")[:8], 16) % folds
        for p in trainable
    }

    def fit(rows: Sequence[dict]) -> TfidfLogisticRegression:
        return TfidfLogisticRegression(c=c, n_jobs=n_jobs).fit(
            rows, [labels[p["parent_asin"]] for p in rows]
        )

    rankings: dict[str, list[str]] = {}
    for fold in range(folds):
        inside = [p for p in trainable if fold_of[p["parent_asin"]] == fold]
        outside = [p for p in trainable if fold_of[p["parent_asin"]] != fold]
        if inside and len({labels[p["parent_asin"]] for p in outside}) > 1:
            model = fit(outside)
            rankings.update(
                zip([p["parent_asin"] for p in inside], model.top_k(inside, k), strict=True)
            )
    others = [p for p in products if p["parent_asin"] not in rankings]
    if others and len({labels[p["parent_asin"]] for p in trainable}) > 1:
        rankings.update(
            zip([p["parent_asin"] for p in others], fit(trainable).top_k(others, k), strict=True)
        )
    return {p["parent_asin"]: rankings.get(p["parent_asin"], []) for p in products}


def build_shortlists(
    products: Sequence[dict],
    taxonomy: Taxonomy,
    labels: Mapping[str, str | None],
    held_out: Collection[str],
    seed: int,
    c: float,
    n_jobs: int = -1,
) -> dict[str, list[str]]:
    """A shortlist of up to TOP_K valid paths for every product."""
    similar = CandidateRetriever(taxonomy.paths()).top_k(products, SIMILARITY_K)
    classified = classifier_rankings(products, labels, held_out, seed, c, n_jobs=n_jobs)
    return {
        p["parent_asin"]: with_ancestors(fuse(row, classified[p["parent_asin"]]))
        for p, row in zip(products, similar, strict=True)
    }


def _is_compatible(candidate: str, gold: str) -> bool:
    """True when one path is an ancestor of, or the same as, the other."""
    return (
        candidate == gold
        or gold.startswith(f"{candidate} > ")
        or candidate.startswith(f"{gold} > ")
    )


def recall_at(
    candidates: Iterable[Sequence[str]], golds: Iterable[str], cutoffs: Iterable[int]
) -> dict:
    """How often the gold path is in the top k, exactly and up to an ancestor or descendant."""
    rows = list(zip(candidates, golds, strict=True))
    if not rows:
        raise ValueError("cannot measure recall on an empty set")
    report = {"n": len(rows)}
    for k in cutoffs:
        exact = sum(gold in row[:k] for row, gold in rows)
        compatible = sum(any(_is_compatible(c, gold) for c in row[:k]) for row, gold in rows)
        report[f"exact_at_{k}"] = exact / len(rows)
        report[f"exact_at_{k}_ci95"] = wilson(exact, len(rows))
        report[f"compatible_at_{k}"] = compatible / len(rows)
        report[f"compatible_at_{k}_ci95"] = wilson(compatible, len(rows))
    return report

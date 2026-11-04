"""TF-IDF plus logistic regression over product text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from eval.metrics import evaluate
from taxon.products import assign_split, seeded_hash
from taxon.taxonomy import Taxonomy

C_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
LEARNING_CURVE = (0.25, 0.5, 1.0)
MAX_FEATURES = 50_000
# Keeps the body vocabulary non-empty when no product in a batch has any body text.
_BODY_SENTINEL = "bodytext"


def title_text(product: dict) -> str:
    return product["title"]


def body_text(product: dict) -> str:
    details = " ".join(f"{k} {v}" for k, v in sorted(product["details"].items()))
    return " ".join([*product["features"], product["description"], details])


class TfidfLogisticRegression:
    """One-vs-rest logistic regression on separate title and body TF-IDF features."""

    def __init__(self, c: float = 10.0, n_jobs: int = -1):
        self.c = c
        self.n_jobs = n_jobs

    def _features(self, products: Sequence[dict], fit: bool = False):
        titles = [title_text(p) for p in products]
        bodies = [f"{body_text(p)} {_BODY_SENTINEL}" for p in products]
        if fit:
            self._title = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=MAX_FEATURES,
            ).fit(titles)
            self._body = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=MAX_FEATURES,
                token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
            ).fit(bodies)
        return sparse.hstack(
            [self._title.transform(titles), self._body.transform(bodies)], format="csr"
        )

    def fit(self, products: Sequence[dict], labels: Sequence[str]) -> TfidfLogisticRegression:
        self._model = OneVsRestClassifier(
            LogisticRegression(C=self.c, solver="liblinear"), n_jobs=self.n_jobs
        ).fit(self._features(products, fit=True), labels)
        return self

    def predict(self, products: Sequence[dict]) -> list[str]:
        return self._model.predict(self._features(products)).tolist()


def most_frequent_label(labels: Iterable[str]) -> str:
    """The commonest label, ties broken alphabetically so the result is deterministic."""
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def _mean_depth(paths: Iterable[str]) -> float:
    return float(np.mean([p.count(" > ") + 1 for p in paths]))


def run_baseline(
    products: list[dict],
    labels: dict[str, str | None],
    taxonomy: Taxonomy,
    seed: int,
    label_source: str,
    n_jobs: int = -1,
) -> dict:
    """Choose C on the validation split, then score once on the test split."""
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for product in products:
        if labels.get(product["parent_asin"]):
            splits[assign_split(product["parent_asin"], seed)].append(product)
    gold = {
        name: [labels[p["parent_asin"]] for p in rows] for name, rows in splits.items()
    }
    valid = set(taxonomy.paths())

    selection = {}
    fitted = {}
    for c in C_GRID:
        fitted[c] = TfidfLogisticRegression(c=c, n_jobs=n_jobs).fit(splits["train"], gold["train"])
        predicted = fitted[c].predict(splits["val"])
        selection[str(c)] = evaluate(gold["val"], predicted)["exact_leaf_accuracy"]
    best_c = max(C_GRID, key=lambda c: (selection[str(c)], -c))
    predictions = fitted[best_c].predict(splits["test"])

    # A learning curve on validation, used to decide whether more data would pay off.
    order = sorted(
        range(len(splits["train"])),
        key=lambda i: seeded_hash(seed, f"curve|{splits['train'][i]['parent_asin']}"),
    )
    curve = []
    for fraction in LEARNING_CURVE:
        chosen = order[: int(len(order) * fraction)]
        if fraction == 1.0:
            model = fitted[best_c]
        else:
            model = TfidfLogisticRegression(c=best_c, n_jobs=n_jobs).fit(
                [splits["train"][i] for i in chosen], [gold["train"][i] for i in chosen]
            )
        scores = evaluate(gold["val"], model.predict(splits["val"]))
        curve.append(
            {
                "train_examples": len(chosen),
                "val_exact_leaf_accuracy": scores["exact_leaf_accuracy"],
                "val_level1_recall": scores["levels"][0]["micro_recall"],
            }
        )

    majority = most_frequent_label(gold["train"])
    train_labels = set(gold["train"])
    return {
        "label_source": label_source,
        "taxonomy_version": taxonomy.version,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "train_classes": len(train_labels),
        "test_labels_unseen_in_train": sum(g not in train_labels for g in gold["test"])
        / len(gold["test"]),
        "mean_gold_depth": _mean_depth(gold["test"]),
        "mean_predicted_depth": _mean_depth(predictions),
        "c_selection_val_exact_leaf_accuracy": selection,
        "best_c": best_c,
        "tfidf_logistic_regression": evaluate(gold["test"], predictions, valid_paths=valid),
        "most_frequent_label": {
            "label": majority,
            **evaluate(gold["test"], [majority] * len(gold["test"]), valid_paths=valid),
        },
        "learning_curve": curve,
    }

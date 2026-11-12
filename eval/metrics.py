"""Hierarchical classification metrics over taxonomy paths.

A prediction is scored at each level by comparing whole path prefixes, so a node
name that recurs under two parents never counts as a match. A prediction that
stops above a level makes no claim there: it lowers recall at that level and
leaves precision alone. Levels are 1-based; level 1 is the vertical.

Every rate comes with a 95% interval. A share of items (accuracy, micro precision and recall,
the invalid rate) gets a Wilson score interval, which stays inside [0, 1] and behaves on small
sets such as a 50-product gold set. Macro averages are not binomial, so they get a percentile
bootstrap over items when one is asked for.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Collection, Sequence

from taxon.taxonomy import path_prefixes

Z95 = 1.959963984540054
BOOTSTRAP = 2_000


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson(successes: int, n: int, z: float = Z95) -> list[float] | None:
    """The Wilson score interval for successes out of n, or None when n is 0."""
    if n == 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n}")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, centre - half), min(1.0, centre + half)]


def percentile_interval(values: Sequence[float]) -> list[float] | None:
    """The middle 95% of bootstrap values, or None if there are none."""
    ordered = sorted(values)
    if not ordered:
        return None
    count = len(ordered)
    return [ordered[int(0.025 * count)], ordered[max(int(0.975 * count) - 1, 0)]]


def bootstrap_interval(
    statistic: Callable[[list[int]], float | None],
    n: int,
    resamples: int = BOOTSTRAP,
    seed: int = 0,
) -> list[float] | None:
    """A percentile interval for a statistic of n items, resampling item indices with replacement.

    Resamples where the statistic is undefined are left out.
    """
    if n == 0:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        value = statistic(rng.choices(range(n), k=n))
        if value is not None:
            values.append(value)
    return percentile_interval(values)


def paired_difference(
    golds: Sequence[str],
    a: Sequence[str | None],
    b: Sequence[str | None],
    resamples: int = BOOTSTRAP,
    seed: int = 0,
) -> dict:
    """Exact-leaf accuracy of a minus b on the same items, with a paired bootstrap interval.

    Resampling items, not systems, keeps each item's two answers together, which is what makes
    a small gold set able to separate two systems at all.
    """
    if not len(golds) == len(a) == len(b):
        raise ValueError("both systems must answer the same items")
    diffs = [(x == g) - (y == g) for g, x, y in zip(golds, a, b, strict=True)]

    def mean_of(indices: list[int]) -> float:
        return sum(diffs[i] for i in indices) / len(indices)

    return {
        "n": len(diffs),
        "difference": sum(diffs) / len(diffs) if diffs else None,
        "ci95": bootstrap_interval(mean_of, len(diffs), resamples, seed),
    }


def _prefix_at(prefixes: list[str], level: int) -> str | None:
    return prefixes[level - 1] if level <= len(prefixes) else None


def _level_scores(gold_prefixes: list[list[str]], pred_prefixes: list[list[str]]) -> list[dict]:
    max_level = max(len(p) for p in [*gold_prefixes, *pred_prefixes])
    levels = []
    for level in range(1, max_level + 1):
        tp: Counter[str] = Counter()
        gold_count: Counter[str] = Counter()
        pred_count: Counter[str] = Counter()
        for gold_path, pred_path in zip(gold_prefixes, pred_prefixes, strict=True):
            gold = _prefix_at(gold_path, level)
            pred = _prefix_at(pred_path, level)
            if gold is not None:
                gold_count[gold] += 1
            if pred is not None:
                pred_count[pred] += 1
            if gold is not None and gold == pred:
                tp[gold] += 1
        classes = gold_count.keys() | pred_count.keys()
        # Macro averages run over classes seen in gold or predictions; an undefined per-class
        # ratio counts as 0, matching scikit-learn's zero_division=0.
        macro_p = sum((_ratio(tp[c], pred_count[c]) or 0.0) for c in classes) / len(classes)
        macro_r = sum((_ratio(tp[c], gold_count[c]) or 0.0) for c in classes) / len(classes)
        levels.append(
            {
                "level": level,
                "gold_support": sum(gold_count.values()),
                "pred_support": sum(pred_count.values()),
                "true_positives": sum(tp.values()),
                "classes": len(classes),
                "micro_precision": _ratio(sum(tp.values()), sum(pred_count.values())),
                "micro_precision_ci95": wilson(sum(tp.values()), sum(pred_count.values())),
                "micro_recall": _ratio(sum(tp.values()), sum(gold_count.values())),
                "micro_recall_ci95": wilson(sum(tp.values()), sum(gold_count.values())),
                "macro_precision": macro_p,
                "macro_recall": macro_r,
            }
        )
    return levels


def evaluate(
    golds: Sequence[str],
    preds: Sequence[str | None],
    valid_paths: Collection[str] | None = None,
    bootstrap: int = 0,
    seed: int = 0,
) -> dict:
    """Score predicted paths against gold paths.

    A prediction of None or "" is an abstention. `exact_leaf_accuracy` is the share of
    items whose predicted path is exactly the gold path, abstentions counting as wrong.
    With `valid_paths`, `invalid_rate` is the share of non-abstaining predictions naming
    a path outside that set; an invalid path can still earn credit for its valid prefix.

    Shares of items carry Wilson intervals (`*_ci95`). With `bootstrap` resamples, the macro
    averages at each level carry percentile intervals too; leave it 0 on large sets, where the
    resampling is slow and the intervals are narrow anyway.
    """
    if len(golds) != len(preds):
        raise ValueError(f"got {len(golds)} golds and {len(preds)} predictions")
    if not golds:
        raise ValueError("cannot evaluate an empty set")
    if not all(golds):
        raise ValueError("every gold path must be non-empty")

    gold_prefixes = [path_prefixes(g) for g in golds]
    pred_prefixes = [path_prefixes(p) if p else [] for p in preds]
    answered = [p for p in preds if p]
    exact = sum(g == p for g, p in zip(golds, preds, strict=True))
    result = {
        "n": len(golds),
        "exact_leaf_accuracy": exact / len(golds),
        "exact_leaf_accuracy_ci95": wilson(exact, len(golds)),
        "abstained": len(golds) - len(answered),
        "levels": _level_scores(gold_prefixes, pred_prefixes),
    }
    if bootstrap:
        _macro_intervals(result["levels"], gold_prefixes, pred_prefixes, bootstrap, seed)
    if valid_paths is not None:
        invalid = sum(p not in valid_paths for p in answered)
        result["invalid_rate"] = _ratio(invalid, len(answered))
        result["invalid_rate_ci95"] = wilson(invalid, len(answered))
    return result


def _macro_intervals(
    levels: list[dict],
    gold_prefixes: list[list[str]],
    pred_prefixes: list[list[str]],
    resamples: int,
    seed: int,
) -> None:
    """Add bootstrap intervals for each level's macro precision and recall, in place."""
    rng = random.Random(seed)
    n = len(gold_prefixes)
    samples: dict[tuple[int, str], list[float]] = {}
    for _ in range(resamples):
        indices = rng.choices(range(n), k=n)
        scores = _level_scores(
            [gold_prefixes[i] for i in indices], [pred_prefixes[i] for i in indices]
        )
        for row in scores:
            for name in ("macro_precision", "macro_recall"):
                samples.setdefault((row["level"], name), []).append(row[name])
    for row in levels:
        for name in ("macro_precision", "macro_recall"):
            row[f"{name}_ci95"] = percentile_interval(samples.get((row["level"], name), []))

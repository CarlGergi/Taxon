"""Hierarchical classification metrics over taxonomy paths.

A prediction is scored at each level by comparing whole path prefixes, so a node
name that recurs under two parents never counts as a match. A prediction that
stops above a level makes no claim there: it lowers recall at that level and
leaves precision alone. Levels are 1-based; level 1 is the vertical.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence

from taxon.taxonomy import path_prefixes


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


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
                "micro_recall": _ratio(sum(tp.values()), sum(gold_count.values())),
                "macro_precision": macro_p,
                "macro_recall": macro_r,
            }
        )
    return levels


def evaluate(
    golds: Sequence[str],
    preds: Sequence[str | None],
    valid_paths: Collection[str] | None = None,
) -> dict:
    """Score predicted paths against gold paths.

    A prediction of None or "" is an abstention. `exact_leaf_accuracy` is the share of
    items whose predicted path is exactly the gold path, abstentions counting as wrong.
    With `valid_paths`, `invalid_rate` is the share of non-abstaining predictions naming
    a path outside that set; an invalid path can still earn credit for its valid prefix.
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
    result = {
        "n": len(golds),
        "exact_leaf_accuracy": sum(g == p for g, p in zip(golds, preds, strict=True)) / len(golds),
        "abstained": len(golds) - len(answered),
        "levels": _level_scores(gold_prefixes, pred_prefixes),
    }
    if valid_paths is not None:
        invalid = sum(p not in valid_paths for p in answered)
        result["invalid_rate"] = _ratio(invalid, len(answered))
    return result

"""Metrics for selective field extraction.

Each row is one request to a model: {"product": id, "requested": [field, ...], "output": text}.
Field compliance is how often a model outputs only the requested fields. Field invariance is how
consistent its answer for a field stays when the rest of the requested schema changes. Neither
needs labels, so both can be measured on any product.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

_MISSING = object()


def parse_fields(text: str) -> dict | None:
    """The JSON object a model replied with, or None if the reply is anything else."""
    try:
        parsed = json.loads(text.strip())
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def field_compliance(rows: Sequence[Mapping]) -> dict:
    parsed = compliant = extra = missing = 0
    for row in rows:
        fields = parse_fields(row["output"])
        if fields is None:
            continue
        parsed += 1
        wanted = set(row["requested"])
        has_extra = bool(set(fields) - wanted)
        has_missing = bool(wanted - set(fields))
        extra += has_extra
        missing += has_missing
        compliant += not has_extra and not has_missing
    n = len(rows)
    return {
        "requests": n,
        "compliance_rate": compliant / n if n else None,
        "parse_rate": parsed / n if n else None,
        "extra_field_rate": extra / n if n else None,
        "missing_field_rate": missing / n if n else None,
    }


def field_invariance(rows: Sequence[Mapping]) -> dict:
    """Over each (product, field) asked under at least two schemas, the share with one answer.

    An unreadable reply or a dropped field counts as a different answer, since the caller got no
    answer for that field under that schema.
    """
    answers: dict[tuple[str, str], list[object]] = defaultdict(list)
    for row in rows:
        fields = parse_fields(row["output"]) or {}
        for name in row["requested"]:
            answers[(row["product"], name)].append(fields.get(name, _MISSING))
    groups: dict[str, list[bool]] = defaultdict(list)
    for (_, name), given in answers.items():
        if len(given) > 1:
            first = given[0]
            groups[name].append(first is not _MISSING and all(a == first for a in given))
    every = [same for outcomes in groups.values() for same in outcomes]
    return {
        "groups": len(every),
        "invariance_rate": sum(every) / len(every) if every else None,
        "per_field": {
            name: {"groups": len(outcomes), "invariance_rate": sum(outcomes) / len(outcomes)}
            for name, outcomes in sorted(groups.items())
        },
    }


def attribute_accuracy(
    golds: Sequence[Mapping[str, str]], predictions: Sequence[Mapping | None]
) -> dict:
    """Share of hand-labeled attribute values a model returned exactly, by attribute handle."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for gold, predicted in zip(golds, predictions, strict=True):
        predicted = predicted or {}
        for handle, value in gold.items():
            totals[handle][0] += predicted.get(handle) == value
            totals[handle][1] += 1
    correct = sum(right for right, _ in totals.values())
    total = sum(n for _, n in totals.values())
    return {
        "values": total,
        "accuracy": correct / total if total else None,
        "per_attribute": {
            handle: {"values": n, "accuracy": right / n}
            for handle, (right, n) in sorted(totals.items())
        },
    }

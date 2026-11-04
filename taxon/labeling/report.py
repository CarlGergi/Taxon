"""Agreement rates and cost projections for a labeling run."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean, stdev

from taxon.labeling.budget import CAPS
from taxon.taxonomy import path_prefixes


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def agreement_report(rows: list[dict]) -> dict:
    """How often annotators agree, overall, per level, per pair, and per domain."""
    if not rows:
        raise ValueError("cannot report on an empty run")
    vendors = sorted(rows[0]["annotators"])
    groups = Counter()
    per_domain: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pair_exact: Counter[str] = Counter()
    pair_seen: Counter[str] = Counter()
    level_agree: Counter[int] = Counter()
    level_seen: Counter[int] = Counter()
    per_vendor = {v: Counter() for v in vendors}

    for row in rows:
        paths = {v: row["annotators"][v]["path"] for v in vendors}
        answered = [p for p in paths.values() if p]
        distinct = len(set(answered))
        if distinct == 1 and len(answered) == len(vendors):
            groups["unanimous"] += 1
        elif answered and max(Counter(answered).values()) >= 2:
            groups["majority"] += 1
        elif answered:
            groups["no_two_agree"] += 1
        else:
            groups["no_answers"] += 1
        per_domain[row["domain"]][0] += row["source"] == "unanimous"
        per_domain[row["domain"]][1] += 1

        for a, b in combinations(vendors, 2):
            pair = f"{a}|{b}"
            pair_seen[pair] += 1
            pair_exact[pair] += bool(paths[a]) and paths[a] == paths[b]
            if paths[a] and paths[b]:
                pa, pb = path_prefixes(paths[a]), path_prefixes(paths[b])
                for level in range(1, max(len(pa), len(pb)) + 1):
                    level_seen[level] += 1
                    same = level <= min(len(pa), len(pb)) and pa[level - 1] == pb[level - 1]
                    level_agree[level] += same

        for vendor in vendors:
            entry = row["annotators"][vendor]
            stats = per_vendor[vendor]
            stats["picked"] += entry["via"] == "pick"
            stats["walked"] += "walk_steps" in entry
            stats["walk_steps"] += entry.get("walk_steps", 0)
            stats["answered"] += bool(entry["path"])
            stats["invalid"] += bool(entry.get("invalid"))
            stats["refused"] += entry["status"] == "refused"
            stats["failed"] += entry["status"] == "failed"
            stats["truncated"] += entry["status"] == "truncated"

    n = len(rows)
    sources = Counter(row["source"] for row in rows)
    return {
        "n": n,
        "agreement": {k: groups[k] / n for k in ("unanimous", "majority", "no_two_agree")},
        "no_answers": groups["no_answers"] / n,
        "sources": dict(sorted(sources.items())),
        "labeled": sum(bool(row["path"]) for row in rows) / n,
        "pairwise_exact": {p: pair_exact[p] / pair_seen[p] for p in sorted(pair_seen)},
        "pairwise_by_level": {
            str(level): level_agree[level] / level_seen[level] for level in sorted(level_seen)
        },
        "per_vendor": {
            v: {
                "shortlist_pick_rate": stats["picked"] / n,
                "walk_rate": stats["walked"] / n,
                "mean_walk_steps": _rate(stats["walk_steps"], stats["walked"]),
                "answer_rate": stats["answered"] / n,
                "invalid": stats["invalid"],
                "refused": stats["refused"],
                "failed": stats["failed"],
                "truncated": stats["truncated"],
            }
            for v, stats in per_vendor.items()
        },
        "unanimous_by_domain": {d: h / t for d, (h, t) in sorted(per_domain.items())},
    }


def cost_projection(rows: list[dict], targets: dict[str, int], spent: dict[str, float]) -> dict:
    """Per-vendor estimates from a sample's per-product costs, with a 95% upper bound.

    `spent` is what the ledger shows already, and the bound decides whether a target fits its cap.
    """
    if len(rows) < 2:
        raise ValueError("need at least two products to project cost")
    report = {}
    for vendor in sorted(CAPS):
        costs = [row["cost"].get(vendor, 0.0) for row in rows]
        per_product = mean(costs)
        upper = per_product + 1.96 * stdev(costs) / math.sqrt(len(costs))
        entry = {
            "per_product": per_product,
            "cap": CAPS[vendor],
            "spent": spent.get(vendor, 0.0),
        }
        for name, n in targets.items():
            estimate = per_product * n
            bound = upper * n
            entry[name] = {
                "products": n,
                "estimate": estimate,
                "upper_95": bound,
                "fits_cap": entry["spent"] + bound <= CAPS[vendor],
            }
        report[vendor] = entry
    return report


def format_projection(report: dict) -> str:
    lines = []
    for vendor, entry in report.items():
        lines.append(
            f"{vendor}: ${entry['per_product']:.5f} per product, "
            f"${entry['spent']:.2f} spent, cap ${entry['cap']:.2f}"
        )
        for name, value in entry.items():
            if isinstance(value, dict):
                verdict = "fits" if value["fits_cap"] else "EXCEEDS CAP"
                lines.append(
                    f"  {name} ({value['products']:,} products): "
                    f"${value['estimate']:.2f}, up to ${value['upper_95']:.2f}, {verdict}"
                )
    return "\n".join(lines)

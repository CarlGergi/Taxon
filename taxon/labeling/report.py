"""Agreement rates and cost projections for a labeling run."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import combinations
from statistics import mean, stdev

from eval.metrics import evaluate
from taxon.labeling.budget import CAPS
from taxon.taxonomy import path_prefixes

ARBITRATOR_VENDOR = "anthropic"


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
    arbitrated = Counter(row["arbitrator_status"] for row in rows if "arbitrator_status" in row)
    return {
        "n": n,
        "agreement": {k: groups[k] / n for k in ("unanimous", "majority", "no_two_agree")},
        "no_answers": groups["no_answers"] / n,
        "sources": dict(sorted(sources.items())),
        "labeled": sum(bool(row["path"]) for row in rows) / n,
        "arbitrator": {"ran": sum(arbitrated.values()), **dict(sorted(arbitrated.items()))},
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


def arbitrator_score(rows: list[dict], gold: Mapping[str, str]) -> dict | None:
    """How the arbitrator does on real disagreements, against hand labels and a majority vote.

    A majority vote costs nothing, so an arbitrator that cannot beat it is not earning its price.
    Abstentions count as wrong for both. Returns None until a disagreement has a hand label.
    """
    disputed = [r for r in rows if "arbitrator_status" in r and r["parent_asin"] in gold]
    if not disputed:
        return None
    golds = [gold[r["parent_asin"]] for r in disputed]

    def majority(row: dict) -> str | None:
        votes = Counter(a["path"] for a in row["annotators"].values() if a["path"])
        top = votes.most_common(1)
        return top[0][0] if top and top[0][1] >= 2 else None

    return {
        "disagreements_with_hand_labels": len(disputed),
        "arbitrator": evaluate(golds, [r["path"] for r in disputed]),
        "majority_vote": evaluate(golds, [majority(r) for r in disputed]),
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
        tokens = [row.get("tokens", {}).get(vendor, {}) for row in rows]
        cached = sum(t.get("cached", 0) for t in tokens)
        all_input = cached + sum(t.get("input", 0) + t.get("written", 0) for t in tokens)
        entry = {
            "per_product": per_product,
            "sample_cost": sum(costs),
            "cached_share_of_input": _rate(cached, all_input),
            "mean_output_tokens": mean(t.get("output", 0) for t in tokens),
            "cap": CAPS[vendor],
            "spent": spent.get(vendor, 0.0),
        }
        if vendor == ARBITRATOR_VENDOR:
            entry["arbitrator_per_product"] = mean(row.get("arbitrator_cost", 0.0) for row in rows)
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
            f"{vendor}: ${entry['sample_cost']:.4f} measured on this sample, "
            f"${entry['per_product']:.5f} per product, ${entry['spent']:.2f} spent in total, "
            f"cap ${entry['cap']:.2f}"
        )
        share = entry["cached_share_of_input"]
        cached = "unknown" if share is None else f"{share:.0%}"
        lines.append(
            f"  input served from cache: {cached}, "
            f"mean output tokens per product: {entry['mean_output_tokens']:.0f}"
        )
        if "arbitrator_per_product" in entry:
            arbitrator = entry["arbitrator_per_product"]
            lines.append(f"  of which arbitrator: ${arbitrator:.5f} per product")
        for name, value in entry.items():
            if isinstance(value, dict):
                verdict = "fits" if value["fits_cap"] else "EXCEEDS CAP"
                lines.append(
                    f"  {name} ({value['products']:,} products): "
                    f"${value['estimate']:.2f}, up to ${value['upper_95']:.2f}, {verdict}"
                )
    return "\n".join(lines)


def attribute_projection(
    rows: list[dict],
    targets: dict[str, int],
    spent: dict[str, float],
    category: dict,
    vendor: str,
    planned: str,
) -> dict:
    """What the attribute pass would cost, per product sampled, with a 95% upper bound.

    It fits only if everything planned does: what is spent, the category calls for `planned`, and
    the attribute pass for `planned`, all on this vendor's cap.
    """
    if len(rows) < 2:
        raise ValueError("need at least two products to project cost")
    costs = [row.get("attribute_cost", 0.0) for row in rows]
    per_product = mean(costs)
    upper = per_product + 1.96 * stdev(costs) / math.sqrt(len(costs))
    entry = {
        "vendor": vendor,
        "per_product": per_product,
        "sample_cost": sum(costs),
        "called_for": sum("attribute_cost" in row for row in rows),
        "cap": CAPS[vendor],
        "spent": spent.get(vendor, 0.0),
    }
    for name, n in targets.items():
        entry[name] = {"products": n, "estimate": per_product * n, "upper_95": upper * n}
    total = entry["spent"] + category[vendor][planned]["upper_95"] + upper * targets[planned]
    entry["everything_planned"] = {"upper_95": total, "fits_cap": total <= CAPS[vendor]}
    return entry


def format_attribute_projection(entry: dict) -> str:
    lines = [
        f"attribute pass ({entry['vendor']}): ${entry['sample_cost']:.4f} measured for "
        f"{entry['called_for']} settled products, ${entry['per_product']:.5f} per product sampled"
    ]
    for name, value in entry.items():
        if isinstance(value, dict) and "estimate" in value:
            lines.append(
                f"  {name} ({value['products']:,} products): ${value['estimate']:.2f}, "
                f"up to ${value['upper_95']:.2f}"
            )
    planned = entry["everything_planned"]
    verdict = "fits" if planned["fits_cap"] else "EXCEEDS CAP"
    lines.append(
        f"  categories and attributes for everything planned, with what is spent: up to "
        f"${planned['upper_95']:.2f} of ${entry['cap']:.2f}, {verdict}"
    )
    return "\n".join(lines)


def attribute_report(rows: list[dict]) -> dict:
    """How the attribute pass went. Unmatched answers are values the taxonomy does not spell."""
    called = [row for row in rows if "attribute_cost" in row]
    statuses = Counter(row.get("attribute_status") for row in rows)
    kept = sum(len(row.get("attributes", {})) for row in called)
    unmatched = [u for row in called for u in row.get("attribute_unmatched", [])]
    return {
        "called": len(called),
        "statuses": dict(sorted((str(k), v) for k, v in statuses.items())),
        "values_per_product": _rate(kept, len(called)),
        "unmatched_share": _rate(len(unmatched), kept + len(unmatched)),
        "unmatched_examples": unmatched[:10],
    }

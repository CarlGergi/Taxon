"""Score CatalogBench: how well the same agent shops each catalog.

For each catalog and run: task success (the agent's first product meets every requirement),
precision of everything it returned, how often it returned nothing, and the searches, model calls,
tokens and dollars a task took. Across runs, the mean and the spread. Against the raw catalog,
the paired difference in success with a bootstrap interval over tasks.

A returned product counts as acceptable if it is one of the task's gold products or the pooled
judging found it meets the task. Scoring refuses while any returned product is still unjudged or
unresolved, so no number rests on a guess.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from taxon.bench.judge import acceptable_with_pool
from taxon.bench.tasks import Task, fingerprint

BASELINE = "raw"
RESAMPLES = 2_000
_METRICS = (
    "success",
    "precision",
    "empty_answers",
    "searches",
    "model_calls",
    "tokens",
    "cost",
)


class NotReady(RuntimeError):
    """Some returned products still need a judgment, or a catalog is missing runs."""


def outcome(returned: Sequence[str], acceptable: set[str]) -> dict:
    hits = sum(1 for asin in returned if asin in acceptable)
    return {
        "success": bool(returned) and returned[0] in acceptable,
        "precision": hits / len(returned) if returned else None,
        "empty": not returned,
    }


def _tokens(row: Mapping) -> int:
    usage = row.get("usage", {})
    return sum(
        usage.get(k, 0)
        for k in ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens")
    )


def acceptable_sets(
    tasks: Sequence[Task],
    rows: Sequence[Mapping],
    judged: Sequence[dict],
    hand: Sequence[dict],
    gold_ids: set[str],
    catalog_ids: set[str],
) -> dict[str, set[str]]:
    """Each task's acceptable products. Raises NotReady listing what still needs a judgment.

    A gold product is settled by its hand label, and an id outside the catalog, such as one the
    agent made up, is simply wrong; neither needs a judgment.
    """
    by_task = {t.id: t for t in tasks}
    accepted: dict[str, set[str]] = {}
    unresolved: set[tuple[str, str]] = set()
    for task in tasks:
        ok, open_ = acceptable_with_pool(task, judged, hand)
        accepted[task.id] = ok
        unresolved |= {(task.id, asin) for asin in open_}
    seen = {(j["task"], j["asin"]) for j in judged} | {(h["task"], h["asin"]) for h in hand}
    missing = set()
    for row in rows:
        task = by_task.get(row["task"])
        if task is None:
            raise NotReady(f"a run names task {row['task']}, which is not in the task set")
        for asin in row["returned"]:
            settled = asin in gold_ids or asin not in catalog_ids
            if not settled and (task.id, asin) not in seen:
                missing.add((task.id, asin))
    returned = {(r["task"], a) for r in rows for a in r["returned"]}
    waiting = (unresolved & returned) | missing
    if waiting:
        raise NotReady(
            f"{len(missing)} returned products have no judgment and "
            f"{len(unresolved & returned)} are unresolved; judge them first"
        )
    return accepted


def run_metrics(rows: Sequence[Mapping], accepted: Mapping[str, set[str]]) -> dict:
    """One run of one catalog over its tasks."""
    outcomes = [outcome(r["returned"], accepted[r["task"]]) for r in rows]
    precisions = [o["precision"] for o in outcomes if o["precision"] is not None]
    n = len(rows)
    return {
        "tasks": n,
        "success": sum(o["success"] for o in outcomes) / n,
        "precision": statistics.mean(precisions) if precisions else None,
        "empty_answers": sum(o["empty"] for o in outcomes) / n,
        "searches": statistics.mean(r["searches"] for r in rows),
        "model_calls": statistics.mean(r["model_calls"] for r in rows),
        "tokens": statistics.mean(_tokens(r) for r in rows),
        "cost": statistics.mean(r["cost"] for r in rows),
        "stopped": dict(sorted(_count(r["stopped"] for r in rows).items())),
    }


def _count(values) -> dict:
    counts: dict = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def spread(values: Sequence[float | None]) -> dict:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(present),
        "sd": statistics.stdev(present) if len(present) > 1 else 0.0,
        "min": min(present),
        "max": max(present),
    }


def per_task_success(rows: Sequence[Mapping], accepted: Mapping[str, set[str]]) -> dict[str, float]:
    """Each task's success averaged over runs, which is what the paired comparison resamples."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(outcome(row["returned"], accepted[row["task"]])["success"])
    return {task: sum(v) / len(v) for task, v in by_task.items()}


def paired_difference(
    a: Mapping[str, float], b: Mapping[str, float], seed: int, resamples: int = RESAMPLES
) -> dict:
    """Mean of a minus b over shared tasks, with a 95% bootstrap interval over tasks."""
    tasks = sorted(set(a) & set(b))
    diffs = [a[t] - b[t] for t in tasks]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return {
        "tasks": len(tasks),
        "difference": statistics.mean(diffs),
        "ci95": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
    }


def check_complete(
    rows: Sequence[Mapping], tasks: Sequence[Task], conditions: Sequence[str]
) -> int:
    """Every catalog must have the same tasks and the same number of runs. Returns that number."""
    expected = {t.id for t in tasks}
    runs_per: dict[str, set[int]] = {}
    for condition in conditions:
        mine = [r for r in rows if r["condition"] == condition]
        runs = {r["run"] for r in mine}
        for run in runs:
            got = {r["task"] for r in mine if r["run"] == run}
            if got != expected:
                raise NotReady(f"{condition} run {run} covers {len(got)} of {len(expected)} tasks")
        runs_per[condition] = runs
    counts = {c: len(r) for c, r in runs_per.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise NotReady(f"catalogs have different numbers of runs: {counts}")
    return next(iter(counts.values()))


def score(
    tasks: Sequence[Task],
    rows: Sequence[Mapping],
    judged: Sequence[dict],
    hand: Sequence[dict],
    conditions: Sequence[str],
    enrichment: Mapping[str, Mapping],
    seed: int,
    gold_ids: set[str],
    catalog_ids: set[str],
) -> dict:
    """The benchmark report. Raises NotReady until every run is in and every product is judged."""
    rows = [r for r in rows if r["condition"] in conditions]
    runs = check_complete(rows, tasks, conditions)
    accepted = acceptable_sets(tasks, rows, judged, hand, gold_ids, catalog_ids)
    report: dict = {
        "tasks": len(tasks),
        "task_set_sha256": fingerprint(tasks),
        "runs_per_catalog": runs,
        "catalogs": {},
    }
    successes = {}
    for condition in conditions:
        mine = [r for r in rows if r["condition"] == condition]
        per_run = [
            run_metrics([r for r in mine if r["run"] == run], accepted)
            for run in sorted({r["run"] for r in mine})
        ]
        report["catalogs"][condition] = {
            **{m: spread([p[m] for p in per_run]) for m in _METRICS},
            "runs": per_run,
            "enrichment": dict(enrichment.get(condition, {})),
        }
        successes[condition] = per_task_success(mine, accepted)
    if BASELINE in conditions:
        report["success_minus_raw"] = {
            c: paired_difference(successes[c], successes[BASELINE], seed)
            for c in conditions
            if c != BASELINE
        }
    return report


def enrichment_cost(
    dollars: float | None = None,
    products: int = 0,
    hours_per_million: float | None = None,
    hardware: Mapping | None = None,
) -> dict:
    """What enriching a million products takes: dollars for an API model, local hours for Taxon."""
    entry: dict = {}
    if dollars is not None and products:
        entry["dollars_per_million_products"] = dollars / products * 1_000_000
    if hours_per_million is not None:
        entry["local_hours_per_million_products"] = hours_per_million
        entry["dollars_per_million_products"] = 0.0
    if hardware:
        entry["hardware"] = dict(hardware)
    return entry


def write_report(report: dict) -> None:
    from taxon.reproduce import write_result

    write_result("catalogbench", report)

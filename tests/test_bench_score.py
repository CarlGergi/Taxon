"""Scoring CatalogBench from logged runs and judgments."""

import json

import pytest

from taxon.bench import cli, score
from taxon.bench.judge import FAILS, MEETS, UNSURE
from taxon.bench.score import (
    NotReady,
    acceptable_sets,
    check_complete,
    enrichment_cost,
    outcome,
    paired_difference,
    per_task_success,
    run_metrics,
    spread,
)
from taxon.bench.tasks import Constraint, Task

CHEAP = Constraint("max_price", amount=30.0)
LIMIT = Constraint("max_price", amount=50.0)
TASKS = [
    Task("t1", [CHEAP, LIMIT], ["G1"], "a cheap tote", True),
    Task("t2", [CHEAP, LIMIT], ["G2"], "a cheap knife", True),
]
GOLD = {"G1", "G2", "G3"}
CATALOG = GOLD | {"D1", "D2"}


def row(condition, run, task, returned, searches=2, cost=0.001, stopped="submitted"):
    return {
        "condition": condition,
        "run": run,
        "task": task,
        "returned": returned,
        "searches": searches,
        "model_calls": searches + 1,
        "usage": {"input_tokens": 100, "cached_tokens": 50, "output_tokens": 10},
        "cost": cost,
        "stopped": stopped,
    }


def judgment(task, asin, verdict):
    return {"task": task, "asin": asin, "verdict": verdict}


# One answer


def test_success_means_the_first_product_is_acceptable():
    assert outcome(["G1", "X"], {"G1"})["success"] is True
    assert outcome(["X", "G1"], {"G1"})["success"] is False
    assert outcome([], {"G1"}) == {"success": False, "precision": None, "empty": True}
    assert outcome(["G1", "X"], {"G1"})["precision"] == 0.5


# Acceptable products


def test_a_judged_distractor_that_meets_the_task_counts():
    rows = [row("raw", 1, "t1", ["D1"])]
    accepted = acceptable_sets(TASKS, rows, [judgment("t1", "D1", MEETS)], [], GOLD, CATALOG)
    assert accepted["t1"] == {"G1", "D1"}


def test_a_hand_judgment_overrides_the_judge():
    rows = [row("raw", 1, "t1", ["D1"])]
    accepted = acceptable_sets(
        TASKS, rows, [judgment("t1", "D1", MEETS)], [judgment("t1", "D1", FAILS)], GOLD, CATALOG
    )
    assert accepted["t1"] == {"G1"}


def test_scoring_waits_for_an_unjudged_distractor():
    with pytest.raises(NotReady, match="no judgment"):
        acceptable_sets(TASKS, [row("raw", 1, "t1", ["D1"])], [], [], GOLD, CATALOG)


def test_scoring_waits_for_an_unresolved_one():
    rows = [row("raw", 1, "t1", ["D1"])]
    with pytest.raises(NotReady, match="unresolved"):
        acceptable_sets(TASKS, rows, [judgment("t1", "D1", UNSURE)], [], GOLD, CATALOG)


def test_gold_and_made_up_ids_need_no_judgment():
    rows = [row("raw", 1, "t1", ["G3", "INVENTED"])]
    accepted = acceptable_sets(TASKS, rows, [], [], GOLD, CATALOG)
    assert accepted["t1"] == {"G1"}


def test_a_run_naming_an_unknown_task_is_refused():
    with pytest.raises(NotReady, match="not in the task set"):
        acceptable_sets(TASKS, [row("raw", 1, "t9", [])], [], [], GOLD, CATALOG)


# Runs and spread


def test_run_metrics_average_over_tasks():
    rows = [row("raw", 1, "t1", ["G1"], searches=2), row("raw", 1, "t2", [], searches=4)]
    metrics = run_metrics(rows, {"t1": {"G1"}, "t2": {"G2"}})
    assert metrics["success"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["empty_answers"] == 0.5
    assert metrics["searches"] == 3
    assert metrics["tokens"] == 160
    assert metrics["stopped"] == {"submitted": 2}


def test_spread_gives_mean_sd_and_range():
    assert spread([0.5, 0.7, 0.6]) == pytest.approx(
        {"mean": 0.6, "sd": 0.1, "min": 0.5, "max": 0.7}
    )
    assert spread([0.5])["sd"] == 0.0
    assert spread([None, None])["mean"] is None


def test_every_catalog_needs_the_same_tasks_and_runs():
    full = [row(c, r, t.id, []) for c in ("raw", "taxon") for r in (1, 2) for t in TASKS]
    assert check_complete(full, TASKS, ["raw", "taxon"]) == 2
    with pytest.raises(NotReady, match="covers 1 of 2"):
        check_complete(full[:-1], TASKS, ["raw", "taxon"])
    one_run_short = [r for r in full if not (r["condition"] == "taxon" and r["run"] == 2)]
    with pytest.raises(NotReady, match="different numbers of runs"):
        check_complete(one_run_short, TASKS, ["raw", "taxon"])


# Comparing catalogs


def test_per_task_success_averages_over_runs():
    rows = [row("raw", 1, "t1", ["G1"]), row("raw", 2, "t1", [])]
    assert per_task_success(rows, {"t1": {"G1"}}) == {"t1": 0.5}


def test_the_paired_difference_has_an_interval_that_holds_it():
    better = {f"t{i}": 1.0 for i in range(40)}
    worse = {f"t{i}": (1.0 if i % 4 == 0 else 0.0) for i in range(40)}
    report = paired_difference(better, worse, seed=1)
    assert report["tasks"] == 40
    assert report["difference"] == pytest.approx(0.75)
    low, high = report["ci95"]
    assert low <= 0.75 <= high and low > 0
    assert report == paired_difference(better, worse, seed=1)


def test_the_full_report_scores_each_catalog_against_raw():
    rows = []
    for run in (1, 2):
        rows += [row("raw", run, "t1", ["D2"]), row("raw", run, "t2", [])]
        rows += [row("taxon", run, "t1", ["G1"]), row("taxon", run, "t2", ["D1"], cost=0.002)]
    judged = [judgment("t1", "D2", FAILS), judgment("t2", "D1", MEETS)]
    costs = {"taxon": enrichment_cost(hours_per_million=2.0, hardware={"chip": "M5 Max"})}
    report = score.score(
        TASKS, rows, judged, [], ["raw", "taxon"], costs, seed=1, gold_ids=GOLD, catalog_ids=CATALOG
    )
    assert report["runs_per_catalog"] == 2
    assert report["catalogs"]["raw"]["success"]["mean"] == 0.0
    assert report["catalogs"]["taxon"]["success"]["mean"] == 1.0
    assert report["catalogs"]["taxon"]["cost"]["mean"] == pytest.approx(0.0015)
    assert report["success_minus_raw"]["taxon"]["difference"] == 1.0
    assert report["catalogs"]["taxon"]["enrichment"]["local_hours_per_million_products"] == 2.0
    assert len(report["task_set_sha256"]) == 64
    json.dumps(report)


def test_enrichment_cost_is_dollars_for_an_api_and_hours_for_local_compute():
    assert enrichment_cost(dollars=10.0, products=2_000) == {
        "dollars_per_million_products": 5_000.0
    }
    local = enrichment_cost(hours_per_million=3.5, hardware={"device": "mps"})
    assert local["dollars_per_million_products"] == 0.0
    assert local["local_hours_per_million_products"] == 3.5


# The command


def test_the_command_says_what_is_missing_instead_of_scoring(tmp_path, monkeypatch, capsys):
    from taxon.bench.agent import RunLog

    log = RunLog(tmp_path / "agent")
    log.add(row("raw", 1, "t1", ["D1"]), [])
    log.add(row("raw", 1, "t2", []), [])
    monkeypatch.setattr(cli.agent, "agent_dir", lambda fake: tmp_path / "agent")
    monkeypatch.setattr(cli.judging, "JUDGING_DIR", tmp_path / "judging")
    monkeypatch.setattr(cli, "_benchmark_ids", lambda derived: sorted(CATALOG))
    written = []
    monkeypatch.setattr(cli.score, "write_report", written.append)

    class Item:
        def __init__(self, asin):
            self.asin = asin

    args = type("Args", (), {"conditions": ["raw"], "taxon_run": None})()
    items = [Item(a) for a in GOLD]
    assert cli._score(args, TASKS, items, tmp_path) == 1
    assert "NOT READY" in capsys.readouterr().out
    assert written == []


def test_reproduce_skips_the_benchmark_until_there_are_runs(tmp_path, monkeypatch, capsys):
    from taxon import reproduce

    monkeypatch.setattr(cli.agent, "agent_dir", lambda fake: tmp_path / "agent")
    reproduce.benchmark_report()
    assert "No agent runs yet" in capsys.readouterr().out


def test_the_latest_taxon_enrichment_is_cited_without_naming_its_run(tmp_path, monkeypatch):
    path = tmp_path / "taxon_enrichment.json"
    path.write_text(json.dumps({"hours_per_million_products": 800.0, "hardware": {"chip": "M5"}}))
    monkeypatch.setattr(cli, "TAXON_ENRICHMENT", path)
    monkeypatch.setattr(cli, "SONNET_ENRICHMENT", tmp_path / "missing.json")
    costs = cli._enrichment_costs(None)
    assert costs["taxon"]["local_hours_per_million_products"] == 800.0
    assert "sonnet" not in costs


# Tasks from the same product


def test_tasks_from_one_product_are_resampled_together():
    from taxon.bench.score import cluster_interval

    # Forty tasks from two products: twenty that always succeed, twenty that never do.
    values = {f"t{i}": float(i < 20) for i in range(40)}
    as_tasks = cluster_interval(values, None, seed=1)
    as_products = cluster_interval(values, {t: ("P1" if v else "P2") for t, v in values.items()}, 1)
    assert as_tasks[1] - as_tasks[0] < 0.4
    assert as_products == [0.0, 1.0]


def test_the_report_counts_source_products_and_gives_every_metric_an_interval():
    tasks = [
        Task("t1", [CHEAP, LIMIT], ["G1"], "a cheap tote", True, source="G1"),
        Task("t2", [CHEAP, LIMIT], ["G1"], "a cheap bag", True, source="G1"),
    ]
    rows = [row(c, r, t.id, ["G1"]) for c in ("raw", "taxon") for r in (1, 2) for t in tasks]
    report = score.score(
        tasks, rows, [], [], ["raw", "taxon"], {}, seed=1, gold_ids=GOLD, catalog_ids=CATALOG
    )
    assert report["source_products"] == 1
    over = report["catalogs"]["taxon"]["over_tasks"]
    assert set(over) == {
        "success",
        "precision",
        "empty_answers",
        "searches",
        "model_calls",
        "tokens",
        "cost",
    }
    assert over["success"] == {"mean": 1.0, "ci95": [1.0, 1.0]}
    assert report["success_minus_raw"]["taxon"]["source_products"] == 1

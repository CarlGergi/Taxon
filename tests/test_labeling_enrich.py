"""Sonnet's zero-shot enrichment of the benchmark catalog, with fake vendors."""

import json

import pytest

from taxon.labeling import cli
from taxon.labeling.attributes import EXACT, SHORTLIST
from taxon.labeling.budget import Budget
from taxon.labeling.cache import ResponseCache
from taxon.labeling.enrich import (
    ASSUMED_WALKS,
    ENRICHER,
    WalkStats,
    enrich,
    enrichment_rows,
    price_enrichment,
    walk_stats,
)
from taxon.labeling.env import Secrets
from taxon.labeling.fakes import FakeBatchClient, answer, demo_responder
from taxon.labeling.llm import LLMResult, Status
from taxon.labeling.runner import BatchRunner
from taxon.products import read_jsonl, write_jsonl

TOTES = "Apparel > Bags > Totes"
KITCHEN = "Home > Kitchen"


def product(asin="A1", **extra):
    return {
        "parent_asin": asin,
        "domain": "d",
        "title": "Canvas tote",
        "features": [],
        "description": "A bag",
        "details": {"Color": "Blue"},
    } | extra


def runner_for(tmp_path, responder=demo_responder):
    client = FakeBatchClient("anthropic", responder)
    runner = BatchRunner(
        {"anthropic": client},
        ResponseCache(tmp_path / "cache"),
        Budget(tmp_path / "ledger.jsonl", {"anthropic": 5.0}),
        tmp_path / "open",
        sleep=lambda _: None,
    )
    return runner, client


def chars(request):
    return (len(request.system) + len(request.user)) // 4


# Enriching


def test_sonnet_alone_labels_and_nothing_is_arbitrated(tmp_path, taxonomy):
    runner, client = runner_for(tmp_path, lambda r: answer(r, 1))
    products = [product(f"A{i}") for i in range(4)]
    shortlists = {p["parent_asin"]: [TOTES, KITCHEN] for p in products}
    rows = enrich(products, shortlists, taxonomy, runner, EXACT)
    assert {r["source"] for r in rows} == {"unanimous"}
    models = {r.config.model for batch in client.submitted for r in batch}
    assert models == {ENRICHER.model}
    purposes = {r.purpose for batch in client.submitted for r in batch}
    assert "arbitrate" not in purposes


def test_attributes_are_added_in_the_chosen_version(tmp_path, taxonomy):
    runner, _ = runner_for(tmp_path)
    products = [product(f"A{i}") for i in range(6)]
    shortlists = {p["parent_asin"]: [TOTES, KITCHEN] for p in products}
    rows = enrich(products, shortlists, taxonomy, runner, SHORTLIST)
    labeled = [r for r in rows if r["path"]]
    assert labeled
    for row in labeled:
        assert set(row["attribute_runs"]) <= {SHORTLIST}
    assert all(r.get("attribute_mode") in (None, SHORTLIST) for r in rows)


def test_an_abstention_leaves_the_listing_unenriched(tmp_path, taxonomy):
    runner, _ = runner_for(tmp_path, lambda r: answer(r, 0))
    rows = enrich([product()], {"A1": [TOTES]}, taxonomy, runner, EXACT)
    assert rows[0]["source"] == "no_proposal"
    assert enrichment_rows(rows) == []


def test_enrichment_rows_are_what_the_search_index_reads():
    rows = [
        {"parent_asin": "A", "path": TOTES, "attributes": {"color": ["Red"]}, "source": "x"},
        {"parent_asin": "B", "path": None, "source": "no_proposal"},
        {"parent_asin": "C", "path": KITCHEN, "source": "x"},
    ]
    assert enrichment_rows(rows) == [
        {"parent_asin": "A", "path": TOTES, "attributes": {"color": ["Red"]}},
        {"parent_asin": "C", "path": KITCHEN, "attributes": {}},
    ]


def test_enrichment_stops_at_the_cap(tmp_path, taxonomy):
    runner, _ = runner_for(tmp_path)
    runner.budget = Budget(tmp_path / "tight.jsonl", {"anthropic": 1e-9})
    from taxon.labeling.budget import BudgetExceeded

    with pytest.raises(BudgetExceeded):
        enrich([product()], {"A1": [TOTES]}, taxonomy, runner, EXACT)


# Pricing


def test_walks_come_from_the_dry_run_or_are_assumed_and_said_so():
    measured = walk_stats({"per_vendor": {"anthropic": {"walk_rate": 0.2, "mean_walk_steps": 2.5}}})
    assert measured == WalkStats(0.2, 2.5, True)
    assert walk_stats(None) == ASSUMED_WALKS
    assert walk_stats({"per_vendor": {"anthropic": {"walk_rate": 0, "mean_walk_steps": None}}}) == (
        ASSUMED_WALKS
    )
    assert ASSUMED_WALKS.measured is False


def test_the_price_counts_sonnet_requests_and_scales_to_the_catalog(taxonomy):
    seen = []

    def count(request):
        seen.append(request)
        return chars(request)

    sample = [product(f"A{i}") for i in range(5)]
    shortlists = {p["parent_asin"]: [TOTES, KITCHEN] for p in sample}
    report = price_enrichment(sample, shortlists, taxonomy, count, WalkStats(0.2, 2, True), 2_000)
    assert {r.config.model for r in seen} == {ENRICHER.model}
    assert {r.purpose for r in seen} >= {"pick", "walk"}
    assert report["products"] == 2_000
    for name in ("expected", "high"):
        assert report["total"][name] == pytest.approx(2_000 * report["per_product"][name])
    assert report["total"]["high"] > report["total"]["expected"] > 0
    assert report["mean_input_tokens"]["system_prompt"] > 0


def test_more_walking_costs_more(taxonomy):
    sample = [product()]
    shortlists = {"A1": [TOTES]}
    few = price_enrichment(sample, shortlists, taxonomy, chars, WalkStats(0.1, 1, True), 100)
    many = price_enrichment(sample, shortlists, taxonomy, chars, WalkStats(0.9, 4, True), 100)
    assert many["total"]["expected"] > few["total"]["expected"]
    assert many["calls_per_product"] > few["calls_per_product"]


def test_the_attribute_version_changes_the_attribute_request_priced(taxonomy):
    sample = [product()]
    shortlists = {"A1": [TOTES]}
    walks = WalkStats(0.0, 0.0, True)
    exact = price_enrichment(sample, shortlists, taxonomy, chars, walks, 1, EXACT)
    listed = price_enrichment(sample, shortlists, taxonomy, chars, walks, 1, SHORTLIST)
    assert exact["mode"] == EXACT and listed["mode"] == SHORTLIST
    assert exact["mean_input_tokens"]["attributes"] != listed["mean_input_tokens"]["attributes"]


# Commands


@pytest.fixture
def inputs(monkeypatch, taxonomy):
    products = [product(f"A{i}") for i in range(30)]
    gold = ["A0", "A1", "A2"]
    shortlists = {p["parent_asin"]: [TOTES, KITCHEN] for p in products}
    monkeypatch.setattr(cli, "_load_inputs", lambda: (taxonomy, products, gold, shortlists))
    monkeypatch.setattr(cli, "FIRST_RUN_PRODUCTS", 10)
    monkeypatch.setattr(cli, "load_secrets", lambda: Secrets({}))
    catalog = ["A0", "A1", "A2", "A20", "A21", "A22"]
    monkeypatch.setattr(
        cli,
        "_benchmark_products",
        lambda products, gold: [p for p in products if p["parent_asin"] in catalog],
    )
    return products, gold


def run(tmp_path, *argv):
    return cli.main(list(argv), root=tmp_path)


def scored(tmp_path, monkeypatch, better=SHORTLIST):
    path = tmp_path / "gold_eval.json"
    path.write_text(json.dumps({"attribute_pass": {"products": 5, "better": better}}))
    original = cli.Workspace.__init__

    def init(self, root):
        original(self, root)
        self.gold_eval = path

    monkeypatch.setattr(cli.Workspace, "__init__", init)


def labeled(tmp_path):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--first", "--fake") == 0


def test_pricing_writes_both_versions_and_what_is_left(tmp_path, inputs, capsys):
    assert run(tmp_path, "enrich-price", "--fake") == 0
    report = json.loads((tmp_path / "results" / "enrichment_price.json").read_text())
    assert set(report["by_mode"]) == {EXACT, SHORTLIST}
    assert report["by_mode"][EXACT]["products"] == 6
    assert report["anthropic_remaining"] == pytest.approx(20.0)
    assert "remaining under the cap" in capsys.readouterr().out


def test_enrichment_waits_for_the_labeling_run(tmp_path, inputs, monkeypatch):
    scored(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="labeling run first"):
        run(tmp_path, "enrich", "--fake")


def unscored(monkeypatch, tmp_path):
    """Point the gold scoring at a file that does not exist, whatever the real results hold."""
    original = cli.Workspace.__init__

    def init(self, root):
        original(self, root)
        self.gold_eval = tmp_path / "missing_gold_eval.json"

    monkeypatch.setattr(cli.Workspace, "__init__", init)


def test_enrichment_waits_for_the_gold_scoring(tmp_path, inputs, monkeypatch):
    unscored(monkeypatch, tmp_path)
    labeled(tmp_path)
    with pytest.raises(SystemExit, match="against hand labels"):
        run(tmp_path, "enrich", "--fake")


def test_enrichment_waits_for_a_price(tmp_path, inputs, monkeypatch):
    scored(tmp_path, monkeypatch)
    labeled(tmp_path)
    with pytest.raises(SystemExit, match="Price the enrichment first"):
        run(tmp_path, "enrich", "--fake")


def test_enrichment_refuses_a_price_above_what_is_left(tmp_path, inputs, monkeypatch):
    scored(tmp_path, monkeypatch)
    labeled(tmp_path)
    assert run(tmp_path, "enrich-price", "--fake") == 0
    path = tmp_path / "results" / "enrichment_price.json"
    report = json.loads(path.read_text())
    report["by_mode"][SHORTLIST]["total"]["expected"] = 25.0
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="more than the"):
        run(tmp_path, "enrich", "--fake")


def test_enrichment_writes_rows_and_the_file_the_index_reads(tmp_path, inputs, monkeypatch, capsys):
    scored(tmp_path, monkeypatch, better=SHORTLIST)
    labeled(tmp_path)
    assert run(tmp_path, "enrich-price", "--fake") == 0
    capsys.readouterr()
    assert run(tmp_path, "enrich", "--fake") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["products"] == 6 and report["mode"] == SHORTLIST
    rows = read_jsonl(tmp_path / "enrichment_rows.jsonl")
    assert {r["parent_asin"] for r in rows} == {"A0", "A1", "A2", "A20", "A21", "A22"}
    saved = json.loads((tmp_path / "results" / "enrichment.json").read_text())
    assert saved["cost"] == pytest.approx(report["cost"]) and saved["products"] == 6
    enrichments = read_jsonl(tmp_path / "enrichments_sonnet.jsonl")
    assert len(enrichments) == report["enriched"]
    assert all(set(e) == {"parent_asin", "path", "attributes"} for e in enrichments)
    assert not (tmp_path / "labels.jsonl").read_text() == ""


def test_the_enriched_file_builds_a_catalog(tmp_path, taxonomy):
    from taxon.bench.catalog import build_catalog

    write_jsonl(tmp_path / "e.jsonl", [{"parent_asin": "A1", "path": TOTES, "attributes": {}}])
    products = [product("A1"), product("A2")]
    rows, report = build_catalog(products, tmp_path / "e.jsonl", taxonomy)
    assert report["enriched"] == 1 and report["not_enriched"] == 1
    assert next(r for r in rows if r["id"] == "A1")["category"] == TOTES


def test_the_failed_requests_are_not_cached_as_answers(tmp_path, taxonomy):
    runner, client = runner_for(tmp_path, lambda r: LLMResult(r.key, Status.FAILED, error="x"))
    rows = enrich([product()], {"A1": [TOTES]}, taxonomy, runner, EXACT)
    assert rows[0]["path"] is None
    assert len(ResponseCache(tmp_path / "cache")) == 0

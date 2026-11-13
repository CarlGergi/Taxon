import ast
import json
from pathlib import Path

import pytest

import taxon.reproduce
from taxon.gold.pool import GoldOverlap, gold_queue
from taxon.gold.store import GoldStore
from taxon.products import write_jsonl
from taxon.reproduce import build_gold_suggestions, gold_report, retrieval_report

TOTES = "Apparel > Bags > Totes"
KNIVES = "Home > Kitchen > Knives"
WORDS = {TOTES: "canvas tote", KNIVES: "chef knife"}


def rows(n=60):
    products, labels = [], {}
    paths = list(WORDS)
    for i in range(n):
        path = paths[i % 2]
        asin = f"A{i}"
        products.append(
            {
                "parent_asin": asin,
                "domain": f"d{i % 3}",
                "title": f"{WORDS[path]} {i}",
                "features": [],
                "description": "",
                "details": {},
            }
        )
        labels[asin] = path
    return products, labels


def test_reproduce_never_imports_anything_that_spends_money():
    tree = ast.parse(Path(taxon.reproduce.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    forbidden = {"taxon.labeling.vendors", "taxon.labeling.cli", "taxon.labeling.env"}
    assert not (names & forbidden)
    assert not any(n in {"anthropic", "openai", "google", "google.genai"} for n in names)


def test_suggestions_come_only_from_the_baseline_on_non_gold_products():
    products, labels = rows()
    queue = gold_queue(products, seed=1, size=6)
    suggestions = build_gold_suggestions(products, queue, labels, c=10.0)
    assert {row["parent_asin"] for row in suggestions} == set(queue)
    assert all(row["suggestions"] for row in suggestions)


def test_suggestions_do_not_train_on_gold(monkeypatch):
    products, labels = rows()
    queue = gold_queue(products, seed=1, size=6)
    original = taxon.reproduce.TfidfLogisticRegression.fit

    def spy(self, training, y):
        assert not ({p["parent_asin"] for p in training} & set(queue))
        return original(self, training, y)

    monkeypatch.setattr(taxon.reproduce.TfidfLogisticRegression, "fit", spy)
    build_gold_suggestions(products, queue, labels, c=10.0)


def test_retrieval_report_is_marked_and_excludes_gold(taxonomy):
    products, labels = rows(20)
    queue = gold_queue(products, seed=1, size=4)
    shortlists = {p["parent_asin"]: taxonomy.paths() for p in products}
    report = retrieval_report(products, taxonomy, labels, queue, shortlists)
    assert report["label_source"] == "silver"
    assert report["shortlist"]["n"] == 16
    assert report["shortlist"]["exact_at_50"] == 1.0
    assert report["similarity_only"]["n"] == 16


@pytest.fixture
def gold_world(tmp_path, taxonomy):
    products, labels = rows()
    queue = gold_queue(products, seed=1, size=6)
    store = GoldStore(tmp_path / "gold.jsonl", queue, taxonomy)
    shortlists = {p["parent_asin"]: [TOTES, KNIVES] for p in products}
    return products, labels, queue, store, shortlists, tmp_path


def test_no_hand_labels_means_no_gold_report(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    report = gold_report(
        products,
        taxonomy,
        queue,
        labels,
        shortlists,
        10.0,
        store.path,
        tmp_path / "none",
        tmp_path / "none2",
    )
    assert report is None


def test_the_gold_report_scores_the_baseline_and_shortlists(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    for asin in queue[:3]:
        store.label(asin, labels[asin], {})
    report = gold_report(
        products,
        taxonomy,
        queue,
        labels,
        shortlists,
        10.0,
        store.path,
        tmp_path / "none",
        tmp_path / "none2",
    )
    assert report["labeled"] == 3
    assert report["baseline_trained_on_silver"]["n"] == 3
    assert report["shortlist_recall"]["exact_at_50"] == 1.0
    assert "pipeline_labels" not in report
    assert "baseline_trained_on_pipeline_labels" not in report


def test_the_gold_report_measures_pipeline_labels_when_present(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    asin = queue[0]
    store.label(asin, labels[asin], {})
    rows_path = tmp_path / "rows.jsonl"
    write_jsonl(rows_path, [{"parent_asin": asin, "path": labels[asin]}])
    report = gold_report(
        products,
        taxonomy,
        queue,
        labels,
        shortlists,
        10.0,
        store.path,
        rows_path,
        tmp_path / "none",
        tmp_path / "ladder",
    )
    assert report["pipeline_labels"]["exact_leaf_accuracy"] == 1.0
    assert "ladder" not in report


def test_the_gold_report_scores_saved_ladder_answers(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    asin = queue[0]
    store.label(asin, labels[asin], {})
    rows_path = tmp_path / "rows.jsonl"
    write_jsonl(rows_path, [{"parent_asin": asin, "path": labels[asin]}])
    ladder = tmp_path / "ladder"
    ladder.mkdir()
    (ladder / "qwen-lora.json").write_text(
        json.dumps({"free": {asin: labels[asin]}, "constrained": {asin: None}})
    )
    report = gold_report(
        products,
        taxonomy,
        queue,
        labels,
        shortlists,
        10.0,
        store.path,
        rows_path,
        tmp_path / "none",
        ladder,
    )
    free, constrained = report["ladder"]["rungs"]
    assert free["category"]["exact_leaf_accuracy"] == 1.0
    assert constrained["minus_pipeline_labels"]["difference"] == -1.0


def test_the_baseline_is_rerun_on_pipeline_labels_when_they_exist(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    asin = queue[0]
    store.label(asin, labels[asin], {})
    training_path = tmp_path / "training.jsonl"
    pool = set(queue)
    write_jsonl(
        training_path,
        [
            {"parent_asin": p["parent_asin"], "path": labels[p["parent_asin"]]}
            for p in products
            if p["parent_asin"] not in pool
        ],
    )
    report = gold_report(
        products,
        taxonomy,
        queue,
        labels,
        shortlists,
        10.0,
        store.path,
        tmp_path / "none",
        training_path,
        tmp_path / "no-ladder",
    )
    assert report["baseline_trained_on_pipeline_labels"]["n"] == 1
    assert "ladder_fit_to_training_labels" not in report


def test_a_run_s_fit_to_its_own_training_labels_is_reported(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    store.label(queue[0], labels[queue[0]], {})
    pool = set(queue)
    trained = [p["parent_asin"] for p in products if p["parent_asin"] not in pool][:4]
    training_path = tmp_path / "training.jsonl"
    write_jsonl(training_path, [{"parent_asin": a, "path": labels[a]} for a in trained])
    ladder = tmp_path / "ladder"
    ladder.mkdir()
    answers = {a: labels[a] for a in trained[:3]} | {trained[3]: None}
    (ladder / "qwen-lora-shortlist.train.json").write_text(json.dumps({"constrained": answers}))
    report = gold_report(
        products,
        taxonomy,
        queue,
        labels,
        shortlists,
        10.0,
        store.path,
        tmp_path / "none",
        training_path,
        ladder,
    )
    fit = report["ladder_fit_to_training_labels"]["qwen-lora-shortlist"]["constrained"]
    assert (fit["products"], fit["exact"]) == (4, 0.75)


def test_a_gold_product_in_the_training_labels_stops_the_report(gold_world, taxonomy):
    products, labels, queue, store, shortlists, tmp_path = gold_world
    asin = queue[0]
    store.label(asin, labels[asin], {})
    training_path = tmp_path / "training.jsonl"
    write_jsonl(training_path, [{"parent_asin": asin, "path": labels[asin]}])
    with pytest.raises(GoldOverlap):
        gold_report(
            products,
            taxonomy,
            queue,
            labels,
            shortlists,
            10.0,
            store.path,
            tmp_path / "none",
            training_path,
        )


def test_make_clean_keeps_hand_labels_and_paid_responses():
    makefile = (taxon.reproduce.DATA_DIR.parent / "Makefile").read_text(encoding="utf-8")
    clean = makefile.split("clean:")[1].split("\n\n")[0]
    assert "data/derived" in clean
    assert "data/gold" not in clean
    assert "data/labeling" not in clean
    assert "rm -rf data\n" not in clean

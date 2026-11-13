"""The evaluation ladder: each trained model against hand labels and the labels it learned from."""

import json

import pytest

from taxon.ladder import ladder_report, rung_report, saved_rungs

PATHS = ["A", "A > B", "A > C", "D", "D > E"]
GOLDS = {"p1": "A > B", "p2": "A > C", "p3": "D > E", "p4": "D"}
LABELS = {"p1": "A > B", "p2": "A", "p3": "D", "p4": "D"}


def rung(name, decoding, predictions):
    return {"name": name, "decoding": decoding, "predictions": predictions}


def test_a_rung_is_set_beside_the_pipeline_labels_on_the_same_products():
    golds = list(GOLDS.values())
    better = golds
    report = rung_report(golds, better, [LABELS[a] for a in GOLDS], PATHS)
    assert report["category"]["exact_leaf_accuracy"] == 1.0
    assert report["minus_pipeline_labels"]["difference"] == pytest.approx(0.5)
    assert report["misses"]["exact"] == 4
    assert "minus_previous_rung" not in report


def test_each_rung_is_compared_with_the_one_below_it_with_the_same_decoding():
    sft = {"p1": "A > B", "p2": "A", "p3": "D", "p4": "D"}
    grpo = dict(GOLDS)
    rungs = [
        rung("sft", "free", sft),
        rung("sft", "constrained", sft),
        rung("grpo", "free", grpo),
        rung("grpo", "constrained", sft),
    ]
    report = ladder_report(GOLDS, LABELS, rungs, PATHS)
    assert report["products"] == 4
    assert report["pipeline_labels"]["exact_leaf_accuracy"] == 0.5
    names = [(r["name"], r["decoding"]) for r in report["rungs"]]
    assert names == [
        ("sft", "free"),
        ("sft", "constrained"),
        ("grpo", "free"),
        ("grpo", "constrained"),
    ]
    assert "minus_previous_rung" not in report["rungs"][0]
    assert report["rungs"][2]["minus_previous_rung"]["difference"] == pytest.approx(0.5)
    assert report["rungs"][3]["minus_previous_rung"]["difference"] == 0.0
    json.dumps(report)


def test_only_products_every_rung_answered_are_scored():
    rungs = [rung("sft", "free", {"p1": "A > B", "p2": "A > C"})]
    assert ladder_report(GOLDS, LABELS, rungs, PATHS)["products"] == 2


def test_saved_answers_are_read_in_ladder_order_and_missing_runs_skipped(tmp_path):
    (tmp_path / "grpo.json").write_text(json.dumps({"free": {"p1": "A"}, "constrained": {}}))
    (tmp_path / "sft.json").write_text(json.dumps({"free": {"p1": "A > B"}}))
    rungs = saved_rungs(["sft", "missing", "grpo"], tmp_path)
    assert [(r["name"], r["decoding"]) for r in rungs] == [
        ("sft", "free"),
        ("grpo", "free"),
        ("grpo", "constrained"),
    ]


def test_a_ladder_with_no_shared_products_has_nothing_to_score():
    assert ladder_report(GOLDS, LABELS, [rung("sft", "free", {"x": "A"})], PATHS) is None


# Products the run trained on


def test_the_fit_report_scores_a_rung_against_the_labels_it_learned():
    from taxon.ladder import fit_report

    labels = {"p1": "A > B", "p2": "A > C", "p3": "D > E", "p4": "D"}
    saved = {"free": {"p1": "A > B", "p2": "A > B", "p3": "D", "p4": None}, "constrained": {}}
    report = fit_report(labels, saved)
    assert report["free"]["products"] == 4
    assert report["free"]["exact"] == 0.25
    assert report["free"]["top_level"] == 0.75
    low, high = report["free"]["exact_ci95"]
    assert low < 0.25 < high
    assert report["constrained"]["exact"] is None


def test_the_training_sample_is_seeded_and_drawn_from_the_training_products():
    from taxon.ladder import training_sample

    ids = [f"p{i}" for i in range(200)]
    sample = training_sample(ids, seed=7)
    assert len(sample) == 50 and set(sample) <= set(ids)
    assert sample == training_sample(list(reversed(ids)), seed=7)
    assert sample != training_sample(ids, seed=8)

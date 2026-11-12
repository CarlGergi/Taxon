import random

import pytest
from sklearn.metrics import precision_recall_fscore_support

from eval.metrics import evaluate


def level(result, n):
    return result["levels"][n - 1]


def test_perfect_predictions():
    golds = ["Apparel > Bags > Totes", "Home > Kitchen"]
    result = evaluate(golds, golds)
    assert result["exact_leaf_accuracy"] == 1.0
    for row in result["levels"]:
        assert row["micro_precision"] == row["micro_recall"] == 1.0
        assert row["macro_precision"] == row["macro_recall"] == 1.0


def test_right_vertical_wrong_leaf_gets_partial_credit_by_level():
    result = evaluate(["Home > Kitchen > Knives"], ["Home > Kitchen > Pans"])
    assert result["exact_leaf_accuracy"] == 0.0
    assert level(result, 1)["micro_recall"] == 1.0
    assert level(result, 2)["micro_recall"] == 1.0
    assert level(result, 3)["micro_recall"] == 0.0
    assert level(result, 3)["micro_precision"] == 0.0


def test_same_node_name_under_another_parent_is_not_a_match():
    result = evaluate(["Apparel > Bags"], ["Home > Bags"])
    assert level(result, 1)["true_positives"] == 0
    assert level(result, 2)["true_positives"] == 0


def test_a_shallow_prediction_keeps_precision_but_loses_recall():
    result = evaluate(["Home > Kitchen > Knives"], ["Home > Kitchen"])
    assert result["exact_leaf_accuracy"] == 0.0
    assert level(result, 2)["micro_precision"] == 1.0
    third = level(result, 3)
    assert third["pred_support"] == 0
    assert third["micro_precision"] is None
    assert third["micro_recall"] == 0.0


def test_an_overdeep_prediction_keeps_recall_but_loses_precision():
    result = evaluate(["Home > Kitchen"], ["Home > Kitchen > Knives"])
    assert result["exact_leaf_accuracy"] == 0.0
    assert level(result, 2)["micro_recall"] == 1.0
    third = level(result, 3)
    assert third["gold_support"] == 0
    assert third["micro_recall"] is None
    assert third["micro_precision"] == 0.0


def test_abstentions_count_against_recall_and_accuracy_only():
    result = evaluate(["Home", "Home > Kitchen"], [None, "Home > Kitchen"])
    assert result["abstained"] == 1
    assert result["exact_leaf_accuracy"] == 0.5
    assert level(result, 1)["micro_precision"] == 1.0
    assert level(result, 1)["micro_recall"] == 0.5


def test_empty_string_is_an_abstention():
    assert evaluate(["Home"], [""])["abstained"] == 1


def test_all_abstaining_has_undefined_precision_rather_than_a_crash():
    row = level(evaluate(["Home"], [None]), 1)
    assert row["micro_precision"] is None
    assert row["micro_recall"] == 0.0


def test_a_mixed_depth_set_has_the_expected_supports():
    result = evaluate(["A > B", "A", "C > D > E"], ["A", "A > B", "C > D > E"])
    assert [row["gold_support"] for row in result["levels"]] == [3, 2, 1]
    assert [row["pred_support"] for row in result["levels"]] == [3, 2, 1]
    assert [row["true_positives"] for row in result["levels"]] == [3, 1, 1]


def test_macro_averages_are_class_balanced_where_micro_is_not():
    golds = ["A"] * 9 + ["B"]
    preds = ["A"] * 10
    row = level(evaluate(golds, preds), 1)
    assert row["micro_recall"] == 0.9
    assert row["macro_recall"] == pytest.approx(0.5)
    assert row["macro_precision"] == pytest.approx(0.45)


def test_invalid_paths_are_reported_and_still_earn_their_valid_prefix():
    valid = {"Home", "Home > Kitchen"}
    result = evaluate(["Home > Kitchen"], ["Home > Attic"], valid_paths=valid)
    assert result["invalid_rate"] == 1.0
    assert level(result, 1)["true_positives"] == 1
    assert level(result, 2)["true_positives"] == 0


def test_abstentions_do_not_count_as_invalid():
    result = evaluate(["Home", "Home"], [None, "Home"], valid_paths={"Home"})
    assert result["invalid_rate"] == 0.0
    assert evaluate(["Home"], [None], valid_paths={"Home"})["invalid_rate"] is None


def test_invalid_rate_is_absent_without_a_taxonomy():
    assert "invalid_rate" not in evaluate(["Home"], ["Home"])


@pytest.mark.parametrize(
    ("golds", "preds", "message"),
    [
        (["A"], ["A", "B"], "1 golds and 2 predictions"),
        ([], [], "empty"),
        ([""], ["A"], "non-empty"),
    ],
)
def test_rejects_bad_input(golds, preds, message):
    with pytest.raises(ValueError, match=message):
        evaluate(golds, preds)


def test_agrees_with_scikit_learn_at_every_level():
    rng = random.Random(7)
    tree = [f"{a} > {b} > {c}" for a in "PQR" for b in "xyz" for c in "123"]

    def shallow(path):
        return " > ".join(path.split(" > ")[: rng.randint(1, 3)])

    golds = [shallow(rng.choice(tree)) for _ in range(400)]
    preds = [shallow(rng.choice(tree)) if rng.random() < 0.5 else g for g in golds]
    result = evaluate(golds, preds)

    missing = "<none>"
    for depth in (1, 2, 3):
        gold_at = [
            " > ".join(g.split(" > ")[:depth]) if g.count(" > ") >= depth - 1 else missing
            for g in golds
        ]
        pred_at = [
            " > ".join(p.split(" > ")[:depth]) if p.count(" > ") >= depth - 1 else missing
            for p in preds
        ]
        labels = sorted(({*gold_at, *pred_at}) - {missing})
        row = level(result, depth)
        for average, p_key, r_key in [
            ("micro", "micro_precision", "micro_recall"),
            ("macro", "macro_precision", "macro_recall"),
        ]:
            p, r, _, _ = precision_recall_fscore_support(
                gold_at, pred_at, labels=labels, average=average, zero_division=0
            )
            assert row[p_key] == pytest.approx(p)
            assert row[r_key] == pytest.approx(r)


# Intervals


def test_wilson_matches_scipy():
    from scipy.stats import binomtest

    from eval.metrics import wilson

    for k, n in ((35, 50), (1, 50), (49, 50), (7, 9)):
        expected = binomtest(k, n).proportion_ci(method="wilson")
        assert wilson(k, n) == pytest.approx([expected.low, expected.high])
    assert wilson(0, 50)[0] == 0.0 and wilson(50, 50)[1] == 1.0
    assert wilson(0, 0) is None
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_every_share_in_an_evaluation_has_an_interval():
    golds = ["A > B", "A > C", "D"] * 10
    preds = ["A > B", "A", None] * 10
    report = evaluate(golds, preds, valid_paths=set(golds) | {"A"})
    low, high = report["exact_leaf_accuracy_ci95"]
    assert low < report["exact_leaf_accuracy"] < high
    for level in report["levels"]:
        assert "micro_precision_ci95" in level and "micro_recall_ci95" in level
        assert "macro_precision_ci95" not in level
    assert report["invalid_rate_ci95"] is not None


def test_macro_intervals_come_from_a_seeded_bootstrap():
    golds = ["A > B", "A > C", "D", "E"] * 5
    preds = ["A > B", "A > B", "D", "D"] * 5
    one = evaluate(golds, preds, bootstrap=200, seed=4)
    assert one == evaluate(golds, preds, bootstrap=200, seed=4)
    first = one["levels"][0]
    low, high = first["macro_recall_ci95"]
    assert low <= first["macro_recall"] <= high


def test_the_paired_difference_keeps_each_items_answers_together():
    from eval.metrics import paired_difference

    golds = [f"P{i}" for i in range(50)]
    better = list(golds)
    worse = [g if i % 5 else "wrong" for i, g in enumerate(golds)]
    report = paired_difference(golds, better, worse, seed=2)
    assert report["n"] == 50
    assert report["difference"] == pytest.approx(0.2)
    low, high = report["ci95"]
    assert 0 < low <= 0.2 <= high
    assert paired_difference(golds, better, better)["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        paired_difference(golds, better, worse[:3])


def test_misses_are_sorted_by_how_they_relate_to_the_gold_path():
    from eval.metrics import miss_kinds

    golds = ["A > B", "A > B", "A > B > C", "A > B", "A > B", "A > B"]
    preds = ["A > B", "A > B > C", "A", "A > D", "E > F", None]
    assert miss_kinds(golds, preds) == {
        "exact": 1,
        "too_deep": 1,
        "too_shallow": 1,
        "other_branch": 2,
        "other_top_level": 1,
        "abstained": 1,
    }

import json

import pytest

from eval.metrics import evaluate
from taxon.baseline import (
    TfidfLogisticRegression,
    body_text,
    most_frequent_label,
    run_baseline,
    title_text,
)
from taxon.products import assign_split
from taxon.reproduce import write_result

SEED = 7
WORDS = {
    "Apparel > Bags > Totes": ["canvas tote", "shopping tote", "market tote"],
    "Home > Kitchen > Knives": ["chef knife", "paring knife", "bread knife"],
    "Home > Bags": ["laundry sack", "storage sack", "linen sack"],
}


def product(asin, title, **extra):
    return {
        "parent_asin": asin,
        "title": title,
        "features": [],
        "description": "",
        "details": {},
        "domain": "d",
        "categories": [],
    } | extra


def toy_set(n=240):
    products, labels = [], {}
    paths = list(WORDS)
    for i in range(n):
        path = paths[i % len(paths)]
        title = WORDS[path][i % 3] + f" item{i}"
        products.append(product(f"A{i}", title))
        labels[f"A{i}"] = path
    return products, labels


def test_body_text_uses_features_description_and_details():
    p = product("A", "T", features=["soft"], description="warm", details={"Color": "Red"})
    assert title_text(p) == "T"
    assert body_text(p) == "soft warm Color Red"


def test_learns_a_separable_toy_problem():
    products, labels = toy_set()
    model = TfidfLogisticRegression(c=10.0, n_jobs=1).fit(
        products, [labels[p["parent_asin"]] for p in products]
    )
    unseen = [product("X1", "bread knife deluxe"), product("X2", "linen sack large")]
    assert model.predict(unseen) == ["Home > Kitchen > Knives", "Home > Bags"]


def test_body_text_contributes_signal():
    products = [
        product("A", "item", description="knife blade steel"),
        product("B", "thing", description="tote canvas handle"),
    ] * 3
    labels = ["Home > Kitchen > Knives", "Apparel > Bags > Totes"] * 3
    model = TfidfLogisticRegression(c=100.0, n_jobs=1).fit(products, labels)
    assert model.predict([product("C", "other", description="steel blade")]) == [labels[0]]


def test_most_frequent_label_breaks_ties_alphabetically():
    assert most_frequent_label(["b", "a", "b", "a", "c"]) == "a"
    assert most_frequent_label(["z", "z", "a"]) == "z"


def test_run_baseline_scores_and_beats_the_majority_class(taxonomy):
    products, labels = toy_set()
    result = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    assert result["label_source"] == "toy"
    assert result["taxonomy_version"] == "toy-1"
    assert sum(result["splits"].values()) == len(products)
    assert result["tfidf_logistic_regression"]["exact_leaf_accuracy"] > 0.9
    assert (
        result["tfidf_logistic_regression"]["exact_leaf_accuracy"]
        > result["most_frequent_label"]["exact_leaf_accuracy"]
    )
    assert result["tfidf_logistic_regression"]["invalid_rate"] == 0.0
    assert result["test_labels_unseen_in_train"] == 0.0


def test_the_test_split_is_held_out_from_training(taxonomy, monkeypatch):
    products, labels = toy_set()
    trained_on = []
    original = TfidfLogisticRegression.fit

    def spy(self, rows, y):
        trained_on.extend(p["parent_asin"] for p in rows)
        return original(self, rows, y)

    monkeypatch.setattr(TfidfLogisticRegression, "fit", spy)
    run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    assert trained_on
    assert {assign_split(asin, SEED) for asin in trained_on} == {"train"}


def test_unlabeled_products_are_left_out(taxonomy):
    products, labels = toy_set()
    products.append(product("NOLABEL", "mystery"))
    labels["NOLABEL"] = None
    result = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    assert sum(result["splits"].values()) == len(products) - 1


def test_run_baseline_is_deterministic(taxonomy):
    products, labels = toy_set()
    first = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    second = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    assert first == second


def test_learning_curve_grows_and_ends_at_the_full_train_split(taxonomy):
    products, labels = toy_set()
    result = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    sizes = [row["train_examples"] for row in result["learning_curve"]]
    assert sizes == sorted(sizes)
    assert sizes[-1] == result["splits"]["train"]


def test_a_test_label_missing_from_train_is_reported_and_scored_wrong(taxonomy):
    products, labels = toy_set()
    test_asin = next(a for a in labels if assign_split(a, SEED) == "test")
    labels[test_asin] = "Apparel > Bag Straps"
    result = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1)
    assert result["test_labels_unseen_in_train"] > 0.0
    assert result["tfidf_logistic_regression"]["exact_leaf_accuracy"] < 1.0


@pytest.mark.parametrize("c", [1.0, 100.0])
def test_predictions_are_always_labels_it_was_trained_on(c):
    products, labels = toy_set(60)
    y = [labels[p["parent_asin"]] for p in products]
    model = TfidfLogisticRegression(c=c, n_jobs=1).fit(products, y)
    predictions = model.predict([product("Q", "completely unrelated words")])
    assert evaluate(predictions, predictions)["n"] == 1
    assert set(predictions) <= set(y)


def test_write_result_is_stable_json(tmp_path):
    write_result("x", {"b": 1, "a": "é"}, results_dir=tmp_path)
    text = (tmp_path / "x.json").read_text(encoding="utf-8")
    assert json.loads(text) == {"a": "é", "b": 1}
    assert text.index('"a"') < text.index('"b"')


def test_products_with_no_body_text_at_all_still_train():
    products, labels = toy_set(30)
    y = [labels[p["parent_asin"]] for p in products]
    assert not any(body_text(p).strip() for p in products)
    assert TfidfLogisticRegression(n_jobs=1).fit(products, y).predict(products[:1])


def test_top_k_orders_labels_and_leads_with_the_prediction():
    products, labels = toy_set(90)
    y = [labels[p["parent_asin"]] for p in products]
    model = TfidfLogisticRegression(c=10.0, n_jobs=1).fit(products, y)
    query = [product("Q", "bread knife deluxe")]
    assert model.top_k(query, 1) == [model.predict(query)]
    assert len(model.top_k(query, 2)[0]) == 2
    assert len(model.top_k(query, 99)[0]) == 3


def test_top_k_works_with_two_classes():
    products = [product("A", "knife"), product("B", "tote")] * 3
    y = ["Home", "Apparel"] * 3
    model = TfidfLogisticRegression(c=10.0, n_jobs=1).fit(products, y)
    assert model.top_k([product("Q", "knife")], 2) == [["Home", "Apparel"]]


def test_held_out_products_reach_no_split(taxonomy, monkeypatch):
    products, labels = toy_set()
    held = {"A0", "A1", "A2"}
    trained_on = []
    original = TfidfLogisticRegression.fit

    def spy(self, rows, y):
        trained_on.extend(p["parent_asin"] for p in rows)
        return original(self, rows, y)

    monkeypatch.setattr(TfidfLogisticRegression, "fit", spy)
    result = run_baseline(products, labels, taxonomy, SEED, "toy", n_jobs=1, held_out=held)
    assert not (set(trained_on) & held)
    assert sum(result["splits"].values()) == len(products) - 3

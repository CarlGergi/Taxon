import pytest

from taxon.labeling.retrieval import (
    CandidateRetriever,
    build_shortlists,
    classifier_rankings,
    fuse,
    path_document,
    product_query,
    recall_at,
    with_ancestors,
)


def product(title, features=(), description=""):
    return {"title": title, "features": list(features), "description": description}


@pytest.fixture
def retriever(taxonomy):
    return CandidateRetriever(taxonomy.paths())


def test_a_path_document_weights_its_own_name_above_its_ancestors():
    document = path_document("Home > Kitchen > Knives")
    assert document.count("Knives") > document.count("Kitchen")


def test_a_product_query_is_led_by_its_title_and_limits_long_text():
    query = product_query(product("Tote", ["f"] * 20, "d" * 5000))
    assert query.startswith("Tote")
    assert query.count("f .") <= 5
    assert len(query) < 1000


def test_the_right_path_ranks_first_for_a_clear_product(retriever):
    assert retriever.top_k([product("Chef knives set")], k=1) == [["Home > Kitchen > Knives"]]


def test_plurals_match(retriever):
    assert retriever.top_k([product("A canvas tote")], k=1) == [["Apparel > Bags > Totes"]]


def test_k_limits_results_and_paths_are_unique_and_valid(retriever, taxonomy):
    results = retriever.top_k([product("Bag"), product("Knife")], k=3)
    assert [len(r) for r in results] == [3, 3]
    for row in results:
        assert len(set(row)) == 3
        assert set(row) <= set(taxonomy.paths())


def test_k_larger_than_the_taxonomy_returns_every_path(retriever, taxonomy):
    assert sorted(retriever.top_k([product("Bag")], k=999)[0]) == taxonomy.paths()


def test_order_is_deterministic_including_ties(taxonomy):
    products = [product("nothing matches this zzz")]
    first = CandidateRetriever(taxonomy.paths()).top_k(products)
    second = CandidateRetriever(taxonomy.paths()).top_k(products)
    assert first == second
    assert first[0] == taxonomy.paths()


def test_batching_does_not_change_results(retriever):
    products = [product(t) for t in ["Bag", "Knife", "Kitchen", "Tote", "Straps"]]
    assert retriever.top_k(products, batch_size=2) == retriever.top_k(products, batch_size=5)


def test_recall_counts_exact_and_compatible_matches():
    candidates = [["A > B", "C"], ["A > B > X", "D"], ["Q", "R"]]
    report = recall_at(candidates, ["A > B", "A > B", "Z"], [1, 2])
    assert report["n"] == 3
    assert report["exact_at_1"] == pytest.approx(1 / 3)
    assert report["compatible_at_1"] == pytest.approx(2 / 3)
    assert report["exact_at_2"] == pytest.approx(1 / 3)
    assert report["compatible_at_2"] == pytest.approx(2 / 3)


def test_a_shared_string_prefix_is_not_compatible():
    report = recall_at([["A > Bags"]], ["A > Bag"], [1])
    assert report["compatible_at_1"] == 0.0


def test_recall_rejects_an_empty_set():
    with pytest.raises(ValueError, match="empty"):
        recall_at([], [], [1])


def test_fusion_prefers_a_path_high_in_both_rankings():
    assert fuse(["a", "b", "c"], ["b", "x"])[0] == "b"


def test_fusion_keeps_every_path_once_and_breaks_ties_by_name():
    assert sorted(fuse(["a", "b"], ["b", "c"])) == ["a", "b", "c"]
    assert fuse(["b"], ["a"]) == ["a", "b"]


def test_fusion_with_an_empty_ranking_keeps_the_other_order():
    assert fuse(["z", "a"], []) == ["z", "a"]


def test_ancestors_follow_the_core_but_not_the_vertical():
    shortlist = with_ancestors(["A > B > C", "X > Y"], k=10, core=2)
    assert shortlist == ["A > B > C", "X > Y", "A > B"]
    assert "A" not in shortlist


def test_ancestors_fill_before_lower_ranked_paths():
    ranked = ["A > B > C", "D", "E"]
    assert with_ancestors(ranked, k=3, core=1) == ["A > B > C", "A > B", "D"]


def test_the_shortlist_never_exceeds_k_or_repeats():
    ranked = [f"A > B > {i}" for i in range(100)]
    shortlist = with_ancestors(ranked, k=50, core=30)
    assert len(shortlist) == 50
    assert len(set(shortlist)) == 50


def rows_with_labels(n=120):
    words = {"Home > Kitchen > Knives": "chef knife", "Apparel > Bags > Totes": "canvas tote"}
    paths = list(words)
    products, labels = [], {}
    for i in range(n):
        path = paths[i % 2]
        asin = f"A{i}"
        products.append(
            {
                "parent_asin": asin,
                "title": f"{words[path]} {i}",
                "features": [],
                "description": "",
                "details": {},
            }
        )
        labels[asin] = path
    return products, labels


def test_no_product_is_ranked_by_a_model_trained_on_it(monkeypatch):
    from taxon import baseline

    products, labels = rows_with_labels()
    fits = []
    original_fit = baseline.TfidfLogisticRegression.fit
    original_top = baseline.TfidfLogisticRegression.top_k

    def spy_fit(self, rows, y):
        self.trained_on = {p["parent_asin"] for p in rows}
        fits.append(self.trained_on)
        return original_fit(self, rows, y)

    def spy_top(self, rows, k):
        assert not (self.trained_on & {p["parent_asin"] for p in rows})
        return original_top(self, rows, k)

    monkeypatch.setattr(baseline.TfidfLogisticRegression, "fit", spy_fit)
    monkeypatch.setattr(baseline.TfidfLogisticRegression, "top_k", spy_top)
    rankings = classifier_rankings(products, labels, set(), seed=1, c=10.0, n_jobs=1)
    assert len(fits) >= 5
    assert all(rankings[p["parent_asin"]] for p in products)


def test_held_out_products_never_train_and_still_get_rankings(monkeypatch):
    from taxon import baseline

    products, labels = rows_with_labels()
    held = {"A0", "A1", "A2", "A3"}
    original = baseline.TfidfLogisticRegression.fit

    def spy(self, rows, y):
        assert not ({p["parent_asin"] for p in rows} & held)
        return original(self, rows, y)

    monkeypatch.setattr(baseline.TfidfLogisticRegression, "fit", spy)
    rankings = classifier_rankings(products, labels, held, seed=1, c=10.0, n_jobs=1)
    assert all(rankings[asin] for asin in held)


def test_unlabeled_products_are_ranked_by_the_full_model():
    products, labels = rows_with_labels()
    labels["A0"] = None
    assert classifier_rankings(products, labels, set(), seed=1, c=10.0, n_jobs=1)["A0"]


def test_nothing_labeled_means_no_classifier_rankings():
    products, labels = rows_with_labels(4)
    empty = dict.fromkeys(labels)
    rankings = classifier_rankings(products, empty, set(), seed=1, c=10.0, n_jobs=1)
    assert all(row == [] for row in rankings.values())


def test_shortlists_are_valid_paths_within_the_limit(taxonomy):
    products, labels = rows_with_labels(40)
    shortlists = build_shortlists(products, taxonomy, labels, set(), seed=1, c=10.0, n_jobs=1)
    for row in shortlists.values():
        assert row
        assert len(row) == len(set(row))
        assert set(row) <= set(taxonomy.paths())


def test_shortlists_work_with_no_labels_at_all(taxonomy):
    products, labels = rows_with_labels(4)
    shortlists = build_shortlists(
        products, taxonomy, dict.fromkeys(labels), set(), seed=1, c=10.0, n_jobs=1
    )
    assert all(shortlists.values())

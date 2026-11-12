"""Checks on the real artifacts. These need `make reproduce` to have fetched the inputs."""

import json
from collections import Counter

import pytest
from tokenizers import Tokenizer

from taxon.baseline import C_GRID
from taxon.bench import search
from taxon.fetch import fetch
from taxon.gold.pool import FIRST_TARGET, GOLD_POOL_SIZE, assert_disjoint, training_products
from taxon.pins import (
    AMAZON_FILE_SIZES,
    DERIVED_DIR,
    RESULTS_DIR,
    SEED,
    TAXONOMY,
    TOKENIZER,
)
from taxon.pins import ROOT as REPO_ROOT
from taxon.products import assign_split, normalized_title, quotas, read_jsonl
from taxon.reproduce import N_PRODUCTS, build_trie
from taxon.silver import tokens
from taxon.taxonomy import Taxonomy, split_path
from taxon.trie import ROOT

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def real_taxonomy():
    return Taxonomy.load(fetch(TAXONOMY))


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer.from_file(str(fetch(TOKENIZER)))


@pytest.fixture(scope="module")
def real_trie(real_taxonomy, tokenizer):
    return build_trie(real_taxonomy, tokenizer)


@pytest.fixture(scope="module")
def products():
    return read_jsonl(DERIVED_DIR / "products.jsonl")


@pytest.fixture(scope="module")
def silver():
    return read_jsonl(DERIVED_DIR / "silver_labels.jsonl")


def test_the_release_is_the_pinned_one(real_taxonomy):
    assert real_taxonomy.version == "2026-10"
    assert len(real_taxonomy) == 64_292
    assert len(real_taxonomy.leaves()) == 50_427
    assert len(real_taxonomy.roots()) == 26


def test_trie_accepts_exactly_the_taxonomy(real_taxonomy, real_trie):
    assert real_trie.n_paths == len(real_taxonomy)
    assert sorted(path for _, path in real_trie.sequences()) == real_taxonomy.paths()


def test_trie_rejects_paths_that_are_not_in_the_taxonomy(real_taxonomy, real_trie, tokenizer):
    roots = real_taxonomy.roots()
    for path in real_taxonomy.leaves()[::997]:
        other_root = roots[0] if not path.startswith(roots[0]) else roots[1]
        tokens = tokenizer.encode(f"{other_root} > {path}", add_special_tokens=False).ids
        assert real_trie.path_for(tokens) is None


def test_generation_starts_clean_after_a_newline(real_taxonomy, tokenizer):
    # The prompt must end in a newline or special token for the trie's first tokens to apply.
    newline = tokenizer.encode("\n", add_special_tokens=False).ids
    for path in real_taxonomy.paths()[::211]:
        whole = tokenizer.encode("\n" + path, add_special_tokens=False).ids
        assert whole == newline + tokenizer.encode(path, add_special_tokens=False).ids


def test_every_vertical_is_reachable_from_the_root(real_taxonomy, real_trie, tokenizer):
    first = {tokenizer.encode(r, add_special_tokens=False).ids[0] for r in real_taxonomy.roots()}
    assert set(real_trie.allowed(ROOT)) == first


def test_the_sample_is_exactly_the_requested_size_and_balanced(products):
    assert len(products) == N_PRODUCTS
    expected = quotas(N_PRODUCTS, AMAZON_FILE_SIZES)
    assert Counter(p["domain"] for p in products) == expected


def test_the_sample_has_no_duplicate_ids_or_titles(products):
    assert len({p["parent_asin"] for p in products}) == len(products)
    assert len({normalized_title(p["title"]) for p in products}) == len(products)


def test_every_sampled_product_has_a_title_and_image(products):
    assert all(p["title"] and p["image_url"].startswith("https://") for p in products)


def test_no_rank_leak_survives_into_the_sample(products):
    assert not any("Best Sellers Rank" in json.dumps(p["details"]) for p in products)


def test_every_silver_label_is_a_valid_path(silver, real_taxonomy):
    assert all(row["path"] is None or row["path"] in real_taxonomy for row in silver)


def test_every_word_of_a_silver_label_comes_from_the_breadcrumb(silver, products):
    for row, product in zip(silver, products, strict=True):
        if row["path"] is not None:
            name = split_path(row["path"])[-1]
            assert tokens(name) <= tokens(" ".join(product["categories"])), row["path"]


def test_silver_labels_cover_exactly_the_sample(silver, products):
    assert [row["parent_asin"] for row in silver] == [p["parent_asin"] for p in products]


@pytest.fixture(scope="module")
def queue():
    return [row["parent_asin"] for row in read_jsonl(DERIVED_DIR / "gold_queue.jsonl")]


@pytest.fixture(scope="module")
def shortlists():
    return {
        row["parent_asin"]: row["shortlist"] for row in read_jsonl(DERIVED_DIR / "shortlists.jsonl")
    }


def test_the_gold_pool_is_reserved_and_balanced(queue, products):
    assert len(queue) == GOLD_POOL_SIZE == len(set(queue))
    domains = {p["parent_asin"]: p["domain"] for p in products}
    counts = Counter(domains[a] for a in queue[:FIRST_TARGET])
    assert len(counts) == len(AMAZON_FILE_SIZES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_the_gold_set_never_overlaps_training_data(queue, products):
    training = training_products(products, queue)
    assert len(training) == len(products) - len(queue)
    assert_disjoint((p["parent_asin"] for p in training), queue)


def test_the_baseline_held_out_the_gold_pool(queue, silver):
    result = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))
    pool = set(queue)
    labeled_outside = sum(
        1 for row in silver if row["path"] is not None and row["parent_asin"] not in pool
    )
    assert sum(result["splits"].values()) == labeled_outside


def test_shortlists_beat_similarity_alone():
    retrieval = json.loads((RESULTS_DIR / "retrieval.json").read_text(encoding="utf-8"))
    assert retrieval["label_source"] == "silver"
    assert retrieval["shortlist"]["exact_at_50"] > retrieval["similarity_only"]["exact_at_50"]


def test_every_shortlist_is_valid_and_within_the_limit(shortlists, products, real_taxonomy):
    assert len(shortlists) == len(products)
    for row in shortlists.values():
        assert 30 <= len(row) <= 50
        assert len(set(row)) == len(row)
        assert all(path in real_taxonomy for path in row)


def test_gold_suggestions_cover_the_pool_with_valid_paths(queue, real_taxonomy):
    rows = read_jsonl(DERIVED_DIR / "gold_suggestions.jsonl")
    assert {row["parent_asin"] for row in rows} == set(queue)
    for row in rows:
        assert len(row["suggestions"]) == 5
        assert all(path in real_taxonomy for path in row["suggestions"])


def test_baseline_result_is_labeled_and_consistent(silver, queue):
    result = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))
    assert result["label_source"] == "silver"
    pool = set(queue)
    expected = Counter(
        assign_split(row["parent_asin"], SEED)
        for row in silver
        if row["path"] is not None and row["parent_asin"] not in pool
    )
    assert result["splits"] == dict(expected)
    assert result["tfidf_logistic_regression"]["invalid_rate"] == 0.0
    assert (
        result["tfidf_logistic_regression"]["exact_leaf_accuracy"]
        > result["most_frequent_label"]["exact_leaf_accuracy"]
    )


def test_the_regularization_search_found_an_interior_optimum():
    result = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))
    assert result["best_c"] != C_GRID[-1], "best C is at the edge of the grid; extend it"


def test_the_raw_catalog_covers_the_sample_and_carries_no_labels(products):
    rows = read_jsonl(DERIVED_DIR / "catalogs" / "raw.jsonl")
    assert sorted(row["id"] for row in rows) == sorted(p["parent_asin"] for p in products)
    allowed = {"id", "title", "store", "features", "description", "details", "price"}
    assert all(set(row) == allowed for row in rows)
    assert not any("Best Sellers Rank" in row["details"] for row in rows)


def test_the_raw_index_holds_the_catalog_and_no_enrichment():
    report = json.loads((RESULTS_DIR / "search.json").read_text(encoding="utf-8"))
    assert report["index"]["documents"] == N_PRODUCTS
    assert not report["index"]["has_category"]
    assert not report["index"]["has_attributes"]


def test_searching_an_exact_title_finds_that_product():
    report = json.loads((RESULTS_DIR / "search.json").read_text(encoding="utf-8"))
    assert report["self_retrieval"]["sample"] == search.SELF_RETRIEVAL_SAMPLE
    assert report["self_retrieval"]["found_in_top_10"] >= 0.98


def test_the_real_index_applies_a_price_filter_and_reports_the_others():
    index = search.index_dir(search.RAW, DERIVED_DIR)
    requests = [
        {"query": "chef knife", "min_price": 10, "max_price": 50, "limit": 50},
        {"query": "chef knife", "category": "Home & Garden", "attributes": {"color": "Red"}},
    ]
    priced, unavailable = search.search_many(index, requests, search.build_binary())
    assert priced["results"]
    assert all(10 <= hit["price"] <= 50 for hit in priced["results"])
    assert unavailable["ignored_filters"] == ["category", "attributes"]
    assert unavailable["results"]


def test_the_readme_reports_the_current_results():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    result = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))
    silver_result = json.loads((RESULTS_DIR / "silver.json").read_text(encoding="utf-8"))
    model = result["tfidf_logistic_regression"]

    def percent(value):
        return f"{value * 100:.1f}%"

    expected = [
        f"| TF-IDF + logistic regression | {percent(model['exact_leaf_accuracy'])} |",
        f"| Always predict the most common label | "
        f"{percent(result['most_frequent_label']['exact_leaf_accuracy'])} |",
        f"Test set of {model['n']:,} products",
        f"{silver_result['labeled']:,} of 10,000 products ({percent(silver_result['coverage'])})",
    ]
    expected += [
        f"| {row['level']} | {row['gold_support']:,} | {percent(row['micro_precision'])} | "
        f"{percent(row['micro_recall'])} | {percent(row['macro_precision'])} | "
        f"{percent(row['macro_recall'])} |"
        for row in model["levels"]
    ]
    expected += [
        f"| {row['train_examples']:,} | {percent(row['val_exact_leaf_accuracy'])} |"
        for row in result["learning_curve"]
    ]
    retrieval = json.loads((RESULTS_DIR / "retrieval.json").read_text(encoding="utf-8"))
    for name, label in [
        ("similarity_only", "TF-IDF similarity alone"),
        ("shortlist", "Similarity, classifier and ancestors"),
    ]:
        row = retrieval[name]
        expected.append(
            f"| {label} | {percent(row['exact_at_30'])} | {percent(row['exact_at_50'])} | "
            f"{percent(row['compatible_at_50'])} |"
        )
    trie = json.loads((RESULTS_DIR / "trie.json").read_text(encoding="utf-8"))
    expected.append(f"| {trie['states']:,} | {trie['mean_tokens']:.1f} | {trie['max_tokens']} |")
    found = json.loads((RESULTS_DIR / "search.json").read_text(encoding="utf-8"))["self_retrieval"]
    expected.append(
        f"| {found['sample']} | {percent(found['found_first'])} | "
        f"{percent(found['found_in_top_10'])} |"
    )
    missing = [line for line in expected if line not in readme]
    assert not missing, f"README is out of date: {missing}"


def test_the_readme_reports_the_current_gold_results():
    gold_path = RESULTS_DIR / "gold_eval.json"
    if not gold_path.exists():
        pytest.skip("no hand labels yet")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    labeling = json.loads((RESULTS_DIR / "labeling.json").read_text(encoding="utf-8"))

    def percent(value):
        return f"{value * 100:.1f}%"

    def interval(bounds):
        return f"[{bounds[0] * 100:.1f}, {bounds[1] * 100:.1f}]"

    def with_ci(value, bounds):
        return f"{percent(value)} {interval(bounds)}"

    labels = gold["pipeline_labels"]
    top = labels["levels"][0]
    misses = gold["pipeline_label_misses"]
    attributes = gold["attribute_pass"]
    paired = attributes["shortlist_minus_exact"]
    shortlist = attributes["versions"]["shortlist"]
    expected = [
        f"| Exact category | "
        f"{with_ci(labels['exact_leaf_accuracy'], labels['exact_leaf_accuracy_ci95'])} |",
        f"| Right at the top level | {with_ci(top['micro_recall'], top['micro_recall_ci95'])} |",
        f"{misses['too_deep'] + misses['too_shallow']} are on the right path at the wrong depth "
        f"({misses['too_deep']} too deep, {misses['too_shallow']} too shallow)",
        f"| Products sent to the arbitrator (of 2,000) | "
        f"{with_ci(labeling['sent_to_arbitrator'], labeling['sent_to_arbitrator_ci95'])} |",
        f"| Shortlist of allowed values | "
        f"{with_ci(shortlist['accuracy'], shortlist['accuracy_ci95'])} |",
        f"{paired['difference'] * 100:.1f} points {interval(paired['ci95'])}",
    ]
    expected += [
        f"| {row['level']} | {row['gold_support']} | "
        f"{with_ci(row['micro_precision'], row['micro_precision_ci95'])} | "
        f"{with_ci(row['micro_recall'], row['micro_recall_ci95'])} |"
        for row in labels["levels"]
        if row["gold_support"] >= 10
    ]
    missing = [line for line in expected if line not in readme]
    assert not missing, f"README is out of date: {missing}"


def test_the_readme_reports_the_current_ladder():
    gold_path = RESULTS_DIR / "gold_eval.json"
    if not gold_path.exists() or "ladder" not in json.loads(gold_path.read_text()):
        pytest.skip("no trained model has been asked about the gold products yet")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    ladder = json.loads(gold_path.read_text(encoding="utf-8"))["ladder"]
    names = {"qwen-lora": "Fine-tuned", "qwen-lora-grpo": "Fine-tuned, then GRPO"}

    def share(value, bounds):
        return f"{value * 100:.1f}% [{bounds[0] * 100:.1f}, {bounds[1] * 100:.1f}]"

    def points(entry):
        low, high = entry["ci95"]
        return f"{entry['difference'] * 100:.1f} [{low * 100:.1f}, {high * 100:.1f}]"

    expected = []
    for rung in ladder["rungs"]:
        category = rung["category"]
        top = category["levels"][0]
        expected.append(
            f"| {names[rung['name']]} | {rung['decoding']} | "
            f"{share(category['exact_leaf_accuracy'], category['exact_leaf_accuracy_ci95'])} | "
            f"{share(top['micro_recall'], top['micro_recall_ci95'])} | "
            f"{share(category['invalid_rate'], category['invalid_rate_ci95'])} | "
            f"{points(rung['minus_pipeline_labels'])} |"
        )
    missing = [line for line in expected if line not in readme]
    assert not missing, f"README is out of date: {missing}"

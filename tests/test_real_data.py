"""Checks on the real artifacts. These need `make reproduce` to have fetched the inputs."""

import json
from collections import Counter

import pytest
from tokenizers import Tokenizer

from taxon.baseline import C_GRID
from taxon.fetch import fetch
from taxon.pins import (
    AMAZON_FILE_SIZES,
    DERIVED_DIR,
    RESULTS_DIR,
    SEED,
    TAXONOMY,
    TOKENIZERS,
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


@pytest.fixture(scope="module", params=sorted(TOKENIZERS))
def tokenizer(request):
    return Tokenizer.from_file(str(fetch(TOKENIZERS[request.param])))


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


def test_baseline_result_is_labeled_and_consistent(silver):
    result = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))
    assert result["label_source"] == "silver"
    expected = Counter(
        assign_split(row["parent_asin"], SEED) for row in silver if row["path"] is not None
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
    for name in TOKENIZERS:
        trie = json.loads((RESULTS_DIR / f"trie_{name}.json").read_text(encoding="utf-8"))
        expected.append(
            f"| {trie['states']:,} | {trie['mean_tokens']:.1f} | {trie['max_tokens']} |"
        )
    missing = [line for line in expected if line not in readme]
    assert not missing, f"README is out of date: {missing}"

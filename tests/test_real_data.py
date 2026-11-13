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


def _documents() -> str:
    """The README and the detailed write-up it links to, where every reported number must match."""
    return "\n".join(
        (REPO_ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "docs/DETAILS.md")
    )


def test_the_readme_reports_the_current_results():
    readme = _documents()
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


def test_the_readme_reports_the_judge_s_validation():
    path = RESULTS_DIR / "catalogbench.json"
    if not path.exists():
        pytest.skip("the benchmark is not scored yet")
    judge = json.loads(path.read_text(encoding="utf-8"))["judge"]
    readme = " ".join(_documents().split())

    def with_ci(value, bounds, digits=1, scale=100):
        return f"{value * scale:.{digits}f}{'%' if scale == 100 else ''} " + (
            f"[{bounds[0] * scale:.{digits}f}, {bounds[1] * scale:.{digits}f}]"
        )

    first, final, verdicts = judge["first_sample"], judge["agreement"], judge["verdicts"]
    first_agreed = (
        first["confusion"]["judge_meets_hand_meets"] + first["confusion"]["judge_fails_hand_fails"]
    )
    final_agreed = (
        final["confusion"]["judge_meets_hand_meets"] + final["confusion"]["judge_fails_hand_fails"]
    )
    expected = [
        f"the judge was unsure of {first['judge_unsure']} of the first "
        f"{first['hand_judged']} pairs",
        f"so the sample grew to {judge['sample']}",
        f"on {first_agreed} of the {first['answered']} it answered in the first "
        f"{first['hand_judged']} ({first['agreement'] * 100:.1f}%)",
        f"on {final_agreed} of the {final['answered']} in the full sample: "
        f"**{with_ci(final['agreement'], final['agreement_ci95'])}**",
        f"Cohen's kappa **{with_ci(final['kappa'], final['kappa_ci95'], 2, 1)}**",
        f"Across the whole pool of {judge['pool']} pairs, the judge said {verdicts['meets']} meet "
        f"their task, {verdicts['fails']} fail and {verdicts['unsure']} are unsure.",
    ]
    names = {"raw": "the raw catalog's", "sonnet": "Sonnet's", "taxon": "Taxon's"}
    for condition, row in final["per_condition"].items():
        expected.append(
            f"{with_ci(row['agreement'], row['agreement_ci95'])} of {names[condition]} "
            f"{row['answered']}"
        )
    missing = [phrase for phrase in expected if phrase not in readme]
    assert not missing, f"README is out of date: {missing}"


def test_the_readme_reports_the_benchmark():
    path = RESULTS_DIR / "catalogbench.json"
    if not path.exists():
        pytest.skip("the benchmark is not scored yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    readme = _documents()
    catalogs = ("raw", "sonnet", "taxon")
    over = {c: report["catalogs"][c]["over_tasks"] for c in catalogs}
    paired = report["paired"]

    def share(v, b):
        return f"{v * 100:.1f}% [{b[0] * 100:.1f}, {b[1] * 100:.1f}]"

    def points(x):
        return f"{x['difference'] * 100:.1f} [{x['ci95'][0] * 100:.1f}, {x['ci95'][1] * 100:.1f}]"

    def plain(v, b):
        return f"{v:.2f} [{b[0]:.2f}, {b[1]:.2f}]"

    def plain_diff(x):
        return plain(x["difference"], x["ci95"])

    def tokens(v, b):
        return f"{v:,.0f} [{b[0]:,.0f}, {b[1]:,.0f}]"

    def tokens_diff(x):
        return tokens(x["difference"], x["ci95"])

    def dollars(v, b):
        return f"${v * 1000:.2f} [{b[0] * 1000:.2f}, {b[1] * 1000:.2f}]"

    def dollars_diff(x):
        low, high = x["ci95"]
        return f"{x['difference'] * 1000:.2f} [{low * 1000:.2f}, {high * 1000:.2f}]"

    rows = [
        ("Tasks solved", "success", share, points),
        ("Precision", "precision", share, points),
        ('"No product matches" answers', "empty_answers", share, points),
        ("Searches per task", "searches", plain, plain_diff),
        ("Model calls per task", "model_calls", plain, plain_diff),
        ("Tokens per task", "tokens", tokens, tokens_diff),
        ("Agent cost per 1,000 tasks", "cost", dollars, dollars_diff),
    ]
    expected = [
        f"{report['tasks']} hand-checked tasks from {report['source_products']} "
        "hand-labeled products"
    ]
    for label, metric, cell, diff in rows:
        cells = " | ".join(cell(over[c][metric]["mean"], over[c][metric]["ci95"]) for c in catalogs)
        expected.append(
            f"| {label} | {cells} | {diff(paired['taxon_minus_raw'][metric])} | "
            f"{diff(paired['taxon_minus_sonnet'][metric])} |"
        )
    enrichment = {c: report["catalogs"][c]["enrichment"] for c in catalogs}
    sonnet = enrichment["sonnet"]["dollars_per_million_products"] * 2_000 / 1_000_000
    hours = enrichment["taxon"]["local_hours_per_million_products"] * 2_000 / 1_000_000
    chip = enrichment["taxon"]["hardware"]["chip"].removeprefix("Apple ")
    expected.append(
        f"| Enriching the 2,000 products | none | ${sonnet:,.2f} in API calls | "
        f"no API cost; {hours:.1f} hours on one {chip} | | |"
    )
    split = report["exploratory"]
    for key, label in (
        ("named_in_title", "Title names a requirement"),
        ("not_named_in_title", "Title names none"),
    ):
        group = split[key]
        cells = " | ".join(
            share(group["success"][c]["mean"], group["success"][c]["ci95"]) for c in catalogs
        )
        expected.append(
            f"| {label} | {group['tasks']} | {cells} | {points(group['taxon_minus']['raw'])} | "
            f"{points(group['taxon_minus']['sonnet'])} |"
        )
    glance = {"raw": "the raw catalog", "sonnet": "Sonnet's catalog", "taxon": "Taxon's catalog"}
    for c in catalogs:
        expected.append(
            f"| Tasks solved on {glance[c]} | "
            f"{share(over[c]['success']['mean'], over[c]['success']['ci95'])} |"
        )
    for other, name in (("raw", "raw"), ("sonnet", "Sonnet")):
        x = paired[f"taxon_minus_{other}"]["success"]
        low, high = (bound * 100 for bound in x["ci95"])
        expected.append(
            f"| Taxon minus {name} | {x['difference'] * 100:.1f} points [{low:.1f}, {high:.1f}]"
        )
    missing = [line for line in expected if line not in readme]
    assert not missing, f"README is out of date: {missing}"


def test_the_readme_reports_attribute_support():
    path = RESULTS_DIR / "attribute_support.json"
    if not path.exists():
        pytest.skip("no enriched catalogs yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    readme = _documents()

    def rate(part, whole, share, bounds):
        low, high = (b * 100 for b in bounds)
        return f"{part:,} of {whole:,}, {share * 100:.1f}% [{low:.1f}, {high:.1f}]"

    cells = {}
    for name in ("taxon", "sonnet"):
        origin, hand, other = (
            report[name]["country_of_origin"],
            report[name]["hand_labels"],
            report[name]["catch_all"],
        )
        labeled = hand["match"] + hand["contradict"]
        cells[name] = [
            rate(
                origin["not_named"],
                origin["specific_countries"],
                origin["not_named_share"],
                origin["not_named_share_ci95"],
            ),
            rate(
                hand["blank_in_hand_label"],
                hand["values_on_shown_attributes"],
                hand["blank_share"],
                hand["blank_share_ci95"],
            ),
            rate(
                hand["contradict"],
                labeled,
                hand["contradict_share_of_hand_labeled"],
                hand["contradict_share_ci95"],
            ),
            f"{other['other']:,} of {other['values']:,}, {other['other_share'] * 100:.1f}%",
        ]
    labels = [
        "Specific countries of origin that the listing text never names",
        "Values on verified products where the hand label is blank",
        "Values on verified products that contradict the hand label",
        'Values that are the catch-all "Other"',
    ]
    expected = [
        f"| {label} | {taxon} | {sonnet} |"
        for label, taxon, sonnet in zip(labels, cells["taxon"], cells["sonnet"], strict=True)
    ]
    expected.append(
        f"(Taxon {report['taxon']['hand_labels']['products']}, "
        f"Sonnet {report['sonnet']['hand_labels']['products']})"
    )
    missing = [line for line in expected if line not in readme]
    assert not missing, f"README is out of date: {missing}"


def test_the_readme_reports_the_current_gold_results():
    gold_path = RESULTS_DIR / "gold_eval.json"
    if not gold_path.exists():
        pytest.skip("no hand labels yet")
    readme = _documents()
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


# The README's name for each trained run.
LADDER_NAMES = {
    "qwen-lora": "Fine-tuned",
    "qwen-lora-shortlist": "Fine-tuned on a shortlist",
    "qwen-lora-grpo": "Fine-tuned, then GRPO",
    "qwen-lora-shortlist-grpo": "Fine-tuned on a shortlist, then GRPO",
}


def test_the_readme_reports_the_current_ladder():
    gold_path = RESULTS_DIR / "gold_eval.json"
    if not gold_path.exists() or "ladder" not in json.loads(gold_path.read_text()):
        pytest.skip("no trained model has been asked about the gold products yet")
    readme = _documents()
    ladder = json.loads(gold_path.read_text(encoding="utf-8"))["ladder"]
    names = LADDER_NAMES

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


def test_the_readme_reports_each_run_s_fit_to_its_training_labels():
    gold_path = RESULTS_DIR / "gold_eval.json"
    if not gold_path.exists():
        pytest.skip("no gold scoring yet")
    fits = json.loads(gold_path.read_text(encoding="utf-8")).get("ladder_fit_to_training_labels")
    if not fits:
        pytest.skip("no trained model has been asked about its training products yet")
    readme = _documents()

    def share(value, bounds):
        return f"{value * 100:.1f}% [{bounds[0] * 100:.1f}, {bounds[1] * 100:.1f}]"

    missing = []
    for name, fit in fits.items():
        for decoding, entry in fit.items():
            line = (
                f"| {LADDER_NAMES[name]} | {decoding} | "
                f"{share(entry['exact'], entry['exact_ci95'])} | "
                f"{share(entry['top_level'], entry['top_level_ci95'])} |"
            )
            if line not in readme:
                missing.append(line)
    assert not missing, f"README is out of date: {missing}"

"""Rebuild every result from the pinned sources. Run through `make reproduce`.

This never spends money or reads API keys: labeling results are recomputed from what is already
on disk, and are skipped if nothing has been labeled yet.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from eval.fields import attribute_accuracy, paired_attribute_difference
from eval.metrics import BOOTSTRAP, evaluate, paired_difference
from taxon.baseline import TfidfLogisticRegression, run_baseline
from taxon.bench import search
from taxon.bench.catalog import raw_catalog
from taxon.fetch import cached_range, fetch
from taxon.gold.pool import (
    FIRST_TARGET,
    GOLD_POOL_SIZE,
    assert_disjoint,
    gold_queue,
    training_products,
)
from taxon.gold.server import LABELS_PATH
from taxon.gold.store import GoldStore
from taxon.hardware import hardware
from taxon.labeling.attributes import EXACT, MODES, SHORTLIST
from taxon.labeling.report import agreement_report, arbitrator_score
from taxon.labeling.retrieval import CandidateRetriever, build_shortlists, recall_at
from taxon.pins import (
    AMAZON_FILE_SIZES,
    AMAZON_REVISION,
    DATA_DIR,
    DERIVED_DIR,
    MODEL,
    RAW_DIR,
    RESULTS_DIR,
    SEED,
    TAXONOMY,
    TOKENIZER,
    amazon_url,
)
from taxon.products import read_candidates, read_jsonl, select, summarize, write_jsonl
from taxon.silver import coverage, label_products, stated_attributes
from taxon.taxonomy import Taxonomy
from taxon.trie import TokenTrie

N_PRODUCTS = 10_000
SUGGESTIONS = 5
RECALL_CUTOFFS = (10, 30, 50)
LABELING_ROWS = DATA_DIR / "labeling" / "labels.jsonl"
GOLD_POOL_ROWS = DATA_DIR / "labeling" / "gold_pool_labels.jsonl"
TRAINING_LABELS = DATA_DIR / "labeling" / "training_labels.jsonl"


def write_result(name: str, payload: dict, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "hardware": hardware()}
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (results_dir / f"{name}.json").write_text(text, encoding="utf-8")
    print(f"wrote results/{name}.json", flush=True)


def build_trie(taxonomy: Taxonomy, tokenizer: Tokenizer) -> TokenTrie:
    return TokenTrie.build(
        taxonomy.paths(),
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
    )


def taxonomy_summary(taxonomy: Taxonomy) -> dict:
    depths = Counter(c.depth for c in taxonomy.categories_by_path.values())
    return {
        "version": taxonomy.version,
        "sha256": TAXONOMY.sha256,
        "categories": len(taxonomy),
        "leaves": len(taxonomy.leaves()),
        "verticals": len(taxonomy.roots()),
        "attributes": len(taxonomy.attributes),
        "categories_per_depth": {str(d): depths[d] for d in sorted(depths)},
    }


def sample_products() -> list[dict]:
    candidates = {}
    for domain, size in sorted(AMAZON_FILE_SIZES.items()):
        cache = RAW_DIR / "amazon" / AMAZON_REVISION / domain
        reader = cached_range(amazon_url(domain), size, cache)
        candidates[domain] = read_candidates(domain, size, reader, SEED)
        print(f"  {domain}: {len(candidates[domain])} candidates", flush=True)
    return select(candidates, N_PRODUCTS, SEED)


def build_gold_suggestions(
    products: list[dict], queue: list[str], labels: dict[str, str | None], c: float
) -> list[dict]:
    """The baseline's top guesses for each gold product, from provisional labels alone."""
    training = [p for p in training_products(products, queue) if labels.get(p["parent_asin"])]
    model = TfidfLogisticRegression(c=c).fit(training, [labels[p["parent_asin"]] for p in training])
    pool = set(queue)
    gold = [p for p in products if p["parent_asin"] in pool]
    return [
        {"parent_asin": p["parent_asin"], "suggestions": row}
        for p, row in zip(gold, model.top_k(gold, SUGGESTIONS), strict=True)
    ]


def retrieval_report(
    products: list[dict],
    taxonomy: Taxonomy,
    labels: dict[str, str | None],
    queue: list[str],
    shortlists: dict[str, list[str]],
) -> dict:
    """Recall against provisional labels, for products outside the gold pool.

    The classifier in the shortlist learned from these same labels and knows only their classes,
    so this flatters it. Recall on the hand-labeled gold set is the honest figure.
    """
    rows = [p for p in training_products(products, queue) if labels.get(p["parent_asin"])]
    golds = [labels[p["parent_asin"]] for p in rows]
    similarity = CandidateRetriever(taxonomy.paths()).top_k(rows, max(RECALL_CUTOFFS))
    fused = [shortlists[p["parent_asin"]] for p in rows]
    return {
        "label_source": "silver",
        "similarity_only": recall_at(similarity, golds, RECALL_CUTOFFS),
        "shortlist": recall_at(fused, golds, RECALL_CUTOFFS),
    }


def attribute_pass_score(
    rows: dict[str, dict], gold: dict[str, dict], products: dict[str, dict], taxonomy: Taxonomy
) -> dict | None:
    """The attribute pass against hand-labeled attribute values, beside the free alternative of
    taking only what a listing's details state. None until some hand label has attributes."""

    def ran(asin: str, mode: str) -> bool:
        return "attributes" in rows.get(asin, {}).get("attribute_runs", {}).get(mode, {})

    asins = [a for a in sorted(gold) if gold[a]["attributes"] and all(ran(a, m) for m in MODES)]
    if not asins:
        return None
    golds = [gold[a]["attributes"] for a in asins]
    stated = [stated_attributes(products[a]["details"], gold[a]["path"], taxonomy) for a in asins]
    versions = {
        mode: attribute_accuracy(
            golds, [rows[a]["attribute_runs"][mode]["attributes"] for a in asins]
        )
        for mode in MODES
    }
    # A tie keeps the exact version, which sends the shorter prompt.
    better = max(MODES, key=lambda m: (versions[m]["accuracy"] or 0.0, m == EXACT))
    return {
        "products": len(asins),
        "same_category_as_hand_label": sum(rows[a]["path"] == gold[a]["path"] for a in asins),
        "versions": versions,
        "better": better,
        "shortlist_minus_exact": paired_attribute_difference(
            golds,
            [rows[a]["attribute_runs"][SHORTLIST]["attributes"] for a in asins],
            [rows[a]["attribute_runs"][EXACT]["attributes"] for a in asins],
        ),
        "details_only": attribute_accuracy(golds, stated),
    }


def gold_report(
    products: list[dict],
    taxonomy: Taxonomy,
    queue: list[str],
    labels: dict[str, str | None],
    shortlists: dict[str, list[str]],
    c: float,
    labels_path: Path = LABELS_PATH,
    rows_path: Path = GOLD_POOL_ROWS,
    training_labels_path: Path = TRAINING_LABELS,
) -> dict | None:
    """Everything measured against hand labels, or None until some exist."""
    gold = GoldStore(labels_path, queue, taxonomy).labels()
    if not gold:
        return None
    by_asin = {p["parent_asin"]: p for p in products}
    asins = sorted(gold)
    golds = [gold[a]["path"] for a in asins]
    training = [p for p in training_products(products, queue) if labels.get(p["parent_asin"])]
    model = TfidfLogisticRegression(c=c).fit(training, [labels[p["parent_asin"]] for p in training])
    predictions = model.predict([by_asin[a] for a in asins])
    report = {
        "labeled": len(asins),
        "baseline_trained_on_silver": evaluate(
            golds, predictions, valid_paths=taxonomy.paths(), bootstrap=BOOTSTRAP
        ),
        "shortlist_recall": recall_at([shortlists[a] for a in asins], golds, RECALL_CUTOFFS),
    }
    if training_labels_path.exists():
        pipeline = {row["parent_asin"]: row["path"] for row in read_jsonl(training_labels_path)}
        assert_disjoint(pipeline, queue)
        rows = [by_asin[a] for a in pipeline]
        better = TfidfLogisticRegression(c=c).fit(rows, [pipeline[p["parent_asin"]] for p in rows])
        better_predictions = better.predict([by_asin[a] for a in asins])
        report["baseline_trained_on_pipeline_labels"] = evaluate(
            golds, better_predictions, valid_paths=taxonomy.paths(), bootstrap=BOOTSTRAP
        )
        report["pipeline_labels_minus_silver_for_the_baseline"] = paired_difference(
            golds, better_predictions, predictions
        )
    if rows_path.exists():
        rows = {row["parent_asin"]: row for row in read_jsonl(rows_path)}
        checked = [a for a in asins if a in rows and rows[a]["path"]]
        if checked:
            report["pipeline_labels"] = evaluate(
                [gold[a]["path"] for a in checked],
                [rows[a]["path"] for a in checked],
                bootstrap=BOOTSTRAP,
            )
        arbitration = arbitrator_score(list(rows.values()), {a: gold[a]["path"] for a in asins})
        if arbitration is not None:
            report["arbitration"] = arbitration
        attributes = attribute_pass_score(rows, gold, by_asin, taxonomy)
        if attributes is not None:
            report["attribute_pass"] = attributes
    return report


def search_report(products: list[dict], derived: Path = DERIVED_DIR) -> dict:
    """Index the raw catalog and check the engine over MCP. Uses no labels and no model."""
    rows = raw_catalog(products)
    manifest = search.index_catalog(rows, search.RAW, derived)
    path = search.catalog_path(search.RAW, derived)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "with_price": sum(row["price"] is not None for row in rows),
        "index": manifest,
        "self_retrieval": search.self_retrieval(rows, search.index_dir(search.RAW, derived), SEED),
    }


def main() -> None:
    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    write_result("taxonomy", taxonomy_summary(taxonomy))

    tokenizer = Tokenizer.from_file(str(fetch(TOKENIZER)))
    stats = build_trie(taxonomy, tokenizer).stats()
    write_result("trie", {"model": MODEL, "tokenizer_sha256": TOKENIZER.sha256, **stats})

    products = sample_products()
    products_path = DERIVED_DIR / "products.jsonl"
    write_jsonl(products_path, products)
    digest = hashlib.sha256(products_path.read_bytes()).hexdigest()
    write_result("sample", {"seed": SEED, "sha256": digest, **summarize(products)})

    queue = gold_queue(products, SEED)
    write_jsonl(DERIVED_DIR / "gold_queue.jsonl", [{"parent_asin": a} for a in queue])
    domains = {p["parent_asin"]: p["domain"] for p in products}
    write_result(
        "gold_pool",
        {
            "pool": GOLD_POOL_SIZE,
            "first_target": FIRST_TARGET,
            "first_target_domains": len({domains[a] for a in queue[:FIRST_TARGET]}),
        },
    )

    silver_rows = label_products(products, taxonomy)
    write_jsonl(DERIVED_DIR / "silver_labels.jsonl", silver_rows)
    write_result("silver", coverage(products, silver_rows))
    labels = {row["parent_asin"]: row["path"] for row in silver_rows}

    baseline = run_baseline(products, labels, taxonomy, SEED, "silver", held_out=set(queue))
    write_result("baseline", baseline)
    c = baseline["best_c"]

    shortlists = build_shortlists(products, taxonomy, labels, set(queue), SEED, c)
    write_jsonl(
        DERIVED_DIR / "shortlists.jsonl",
        [{"parent_asin": a, "shortlist": row} for a, row in sorted(shortlists.items())],
    )
    write_result("retrieval", retrieval_report(products, taxonomy, labels, queue, shortlists))
    write_jsonl(
        DERIVED_DIR / "gold_suggestions.jsonl",
        build_gold_suggestions(products, queue, labels, c),
    )

    write_result("search", search_report(products))

    report = gold_report(products, taxonomy, queue, labels, shortlists, c)
    if report is None:
        print("No hand labels yet, so gold evaluation is skipped.", flush=True)
    else:
        write_result("gold_eval", report)
    if LABELING_ROWS.exists():
        write_result("labeling", agreement_report(read_jsonl(LABELING_ROWS)))
    else:
        print("Nothing has been labeled yet, so agreement rates are skipped.", flush=True)
    benchmark_report()


def benchmark_report() -> None:
    """Score CatalogBench from the logged agent runs and judgments, once there are any."""
    from taxon.bench import agent
    from taxon.bench import cli as bench

    if not agent.RunLog(agent.agent_dir(False)).path.exists():
        print("No agent runs yet, so the benchmark is not scored.", flush=True)
        return
    bench.main(["score"])


if __name__ == "__main__":
    main()

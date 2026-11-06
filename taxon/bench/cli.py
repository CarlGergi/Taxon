"""Commands for building the CatalogBench task set and its search indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taxon.bench import search
from taxon.bench.catalog import build_catalog
from taxon.bench.tasks import (
    FIRST_TASK_COUNT,
    check_tasks,
    draft_tasks,
    gold_items,
    load_tasks,
    merge_drafts,
    review,
    save_tasks,
)
from taxon.fetch import fetch
from taxon.gold.pool import training_products
from taxon.gold.server import LABELS_PATH
from taxon.gold.store import GoldStore
from taxon.pins import DATA_DIR, DERIVED_DIR, SEED, TAXONOMY
from taxon.products import read_jsonl
from taxon.taxonomy import Taxonomy

TASKS_PATH = DATA_DIR / "bench" / "tasks.jsonl"


def _inputs(labels_path: Path, derived: Path):
    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    products = read_jsonl(derived / "products.jsonl")
    queue = [row["parent_asin"] for row in read_jsonl(derived / "gold_queue.jsonl")]
    labels = GoldStore(labels_path, queue, taxonomy).labels()
    by_asin = {p["parent_asin"]: p for p in products}
    items = gold_items(labels, by_asin)
    training_ids = {p["parent_asin"] for p in training_products(products, queue)}
    return taxonomy, by_asin, items, training_ids


def main(
    argv: list[str] | None = None,
    tasks_path: Path = TASKS_PATH,
    labels_path: Path = LABELS_PATH,
    derived: Path = DERIVED_DIR,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m taxon.bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="propose tasks from the hand labels")
    draft.add_argument("--n", type=int, default=FIRST_TASK_COUNT)
    commands.add_parser("review", help="hand-check written tasks in the terminal")
    check = commands.add_parser("check", help="validate the finished task set")
    check.add_argument("--need", type=int, default=FIRST_TASK_COUNT)
    index = commands.add_parser("search-index", help="build a catalog and its search index")
    index.add_argument("--name", default=search.RAW)
    index.add_argument("--enrichments", type=Path, help="JSONL of parent_asin, path, attributes")
    query = commands.add_parser("search", help="run one search_catalog request, given as JSON")
    query.add_argument("request", type=json.loads)
    query.add_argument("--name", default=search.RAW)
    args = parser.parse_args(argv)

    if args.command == "search-index":
        if (args.name == search.RAW) != (args.enrichments is None):
            parser.error("the raw catalog takes no enrichments, and any other needs them")
        products = read_jsonl(derived / "products.jsonl")
        taxonomy = Taxonomy.load(fetch(TAXONOMY)) if args.enrichments else None
        rows, report = build_catalog(products, args.enrichments, taxonomy)
        manifest = search.index_catalog(rows, args.name, derived)
        print(json.dumps({**manifest, **report}, indent=2))
        return 0
    if args.command == "search":
        response = search.search_many(search.index_dir(args.name, derived), [args.request])[0]
        print(json.dumps(response, indent=2, ensure_ascii=False))
        return 0

    taxonomy, by_asin, items, training_ids = _inputs(labels_path, derived)
    existing = load_tasks(tasks_path) if tasks_path.exists() else []
    if args.command == "draft":
        tasks = merge_drafts(existing, draft_tasks(items, args.n, SEED))
        save_tasks(tasks_path, tasks)
        print(f"{len(tasks)} tasks in {tasks_path}, from {len(items)} hand-labeled products")
        if len(tasks) < args.n:
            print(f"Only {len(tasks)} of {args.n} are possible. Label more products to draft more.")
    elif args.command == "review":
        titles = {asin: p["title"] for asin, p in by_asin.items()}
        approved = review(existing, titles, input, print)
        save_tasks(tasks_path, existing)
        print(f"{approved} tasks approved")
    else:
        print(json.dumps(check_tasks(existing, items, taxonomy, training_ids, args.need), indent=2))
    return 0

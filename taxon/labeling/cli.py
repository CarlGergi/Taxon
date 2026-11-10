"""The labeling commands: a dry run with a cost estimate, then staged runs.

`--fake` runs everything against in-memory vendors, in its own directory, spending nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taxon.fetch import fetch
from taxon.gold.pool import assert_disjoint, stratified_order, training_products
from taxon.labeling.attributes import (
    ATTRIBUTE_ANNOTATOR,
    MODES,
    AttributeExtractor,
    use_mode,
)
from taxon.labeling.budget import CAPS, Budget, BudgetExceeded
from taxon.labeling.cache import ResponseCache
from taxon.labeling.env import MissingKey, Secrets, load_secrets
from taxon.labeling.fakes import FakeBatchClient
from taxon.labeling.images import ImageStore
from taxon.labeling.pipeline import Labeler
from taxon.labeling.report import (
    agreement_report,
    attribute_projection,
    attribute_report,
    cost_projection,
    format_attribute_projection,
    format_projection,
)
from taxon.labeling.runner import BatchRunner
from taxon.labeling.tracing import build_tracer
from taxon.labeling.vendors import build_clients
from taxon.pins import DATA_DIR, DERIVED_DIR, RESULTS_DIR, SEED, TAXONOMY
from taxon.products import read_jsonl, write_jsonl
from taxon.taxonomy import Taxonomy

DRY_RUN_PRODUCTS = 100
FIRST_RUN_PRODUCTS = 2_000
EXIT_BUDGET = 3
# Everything this milestone plans to label: the first run plus the gold pool.
PLANNED = "first_run_and_gold_pool"
EXIT_USAGE = 2


def labeling_order(products: list[dict], gold: list[str]) -> list[dict]:
    """Training products in a seeded, domain-balanced order, so each stage extends the last."""
    return stratified_order(training_products(products, gold), SEED, "labeling")


class Workspace:
    """Paths for a real or fake run. Fake runs never touch the real cache or ledger."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = root / "responses"
        self.ledger = root / "ledger.jsonl"
        self.journal = root / "open_batches"
        self.images = DATA_DIR / "images"
        self.labels = root / "labels.jsonl"
        self.training_labels = root / "training_labels.jsonl"
        self.gold_rows = root / "gold_pool_labels.jsonl"
        self.results = root / "results"
        self.dry_run = self.results / "labeling_dry_run.json"
        # Where make reproduce scores the attribute pass against hand labels.
        self.gold_eval = RESULTS_DIR / "gold_eval.json"

    def write_result(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")


def _load_inputs() -> tuple[Taxonomy, list[dict], list[str], dict[str, list[str]]]:
    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    products = read_jsonl(DERIVED_DIR / "products.jsonl")
    gold = [row["parent_asin"] for row in read_jsonl(DERIVED_DIR / "gold_queue.jsonl")]
    shortlists = {
        row["parent_asin"]: row["shortlist"]
        for row in read_jsonl(DERIVED_DIR / "shortlists.jsonl")
    }
    return taxonomy, products, gold, shortlists


def _build_labeler(taxonomy: Taxonomy, space: Workspace, fake: bool, secrets: Secrets):
    budget = Budget(space.ledger, CAPS)
    if fake:
        clients = {v: FakeBatchClient(v) for v in CAPS}
        tracer = None
        image_for = None
    else:
        clients = build_clients(secrets, charge=budget.charge)
        tracer = build_tracer(secrets)
        image_for = ImageStore(space.images).image_for
    runner = BatchRunner(clients, ResponseCache(space.cache), budget, space.journal, tracer)
    args = {"image_for": image_for} if image_for else {}
    extractor = AttributeExtractor(taxonomy, runner, **args)
    return Labeler(taxonomy, runner, **args), extractor, runner, tracer


def export_training_labels(rows: list[dict], gold: list[str], path: Path) -> int:
    """Write labels usable for training. Refuses outright if a gold product is among them."""
    usable = [
        {
            "parent_asin": r["parent_asin"],
            "path": r["path"],
            "source": r["source"],
            **({"attributes": r["attributes"]} if "attributes" in r else {}),
        }
        for r in rows
        if r["path"]
    ]
    assert_disjoint((row["parent_asin"] for row in usable), gold)
    write_jsonl(path, usable)
    return len(usable)


def _spent(budget: Budget) -> dict[str, float]:
    return {vendor: budget.settled(vendor) for vendor in CAPS}


def dry_run(n: int, fake: bool, space: Workspace, secrets: Secrets) -> dict:
    taxonomy, products, gold, shortlists = _load_inputs()
    order = labeling_order(products, gold)
    sample = order[:n]
    labeler, extractor, runner, tracer = _build_labeler(taxonomy, space, fake, secrets)
    if not fake:
        print(ImageStore(space.images).ensure_all(sample))
    rows = labeler.label(sample, shortlists)
    for mode in MODES:
        extractor.extract(sample, rows, mode)
    if tracer is not None:
        tracer.flush()
    targets = {
        "first_run": FIRST_RUN_PRODUCTS,
        "first_run_and_gold_pool": FIRST_RUN_PRODUCTS + len(gold),
        "all_training": len(order),
    }
    spent = _spent(runner.budget)
    cost = cost_projection(rows, targets, spent)
    attribute_targets = {
        "gold_pool": len(gold),
        "first_run": FIRST_RUN_PRODUCTS,
        "first_run_and_gold_pool": FIRST_RUN_PRODUCTS + len(gold),
    }
    vendor = ATTRIBUTE_ANNOTATOR.vendor
    alone = {
        mode: attribute_projection(rows, attribute_targets, spent, cost, vendor, PLANNED, mode)
        for mode in MODES
    }
    # Both versions run on the gold pool, so each one's plan carries the other's gold-pool cost.
    attribute_cost = {
        mode: attribute_projection(
            rows,
            attribute_targets,
            spent,
            cost,
            vendor,
            PLANNED,
            mode,
            extra=sum(alone[m]["gold_pool"]["upper_95"] for m in MODES if m != mode),
        )
        for mode in MODES
    }
    report = {
        "sample": n,
        "fake": fake,
        "cost": cost,
        "attribute_cost": attribute_cost,
        "attributes": {mode: attribute_report(rows, mode) for mode in MODES},
        "agreement": agreement_report(rows),
    }
    space.write_result(space.dry_run, report)
    return report


def _require_estimate(space: Workspace, stage: str) -> None:
    """Refuse a stage until a dry run has been seen, or if its estimate would pass a cap."""
    if not space.dry_run.exists():
        raise SystemExit("Run the dry run first, so its cost estimate has been seen.")
    estimate = json.loads(space.dry_run.read_text(encoding="utf-8"))
    over = [v for v, e in estimate["cost"].items() if not e[stage]["fits_cap"]]
    if over:
        raise SystemExit(f"The dry run estimates this stage passes the cap for: {', '.join(over)}")


def _require_attribute_estimate(space: Workspace) -> dict:
    _require_estimate(space, "first_run_and_gold_pool")
    estimate = json.loads(space.dry_run.read_text(encoding="utf-8"))
    if set(estimate.get("attribute_cost", {})) != set(MODES):
        raise SystemExit("Rerun the dry run, so it prices both versions of the attribute pass.")
    if not all(e["everything_planned"]["fits_cap"] for e in estimate["attribute_cost"].values()):
        raise SystemExit("The dry run estimates the attribute pass would pass its cap.")
    return estimate


def _require_attribute_score(space: Workspace) -> dict:
    """The pass runs on the 2,000 only after it has been scored against hand labels."""
    report = json.loads(space.gold_eval.read_text()) if space.gold_eval.exists() else {}
    if "better" not in report.get("attribute_pass", {}):
        raise SystemExit(
            "Score the attribute pass against hand labels first: make label-gold-pool, "
            "then make reproduce."
        )
    return report["attribute_pass"]


def run_attributes(fake: bool, space: Workspace, secrets: Secrets) -> dict:
    """Add attributes to the labeled training products. Their category calls are not repeated."""
    _require_attribute_estimate(space)
    score = _require_attribute_score(space)
    if not space.labels.exists():
        raise SystemExit("Label the first products before adding their attributes.")
    taxonomy, products, gold, _ = _load_inputs()
    rows = read_jsonl(space.labels)
    by_asin = {p["parent_asin"]: p for p in products}
    sample = [by_asin[row["parent_asin"]] for row in rows]
    _, extractor, runner, tracer = _build_labeler(taxonomy, space, fake, secrets)
    if not fake:
        print(ImageStore(space.images).ensure_all(sample))
    # Only the version that scored better against hand labels runs on training products.
    mode = score["better"]
    extractor.extract(sample, rows, mode)
    use_mode(rows, mode)
    if tracer is not None:
        tracer.flush()
    write_jsonl(space.labels, rows)
    exported = export_training_labels(rows, gold, space.training_labels)
    return {
        "products": len(rows),
        "training_labels": exported,
        "scored_against_hand_labels": score,
        "mode": mode,
        "attributes": attribute_report(rows, mode),
        "spent": _spent(runner.budget),
        "fake": fake,
    }


def run_stage(n: int | None, fake: bool, space: Workspace, secrets: Secrets) -> dict:
    """Label the first `n` training products, or all of them. Requires a passing dry run."""
    _require_estimate(space, "first_run" if n == FIRST_RUN_PRODUCTS else "all_training")
    taxonomy, products, gold, shortlists = _load_inputs()
    order = labeling_order(products, gold)
    sample = order[:n] if n else order
    labeler, extractor, runner, tracer = _build_labeler(taxonomy, space, fake, secrets)
    if not fake:
        print(ImageStore(space.images).ensure_all(sample))
    rows = labeler.label(sample, shortlists)
    if tracer is not None:
        tracer.flush()
    write_jsonl(space.labels, rows)
    exported = export_training_labels(rows, gold, space.training_labels)
    report = {
        "products": len(rows),
        "training_labels": exported,
        "fake": fake,
        "spent": _spent(runner.budget),
        "caps": CAPS,
        "agreement": agreement_report(rows),
    }
    space.write_result(space.results / f"labeling_{len(rows)}.json", report)
    return report


def run_gold_pool(fake: bool, space: Workspace, secrets: Secrets) -> dict:
    """Label the gold pool too, only to score the pipeline against hand labels later.

    These rows stay in their own file. They never join the training labels, and the labeling page
    never reads them.
    """
    _require_attribute_estimate(space)
    taxonomy, products, gold, shortlists = _load_inputs()
    pool = set(gold)
    sample = [p for p in products if p["parent_asin"] in pool]
    labeler, extractor, runner, tracer = _build_labeler(taxonomy, space, fake, secrets)
    if not fake:
        print(ImageStore(space.images).ensure_all(sample))
    rows = labeler.label(sample, shortlists)
    # Both attribute versions too, so each can be scored against hand labels before either runs
    # on training products.
    for mode in MODES:
        extractor.extract(sample, rows, mode)
    if tracer is not None:
        tracer.flush()
    write_jsonl(space.gold_rows, rows)
    return {"products": len(rows), "spent": _spent(runner.budget), "fake": fake}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m taxon.labeling", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("dry-run", help="label a small sample and print a cost estimate")
    dry.add_argument("--n", type=int, default=DRY_RUN_PRODUCTS)
    dry.add_argument("--fake", action="store_true")
    run = commands.add_parser("run", help="label a stage, after a passing dry run")
    stage = run.add_mutually_exclusive_group(required=True)
    stage.add_argument("--first", action="store_true", help=f"{FIRST_RUN_PRODUCTS:,} products")
    stage.add_argument(
        "--rest", action="store_true", help="every training product, only after milestone 3 decides"
    )
    stage.add_argument(
        "--gold-pool", action="store_true", help="the gold pool, kept apart, to score the pipeline"
    )
    stage.add_argument(
        "--attributes",
        action="store_true",
        help="the attribute pass on labeled training products, after it is scored on gold",
    )
    run.add_argument("--fake", action="store_true")
    return parser


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    args = _parser().parse_args(argv)
    default_root = DATA_DIR / ("labeling_fake" if args.fake else "labeling")
    space = Workspace(root or default_root)
    secrets = load_secrets()
    try:
        if args.command == "dry-run":
            report = dry_run(args.n, args.fake, space, secrets)
            print(format_projection(report["cost"]))
            for entry in report["attribute_cost"].values():
                print(format_attribute_projection(entry))
            print(json.dumps(report["agreement"]["agreement"], indent=2))
            print(f"Full report: {space.dry_run}")
        elif args.attributes:
            report = run_attributes(args.fake, space, secrets)
            print(json.dumps(report["attributes"], indent=2))
            print(f"Spent by vendor: {report['spent']}")
        elif args.gold_pool:
            report = run_gold_pool(args.fake, space, secrets)
            print(f"Labeled {report['products']} gold-pool products, kept apart from training.")
            print(f"Spent by vendor: {report['spent']}")
        else:
            n = FIRST_RUN_PRODUCTS if args.first else None
            report = run_stage(n, args.fake, space, secrets)
            print(json.dumps(report["agreement"]["agreement"], indent=2))
            print(f"Spent by vendor: {report['spent']}")
    except BudgetExceeded as stop:
        print(f"STOPPED: {secrets.redact(str(stop))}")
        return EXIT_BUDGET
    except MissingKey as missing:
        print(f"STOPPED: {missing}")
        return EXIT_USAGE
    return 0

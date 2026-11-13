"""Commands for reinforcement learning: `train` with GRPO, and `decode` to compare constrained and
free decoding."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from taxon.fetch import fetch
from taxon.hardware import accelerator
from taxon.pins import DATA_DIR, DERIVED_DIR, SEED, TAXONOMY
from taxon.products import read_jsonl, write_jsonl
from taxon.taxonomy import Taxonomy
from taxon.training.cli import RUNS_DIR as SFT_RUNS
from taxon.training.cli import image_lookup
from taxon.training.examples import (
    LABEL_SOURCES,
    SILVER,
    LabelSource,
    load_labeled,
    load_shortlists,
    request_row,
    shortlist_answer,
    training_examples,
)
from taxon.training.model import TINY

RL_RUNS = DATA_DIR / "rl"
ENRICHMENTS_FILE = "enrichments_taxon.jsonl"
BENCH_TAXON_ENRICHMENT = DATA_DIR / "bench" / "taxon_enrichment.json"
EXIT_REFUSED = 2
TINY_NOTE = "Tiny random weights: this shows the pipeline runs, and says nothing about quality."


def _train(args, source, products, queue, taxonomy, processor_factory, derived) -> int:
    from taxon.rl.enrich import run_shortlist
    from taxon.rl.grpo import GrpoSettings, pick_length, run_grpo
    from taxon.training.fields import CATEGORY
    from taxon.training.train import RunRefused, check_settings

    settings = GrpoSettings(
        name=args.name,
        gpu=args.gpu,
        max_steps=args.max_steps,
        num_generations=args.generations,
        batch_size=args.generations * args.prompts,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
    )
    try:
        check_settings(settings, source, accelerator())
    except RunRefused as error:
        print(f"STOPPED: {error}")
        return EXIT_REFUSED
    sft_dir = None
    if args.start_from:
        sft_dir = args.sft_runs / args.start_from
        if not (sft_dir / "run.json").exists():
            print(f"STOPPED: no fine-tuning run named {args.start_from}; train one first")
            return EXIT_REFUSED
    if not source.path.exists():
        print(f"STOPPED: no {source.name} labels at {source.path}")
        return EXIT_REFUSED
    items = load_labeled(source, products, queue, taxonomy, SEED, "train", args.products)
    image_for = image_lookup([i.product for i in items], args.images)
    shortlists = None
    if sft_dir is not None:
        k = run_shortlist(json.loads((sft_dir / "run.json").read_text(encoding="utf-8")))
        if k:
            settings = replace(settings, shortlist=k)
            ranked = load_shortlists(derived / "shortlists.jsonl")
            shortlists = {item.asin: ranked[item.asin][:k] for item in items}
    if shortlists:
        # On a shortlist the answer is one pick: ask for the category alone and stop generating
        # once the longest pick could be done.
        rows = [
            request_row(item, [CATEGORY], image_for, candidates=shortlists[item.asin])
            for item in items
        ]
        processor = processor_factory()
        if not args.keep_length:
            settings = replace(
                settings, max_completion_length=pick_length(processor.tokenizer, rows)
            )
    else:
        rows = training_examples(items, taxonomy, SEED + 1, image_for)
        processor = processor_factory()
    golds = {
        item.asin: shortlist_answer(item.category, shortlists[item.asin])
        if shortlists
        else item.category
        for item in items
    }
    record = run_grpo(
        rows,
        processor,
        taxonomy,
        settings,
        source,
        args.runs / args.name,
        golds,
        sft_dir,
        resume=args.resume,
    )
    print(json.dumps({k: record[k] for k in ("model", "label_source", "prompts")}))
    print(f"Depth plot: {args.runs / args.name / 'depth.png'}")
    if record["model"] == TINY:
        print(TINY_NOTE)
    return 0


def _decode(args, source, products, queue, taxonomy, processor_factory, derived) -> int:
    from taxon.rl.decoding import category_trie, decoding_report
    from taxon.rl.enrich import load_run_model, run_shortlist

    run_dir = args.runs / args.name
    if not (run_dir / "run.json").exists():
        print(f"STOPPED: no run named {args.name} in {args.runs}")
        return EXIT_REFUSED
    processor = processor_factory()
    model, record = load_run_model(run_dir, processor)
    items = load_labeled(source, products, queue, taxonomy, SEED, "val", args.products)
    image_for = image_lookup([i.product for i in items], args.images)
    k = run_shortlist(record)
    shortlists = None
    if k:
        ranked = load_shortlists(derived / "shortlists.jsonl")
        shortlists = {item.asin: ranked[item.asin][:k] for item in items}
    rows = training_examples(items, taxonomy, SEED + 2, image_for, shortlists=shortlists)
    trie = category_trie(processor.tokenizer, taxonomy.paths())
    reports = [
        decoding_report(model, processor, rows, taxonomy, trie if constrained else None)
        for constrained in (False, True)
    ]
    (run_dir / "decoding.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    if record["model"] == TINY:
        print(TINY_NOTE)
    return 0


def _enrich(args, products, queue, taxonomy, processor_factory, derived) -> int:
    """Taxon enriches the benchmark catalog with a trained run."""
    from taxon.bench.agent import benchmark_catalog_ids
    from taxon.hardware import hardware
    from taxon.rl.decoding import category_trie
    from taxon.rl.enrich import (
        enrich_catalog,
        enrichment_rows,
        load_run_model,
        model_generate,
        run_shortlist,
    )

    run_dir = args.runs / args.name
    if not (run_dir / "run.json").exists():
        print(f"STOPPED: no run named {args.name} in {args.runs}")
        return EXIT_REFUSED
    ids = set(benchmark_catalog_ids(products, queue))
    catalog = [p for p in products if p["parent_asin"] in ids]
    if args.limit:
        catalog = catalog[: args.limit]
    processor = processor_factory()
    model, record = load_run_model(run_dir, processor)
    trie = category_trie(processor.tokenizer, taxonomy.paths())
    image_for = image_lookup(catalog, args.images)
    k = run_shortlist(record)
    shortlists = None
    if k:
        ranked = load_shortlists(derived / "shortlists.jsonl")
        shortlists = {p["parent_asin"]: ranked[p["parent_asin"]][:k] for p in catalog}
    rows, report = enrich_catalog(
        catalog,
        taxonomy,
        model_generate(model, processor, trie),
        image_for,
        shortlists=shortlists,
    )
    enriched = enrichment_rows(rows)
    write_jsonl(run_dir / ENRICHMENTS_FILE, enriched)
    report = {
        "run": args.name,
        "model": record["model"],
        "hardware": hardware(model.device.type),
        "enrichments": str(run_dir / ENRICHMENTS_FILE),
        **report,
    }
    text = json.dumps(report, indent=2)
    (run_dir / "enrichment.json").write_text(text, encoding="utf-8")
    # The benchmark cites the latest Taxon enrichment, whichever run produced it.
    BENCH_TAXON_ENRICHMENT.parent.mkdir(parents=True, exist_ok=True)
    BENCH_TAXON_ENRICHMENT.write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if record["model"] == TINY:
        print(TINY_NOTE)
    return 0


def main(
    argv: list[str] | None = None,
    derived: Path = DERIVED_DIR,
    sources: dict[str, LabelSource] = LABEL_SOURCES,
    processor_factory: Callable | None = None,
    load_taxonomy: Callable[[], Taxonomy] = lambda: Taxonomy.load(fetch(TAXONOMY)),
) -> int:
    parser = argparse.ArgumentParser(prog="python -m taxon.rl", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="GRPO on top of a fine-tuned adapter")
    train.add_argument("--name", default="grpo-tiny")
    train.add_argument("--from", dest="start_from", help="fine-tuning run to start from")
    train.add_argument("--sft-runs", type=Path, default=SFT_RUNS)
    train.add_argument("--labels", choices=sorted(sources), default=SILVER.name)
    train.add_argument("--gpu", action="store_true", help="the real model on a GPU, vLLM rollouts")
    train.add_argument("--products", type=int, default=16)
    train.add_argument("--max-steps", type=int, default=4)
    train.add_argument("--generations", type=int, default=4, help="replies per prompt")
    train.add_argument("--prompts", type=int, default=2, help="prompts per step")
    train.add_argument("--max-completion-length", type=int, default=128)
    train.add_argument(
        "--keep-length",
        action="store_true",
        help="on a shortlist, keep --max-completion-length instead of the longest pick's length",
    )
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--save-steps", type=int, default=50, help="checkpoint interval; 0 for none")
    train.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    decode = commands.add_parser("decode", help="free against trie-constrained decoding")
    decode.add_argument("--name", default="grpo-tiny")
    decode.add_argument("--labels", choices=sorted(sources), default=SILVER.name)
    decode.add_argument("--products", type=int, default=8)
    enrich = commands.add_parser("enrich", help="Taxon enriches the benchmark catalog")
    enrich.add_argument("--name", default="grpo-tiny", help="the trained run to enrich with")
    enrich.add_argument("--limit", type=int, default=0, help="only the first N products")
    for command in (train, decode, enrich):
        command.add_argument("--runs", type=Path, default=RL_RUNS)
        command.add_argument("--images", type=Path, default=None, help="folder of cached images")
    args = parser.parse_args(argv)

    if processor_factory is None:
        from taxon.training.model import load_processor

        processor_factory = load_processor
    taxonomy = load_taxonomy()
    products = read_jsonl(derived / "products.jsonl")
    queue = [row["parent_asin"] for row in read_jsonl(derived / "gold_queue.jsonl")]
    if args.command == "enrich":
        return _enrich(args, products, queue, taxonomy, processor_factory, derived)
    source = sources[args.labels]
    if args.command == "train":
        return _train(args, source, products, queue, taxonomy, processor_factory, derived)
    return _decode(args, source, products, queue, taxonomy, processor_factory, derived)

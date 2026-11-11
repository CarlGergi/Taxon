"""Commands for fine-tuning: `train` and `evaluate`."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from taxon.fetch import fetch
from taxon.gold.server import LABELS_PATH
from taxon.gold.store import GoldStore
from taxon.hardware import accelerator
from taxon.labeling.images import ImageStore
from taxon.pins import DATA_DIR, DERIVED_DIR, RESULTS_DIR, SEED, TAXONOMY
from taxon.products import read_jsonl
from taxon.taxonomy import Taxonomy
from taxon.training.examples import (
    LABEL_SOURCES,
    SILVER,
    LabeledProduct,
    LabelSource,
    load_labeled,
    training_examples,
)
from taxon.training.model import TINY

RUNS_DIR = DATA_DIR / "training"
IMAGES_DIR = DATA_DIR / "images"
EXIT_REFUSED = 2


def image_lookup(products: list[dict], images_dir: Path | None) -> Callable:
    if images_dir is None:
        return lambda _product: None
    store = ImageStore(images_dir)
    print(store.ensure_all(products), flush=True)

    def image_for(product: dict) -> str | None:
        found = store.image_for(product)
        return found[0] if found else None

    return image_for


def _gold_items(products: list[dict], labels: dict) -> list[LabeledProduct]:
    return [
        LabeledProduct(p, labels[p["parent_asin"]]["path"], labels[p["parent_asin"]]["attributes"])
        for p in products
        if p["parent_asin"] in labels
    ]


def _train(args, source, products, queue, taxonomy, processor_factory) -> int:
    from taxon.training.train import RunRefused, TrainSettings, check_settings, run_training

    settings = TrainSettings(
        name=args.name,
        gpu=args.gpu,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    try:
        check_settings(settings, source, accelerator())
    except RunRefused as error:
        print(f"STOPPED: {error}")
        return EXIT_REFUSED
    if not source.path.exists():
        print(f"STOPPED: no {source.name} labels at {source.path}")
        return EXIT_REFUSED
    items = load_labeled(source, products, queue, taxonomy, SEED, "train", args.products)
    image_for = image_lookup([i.product for i in items], args.images)
    rows = training_examples(items, taxonomy, SEED, image_for, args.variants)
    record = run_training(rows, processor_factory(), settings, source, args.runs / args.name)
    print(json.dumps({k: record[k] for k in ("model", "label_source", "examples")}))
    print(f"Loss went from {record['losses'][0]:.3f} to {record['losses'][-1]:.3f}")
    if record["model"] == TINY:
        print("Tiny random weights: this shows the pipeline runs, and says nothing about quality.")
    return 0


def _evaluate(args, source, products, queue, taxonomy, labels_path, processor_factory) -> int:
    from taxon.training.evaluate import evaluate_run, load_trained, model_generate

    processor = processor_factory()
    run_dir = args.runs / args.name
    if not (run_dir / "run.json").exists():
        print(f"STOPPED: no run named {args.name}; train one first")
        return EXIT_REFUSED
    hand_labels = None
    if args.on == "gold":
        hand_labels = GoldStore(labels_path, queue, taxonomy).labels()
        if not hand_labels:
            print("STOPPED: no hand labels yet, so there is nothing to score against")
            return EXIT_REFUSED
        items = _gold_items(products, hand_labels)[: args.products]
    else:
        items = load_labeled(source, products, queue, taxonomy, SEED, "val", args.products)
    image_for = image_lookup([i.product for i in items], args.images)
    model, record = load_trained(run_dir, processor)
    report = evaluate_run(
        run_dir,
        record,
        model_generate(model, processor),
        items,
        image_for,
        args.on,
        hand_labels,
        taxonomy.paths(),
    )
    print(json.dumps({k: report[k] for k in ("compliance", "invariance")}, indent=2))
    if record["model"] == TINY:
        print("Tiny random weights: this shows the pipeline runs, and says nothing about quality.")
    return 0


def _measure(args, products, queue, taxonomy, processor_factory) -> int:
    """A few real steps of fine-tuning and of GRPO, timed, to size runs on this machine."""
    import gc

    import torch

    from taxon.rl.grpo import GrpoSettings, run_grpo
    from taxon.training.measure import StepMeter, measurement_record, write_measurement
    from taxon.training.model import align_end_tokens, real_model
    from taxon.training.train import TrainSettings, run_training

    device = accelerator()
    if device is None:
        print("STOPPED: measuring needs a GPU, NVIDIA or Apple")
        return EXIT_REFUSED
    items = load_labeled(SILVER, products, queue, taxonomy, SEED, "train", args.products)
    image_for = image_lookup([i.product for i in items], args.images)
    rows = training_examples(items, taxonomy, SEED, image_for)
    processor = processor_factory()
    scratch = args.runs / "measure"
    phases = {}

    started = time.perf_counter()
    model = real_model(device)
    align_end_tokens(model, processor.tokenizer)
    load_s = time.perf_counter() - started
    meter = StepMeter(device)
    sft = TrainSettings(
        name="measure-sft", gpu=True, max_steps=args.sft_steps, batch_size=args.batch_size
    )
    run_training(rows, processor, sft, SILVER, scratch / "sft", model=model, callbacks=[meter])
    phases["fine_tuning"] = {"load_s": load_s, "batch_size": args.batch_size, **meter.summary()}
    settings = {k: v for k, v in vars(args).items() if isinstance(v, int | float | str)}
    # Written now, so a failure in GRPO cannot lose these.
    write_measurement(args.out, measurement_record(device, phases, settings, rows, args.hours))
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    meter = StepMeter(device)
    grpo = GrpoSettings(
        name="measure-grpo",
        gpu=True,
        max_steps=args.grpo_steps,
        batch_size=args.generations * args.prompts,
        num_generations=args.generations,
        max_completion_length=args.completion_length,
    )
    gold = {item.asin: item.category for item in items}
    run_grpo(rows, processor, taxonomy, grpo, SILVER, scratch / "grpo", gold, callbacks=[meter])
    phases["grpo"] = {
        "prompts_per_step": args.prompts,
        "generations": args.generations,
        "max_completion_length": args.completion_length,
        **meter.summary(),
    }
    record = measurement_record(device, phases, settings, rows, args.hours)
    write_measurement(args.out, record)
    print(json.dumps(record, indent=2))
    print("These runs used provisional labels only to time steps. Nothing they learned is used.")
    return 0


def main(
    argv: list[str] | None = None,
    derived: Path = DERIVED_DIR,
    labels_path: Path = LABELS_PATH,
    sources: dict[str, LabelSource] = LABEL_SOURCES,
    processor_factory: Callable | None = None,
    load_taxonomy: Callable[[], Taxonomy] = lambda: Taxonomy.load(fetch(TAXONOMY)),
) -> int:
    parser = argparse.ArgumentParser(prog="python -m taxon.training", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fine-tune an adapter")
    train.add_argument("--name", default="tiny")
    train.add_argument("--labels", choices=sorted(sources), default=SILVER.name)
    train.add_argument("--gpu", action="store_true", help="use the real model on a GPU")
    train.add_argument("--products", type=int, default=32)
    train.add_argument("--variants", type=int, default=1, help="field choices per product")
    train.add_argument("--max-steps", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--runs", type=Path, default=RUNS_DIR)
    train.add_argument("--images", type=Path, default=IMAGES_DIR)
    train.add_argument("--no-images", dest="images", action="store_const", const=None)
    check = commands.add_parser("evaluate", help="measure a trained adapter")
    check.add_argument("--name", default="tiny")
    check.add_argument("--labels", choices=sorted(sources), default=SILVER.name)
    check.add_argument("--on", choices=["val", "gold"], default="val")
    check.add_argument("--products", type=int, default=4)
    check.add_argument("--runs", type=Path, default=RUNS_DIR)
    check.add_argument("--images", type=Path, default=IMAGES_DIR)
    check.add_argument("--no-images", dest="images", action="store_const", const=None)
    measure = commands.add_parser("measure", help="time real steps on this machine to size runs")
    measure.add_argument("--products", type=int, default=32)
    measure.add_argument("--batch-size", type=int, default=4)
    measure.add_argument("--sft-steps", type=int, default=6)
    measure.add_argument("--grpo-steps", type=int, default=3)
    measure.add_argument("--prompts", type=int, default=2, help="prompts per GRPO step")
    measure.add_argument("--generations", type=int, default=4, help="replies per prompt")
    measure.add_argument("--completion-length", type=int, default=128)
    measure.add_argument("--hours", type=float, default=8.0)
    measure.add_argument("--out", type=Path, default=RESULTS_DIR / "training_hardware.json")
    measure.add_argument("--runs", type=Path, default=RUNS_DIR)
    measure.add_argument("--images", type=Path, default=IMAGES_DIR)
    measure.add_argument("--no-images", dest="images", action="store_const", const=None)
    args = parser.parse_args(argv)

    if processor_factory is None:
        from taxon.training.model import load_processor

        processor_factory = load_processor
    taxonomy = load_taxonomy()
    products = read_jsonl(derived / "products.jsonl")
    queue = [row["parent_asin"] for row in read_jsonl(derived / "gold_queue.jsonl")]
    source = sources.get(getattr(args, "labels", SILVER.name), SILVER)
    if args.command == "measure":
        return _measure(args, products, queue, taxonomy, processor_factory)
    if args.command == "train":
        return _train(args, source, products, queue, taxonomy, processor_factory)
    return _evaluate(args, source, products, queue, taxonomy, labels_path, processor_factory)

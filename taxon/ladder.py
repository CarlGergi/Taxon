"""The evaluation ladder: does each trained model beat the labels it learned from?

Each rung is a trained run (fine-tuning, then GRPO on top), asked for the category alone of every
hand-labeled product, greedy, once free and once constrained by the token trie. Its answers are
scored against the hand labels, and set beside the labeling pipeline's own labels for the same
products with a paired bootstrap: a model trained on noisy labels only earns its keep if it does
better than those labels, or as well at far less cost.

Asking the models needs the GPU, so `generate` saves each rung's answers under data/ladder/.
`make reproduce` scores whatever answers are saved there, with no model in sight.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval.metrics import BOOTSTRAP, evaluate, miss_kinds, paired_difference
from taxon.pins import DATA_DIR, DERIVED_DIR, TAXONOMY

LADDER_DIR = DATA_DIR / "ladder"
DECODINGS = ("free", "constrained")


def rung_report(
    golds: Sequence[str],
    predictions: Sequence[str | None],
    label_predictions: Sequence[str | None],
    valid_paths: Sequence[str],
    previous: Sequence[str | None] | None = None,
) -> dict:
    """One rung against the hand labels, against the pipeline labels, and against the rung below."""
    report = {
        "category": evaluate(golds, predictions, valid_paths=valid_paths, bootstrap=BOOTSTRAP),
        "misses": miss_kinds(golds, predictions),
        "minus_pipeline_labels": paired_difference(golds, predictions, label_predictions),
    }
    if previous is not None:
        report["minus_previous_rung"] = paired_difference(golds, predictions, previous)
    return report


def ladder_report(
    golds: Mapping[str, str],
    labels: Mapping[str, str | None],
    rungs: Sequence[Mapping],
    valid_paths: Sequence[str],
) -> dict | None:
    """Score saved rungs, in order, on the products every rung and the labels answered.

    Returns None when no product was answered by all of them.

    Each rung is {"name", "decoding", "predictions": {asin: path or None}}. A rung is compared
    with the rung below it that used the same decoding.
    """
    asins = sorted(set(golds) & set(labels))
    for rung in rungs:
        asins = [a for a in asins if a in rung["predictions"]]
    if not asins:
        return None
    gold_paths = [golds[a] for a in asins]
    label_paths = [labels[a] for a in asins]
    report: dict = {
        "products": len(asins),
        "pipeline_labels": evaluate(gold_paths, label_paths, valid_paths=valid_paths),
        "rungs": [],
    }
    below: dict[str, list[str | None]] = {}
    for rung in rungs:
        predictions = [rung["predictions"][a] for a in asins]
        entry = rung_report(
            gold_paths, predictions, label_paths, valid_paths, below.get(rung["decoding"])
        )
        report["rungs"].append({"name": rung["name"], "decoding": rung["decoding"], **entry})
        below[rung["decoding"]] = predictions
    return report


def saved_rungs(names: Sequence[str], root: Path = LADDER_DIR) -> list[dict]:
    """The saved answers of the named runs, both decodings each, in ladder order."""
    rungs = []
    for name in names:
        path = root / f"{name}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            rungs += [
                {"name": name, "decoding": d, "predictions": saved[d]}
                for d in DECODINGS
                if d in saved
            ]
    return rungs


def generate(run_dir: Path, images: Path | None, root: Path = LADDER_DIR) -> Path:
    """Ask a trained run for every hand-labeled product's category, free and constrained."""
    from taxon.fetch import fetch
    from taxon.gold.server import LABELS_PATH
    from taxon.gold.store import GoldStore
    from taxon.products import read_jsonl
    from taxon.rl.decoding import category_trie, category_value
    from taxon.rl.enrich import CATEGORY_TOKENS, inference_row, load_run_model, model_generate
    from taxon.taxonomy import Taxonomy
    from taxon.training.cli import image_lookup
    from taxon.training.model import load_processor

    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    queue = [row["parent_asin"] for row in read_jsonl(DERIVED_DIR / "gold_queue.jsonl")]
    labels = GoldStore(LABELS_PATH, queue, taxonomy).labels()
    by_asin = {p["parent_asin"]: p for p in read_jsonl(DERIVED_DIR / "products.jsonl")}
    chosen = [by_asin[a] for a in sorted(labels)]
    image_for = image_lookup(chosen, images)
    rows = {p["parent_asin"]: inference_row(p, ["category"], image_for) for p in chosen}

    processor = load_processor()
    model, _ = load_run_model(run_dir, processor)
    trie = category_trie(processor.tokenizer, taxonomy.paths())
    saved = {}
    for decoding in DECODINGS:
        reply = model_generate(model, processor, trie if decoding == "constrained" else None)
        saved[decoding] = {
            asin: category_value(reply(row, CATEGORY_TOKENS)) for asin, row in rows.items()
        }
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_dir.name}.json"
    path.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    from taxon.training.cli import IMAGES_DIR

    parser = argparse.ArgumentParser(prog="python -m taxon.ladder", description=__doc__)
    parser.add_argument("run", type=Path, help="a trained run's folder")
    parser.add_argument("--images", type=Path, default=IMAGES_DIR)
    args = parser.parse_args(argv)
    if not (args.run / "run.json").exists():
        print(f"STOPPED: {args.run} is not a trained run")
        return 1
    print(f"Saved {generate(args.run, args.images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

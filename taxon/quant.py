"""The quantization study: the final model at full precision, 8-bit and 4-bit, on this Mac.

The trained adapters are merged into the base weights, and MLX converts the merged model three
ways: bfloat16, and 8-bit and 4-bit affine weight quantization. Each version answers the same gold
products the same way: the category alone, greedy, constrained by the same token trie. The report
gives accuracy per taxonomy level, p50 and p95 latency, throughput, size, peak memory, and the local
hours a million products would take. FP8 needs an NVIDIA GPU and is not measured here.

MLX is an optional dependency group, installed only for this study: `uv sync --group quant`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median, quantiles

from eval.metrics import BOOTSTRAP, bootstrap_interval, evaluate, paired_difference
from taxon.hardware import hardware
from taxon.pins import DATA_DIR, DERIVED_DIR, TAXONOMY
from taxon.rl.decoding import TrieGuide, category_trie, shortlist_trie, trie_guide
from taxon.rl.enrich import CATEGORY_TOKENS, inference_row, picked_category, run_shortlist
from taxon.taxonomy import Taxonomy
from taxon.training.cli import IMAGES_DIR

QUANT_DIR = DATA_DIR / "quant"
MERGED = "merged"
# Which run was converted, and whether it picks its category from a shortlist.
SOURCE_FILE = "source.json"
# Name, and bits per weight (None keeps bfloat16).
PRECISIONS = {"bf16": None, "int8": 8, "int4": 4}
GROUP_SIZE = 64

# Generate one reply: the prompt text, an image path or None, and a trie guide for the category.
Generate = Callable[[str, str | None, TrieGuide], str]


def merge_run(run_dir: Path, out: Path) -> Path:
    """Fold a run's adapters into full weights, as MLX needs plain Hugging Face weights."""
    from taxon.rl.enrich import load_run_model
    from taxon.training.model import load_processor

    processor = load_processor()
    model, _ = load_run_model(run_dir, processor)
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    return out


def convert_all(merged: Path, root: Path) -> dict[str, Path]:
    from mlx_vlm import convert

    paths = {}
    for name, bits in PRECISIONS.items():
        path = root / name
        if not (path / "config.json").exists():
            convert(
                hf_path=str(merged),
                mlx_path=str(path),
                quantize=bits is not None,
                q_bits=bits or 4,
                q_group_size=GROUP_SIZE,
                dtype="bfloat16",
            )
        paths[name] = path
    return paths


class MlxTrie:
    """The trie constraint as an MLX logits processor, for one reply.

    The first call comes before any reply token; every later call first takes the newest token.
    """

    def __init__(self, guide: TrieGuide):
        self.guide = guide
        self.started = False

    def __call__(self, tokens, logits):
        import mlx.core as mx

        if self.started:
            self.guide.advance(int(tokens[-1].item()))
        self.started = True
        allowed = self.guide.allowed()
        if allowed is None:
            return logits
        mask = mx.full(logits.shape, -mx.inf, dtype=logits.dtype)
        mask[..., mx.array(allowed)] = 0
        return logits + mask


def _p95(values: Sequence[float]) -> float:
    return quantiles(values, n=20)[18] if len(values) > 1 else values[0]


def latency_summary(latencies: Sequence[float], seed: int = 0) -> dict:
    """Percentiles, throughput and hours per million products, each with a bootstrap interval.

    Throughput and hours come from the mean latency, so their intervals are the mean's, inverted.
    """
    seconds = mean(latencies)

    def over(statistic):
        return bootstrap_interval(
            lambda idx: statistic([latencies[i] for i in idx]), len(latencies), BOOTSTRAP, seed
        )

    mean_low, mean_high = over(mean)
    return {
        "replies": len(latencies),
        "latency_p50_s": median(latencies),
        "latency_p50_s_ci95": over(median),
        "latency_p95_s": _p95(latencies),
        "latency_p95_s_ci95": over(_p95),
        "products_per_hour": 3_600 / seconds,
        "products_per_hour_ci95": [3_600 / mean_high, 3_600 / mean_low],
        "hours_per_million_products": seconds * 1_000_000 / 3_600,
        "hours_per_million_products_ci95": [
            mean_low * 1_000_000 / 3_600,
            mean_high * 1_000_000 / 3_600,
        ],
    }


def measure_rows(
    rows: Sequence[Mapping],
    golds: Sequence[str],
    prompt_for: Callable[[Mapping], str],
    generate: Generate,
    new_guide: Callable[[Mapping], TrieGuide],
    valid_paths: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
    warmup: int = 1,
) -> dict:
    """Ask each row for its category, one at a time, timing each reply.

    The first `warmup` replies are made and thrown away first, so one-time compilation is not
    counted as latency.
    """
    for row in rows[:warmup]:
        generate(prompt_for(row), row["image_path"], new_guide(row))
    predictions, latencies = [], []
    for row in rows:
        prompt = prompt_for(row)
        start = clock()
        text = generate(prompt, row["image_path"], new_guide(row))
        latencies.append(clock() - start)
        predictions.append(picked_category(text))
    return {
        "accuracy": evaluate(
            list(golds), predictions, valid_paths=valid_paths, bootstrap=BOOTSTRAP
        ),
        **latency_summary(latencies),
        "predictions": predictions,
    }


def against_full_precision(golds: Sequence[str], precisions: Mapping[str, Mapping]) -> dict:
    """Each quantized version's exact accuracy minus full precision's, paired over products."""
    full = precisions["bf16"]["predictions"]
    return {
        name: paired_difference(golds, report["predictions"], full)
        for name, report in precisions.items()
        if name != "bf16"
    }


def _size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def measure_precision(path: Path, rows, golds, taxonomy: Taxonomy) -> dict:
    import mlx.core as mx
    from mlx_vlm import generate, load

    from taxon.training.collate import prompt_text

    model, processor = load(str(path))
    trie = category_trie(processor.tokenizer, taxonomy.paths())
    mx.reset_peak_memory()

    def reply(prompt: str, image: str | None, guide: TrieGuide) -> str:
        result = generate(
            model,
            processor,
            prompt,
            image=[image] if image else None,
            max_tokens=CATEGORY_TOKENS,
            temperature=0.0,
            logits_processors=[MlxTrie(guide)],
        )
        return result.text

    report = measure_rows(
        rows,
        golds,
        lambda row: prompt_text(processor, row["messages"]),
        reply,
        lambda row: trie_guide(processor.tokenizer, row_trie(processor.tokenizer, trie, row)),
        taxonomy.paths(),
    )
    report["size_gb"] = _size_gb(path)
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    return report


def row_trie(tokenizer, trie, row: Mapping):
    """The whole taxonomy's trie, or the row's own candidates and "none" when it has a shortlist."""
    if row.get("candidates") is not None:
        return shortlist_trie(tokenizer, tuple(row["candidates"]))
    return trie


def gold_rows(taxonomy: Taxonomy, images: Path | None, limit: int, shortlist: int = 0):
    """The hand-labeled products, each asked for its category alone, from its candidates when the
    run was trained with a shortlist."""
    from taxon.gold.server import LABELS_PATH
    from taxon.gold.store import GoldStore
    from taxon.products import read_jsonl
    from taxon.training.cli import image_lookup
    from taxon.training.examples import load_shortlists

    products = {p["parent_asin"]: p for p in read_jsonl(DERIVED_DIR / "products.jsonl")}
    queue = [row["parent_asin"] for row in read_jsonl(DERIVED_DIR / "gold_queue.jsonl")]
    labels = GoldStore(LABELS_PATH, queue, taxonomy).labels()
    asins = sorted(labels)[:limit] if limit else sorted(labels)
    chosen = [products[a] for a in asins]
    image_for = image_lookup(chosen, images)
    ranked = load_shortlists() if shortlist else {}
    rows = [
        inference_row(
            p, ["category"], image_for, ranked[p["parent_asin"]][:shortlist] if shortlist else None
        )
        for p in chosen
    ]
    return rows, [labels[a]["path"] for a in asins]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m taxon.quant", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="merge a run and convert it three ways")
    prepare.add_argument("--run", type=Path, required=True, help="the trained run's folder")
    measure = commands.add_parser("measure", help="measure every version on the gold set")
    measure.add_argument("--limit", type=int, default=0)
    measure.add_argument("--images", type=Path, default=IMAGES_DIR)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        merged = merge_run(args.run, QUANT_DIR / MERGED)
        paths = convert_all(merged, QUANT_DIR)
        record = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
        source = {"run": str(args.run), "shortlist": run_shortlist(record)}
        (QUANT_DIR / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8")
        print(json.dumps({name: str(p) for name, p in paths.items()}, indent=2))
        return 0

    from taxon.fetch import fetch
    from taxon.reproduce import write_result

    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    source = json.loads((QUANT_DIR / SOURCE_FILE).read_text(encoding="utf-8"))
    rows, golds = gold_rows(taxonomy, args.images, args.limit, source["shortlist"])
    if not rows:
        print("No hand labels yet, so there is nothing to measure against.")
        return 1
    precisions = {
        name: measure_precision(QUANT_DIR / name, rows, golds, taxonomy) for name in PRECISIONS
    }
    report = {
        "hardware": hardware("mps"),
        "run": source["run"],
        "decoding": "greedy, category alone, trie-constrained"
        + (f", picked from {source['shortlist']} candidates" if source["shortlist"] else ""),
        "group_size": GROUP_SIZE,
        "fp8": "not measured: it needs an NVIDIA GPU",
        "precisions": precisions,
        "accuracy_minus_bf16": against_full_precision(golds, precisions),
    }
    write_result("quantization", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

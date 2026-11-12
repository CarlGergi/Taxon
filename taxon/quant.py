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

from eval.metrics import evaluate
from taxon.hardware import hardware
from taxon.pins import DATA_DIR, DERIVED_DIR, TAXONOMY
from taxon.rl.decoding import TrieGuide, category_trie, category_value, trie_guide
from taxon.rl.enrich import CATEGORY_TOKENS, inference_row
from taxon.taxonomy import Taxonomy
from taxon.training.cli import IMAGES_DIR

QUANT_DIR = DATA_DIR / "quant"
MERGED = "merged"
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


def latency_summary(latencies: Sequence[float]) -> dict:
    cut = quantiles(latencies, n=20) if len(latencies) > 1 else list(latencies) * 19
    seconds = mean(latencies)
    return {
        "replies": len(latencies),
        "latency_p50_s": median(latencies),
        "latency_p95_s": cut[18],
        "products_per_hour": 3_600 / seconds,
        "hours_per_million_products": seconds * 1_000_000 / 3_600,
    }


def measure_rows(
    rows: Sequence[Mapping],
    golds: Sequence[str],
    prompt_for: Callable[[Mapping], str],
    generate: Generate,
    new_guide: Callable[[], TrieGuide],
    valid_paths: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
    warmup: int = 1,
) -> dict:
    """Ask each row for its category, one at a time, timing each reply.

    The first `warmup` replies are made and thrown away first, so one-time compilation is not
    counted as latency.
    """
    for row in rows[:warmup]:
        generate(prompt_for(row), row["image_path"], new_guide())
    predictions, latencies = [], []
    for row in rows:
        prompt = prompt_for(row)
        start = clock()
        text = generate(prompt, row["image_path"], new_guide())
        latencies.append(clock() - start)
        predictions.append(category_value(text))
    return {
        "accuracy": evaluate(list(golds), predictions, valid_paths=valid_paths),
        **latency_summary(latencies),
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
        lambda: trie_guide(processor.tokenizer, trie),
        taxonomy.paths(),
    )
    report["size_gb"] = _size_gb(path)
    report["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    return report


def gold_rows(taxonomy: Taxonomy, images: Path | None, limit: int):
    """The hand-labeled products, each asked for its category alone."""
    from taxon.gold.server import LABELS_PATH
    from taxon.gold.store import GoldStore
    from taxon.products import read_jsonl
    from taxon.training.cli import image_lookup

    products = {p["parent_asin"]: p for p in read_jsonl(DERIVED_DIR / "products.jsonl")}
    queue = [row["parent_asin"] for row in read_jsonl(DERIVED_DIR / "gold_queue.jsonl")]
    labels = GoldStore(LABELS_PATH, queue, taxonomy).labels()
    asins = sorted(labels)[:limit] if limit else sorted(labels)
    chosen = [products[a] for a in asins]
    image_for = image_lookup(chosen, images)
    rows = [inference_row(p, ["category"], image_for) for p in chosen]
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
        print(json.dumps({name: str(p) for name, p in paths.items()}, indent=2))
        return 0

    from taxon.fetch import fetch
    from taxon.reproduce import write_result

    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    rows, golds = gold_rows(taxonomy, args.images, args.limit)
    if not rows:
        print("No hand labels yet, so there is nothing to measure against.")
        return 1
    report = {
        "hardware": hardware("mps"),
        "decoding": "greedy, category alone, trie-constrained",
        "group_size": GROUP_SIZE,
        "fp8": "not measured: it needs an NVIDIA GPU",
        "precisions": {
            name: measure_precision(QUANT_DIR / name, rows, golds, taxonomy) for name in PRECISIONS
        },
    }
    write_result("quantization", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

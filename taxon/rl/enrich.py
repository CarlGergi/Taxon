"""Taxon enriches the benchmark catalog with its trained model.

Two passes per product, as with the frontier model: first the category alone, decoded through the
token trie so it is always a real path, then that category's allowed attributes. Only values the
taxonomy allows are kept. Timing is recorded per product, so the report can give the local
compute behind a million products.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from eval.fields import parse_fields
from taxon.rl.decoding import category_value
from taxon.taxonomy import Taxonomy
from taxon.training.examples import NONE, ImageFor, messages_for
from taxon.training.fields import CATEGORY, ordered

# Generate one reply to a request row, within a number of new tokens.
Generate = Callable[[Mapping, int], str]
CATEGORY_TOKENS = 64
_TOKENS_PER_ATTRIBUTE = 24
_MAX_ATTRIBUTE_TOKENS = 512


def inference_row(
    product: Mapping,
    requested: Sequence[str],
    image_for: ImageFor,
    candidates: Sequence[str] | None = None,
) -> dict:
    """A request in exactly the training format, for a product with no label."""
    image_path = image_for(product)
    return {
        "asin": product["parent_asin"],
        "requested": list(requested),
        "schema": "inference",
        "messages": messages_for(product, requested, image_path is not None, candidates),
        "image_path": image_path,
        "candidates": list(candidates) if candidates is not None else None,
    }


def run_shortlist(record: Mapping) -> int:
    """How many candidate paths a run picks its category from; 0 when it writes it from memory.
    A GRPO run inherits this from the fine-tuning run it started from."""
    if record.get("shortlist"):
        return int(record["shortlist"])
    if record.get("fine_tuned_from"):
        source = Path(record["fine_tuned_from"]) / "run.json"
        if source.exists():
            return run_shortlist(json.loads(source.read_text(encoding="utf-8")))
    return 0


def picked_category(reply: str) -> str | None:
    """The category a reply gives, with a shortlist's "none" read as no answer."""
    value = category_value(reply)
    return None if value == NONE else value


def attribute_tokens(n: int) -> int:
    return min(_MAX_ATTRIBUTE_TOKENS, 32 + _TOKENS_PER_ATTRIBUTE * n)


def enrich_product(
    product: Mapping,
    taxonomy: Taxonomy,
    generate: Generate,
    image_for: ImageFor,
    candidates: Sequence[str] | None = None,
) -> dict:
    """With candidates, the category is picked from them, and the attribute request also asks for
    the category, as every shortlist training example did; that second category is ignored."""
    asin = product["parent_asin"]
    reply = generate(inference_row(product, [CATEGORY], image_for, candidates), CATEGORY_TOKENS)
    path = picked_category(reply)
    if path not in taxonomy:
        return {"parent_asin": asin, "path": None, "attributes": {}, "status": "no_category"}
    allowed = taxonomy.allowed_values(path)
    row = {"parent_asin": asin, "path": path, "attributes": {}, "status": "ok", "dropped": 0}
    if not allowed:
        return row
    handles = ordered(allowed)
    requested = handles if candidates is None else ordered([CATEGORY, *handles])
    request = inference_row(product, requested, image_for, candidates)
    fields = parse_fields(generate(request, attribute_tokens(len(requested))))
    if fields is None:
        row["status"] = "unparsed_attributes"
        return row
    for handle in handles:
        value = fields.get(handle)
        if not isinstance(value, str):
            continue
        if value in allowed[handle]:
            row["attributes"][handle] = value
        else:
            row["dropped"] += 1
    return row


def enrich_catalog(
    products: Sequence[Mapping],
    taxonomy: Taxonomy,
    generate: Generate,
    image_for: ImageFor,
    clock: Callable[[], float] = time.perf_counter,
    shortlists: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[dict], dict]:
    rows = []
    start = clock()
    for product in products:
        candidates = shortlists[product["parent_asin"]] if shortlists is not None else None
        rows.append(enrich_product(product, taxonomy, generate, image_for, candidates))
    seconds = clock() - start
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    per_product = seconds / len(rows) if rows else 0.0
    report = {
        "products": len(rows),
        "with_category": sum(row["path"] is not None for row in rows),
        "statuses": dict(sorted(statuses.items())),
        "attribute_values_kept": sum(len(row["attributes"]) for row in rows),
        "attribute_values_dropped": sum(row.get("dropped", 0) for row in rows),
        "seconds": seconds,
        "seconds_per_product": per_product,
        "hours_per_million_products": per_product * 1_000_000 / 3_600,
    }
    return rows, report


def enrichment_rows(rows: Sequence[Mapping]) -> list[dict]:
    """What the search index reads, as for the frontier model."""
    return [
        {"parent_asin": r["parent_asin"], "path": r["path"], "attributes": r["attributes"]}
        for r in rows
        if r["path"]
    ]


def load_run_model(run_dir: Path, processor):
    """A trained run's model as it was trained: a GRPO run started from a fine-tuned model gets
    that model's adapter merged in first, then its own adapter on top."""
    from peft import PeftModel

    from taxon.training.train import ADAPTER_DIR, base_model

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model = base_model(processor, record["gpu"], record["seed"])
    if record.get("fine_tuned_from"):
        sft = Path(record["fine_tuned_from"]) / ADAPTER_DIR
        model = PeftModel.from_pretrained(model, sft).merge_and_unload()
    return PeftModel.from_pretrained(model, run_dir / ADAPTER_DIR).eval(), record


def model_generate(model, processor, trie) -> Generate:
    """Greedy replies; the trie constrains any reply that asks for the category. A request with
    candidates is constrained to them and "none" instead, when a trie is given at all."""
    from taxon.rl.decoding import generate_one, shortlist_trie

    def generate(row: Mapping, max_new_tokens: int) -> str:
        row_trie = trie
        if trie is not None and row.get("candidates") is not None:
            row_trie = shortlist_trie(processor.tokenizer, tuple(row["candidates"]))
        return generate_one(model, processor, row, row_trie, max_new_tokens)

    return generate

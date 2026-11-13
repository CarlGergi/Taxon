"""Measure a trained adapter.

Field compliance and invariance need no labels. Accuracy is computed only against hand labels, so a
provisional label can never produce an accuracy figure here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from peft import PeftModel

from eval.fields import attribute_accuracy, field_compliance, field_invariance, parse_fields
from eval.metrics import BOOTSTRAP, evaluate
from taxon.gold.pool import assert_disjoint
from taxon.hardware import hardware
from taxon.products import seeded_hash
from taxon.training.collate import load_images, prompt_text
from taxon.training.examples import NONE, ImageFor, LabeledProduct, request_row
from taxon.training.fields import CATEGORY, ordered
from taxon.training.train import ADAPTER_DIR, RUN_FILE, TRAIN_IDS_FILE, base_model, run_device

ALONE = "alone"
FULL = "full"
MAX_NEW_TOKENS = 128
_DISTRACTORS = ("color", "material", "size", "pattern")

Generate = Callable[[Sequence[Mapping]], list[str]]


def evaluation_requests(
    items: Sequence[LabeledProduct],
    image_for: ImageFor,
    shortlists: Mapping[str, Sequence[str]] | None = None,
) -> list[dict]:
    """Each field asked alone, then everything at once plus one field the product does not state.

    So every field is asked under at least two schemas, which is what invariance compares. With
    shortlists, every request lists the product's candidates and also asks for the category, as
    every shortlist training example did.
    """
    rows = []
    for item in items:
        fields = ordered(item.values)
        options = [name for name in _DISTRACTORS if name not in fields]
        extra = sorted(options, key=lambda name: seeded_hash(0, f"{item.asin}|{name}"))[:1]
        candidates = shortlists[item.asin] if shortlists is not None else None
        for name in fields:
            alone = [name] if candidates is None else ordered([CATEGORY, name])
            rows.append(request_row(item, alone, image_for, ALONE, candidates))
        rows.append(request_row(item, ordered([*fields, *extra]), image_for, FULL, candidates))
    return rows


def run_requests(requests: Sequence[Mapping], generate: Generate) -> list[dict]:
    outputs = generate(requests)
    return [
        {
            "product": row["asin"],
            "requested": row["requested"],
            "schema": row["schema"],
            "output": output,
        }
        for row, output in zip(requests, outputs, strict=True)
    ]


def gold_scores(
    rows: Sequence[Mapping], hand_labels: Mapping[str, Mapping], valid_paths: Sequence[str]
) -> dict:
    """Accuracy from each product's full-schema reply, against hand labels."""
    replies = {row["product"]: row for row in rows if row["schema"] == FULL}
    asins = sorted(set(replies) & set(hand_labels))
    parsed = [parse_fields(replies[asin]["output"]) or {} for asin in asins]
    predictions = [fields.get(CATEGORY) for fields in parsed]
    predictions = [None if p == NONE else p for p in predictions]
    return {
        "hand_labeled": len(asins),
        # Kept per product, so two runs can be compared on the same products (paired_difference).
        "category_predictions": {
            asin: p if isinstance(p, str) else None
            for asin, p in zip(asins, predictions, strict=True)
        },
        "category": evaluate(
            [hand_labels[asin]["path"] for asin in asins],
            [p if isinstance(p, str) else None for p in predictions],
            valid_paths=valid_paths,
            bootstrap=BOOTSTRAP,
        ),
        "attributes": attribute_accuracy(
            [hand_labels[asin]["attributes"] for asin in asins], parsed
        ),
    }


def field_report(rows: Sequence[Mapping]) -> dict:
    return {"compliance": field_compliance(rows), "invariance": field_invariance(rows)}


def model_generate(model, processor, batch_size: int = 4) -> Generate:
    """Greedy generation, so the same adapter always gives the same replies."""

    def generate(rows: Sequence[Mapping]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = processor(
                text=[prompt_text(processor, row["messages"]) for row in batch],
                images=load_images(batch),
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            new = ids[:, inputs["input_ids"].shape[1] :]
            outputs.extend(processor.tokenizer.batch_decode(new, skip_special_tokens=True))
        return outputs

    return generate


def load_trained(run_dir: Path, processor):
    """Rebuild a run's model from its record and adapter."""
    record = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    base = base_model(processor, record["gpu"], record["seed"])
    model = PeftModel.from_pretrained(base, run_dir / ADAPTER_DIR)
    model.eval()
    return model, record


def evaluate_run(
    run_dir: Path,
    record: Mapping,
    generate: Generate,
    items: Sequence[LabeledProduct],
    image_for: ImageFor,
    on: str,
    hand_labels: Mapping[str, Mapping] | None = None,
    valid_paths: Sequence[str] = (),
    shortlists: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Write a report next to the run. Accuracy appears only when hand labels are passed in."""
    if hand_labels is not None:
        train_ids = json.loads((run_dir / TRAIN_IDS_FILE).read_text(encoding="utf-8"))
        assert_disjoint(train_ids, hand_labels)
    rows = run_requests(evaluation_requests(items, image_for, shortlists), generate)
    report = {
        "run": record["name"],
        "model": record["model"],
        "trained_on": record["label_source"],
        "evaluated_on": on,
        "hardware": hardware(run_device(record.get("gpu", False))),
        "products": len(items),
        **field_report(rows),
    }
    if hand_labels is not None:
        report["gold"] = gold_scores(rows, hand_labels, valid_paths)
    path = run_dir / f"eval_{on}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

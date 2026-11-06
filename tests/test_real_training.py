"""Checks with the real tokenizer, chat template and labels. Weights stay tiny and random."""

import json

import pytest
import torch
from PIL import Image

from taxon.fetch import fetch
from taxon.gold.pool import assert_disjoint
from taxon.pins import DERIVED_DIR, SEED, TAXONOMY
from taxon.products import read_jsonl
from taxon.taxonomy import Taxonomy
from taxon.training.collate import END_OF_TURN, IGNORED, FieldCollator, prompt_text
from taxon.training.examples import (
    SILVER,
    SYSTEM,
    load_labeled,
    request_row,
    training_examples,
)
from taxon.training.fields import CATEGORY
from taxon.training.model import load_processor, tiny_model

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def processor():
    return load_processor()


@pytest.fixture(scope="module")
def taxonomy():
    return Taxonomy.load(fetch(TAXONOMY))


@pytest.fixture(scope="module")
def queue():
    return [row["parent_asin"] for row in read_jsonl(DERIVED_DIR / "gold_queue.jsonl")]


@pytest.fixture(scope="module")
def items(taxonomy, queue):
    products = read_jsonl(DERIVED_DIR / "products.jsonl")
    return load_labeled(SILVER, products, queue, taxonomy, SEED, "train")


def no_image(_product):
    return None


@pytest.fixture(scope="module")
def rows(items, taxonomy):
    return training_examples(items[:200], taxonomy, SEED, no_image)


def test_training_data_stays_out_of_the_gold_pool(items, queue):
    assert items
    assert_disjoint((item.asin for item in items), queue)


def test_real_examples_vary_their_requested_fields(items, taxonomy):
    with_attributes = [item for item in items if item.attributes][:200]
    rows = training_examples(with_attributes, taxonomy, SEED, no_image)
    assert any(len(row["requested"]) > 1 for row in rows)
    assert any(CATEGORY not in row["requested"] for row in rows)
    assert any(None in json.loads(row["target"]).values() for row in rows)


def test_the_real_template_renders_the_prompt_as_plain_text(processor, rows):
    text = prompt_text(processor, rows[0]["messages"])
    assert text.startswith("<|im_start|>system\n" + SYSTEM.strip() + "<|im_end|>")
    assert "<product>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "'type'" not in text


def test_the_real_template_marks_where_an_image_goes(processor, items):
    row = request_row(items[0], [CATEGORY], lambda _p: "unused.jpg")
    assert "<|vision_start|><|image_pad|><|vision_end|>" in prompt_text(processor, row["messages"])


def test_the_real_tokenizer_scores_exactly_the_reply(processor, rows):
    collate = FieldCollator(processor)
    for start in range(0, 64, 8):
        batch_rows = rows[start : start + 8]
        batch = collate(batch_rows)
        for i, row in enumerate(batch_rows):
            ids = batch["input_ids"][i][batch["labels"][i] != IGNORED]
            assert processor.tokenizer.decode(ids) == row["target"] + END_OF_TURN


def test_a_reply_starts_straight_after_the_prompt(processor, rows):
    batch = FieldCollator(processor)(rows[:1])
    scored = (batch["labels"][0] != IGNORED).nonzero()[0].item()
    before = processor.tokenizer.decode(batch["input_ids"][0][:scored])
    assert before == prompt_text(processor, rows[0]["messages"])


def test_a_real_batch_with_an_image_runs_through_the_tiny_model(processor, items, tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (512, 384), (10, 200, 10)).save(path)
    rows = [request_row(items[0], [CATEGORY], lambda _p: str(path))]
    batch = FieldCollator(processor)(rows)
    assert batch["pixel_values"].shape[0] > 0
    loss = tiny_model(processor, SEED)(**batch).loss
    assert torch.isfinite(loss)

"""Taxon enriching the benchmark catalog, with scripted replies and with the tiny model."""

import json
import shutil

import pytest
import torch

from taxon.rl import cli
from taxon.rl.enrich import (
    CATEGORY_TOKENS,
    attribute_tokens,
    enrich_catalog,
    enrich_product,
    enrichment_rows,
    inference_row,
    load_run_model,
)
from taxon.taxonomy import Taxonomy
from taxon.training.examples import SILVER
from taxon.training.train import ADAPTER_DIR, TrainSettings, base_model, run_training
from tests.conftest import toy_release
from tests.test_rl_run import rl_rows, run_cli, world  # noqa: F401 - the shared fixture
from tests.tiny_processor import tiny_processor

TOTES = "Apparel > Bags > Totes"


@pytest.fixture(scope="module")
def taxonomy():
    return Taxonomy.from_release(toy_release())


def product(asin="A"):
    return {
        "parent_asin": asin,
        "title": "Red tote",
        "features": [],
        "description": "",
        "details": {},
    }


def scripted(category_reply, attribute_reply=None):
    calls = []

    def generate(row, max_new_tokens):
        calls.append((row, max_new_tokens))
        return category_reply if len(calls) == 1 else attribute_reply

    return generate, calls


def no_image(_product):
    return None


# One product


def test_the_category_comes_first_then_that_categorys_attributes(taxonomy):
    allowed = taxonomy.allowed_values(TOTES)
    handle = sorted(allowed)[0]
    value = sorted(allowed[handle])[0]
    generate, calls = scripted(json.dumps({"category": TOTES}), json.dumps({handle: value}))
    row = enrich_product(product(), taxonomy, generate, no_image)
    assert row["path"] == TOTES and row["status"] == "ok"
    assert row["attributes"] == {handle: value}
    assert calls[0][0]["requested"] == ["category"]
    assert calls[0][1] == CATEGORY_TOKENS
    assert calls[1][0]["requested"] == sorted(allowed)
    assert calls[1][1] == attribute_tokens(len(allowed))


def test_values_the_taxonomy_does_not_allow_are_dropped_and_counted(taxonomy):
    allowed = taxonomy.allowed_values(TOTES)
    handle = sorted(allowed)[0]
    generate, _ = scripted(
        json.dumps({"category": TOTES}), json.dumps({handle: "Plaid sunset", "x": None})
    )
    row = enrich_product(product(), taxonomy, generate, no_image)
    assert row["attributes"] == {}
    assert row["dropped"] == 1


def test_no_real_category_means_no_second_call(taxonomy):
    generate, calls = scripted(json.dumps({"category": "Nowhere > Else"}))
    row = enrich_product(product(), taxonomy, generate, no_image)
    assert (row["path"], row["status"]) == (None, "no_category")
    assert len(calls) == 1


def test_a_category_without_attributes_needs_one_call(taxonomy):
    bare = next(p for p in taxonomy.paths() if not taxonomy.allowed_values(p))
    generate, calls = scripted(json.dumps({"category": bare}))
    row = enrich_product(product(), taxonomy, generate, no_image)
    assert row["path"] == bare and row["attributes"] == {}
    assert len(calls) == 1


def test_an_unreadable_attribute_reply_keeps_the_category(taxonomy):
    generate, _ = scripted(json.dumps({"category": TOTES}), "not json")
    row = enrich_product(product(), taxonomy, generate, no_image)
    assert row["path"] == TOTES and row["status"] == "unparsed_attributes"


def test_requests_use_the_training_format_with_the_image_when_there_is_one():
    row = inference_row(product(), ["category"], lambda _p: "/tmp/a.jpg")
    assert row["image_path"] == "/tmp/a.jpg"
    assert row["messages"][1]["content"][0] == {"type": "image"}
    assert "Fields: category" in row["messages"][1]["content"][-1]["text"]


def test_attribute_budgets_grow_with_the_attributes_and_stop_at_a_ceiling():
    assert attribute_tokens(1) < attribute_tokens(5) <= attribute_tokens(100) == 512


# The catalog


def test_the_catalog_report_counts_and_times_every_product(taxonomy):
    ticks = iter([0.0, 6.0])
    replies = {"A": json.dumps({"category": "Home > Kitchen"}), "B": "nonsense"}

    def generate(row, _tokens):
        return replies[row["asin"]]

    rows, report = enrich_catalog(
        [product("A"), product("B")], taxonomy, generate, no_image, clock=lambda: next(ticks)
    )
    assert report["products"] == 2 and report["with_category"] == 1
    assert report["statuses"] == {"no_category": 1, "ok": 1}
    assert report["seconds_per_product"] == 3.0
    assert report["hours_per_million_products"] == pytest.approx(3.0 * 1_000_000 / 3_600)
    assert enrichment_rows(rows) == [
        {"parent_asin": "A", "path": "Home > Kitchen", "attributes": {}}
    ]


# Loading a run


def test_a_grpo_run_is_loaded_on_top_of_the_model_it_was_trained_from(tmp_path, taxonomy):
    processor = tiny_processor()
    run_training(
        rl_rows(taxonomy),
        processor,
        TrainSettings(name="t", max_steps=2, learning_rate=5e-3),
        SILVER,
        tmp_path / "sft",
    )
    grpo = tmp_path / "grpo"
    shutil.copytree(tmp_path / "sft" / ADAPTER_DIR, grpo / ADAPTER_DIR)
    record = {"gpu": False, "seed": 0, "model": "tiny", "fine_tuned_from": str(tmp_path / "sft")}
    (grpo / "run.json").write_text(json.dumps(record))
    stacked, _ = load_run_model(grpo, processor)

    from peft import PeftModel

    alone = PeftModel.from_pretrained(base_model(processor, False, 0), grpo / ADAPTER_DIR).eval()
    batch = processor(text=["hello"], return_tensors="pt")
    with torch.no_grad():
        assert not torch.equal(stacked(**batch).logits, alone(**batch).logits)


def test_a_fine_tuning_run_loads_as_it_is(tmp_path, taxonomy):
    processor = tiny_processor()
    run_training(
        rl_rows(taxonomy), processor, TrainSettings(name="t", max_steps=1), SILVER, tmp_path / "sft"
    )
    model, record = load_run_model(tmp_path / "sft", processor)
    assert not record.get("fine_tuned_from")
    assert model.training is False


# The command


def test_the_command_enriches_the_catalog_with_a_trained_run(world, capsys, monkeypatch):  # noqa: F811
    root = world[0]
    assert (
        run_cli(
            world, "train", "--products", "4", "--max-steps", "1", "--max-completion-length", "8"
        )
        == 0
    )
    monkeypatch.setattr(
        "taxon.bench.agent.benchmark_catalog_ids",
        lambda products, queue: [p["parent_asin"] for p in products[:5]],
    )
    capsys.readouterr()
    assert run_cli(world, "enrich", "--limit", "3") == 0
    out = capsys.readouterr().out
    assert "Tiny random weights" in out
    report = json.loads((root / "rl" / "grpo-tiny" / "enrichment.json").read_text())
    assert report["products"] == 3
    assert report["with_category"] == 3
    assert report["hardware"]["device"] == "cpu"
    rows = [json.loads(line) for line in (root / "rl" / "grpo-tiny" / cli.ENRICHMENTS_FILE).open()]
    assert len(rows) == 3 and all(set(r) == {"parent_asin", "path", "attributes"} for r in rows)


def test_enriching_needs_a_trained_run(world, capsys):  # noqa: F811
    assert run_cli(world, "enrich", "--name", "nope") == cli.EXIT_REFUSED
    assert "no run named nope" in capsys.readouterr().out

"""The whole fine-tuning flow on CPU, with tiny random weights and a byte-level vocabulary."""

import json

import pytest
import torch
from peft import get_peft_model
from PIL import Image
from safetensors.torch import load_file

from taxon.gold.pool import GoldOverlap
from taxon.products import write_jsonl
from taxon.taxonomy import Taxonomy
from taxon.training import cli, train
from taxon.training.collate import END_OF_TURN, IGNORED, FieldCollator, load_images, prompt_text
from taxon.training.evaluate import (
    ALONE,
    FULL,
    evaluate_run,
    evaluation_requests,
    gold_scores,
    load_trained,
    model_generate,
    run_requests,
)
from taxon.training.examples import PIPELINE, SILVER, LabeledProduct, LabelSource, request_row
from taxon.training.fields import CATEGORY
from taxon.training.model import TINY, lora_config, tiny_model
from taxon.training.train import (
    RunRefused,
    TrainSettings,
    base_model,
    check_settings,
    run_training,
    training_arguments,
)
from tests.conftest import toy_release
from tests.tiny_processor import tiny_processor

TOTES = "Apparel > Bags > Totes"
FAST = TrainSettings(name="t", max_steps=6, learning_rate=5e-3, seed=11)


@pytest.fixture(scope="module")
def processor():
    return tiny_processor()


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("images") / "a.jpg"
    Image.new("RGB", (64, 48), (200, 10, 10)).save(path)
    return str(path)


def product(asin="A", **extra):
    return {
        "parent_asin": asin,
        "title": f"Red tote {asin}",
        "features": ["Roomy"],
        "description": "A bag.",
        "details": {"Color": "Red"},
        **extra,
    }


ITEM = LabeledProduct(product(), TOTES, {"color": "Red"})


def no_image(_product):
    return None


@pytest.fixture(scope="module")
def rows(image_path):
    return [
        request_row(ITEM, [CATEGORY, "color"], lambda _p: image_path),
        request_row(ITEM, ["color"], no_image),
    ]


def scored_text(processor, batch, index):
    ids = batch["input_ids"][index][batch["labels"][index] != IGNORED]
    return processor.tokenizer.decode(ids)


def test_the_prompt_stops_where_the_reply_starts(processor, rows):
    text = prompt_text(processor, rows[0]["messages"])
    assert text.endswith("<think>\n\n</think>\n\n")
    assert rows[0]["target"] not in text


def test_the_model_is_scored_on_the_reply_and_nothing_else(processor, rows):
    batch = FieldCollator(processor)(rows)
    for i, row in enumerate(rows):
        assert scored_text(processor, batch, i) == row["target"] + END_OF_TURN


def test_padding_is_never_scored(processor, rows):
    batch = FieldCollator(processor)(rows)
    assert (batch["attention_mask"] == 0).any()
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORED).all()


def test_a_batch_can_mix_products_with_and_without_images(processor, rows):
    batch = FieldCollator(processor)(rows)
    assert batch["image_grid_thw"].shape[0] == 1
    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    has_image = (batch["input_ids"] == image_token).any(dim=1).tolist()
    assert has_image == [True, False]


def test_a_batch_without_images_sends_no_pixels(processor, rows):
    batch = FieldCollator(processor)([rows[1]])
    assert "pixel_values" not in batch
    assert load_images([rows[1]]) is None


def test_the_tiny_model_is_the_same_for_the_same_seed(processor):
    first = tiny_model(processor, 3).state_dict()
    second = tiny_model(processor, 3).state_dict()
    third = tiny_model(processor, 4).state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not all(torch.equal(first[k], third[k]) for k in first)


def test_only_the_adapter_in_the_text_model_trains(processor):
    model = get_peft_model(tiny_model(processor, 1), lora_config())
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in name and "language_model" in name for name in trainable)
    assert not any("visual" in name for name in trainable)


def test_the_adapter_reaches_both_kinds_of_attention_layer(processor):
    model = get_peft_model(tiny_model(processor, 1), lora_config())
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any("linear_attn" in name for name in names)
    assert any("self_attn" in name for name in names)


def test_training_lowers_the_loss(processor, rows, tmp_path):
    settings = TrainSettings(name="t", max_steps=12, learning_rate=5e-3, seed=11)
    losses = run_training(rows * 6, processor, settings, SILVER, tmp_path)["losses"]
    assert len(losses) == 12
    assert sum(losses[-3:]) < sum(losses[:3])


def test_the_same_seed_gives_the_same_run(processor, rows, tmp_path):
    first = run_training(rows * 4, processor, FAST, SILVER, tmp_path / "a")
    second = run_training(rows * 4, processor, FAST, SILVER, tmp_path / "b")
    assert first["losses"] == second["losses"]


def test_the_record_says_how_the_run_was_made(processor, rows, tmp_path):
    run_training(rows * 4, processor, FAST, SILVER, tmp_path)
    record = json.loads((tmp_path / train.RUN_FILE).read_text())
    assert record["model"] == TINY
    assert record["hardware"]["device"] == "cpu"
    assert record["label_source"] == "silver"
    assert record["provisional_labels"] is True
    assert (record["examples"], record["with_image"]) == (8, 4)
    assert set(record["versions"]) == {"torch", "transformers", "peft", "trl"}
    assert json.loads((tmp_path / train.TRAIN_IDS_FILE).read_text()) == ["A"]


def test_no_accuracy_is_recorded_or_printed_during_training(processor, rows, tmp_path, capsys):
    run_training(rows * 4, processor, FAST, SILVER, tmp_path)
    printed = capsys.readouterr().out
    assert "loss" in printed
    assert "accuracy" not in printed
    assert "accuracy" not in (tmp_path / train.RUN_FILE).read_text()


def test_development_runs_on_cpu_by_default(tmp_path):
    arguments = training_arguments(TrainSettings(), tmp_path)
    assert arguments.use_cpu is True
    assert arguments.bf16 is False
    assert arguments.gradient_checkpointing is False


def test_the_gpu_flag_turns_on_the_gpu_settings(tmp_path):
    arguments = training_arguments(TrainSettings(gpu=True), tmp_path)
    assert arguments.use_cpu is False
    assert arguments.bf16 is True
    assert arguments.gradient_checkpointing is True


def test_gradient_checkpointing_still_trains_the_adapter(processor, rows, tmp_path, monkeypatch):
    """The GPU setting most likely to break silently, checked here rather than on a rented GPU."""
    original = train.training_arguments

    def with_checkpointing(settings, out_dir, device):
        arguments = original(settings, out_dir, device)
        arguments.gradient_checkpointing = True
        return arguments

    monkeypatch.setattr(train, "training_arguments", with_checkpointing)
    run_training(rows * 4, processor, FAST, SILVER, tmp_path)
    weights = load_file(tmp_path / train.ADAPTER_DIR / "adapter_model.safetensors")
    # The second half of each adapter starts at zero, so a nonzero value means gradients arrived.
    lora_b = [value for name, value in weights.items() if "lora_B" in name]
    assert lora_b and all(value.abs().sum() > 0 for value in lora_b)


def test_real_weights_load_only_behind_the_flag(processor, monkeypatch):
    def real(device):
        return f"real weights on {device}"

    aligned = []
    monkeypatch.setattr(train, "real_model", real)
    monkeypatch.setattr(train, "accelerator", lambda: "mps")
    monkeypatch.setattr(train, "align_end_tokens", lambda model, _tokenizer: aligned.append(model))
    assert base_model(processor, True, 1) == "real weights on mps"
    assert aligned == ["real weights on mps"]
    assert not isinstance(base_model(processor, False, 1), str)


def test_the_gpu_flag_needs_a_gpu():
    with pytest.raises(RunRefused, match="no GPU"):
        check_settings(TrainSettings(gpu=True), PIPELINE, device=None)


def test_provisional_labels_do_not_get_a_gpu_run():
    with pytest.raises(RunRefused, match="provisional"):
        check_settings(TrainSettings(gpu=True), SILVER, device="mps")


def test_real_labels_on_a_gpu_are_allowed_and_cpu_needs_nothing():
    check_settings(TrainSettings(gpu=True), PIPELINE, device="cuda")
    check_settings(TrainSettings(gpu=True), PIPELINE, device="mps")
    check_settings(TrainSettings(), SILVER, device=None)


@pytest.fixture(scope="module")
def trained(processor, rows, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("runs") / "t"
    run_training(rows * 4, processor, FAST, SILVER, run_dir)
    return run_dir


def logits_of(model, batch):
    with torch.no_grad():
        return model(**{k: v for k, v in batch.items() if k != "labels"}).logits


def test_a_saved_run_reloads_to_the_same_model(processor, rows, trained):
    batch = FieldCollator(processor)(rows)
    first, _ = load_trained(trained, processor)
    second, _ = load_trained(trained, processor)
    assert torch.equal(logits_of(first, batch), logits_of(second, batch))


def test_the_adapter_changes_the_model(processor, rows, trained):
    batch = FieldCollator(processor)(rows)
    model, record = load_trained(trained, processor)
    plain = tiny_model(processor, record["seed"]).eval()
    assert not torch.equal(logits_of(model, batch), logits_of(plain, batch))


def test_generation_is_repeatable_and_gives_one_reply_per_request(processor, rows, trained):
    model, _ = load_trained(trained, processor)
    generate = model_generate(model, processor, batch_size=1)
    first = generate(rows)
    assert len(first) == 2
    assert all(isinstance(text, str) for text in first)
    assert first == generate(rows)


def test_batching_does_not_change_what_is_generated(processor, rows, trained):
    model, _ = load_trained(trained, processor)
    alone = model_generate(model, processor, batch_size=1)(rows)
    together = model_generate(model, processor, batch_size=2)(rows)
    assert alone == together


def test_every_field_is_asked_alone_and_in_the_full_schema():
    requests = evaluation_requests([ITEM], no_image)
    alone = [r["requested"] for r in requests if r["schema"] == ALONE]
    full = [r["requested"] for r in requests if r["schema"] == FULL]
    assert alone == [[CATEGORY], ["color"]]
    assert len(full) == 1
    assert {CATEGORY, "color"} < set(full[0])


def test_a_product_with_only_a_category_is_still_asked_under_two_schemas():
    requests = evaluation_requests([LabeledProduct(product(), TOTES, {})], no_image)
    assert len(requests) == 2
    assert all(CATEGORY in r["requested"] for r in requests)


def perfect_replies(item):
    def generate(requests):
        return [
            json.dumps({name: item.values.get(name) for name in r["requested"]}) for r in requests
        ]

    return generate


def record_for():
    return {"name": "t", "model": TINY, "label_source": "silver"}


def test_a_report_on_provisional_data_has_field_metrics_and_no_accuracy(trained):
    report = evaluate_run(trained, record_for(), perfect_replies(ITEM), [ITEM], no_image, "val")
    assert report["compliance"]["compliance_rate"] == 1.0
    assert report["invariance"]["invariance_rate"] == 1.0
    assert "gold" not in report
    assert "accuracy" not in json.dumps(report)
    assert (trained / "eval_val.json").exists()


def test_a_report_on_hand_labels_scores_accuracy(trained):
    gold_item = LabeledProduct(product("G"), TOTES, {"color": "Red"})
    labels = {"G": {"path": TOTES, "attributes": {"color": "Red"}}}
    report = evaluate_run(
        trained,
        record_for(),
        perfect_replies(gold_item),
        [gold_item],
        no_image,
        "gold",
        labels,
        [TOTES, "Apparel", "Apparel > Bags"],
    )
    assert report["gold"]["category"]["exact_leaf_accuracy"] == 1.0
    assert report["gold"]["attributes"]["accuracy"] == 1.0


def test_scoring_refuses_a_run_that_trained_on_a_hand_labeled_product(trained):
    labels = {"A": {"path": TOTES, "attributes": {}}}
    with pytest.raises(GoldOverlap):
        evaluate_run(trained, record_for(), perfect_replies(ITEM), [ITEM], no_image, "gold", labels)


def test_gold_scores_treat_an_unreadable_reply_as_no_prediction():
    rows = [{"product": "G", "requested": [CATEGORY], "schema": FULL, "output": "junk"}]
    labels = {"G": {"path": TOTES, "attributes": {"color": "Red"}}}
    scores = gold_scores(rows, labels, [TOTES])
    assert scores["category"]["exact_leaf_accuracy"] == 0.0
    assert scores["attributes"]["accuracy"] == 0.0


def test_run_requests_keeps_replies_with_their_requests():
    requests = evaluation_requests([ITEM], no_image)
    rows = run_requests(requests, lambda batch: [str(i) for i in range(len(batch))])
    assert [row["output"] for row in rows] == [str(i) for i in range(len(requests))]
    assert all(row["product"] == "A" for row in rows)


@pytest.fixture
def world(tmp_path):
    products = [product(f"P{i:03}") for i in range(200)]
    write_jsonl(tmp_path / "products.jsonl", products)
    write_jsonl(tmp_path / "gold_queue.jsonl", [{"parent_asin": "P000"}])
    labels = tmp_path / "silver.jsonl"
    write_jsonl(labels, [{"parent_asin": p["parent_asin"], "path": TOTES} for p in products])
    sources = {
        "silver": LabelSource("silver", labels, True),
        "pipeline": LabelSource("pipeline", tmp_path / "missing.jsonl", False),
    }
    return tmp_path, sources


def run_cli(world, *argv):
    root, sources = world
    return cli.main(
        [*argv, "--no-images", "--runs", str(root / "runs")],
        derived=root,
        labels_path=root / "gold.jsonl",
        sources=sources,
        processor_factory=tiny_processor,
        load_taxonomy=lambda: Taxonomy.from_release(toy_release()),
    )


def test_the_commands_train_then_evaluate(world, capsys):
    assert run_cli(world, "train", "--products", "4", "--max-steps", "2") == 0
    trained_out = capsys.readouterr().out
    assert "Tiny random weights" in trained_out
    assert "accuracy" not in trained_out
    assert run_cli(world, "evaluate", "--products", "1") == 0
    evaluated = capsys.readouterr().out
    assert "compliance_rate" in evaluated
    assert "Tiny random weights" in evaluated
    assert (world[0] / "runs" / "tiny" / "eval_val.json").exists()
    assert "gold" not in json.loads((world[0] / "runs" / "tiny" / "eval_val.json").read_text())


def test_training_never_uses_the_gold_pool(world):
    assert run_cli(world, "train", "--products", "50", "--max-steps", "1") == 0
    train_ids = json.loads((world[0] / "runs" / "tiny" / train.TRAIN_IDS_FILE).read_text())
    assert "P000" not in train_ids


def test_evaluating_on_gold_stops_without_hand_labels(world, capsys):
    assert run_cli(world, "train", "--products", "2", "--max-steps", "1") == 0
    capsys.readouterr()
    assert run_cli(world, "evaluate", "--on", "gold") == cli.EXIT_REFUSED
    assert "no hand labels" in capsys.readouterr().out


def test_evaluating_a_run_that_does_not_exist_stops(world, capsys):
    assert run_cli(world, "evaluate", "--name", "nope") == cli.EXIT_REFUSED
    assert "no run named" in capsys.readouterr().out


def test_the_gpu_flag_stops_without_a_gpu(world, capsys, monkeypatch):
    monkeypatch.setattr(cli, "accelerator", lambda: None)
    assert run_cli(world, "train", "--gpu", "--labels", "pipeline") == cli.EXIT_REFUSED
    assert "no GPU" in capsys.readouterr().out


def test_the_gpu_flag_stops_on_provisional_labels(world, capsys, monkeypatch):
    monkeypatch.setattr(cli, "accelerator", lambda: "mps")
    assert run_cli(world, "train", "--gpu") == cli.EXIT_REFUSED
    assert "provisional" in capsys.readouterr().out


def test_training_on_labels_that_do_not_exist_yet_stops(world, capsys):
    assert run_cli(world, "train", "--labels", "pipeline") == cli.EXIT_REFUSED
    assert "no pipeline labels" in capsys.readouterr().out


def test_replies_end_at_the_end_of_the_chat_turn(processor):
    from taxon.training.model import align_end_tokens

    model = tiny_model(processor, 1)
    model.generation_config.eos_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|endoftext|>"
    )
    align_end_tokens(model, processor.tokenizer)
    end_of_turn = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert end_of_turn in model.generation_config.eos_token_id
    assert model.config.get_text_config().eos_token_id == end_of_turn


# Picking from a shortlist


def test_with_shortlists_every_evaluation_request_lists_candidates_and_asks_for_the_category():
    requests = evaluation_requests([ITEM], no_image, {"A": [TOTES, "Home"]})
    assert [r["requested"] for r in requests if r["schema"] == ALONE] == [
        [CATEGORY],
        [CATEGORY, "color"],
    ]
    assert all(r["candidates"] == [TOTES, "Home"] for r in requests)


def test_a_none_pick_scores_as_no_answer():
    rows = [{"product": "A", "schema": FULL, "output": json.dumps({CATEGORY: "none"})}]
    scores = gold_scores(rows, {"A": {"path": TOTES, "attributes": {}}}, [TOTES])
    assert scores["category_predictions"] == {"A": None}


def test_a_shortlist_run_is_trained_and_evaluated_with_its_candidates(world, capsys):
    root, _ = world
    write_jsonl(
        root / "shortlists.jsonl",
        [{"parent_asin": f"P{i:03}", "shortlist": [TOTES, "Home", "Apparel"]} for i in range(200)],
    )
    assert run_cli(world, "train", "--products", "2", "--max-steps", "1", "--shortlist", "2") == 0
    record = json.loads((root / "runs" / "tiny" / "run.json").read_text())
    assert record["shortlist"] == 2
    assert run_cli(world, "evaluate", "--products", "1") == 0
    assert "compliance_rate" in capsys.readouterr().out

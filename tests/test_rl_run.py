"""Constrained decoding and GRPO on CPU, with tiny random weights and a byte-level vocabulary."""

import json

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from taxon.products import write_jsonl
from taxon.rl import cli, grpo
from taxon.rl.decoding import (
    CATEGORY_OPEN,
    TrieLogitsProcessor,
    category_trie,
    category_value,
    decoding_report,
    generate_one,
)
from taxon.rl.grpo import (
    DEPTH_FILE,
    DEPTH_PLOT,
    REWARD_NAMES,
    GrpoSettings,
    grpo_arguments,
    plot_depths,
    prompt_dataset,
    reward_functions,
    run_grpo,
    starting_model,
)
from taxon.rl.rewards import RewardWeights
from taxon.taxonomy import Taxonomy
from taxon.training.examples import (
    PIPELINE,
    SILVER,
    LabeledProduct,
    LabelSource,
    request_row,
    training_examples,
)
from taxon.training.fields import CATEGORY
from taxon.training.model import tiny_model
from taxon.training.train import ADAPTER_DIR, TrainSettings, run_training
from tests.conftest import toy_release
from tests.tiny_processor import tiny_processor

TOTES = "Apparel > Bags > Totes"
FAST = GrpoSettings(max_steps=2, learning_rate=1e-3, max_completion_length=16)


@pytest.fixture(scope="module")
def processor():
    return tiny_processor()


@pytest.fixture(scope="module")
def taxonomy():
    return Taxonomy.from_release(toy_release())


@pytest.fixture(scope="module")
def trie(processor, taxonomy):
    return category_trie(processor.tokenizer, taxonomy.paths())


def product(asin="A"):
    return {
        "parent_asin": asin,
        "title": f"Red tote {asin}",
        "features": [],
        "description": "",
        "details": {},
    }


def no_image(_product):
    return None


ITEMS = [LabeledProduct(product(f"A{i}"), TOTES, {"color": "Red"}) for i in range(8)]
GOLD = {item.asin: item.category for item in ITEMS}


def row(requested=(CATEGORY,), item=ITEMS[0]):
    return request_row(item, list(requested), no_image)


def test_a_constrained_reply_opens_with_the_category(processor, taxonomy, trie):
    model = tiny_model(processor, 1).eval()
    text = generate_one(model, processor, row(), trie, 60)
    assert text.startswith(CATEGORY_OPEN)
    assert category_value(text) in taxonomy


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_constrained_category_is_a_real_path(processor, taxonomy, trie, seed):
    model = tiny_model(processor, seed).eval()
    report = decoding_report(model, processor, [row(), row(item=ITEMS[1])], taxonomy, trie, 60)
    assert report["valid_category_rate"] == 1.0


def test_free_decoding_from_random_weights_finds_no_real_path(processor, taxonomy):
    model = tiny_model(processor, 1).eval()
    report = decoding_report(model, processor, [row()], taxonomy, None, 30)
    assert report["constrained"] is False
    assert report["valid_category_rate"] == 0.0


def test_the_constraint_stays_off_when_category_is_not_asked(processor, trie):
    model = tiny_model(processor, 1).eval()
    assert not generate_one(model, processor, row(["color"]), trie, 20).startswith(CATEGORY_OPEN)


def test_the_report_times_each_reply(processor, taxonomy, trie):
    model = tiny_model(processor, 1).eval()
    report = decoding_report(model, processor, [row()] * 3, taxonomy, trie, 20)
    assert 0 < report["latency_p50_s"] <= report["latency_p95_s"]
    assert report["replies"] == 3


def allowed_now(processor, trie, generated_text, prompt=(5, 6)):
    """The tokens the processor allows next, after `generated_text`."""
    tokenizer = processor.tokenizer
    constraint = TrieLogitsProcessor(tokenizer, trie, len(prompt))
    ids = list(prompt)
    scores = constraint(torch.tensor([ids]), torch.zeros(1, len(tokenizer)))
    for token in tokenizer.encode(generated_text, add_special_tokens=False):
        ids.append(token)
        scores = constraint(torch.tensor([ids]), torch.zeros(1, len(tokenizer)))
    return {tokenizer.decode([i]) for i in torch.nonzero(scores[0] == 0).ravel().tolist()}


def test_only_the_opening_is_allowed_at_first(processor, trie):
    assert allowed_now(processor, trie, "") == {"{"}
    assert allowed_now(processor, trie, '{"cat') == {"e"}


def test_inside_the_value_only_continuations_of_real_paths_are_allowed(processor, trie):
    assert allowed_now(processor, trie, CATEGORY_OPEN) == {"A", "H"}
    assert allowed_now(processor, trie, CATEGORY_OPEN + "Apparel > Bag") == {"s", " "}


def test_the_value_may_close_only_where_a_path_ends(processor, trie):
    assert '"' not in allowed_now(processor, trie, CATEGORY_OPEN + "Appar")
    assert '"' in allowed_now(processor, trie, CATEGORY_OPEN + "Apparel")
    assert '"' in allowed_now(processor, trie, CATEGORY_OPEN + TOTES)


def test_after_the_closing_quote_generation_is_free(processor, trie):
    allowed = allowed_now(processor, trie, CATEGORY_OPEN + TOTES + '"')
    assert {"}", ",", "x", '"', "<|im_end|>"} <= allowed


def test_rows_in_a_batch_are_constrained_separately(processor, trie):
    tokenizer = processor.tokenizer
    constraint = TrieLogitsProcessor(tokenizer, trie, 1)
    first = tokenizer.encode(CATEGORY_OPEN + "A", add_special_tokens=False)
    second = tokenizer.encode(CATEGORY_OPEN + "H", add_special_tokens=False)
    batch = [[0], [0]]
    constraint(torch.tensor(batch), torch.zeros(2, len(tokenizer)))
    for a, b in zip(first, second, strict=True):
        batch = [batch[0] + [a], batch[1] + [b]]
        scores = constraint(torch.tensor(batch), torch.zeros(2, len(tokenizer)))
    nxt = [{tokenizer.decode([i]) for i in torch.nonzero(s == 0).ravel().tolist()} for s in scores]
    assert nxt == [{"p"}, {"o"}]


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"category": "Home"}', "Home"),
        ('{"category": "Home""""', "Home"),
        ('{"category": "Hom', None),
        ('{"color": "Red"}', None),
    ],
)
def test_category_value_reads_even_broken_replies(text, expected):
    assert category_value(text) == expected


def completions(*texts):
    return [[{"role": "assistant", "content": t}] for t in texts]


def test_reward_functions_follow_trl_s_shape_and_order(taxonomy):
    log = []
    funcs = reward_functions(taxonomy, log)
    assert [f.__name__ for f in funcs] == list(REWARD_NAMES)
    texts = completions(json.dumps({CATEGORY: TOTES}), json.dumps({CATEGORY: "Apparel"}), "junk")
    kwargs = {"requested": [[CATEGORY]] * 3, "gold_category": [TOTES] * 3}
    by_name = {f.__name__: f(texts, **kwargs) for f in funcs}
    assert by_name["compliance"] == [1.0, 1.0, 0.0]
    assert by_name["path_match"][0] == 1.0
    assert by_name["depth_penalty"][1] < 0
    assert log[-1]["mean_predicted_depth"] == 2
    assert log[-1]["mean_gold_depth"] == 3
    assert (log[-1]["valid_categories"], log[-1]["asked"]) == (2, 3)


def test_the_weights_follow_the_reward_order():
    assert len(RewardWeights().as_list()) == len(REWARD_NAMES)


def test_prompts_carry_the_labeled_category_even_when_it_is_not_requested(taxonomy):
    rows = [row(["color"])]
    dataset = prompt_dataset(rows, GOLD)
    assert dataset[0]["gold_category"] == TOTES
    assert dataset[0]["requested"] == ["color"]
    assert dataset[0]["prompt"][1]["content"].startswith("<product>")
    assert "image" not in dataset.column_names


def test_prompts_carry_images_when_every_row_has_one(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (9, 9, 9)).save(path)
    rows = [request_row(ITEMS[0], [CATEGORY], lambda _p: str(path))]
    dataset = prompt_dataset(rows, GOLD)
    assert dataset[0]["prompt"][1]["content"][0]["type"] == "image"
    assert "image" in dataset.column_names


def test_the_depth_plot_is_written_even_with_gaps(tmp_path):
    log = [
        {"step": 0, "mean_predicted_depth": None, "mean_gold_depth": 3},
        {"step": 1, "mean_predicted_depth": 2.5, "mean_gold_depth": 3},
    ]
    plot_depths(log, tmp_path / "depth.png")
    assert (tmp_path / "depth.png").read_bytes()[:4] == b"\x89PNG"


def test_development_runs_on_cpu_and_the_gpu_flag_turns_on_vllm(tmp_path):
    cpu = grpo_arguments(GrpoSettings(), tmp_path, RewardWeights())
    assert (cpu.use_cpu, cpu.use_vllm, cpu.bf16) == (True, False, False)
    assert cpu.chat_template_kwargs == {"enable_thinking": False}
    assert cpu.reward_weights == RewardWeights().as_list()
    gpu = grpo_arguments(GrpoSettings(gpu=True), tmp_path, RewardWeights())
    assert (gpu.use_cpu, gpu.use_vllm, gpu.bf16, gpu.gradient_checkpointing) == (
        False,
        True,
        True,
        True,
    )


def rl_rows(taxonomy):
    return training_examples(ITEMS, taxonomy, 7, no_image)


def test_a_grpo_run_saves_its_adapter_depths_and_plot(processor, taxonomy, tmp_path, capsys):
    record = run_grpo(rl_rows(taxonomy), processor, taxonomy, FAST, SILVER, tmp_path, GOLD)
    assert (tmp_path / ADAPTER_DIR / "adapter_model.safetensors").exists()
    assert json.loads((tmp_path / DEPTH_FILE).read_text())
    assert (tmp_path / DEPTH_PLOT).exists()
    assert record["provisional_labels"] is True
    assert len(record["losses"]) == 2
    printed = capsys.readouterr().out
    assert "accuracy" not in printed and "reward" not in printed
    assert "accuracy" not in (tmp_path / grpo.RUN_FILE).read_text()


def longer_is_better(completions, **_kwargs):
    return [float(len(c[0]["content"])) for c in completions]


def test_gradients_reach_the_adapter_when_rewards_vary(processor, taxonomy, tmp_path):
    settings = GrpoSettings(max_steps=2, learning_rate=1e-2, max_completion_length=16)
    run_grpo(
        rl_rows(taxonomy),
        processor,
        taxonomy,
        settings,
        SILVER,
        tmp_path,
        GOLD,
        extra_rewards=[longer_is_better],
    )
    weights = load_file(tmp_path / ADAPTER_DIR / "adapter_model.safetensors")
    lora_b = [value for name, value in weights.items() if "lora_B" in name]
    assert lora_b and any(value.abs().sum() > 0 for value in lora_b)


def test_grpo_starts_from_the_fine_tuned_model(processor, taxonomy, tmp_path):
    rows = rl_rows(taxonomy)
    run_training(
        rows,
        processor,
        TrainSettings(name="t", max_steps=2, learning_rate=5e-3),
        SILVER,
        tmp_path / "sft",
    )
    merged = starting_model(processor, FAST, tmp_path / "sft").eval()
    plain = starting_model(processor, FAST, None).eval()
    batch = processor(text=["hello"], return_tensors="pt")
    with torch.no_grad():
        assert not torch.equal(merged(**batch).logits, plain(**batch).logits)


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
        [*argv, "--runs", str(root / "rl")],
        derived=root,
        sources=sources,
        processor_factory=tiny_processor,
        load_taxonomy=lambda: Taxonomy.from_release(toy_release()),
    )


def test_the_commands_train_then_compare_decoding(world, capsys):
    assert (
        run_cli(
            world, "train", "--products", "4", "--max-steps", "1", "--max-completion-length", "8"
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Tiny random weights" in out and "accuracy" not in out
    assert run_cli(world, "decode", "--products", "2") == 0
    reports = json.loads((world[0] / "rl" / "grpo-tiny" / "decoding.json").read_text())
    assert [r["constrained"] for r in reports] == [False, True]
    assert reports[1]["valid_category_rate"] == 1.0


def test_grpo_needs_the_fine_tuning_run_it_names(world, capsys):
    assert run_cli(world, "train", "--from", "nope") == cli.EXIT_REFUSED
    assert "no fine-tuning run named nope" in capsys.readouterr().out


def test_the_gpu_flag_stops_without_a_gpu(world, capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run_cli(world, "train", "--gpu", "--labels", "pipeline") == cli.EXIT_REFUSED
    assert "no GPU" in capsys.readouterr().out


def test_the_gpu_flag_stops_on_provisional_labels(world, capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert run_cli(world, "train", "--gpu") == cli.EXIT_REFUSED
    assert "provisional" in capsys.readouterr().out


def test_pipeline_labels_are_not_provisional():
    assert PIPELINE.provisional is False


def test_image_markers_sit_on_the_image_tokens_in_grpo(processor, taxonomy, tmp_path, monkeypatch):
    """Prompts of very different lengths need lots of padding, which once moved the markers."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    original = modeling_qwen3_5.Qwen3_5Model.get_rope_index
    checked = []

    def get_rope_index(self, input_ids, mm_token_type_ids, *args, **kwargs):
        image = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert torch.equal(input_ids == image, mm_token_type_ids == 1)
        checked.append(input_ids.shape[0])
        return original(self, input_ids, mm_token_type_ids, *args, **kwargs)

    monkeypatch.setattr(modeling_qwen3_5.Qwen3_5Model, "get_rope_index", get_rope_index)
    path = tmp_path / "a.jpg"
    Image.new("RGB", (48, 32), (9, 9, 9)).save(path)
    items = [
        LabeledProduct({**product(f"L{i}"), "title": "tote " + "long words " * (40 * i)}, TOTES, {})
        for i in range(4)
    ]
    rows = training_examples(items, taxonomy, 7, lambda _p: str(path))
    gold = {item.asin: item.category for item in items}
    two_prompts_a_step = GrpoSettings(max_steps=1, batch_size=8, max_completion_length=8)
    run_grpo(rows, processor, taxonomy, two_prompts_a_step, SILVER, tmp_path / "run", gold)
    assert checked


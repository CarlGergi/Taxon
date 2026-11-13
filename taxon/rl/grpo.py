"""GRPO after fine-tuning, with TRL. Rewards come from the taxonomy (taxon.rl.rewards).

The default is CPU and tiny random weights. A GPU run takes --gpu, uses vLLM for rollouts, and
refuses provisional labels, like fine-tuning does.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from datasets import Dataset
from peft import PeftModel
from PIL import Image
from transformers import PrinterCallback, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from taxon.hardware import hardware
from taxon.pins import MODEL, MODEL_REVISION, SEED
from taxon.rl.rewards import RewardWeights, predicted_category, score
from taxon.taxonomy import Taxonomy, split_path
from taxon.training.examples import NONE, LabelSource
from taxon.training.fields import CATEGORY, target_text
from taxon.training.model import LORA_RANK, TINY, lora_config
from taxon.training.train import ADAPTER_DIR, base_model, run_device

RUN_FILE = "run.json"
DEPTH_FILE = "depth.json"
DEPTH_PLOT = "depth.png"
REWARD_NAMES = ("compliance", "valid_node", "path_match", "allowed_values", "depth_penalty")
_PACKAGES = ("torch", "transformers", "peft", "trl")


@dataclass(frozen=True)
class GrpoSettings:
    name: str = "grpo-tiny"
    gpu: bool = False
    max_steps: int = 4
    batch_size: int = 4
    num_generations: int = 4
    max_completion_length: int = 64
    learning_rate: float = 1e-5
    beta: float = 0.04
    lora_rank: int = LORA_RANK
    seed: int = SEED
    # A checkpoint every this many steps, so a stopped run can resume. 0 saves none.
    save_steps: int = 50
    # Candidate paths per request, from the fine-tuning run it starts from; 0 for none.
    shortlist: int = 0


# Checkpoints kept on disk; older ones are deleted as new ones are written.
CHECKPOINTS_KEPT = 2


def _message(content_parts: Sequence[Mapping], with_image: bool, as_parts: bool = False):
    """Plain text, or a list of parts when the prompt holds images, since every message in a column
    must have the same shape."""
    text = "".join(part["text"] for part in content_parts if part["type"] == "text")
    parts = [{"type": "text", "text": text}]
    if with_image:
        return [{"type": "image"}, *parts]
    return parts if as_parts else text


def prompt_dataset(rows: Sequence[Mapping], gold_paths: Mapping[str, str]) -> Dataset:
    """Rows from taxon.training.examples, as TRL prompts, with each product's labeled category.
    Either every row has an image or none does, since a column cannot mix the two message
    shapes."""
    with_images = all(row["image_path"] for row in rows)
    records = []
    for row in rows:
        system, user = row["messages"]
        record = {
            "prompt": [
                {"role": "system", "content": _message(system["content"], False, with_images)},
                {"role": "user", "content": _message(user["content"], with_images)},
            ],
            "requested": list(row["requested"]),
            "gold_category": gold_paths[row["asin"]],
        }
        if with_images:
            with Image.open(row["image_path"]) as image:
                record["image"] = image.convert("RGB")
        records.append(record)
    return Dataset.from_list(records)


def _text(completion) -> str:
    return completion[0]["content"] if isinstance(completion, list) else completion


def reward_functions(taxonomy: Taxonomy, depth_log: list[dict]) -> list[Callable]:
    """One TRL reward function per reward, in REWARD_NAMES order. The depth penalty's function
    also records predicted and gold depths at each step, for the plot."""

    def make(name: str) -> Callable:
        def reward(completions, requested, gold_category, trainer_state=None, **_kwargs):
            texts = [_text(c) for c in completions]
            parts = [
                score(t, r, {CATEGORY: g}, taxonomy)
                for t, r, g in zip(texts, requested, gold_category, strict=True)
            ]
            if name == "depth_penalty":
                _record_depths(texts, requested, gold_category, taxonomy, trainer_state, depth_log)
            return [p[name] for p in parts]

        reward.__name__ = name
        return reward

    return [make(name) for name in REWARD_NAMES]


def _record_depths(texts, requested, golds, taxonomy, state, depth_log) -> None:
    predicted, gold = [], []
    for text, asked, path in zip(texts, requested, golds, strict=True):
        if CATEGORY not in asked or path == NONE:
            continue
        gold.append(len(split_path(path)))
        category = predicted_category(text)
        if category in taxonomy:
            predicted.append(len(split_path(category)))
    depth_log.append(
        {
            "step": state.global_step if state is not None else len(depth_log),
            "mean_predicted_depth": mean(predicted) if predicted else None,
            "mean_gold_depth": mean(gold) if gold else None,
            "valid_categories": len(predicted),
            "asked": len(gold),
        }
    )


def plot_depths(depth_log: Sequence[Mapping], path: Path) -> None:
    """Predicted against gold depth over training. A predicted line sinking below gold is the
    model retreating to shallow, safe categories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in depth_log]
    figure, axis = plt.subplots(figsize=(6, 3.5))
    for key, label in (("mean_predicted_depth", "predicted"), ("mean_gold_depth", "gold")):
        axis.plot(steps, [row[key] for row in depth_log], marker="o", label=label)
    axis.set_xlabel("training step")
    axis.set_ylabel("mean category depth")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


class StepPrinter(TrainerCallback):
    """Print only the step and loss. Reward means are measured against the training labels, and
    must not read as results when those labels are provisional."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step {state.global_step}: loss {float(logs['loss']):.4f}", flush=True)


def grpo_arguments(
    settings: GrpoSettings, out_dir: Path, weights: RewardWeights, device: str = "cuda"
) -> GRPOConfig:
    """vLLM rollouts need CUDA. On Apple's GPU, and on CPU, replies come from plain generation."""
    return GRPOConfig(
        output_dir=str(out_dir),
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.batch_size,
        num_generations=settings.num_generations,
        max_completion_length=settings.max_completion_length,
        learning_rate=settings.learning_rate,
        beta=settings.beta,
        reward_weights=weights.as_list(),
        chat_template_kwargs={"enable_thinking": False},
        seed=settings.seed,
        data_seed=settings.seed,
        use_cpu=not settings.gpu,
        bf16=settings.gpu,
        gradient_checkpointing=settings.gpu,
        use_vllm=settings.gpu and device == "cuda",
        vllm_mode="colocate",
        logging_steps=1,
        save_strategy="steps" if settings.save_steps else "no",
        save_steps=settings.save_steps or 500,
        save_total_limit=CHECKPOINTS_KEPT,
        report_to="none",
        disable_tqdm=True,
    )


def pick_length(tokenizer, rows: Sequence[Mapping]) -> int:
    """Tokens in the longest one-pick reply any row allows, plus the end token."""
    replies = {
        target_text([CATEGORY], {CATEGORY: pick})
        for row in rows
        for pick in [*row["candidates"], NONE]
    }
    return 1 + max(len(tokenizer.encode(r, add_special_tokens=False)) for r in replies)


def latest_checkpoint(out_dir: Path) -> Path | None:
    """The run's newest checkpoint folder, by step, or None."""
    found = [p for p in out_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    return max(found, key=lambda p: int(p.name.split("-")[-1]), default=None)


def starting_model(processor, settings: GrpoSettings, sft_dir: Path | None):
    """The fine-tuned model with its adapter merged in, so GRPO adds a fresh adapter on top."""
    base = base_model(processor, settings.gpu, settings.seed)
    if sft_dir is None:
        return base
    return PeftModel.from_pretrained(base, sft_dir / ADAPTER_DIR).merge_and_unload()


def run_grpo(
    rows: Sequence[Mapping],
    processor,
    taxonomy: Taxonomy,
    settings: GrpoSettings,
    source: LabelSource,
    out_dir: Path,
    gold_paths: Mapping[str, str],
    sft_dir: Path | None = None,
    weights: RewardWeights | None = None,
    extra_rewards: Sequence[Callable] = (),
    callbacks: Sequence[TrainerCallback] = (),
    resume: bool = False,
) -> dict:
    """Train with GRPO and record how. The record holds losses and depths, never an accuracy.

    With `resume`, training continues from the run's latest checkpoint when it has one. The depth
    log then covers only the steps after it, and the record says where it resumed.
    """
    weights = weights or RewardWeights()
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(settings.seed)
    device = run_device(settings.gpu)
    model = starting_model(processor, settings, sft_dir)
    depth_log: list[dict] = []
    # TRL generates from left-padded prompts, then processes the prompts again to mark which tokens
    # are image tokens, padding with the tokenizer's own side. Unless that side is also left, image
    # markers land beside the image tokens instead of on them: position ids go silently wrong, or
    # training crashes once the shift crosses the padding.
    processor.tokenizer.padding_side = "left"
    funcs = [*reward_functions(taxonomy, depth_log), *extra_rewards]
    arguments = grpo_arguments(settings, out_dir, weights, device)
    arguments.reward_weights = [*weights.as_list(), *[1.0] * len(extra_rewards)]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=funcs,
        args=arguments,
        train_dataset=prompt_dataset(rows, gold_paths),
        processing_class=processor,
        peft_config=lora_config(settings.lora_rank),
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(StepPrinter())
    for callback in callbacks:
        trainer.add_callback(callback)
    checkpoint = latest_checkpoint(out_dir) if resume else None
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.model.save_pretrained(out_dir / ADAPTER_DIR)
    (out_dir / DEPTH_FILE).write_text(json.dumps(depth_log, indent=2), encoding="utf-8")
    plot_depths(depth_log, out_dir / DEPTH_PLOT)
    record = {
        **asdict(settings),
        "hardware": hardware(device),
        "rollouts": "vllm" if arguments.use_vllm else "generate",
        "model": f"{MODEL}@{MODEL_REVISION}" if settings.gpu else TINY,
        "fine_tuned_from": str(sft_dir) if sft_dir else None,
        "resumed_from": checkpoint.name if checkpoint else None,
        "label_source": source.name,
        "provisional_labels": source.provisional,
        "prompts": len(rows),
        "reward_weights": dict(zip(REWARD_NAMES, weights.as_list(), strict=True)),
        "losses": [e["loss"] for e in trainer.state.log_history if "loss" in e],
        "versions": {name: importlib.metadata.version(name) for name in _PACKAGES},
    }
    (out_dir / RUN_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record

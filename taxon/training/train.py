"""Fine-tune with LoRA. The default is CPU and tiny random weights, and a GPU takes a flag."""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Dataset
from peft import get_peft_model
from transformers import PrinterCallback, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from taxon.hardware import accelerator, hardware
from taxon.pins import MODEL, MODEL_REVISION, SEED
from taxon.training.collate import FieldCollator
from taxon.training.examples import LabelSource
from taxon.training.model import (
    LORA_RANK,
    TINY,
    align_end_tokens,
    lora_config,
    real_model,
    tiny_model,
)

RUN_FILE = "run.json"
TRAIN_IDS_FILE = "train_ids.json"
ADAPTER_DIR = "adapter"
_PACKAGES = ("torch", "transformers", "peft", "trl")


class RunRefused(ValueError):
    """These settings must not run."""


@dataclass(frozen=True)
class TrainSettings:
    name: str = "tiny"
    gpu: bool = False
    max_steps: int = 10
    batch_size: int = 2
    learning_rate: float = 2e-4
    lora_rank: int = LORA_RANK
    seed: int = SEED


def check_settings(settings, source: LabelSource, device: str | None) -> None:
    """A real run is for real labels. Provisional ones are only for testing the pipeline.

    `device` is the accelerator found here: "cuda", "mps", or None.
    """
    if not settings.gpu:
        return
    if device is None:
        raise RunRefused("--gpu was given, but no GPU is available, NVIDIA or Apple")
    if source.provisional:
        raise RunRefused(f"{source.name} labels are provisional, so they do not get a GPU run")


class LossPrinter(TrainerCallback):
    """Print only the loss. The trainer's own log also shows token accuracy on the training labels,
    which must not read as a result when those labels are provisional."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"step {state.global_step}: loss {float(logs['loss']):.3f}", flush=True)


def training_arguments(settings: TrainSettings, out_dir: Path, device: str = "cuda") -> SFTConfig:
    return SFTConfig(
        output_dir=str(out_dir),
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        seed=settings.seed,
        data_seed=settings.seed,
        use_cpu=not settings.gpu,
        bf16=settings.gpu,
        gradient_checkpointing=settings.gpu,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_pin_memory=settings.gpu and device == "cuda",
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def fine_tune(
    rows: Sequence[dict],
    processor,
    model,
    settings: TrainSettings,
    out_dir: Path,
    device: str,
    callbacks: Sequence[TrainerCallback] = (),
):
    """Train an adapter and save it under `out_dir`. Returns the loss at each step."""
    set_seed(settings.seed)
    peft_model = get_peft_model(model, lora_config(settings.lora_rank))
    collate = FieldCollator(processor)
    # The dataset holds only row numbers. Arrow would pad each message part with the keys of the
    # others, which changes what a chat template sees.
    trainer = SFTTrainer(
        model=peft_model,
        args=training_arguments(settings, out_dir, device),
        train_dataset=Dataset.from_dict({"row": list(range(len(rows)))}),
        data_collator=lambda batch: collate([rows[item["row"]] for item in batch]),
        processing_class=processor,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LossPrinter())
    for callback in callbacks:
        trainer.add_callback(callback)
    trainer.train()
    peft_model.save_pretrained(out_dir / ADAPTER_DIR)
    return [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]


def run_device(gpu: bool) -> str:
    """Where a run trains: the accelerator here for a real run, the CPU otherwise."""
    return (accelerator() or "cpu") if gpu else "cpu"


def base_model(processor, gpu: bool, seed: int):
    if not gpu:
        return tiny_model(processor, seed)
    model = real_model(run_device(True))
    align_end_tokens(model, processor.tokenizer)
    return model


def run_training(
    rows: Sequence[dict],
    processor,
    settings: TrainSettings,
    source: LabelSource,
    out_dir: Path,
    model=None,
    callbacks: Sequence[TrainerCallback] = (),
) -> dict:
    """Train and record how the run was made. The record holds losses, and never any accuracy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = run_device(settings.gpu)
    model = base_model(processor, settings.gpu, settings.seed) if model is None else model
    losses = fine_tune(rows, processor, model, settings, out_dir, device, callbacks)
    record = {
        **asdict(settings),
        "hardware": hardware(device),
        "model": f"{MODEL}@{MODEL_REVISION}" if settings.gpu else TINY,
        "label_source": source.name,
        "provisional_labels": source.provisional,
        "examples": len(rows),
        "with_image": sum(bool(row["image_path"]) for row in rows),
        "losses": losses,
        "versions": {name: importlib.metadata.version(name) for name in _PACKAGES},
    }
    (out_dir / RUN_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    train_ids = sorted({row["asin"] for row in rows})
    (out_dir / TRAIN_IDS_FILE).write_text(json.dumps(train_ids), encoding="utf-8")
    return record

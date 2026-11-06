"""Build training batches: the model is scored only on the reply, never on the prompt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from PIL import Image

END_OF_TURN = "<|im_end|>"
IGNORED = -100


def prompt_text(processor, messages: Sequence[Mapping]) -> str:
    """The prompt, ending where the reply starts, with thinking turned off."""
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def load_images(rows: Sequence[Mapping]) -> list[Image.Image] | None:
    """Every row's image in order. Rows without one are skipped, and None means no images at all."""
    images = []
    for row in rows:
        if row["image_path"]:
            with Image.open(row["image_path"]) as image:
                images.append(image.convert("RGB"))
    return images or None


class FieldCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, rows: Sequence[Mapping]) -> dict:
        replies = [f"{row['target']}{END_OF_TURN}" for row in rows]
        texts = [
            prompt_text(self.processor, row["messages"]) + reply
            for row, reply in zip(rows, replies, strict=True)
        ]
        batch = self.processor(
            text=texts,
            images=load_images(rows),
            padding=True,
            padding_side="right",
            add_special_tokens=False,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], IGNORED)
        reply_ids = self.processor.tokenizer(replies, add_special_tokens=False)["input_ids"]
        for i, ids in enumerate(reply_ids):
            end = int(batch["attention_mask"][i].sum())
            labels[i, end - len(ids) : end] = batch["input_ids"][i, end - len(ids) : end]
        batch["labels"] = labels
        return batch

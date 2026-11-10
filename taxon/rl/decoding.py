"""Trie-constrained decoding: every requested category is a real taxonomy path by construction.

Replies list category first (see taxon.training.fields.ordered), so when category is requested the
reply is made to open with CATEGORY_OPEN, then only tokens that continue some valid path are
allowed, and the value may close only where a path ends. After that, generation is free. A path's
tokens there are its standalone tokens: the opening quote merges with the space before it, never
with the path's first letter. A data test checks this on the real tokenizer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from functools import lru_cache
from statistics import median, quantiles

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from eval.fields import parse_fields
from taxon.taxonomy import Taxonomy
from taxon.training.collate import load_images, prompt_text
from taxon.training.fields import CATEGORY
from taxon.trie import ROOT, TokenTrie

CATEGORY_OPEN = '{"category": "'


def category_trie(tokenizer, paths: Sequence[str]) -> TokenTrie:
    return TokenTrie.build(
        paths,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids),
    )


@lru_cache(maxsize=4)
def _closers(tokenizer) -> tuple[int, ...]:
    """Tokens that end a JSON string value: those whose text starts with a quote."""
    return tuple(i for i in range(len(tokenizer)) if tokenizer.decode([i]).startswith('"'))


class TrieLogitsProcessor(LogitsProcessor):
    """Forces the category opening, then keeps each row's value inside the trie. Rows are tracked
    separately, each as ("open", tokens forced so far), ("path", trie state) or ("free", None)."""

    def __init__(self, tokenizer, trie: TokenTrie, prompt_length: int):
        self.trie = trie
        self.prompt_length = prompt_length
        self.opening = tokenizer.encode(CATEGORY_OPEN, add_special_tokens=False)
        self.closers = torch.tensor(_closers(tokenizer))
        self.states: list[tuple[str, int | None]] = []

    def _advance(self, row: int, token: int) -> None:
        phase, position = self.states[row]
        if phase == "open":
            position += 1
            self.states[row] = (
                ("open", position) if position < len(self.opening) else ("path", ROOT)
            )
        elif phase == "path":
            following = self.trie.step(position, token)
            # Anything else allowed in this phase is a closing quote after a complete path.
            self.states[row] = ("path", following) if following is not None else ("free", None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if not self.states:
            self.states = [("open", 0)] * input_ids.shape[0]
        elif input_ids.shape[1] > self.prompt_length:
            for row in range(input_ids.shape[0]):
                self._advance(row, int(input_ids[row, -1]))
        for row, (phase, position) in enumerate(self.states):
            if phase == "free":
                continue
            if phase == "open":
                allowed = torch.tensor([self.opening[position]])
            else:
                allowed = torch.tensor(self.trie.allowed(position), dtype=torch.long)
                if self.trie.terminal(position) is not None:
                    allowed = torch.cat([allowed, self.closers])
            mask = torch.full_like(scores[row], float("-inf"))
            mask[allowed] = 0.0
            scores[row] = scores[row] + mask
        return scores


def category_value(text: str) -> str | None:
    """The category a reply gives, read even when the rest of the reply is not valid JSON."""
    value = (parse_fields(text) or {}).get(CATEGORY)
    if isinstance(value, str):
        return value
    if text.startswith(CATEGORY_OPEN):
        rest = text[len(CATEGORY_OPEN) :]
        return rest.split('"', 1)[0] if '"' in rest else None
    return None


def generate_one(model, processor, row: Mapping, trie: TokenTrie | None, max_new_tokens: int):
    """One greedy reply. The constraint applies only when category is requested."""
    inputs = processor(
        text=[prompt_text(processor, row["messages"])],
        images=load_images([row]),
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    length = inputs["input_ids"].shape[1]
    kwargs = {}
    if trie is not None and CATEGORY in row["requested"]:
        constraint = TrieLogitsProcessor(processor.tokenizer, trie, length)
        kwargs["logits_processor"] = LogitsProcessorList([constraint])
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kwargs)
    return processor.tokenizer.decode(ids[0, length:], skip_special_tokens=True)


def decoding_report(
    model,
    processor,
    rows: Sequence[Mapping],
    taxonomy: Taxonomy,
    trie: TokenTrie | None,
    max_new_tokens: int = 128,
) -> dict:
    """Latency per reply, and how often a requested category is a real path.

    One request at a time, so latency is per reply. A constrained reply can still be invalid if it
    runs out of tokens mid-path. Accuracy needs hand labels, so it is not here.
    """
    latencies, valid, asked, parsed = [], 0, 0, 0
    for row in rows:
        start = time.perf_counter()
        text = generate_one(model, processor, row, trie, max_new_tokens)
        latencies.append(time.perf_counter() - start)
        parsed += parse_fields(text) is not None
        if CATEGORY in row["requested"]:
            asked += 1
            valid += category_value(text) in taxonomy
    cut = quantiles(latencies, n=20) if len(latencies) > 1 else latencies * 19
    return {
        "constrained": trie is not None,
        "replies": len(rows),
        "category_asked": asked,
        "valid_category_rate": valid / asked if asked else None,
        "json_parse_rate": parsed / len(rows) if rows else None,
        "latency_p50_s": median(latencies),
        "latency_p95_s": cut[18],
    }

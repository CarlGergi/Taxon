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
from taxon.hardware import hardware
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


class TrieGuide:
    """Which tokens one reply may take next, whatever framework generates it.

    It forces the category opening, then keeps the value inside the trie, and lets it close only
    where a path ends; after that the reply is free. `allowed()` is None once it is free.
    """

    def __init__(self, opening: Sequence[int], trie: TokenTrie, closers: Sequence[int]):
        self.opening = list(opening)
        self.trie = trie
        self.closers = list(closers)
        self.phase, self.position = "open", 0

    def advance(self, token: int) -> None:
        if self.phase == "open":
            self.position += 1
            if self.position == len(self.opening):
                self.phase, self.position = "path", ROOT
        elif self.phase == "path":
            following = self.trie.step(self.position, token)
            # Anything else allowed in this phase is a closing quote after a complete path.
            if following is None:
                self.phase, self.position = "free", None
            else:
                self.position = following

    def allowed(self) -> list[int] | None:
        if self.phase == "free":
            return None
        if self.phase == "open":
            return [self.opening[self.position]]
        tokens = self.trie.allowed(self.position)
        if self.trie.terminal(self.position) is not None:
            tokens = tokens + self.closers
        return tokens


def trie_guide(tokenizer, trie: TokenTrie) -> TrieGuide:
    opening = tokenizer.encode(CATEGORY_OPEN, add_special_tokens=False)
    return TrieGuide(opening, trie, _closers(tokenizer))


class TrieLogitsProcessor(LogitsProcessor):
    """A TrieGuide per row of a batch, applied to PyTorch scores."""

    def __init__(self, tokenizer, trie: TokenTrie, prompt_length: int):
        self.tokenizer = tokenizer
        self.trie = trie
        self.prompt_length = prompt_length
        self.guides: list[TrieGuide] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if not self.guides:
            self.guides = [trie_guide(self.tokenizer, self.trie) for _ in range(input_ids.shape[0])]
        elif input_ids.shape[1] > self.prompt_length:
            for row, guide in enumerate(self.guides):
                guide.advance(int(input_ids[row, -1]))
        for row, guide in enumerate(self.guides):
            allowed = guide.allowed()
            if allowed is None:
                continue
            mask = torch.full_like(scores[row], float("-inf"))
            mask[torch.tensor(allowed, dtype=torch.long)] = 0.0
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
        "hardware": hardware(model.device.type),
        "constrained": trie is not None,
        "replies": len(rows),
        "category_asked": asked,
        "valid_category_rate": valid / asked if asked else None,
        "json_parse_rate": parsed / len(rows) if rows else None,
        "latency_p50_s": median(latencies),
        "latency_p95_s": cut[18],
    }

"""Batch prices in USD per million tokens, read from each vendor's pricing page on 2026-11-03.

https://platform.claude.com/docs/en/about-claude/pricing
https://developers.openai.com/api/docs/pricing
https://aistudio.google.com/docs/pricing
"""

from __future__ import annotations

from dataclasses import dataclass

from taxon.labeling.llm import LLMRequest, ModelConfig, Usage

MILLION = 1_000_000
# Deliberately high for the worst-case guard: real text runs near 4 characters per token, and a
# 512 px image costs under 1,000 tokens on every vendor.
_WORST_CASE_CHARS_PER_TOKEN = 2
_WORST_CASE_IMAGE_TOKENS = 1_600


@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    cache_write: float
    output: float


# Anthropic's cache multipliers stack on the batch discount; writes use the 1-hour rate. OpenAI
# bills cached writes at the input rate. Google bills cache hits at its standard caching rate.
PRICES = {
    ("anthropic", "claude-haiku-5-5"): Price(0.05, 0.005, 0.10, 0.25),
    ("anthropic", "claude-sonnet-5-5"): Price(1.00, 0.05, 2.00, 5.00),
    ("anthropic", "claude-opus-5-5"): Price(2.00, 0.10, 4.00, 10.00),
    ("openai", "gpt-6-luna"): Price(0.05, 0.005, 0.05, 0.25),
    ("google", "gemini-3.5-flash-lite"): Price(0.15, 0.03, 0.15, 1.25),
}


def price_for(config: ModelConfig) -> Price:
    """Raises for an unknown model, so a missing price can never count as free."""
    try:
        return PRICES[(config.vendor, config.model)]
    except KeyError:
        raise KeyError(f"no price recorded for {config.vendor} {config.model}") from None


def cost_of(usage: Usage, config: ModelConfig) -> float:
    price = price_for(config)
    return (
        usage.input_tokens * price.input
        + usage.cached_tokens * price.cached_input
        + usage.cache_write_tokens * price.cache_write
        + usage.output_tokens * price.output
    ) / MILLION


def worst_case_cost(request: LLMRequest) -> float:
    """An upper bound on what a request can cost: no cache hits and every output token used."""
    price = price_for(request.config)
    tokens = (len(request.system) + len(request.user)) / _WORST_CASE_CHARS_PER_TOKEN
    if request.image_path is not None:
        tokens += _WORST_CASE_IMAGE_TOKENS
    input_rate = max(price.input, price.cache_write)
    return (tokens * input_rate + request.config.max_tokens * price.output) / MILLION

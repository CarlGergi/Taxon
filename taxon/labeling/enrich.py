"""Zero-shot enrichment by a frontier model: the benchmark's middle condition.

Sonnet, alone and with no fine-tuning, labels the benchmark catalog through the same prompts the
annotators use: pick from the retrieved shortlist, or walk down the tree when nothing fits. It then
fills in attributes through the same attribute pass, in the version that scored better on the hand
labels, so the only thing that differs from the Taxon catalog is the model. On gold products the
same output is the zero-shot baseline of the classifier ladder, so it is never paid for twice.

The cost is priced before anything runs, from token counts of the real requests.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from taxon.labeling.attributes import EXACT, AttributeExtractor, use_mode
from taxon.labeling.llm import LLMRequest, ModelConfig, Usage
from taxon.labeling.pipeline import Labeler
from taxon.labeling.pricing import cost_at, price_for
from taxon.labeling.prompts import pick_request, stable_prefix, walk_request
from taxon.labeling.runner import BatchRunner
from taxon.taxonomy import Taxonomy

ENRICHER = ModelConfig("anthropic", "claude-sonnet-5-5", max_tokens=2_048, effort="low")
ImageFor = Callable[[dict], tuple | None]
# Counts a request's input tokens: the vendor's free counting endpoint, or an estimate in tests.
Counter = Callable[[LLMRequest], int]


def _no_image(_product: dict) -> None:
    return None


def enrich(
    products: Sequence[dict],
    shortlists: Mapping[str, list[str]],
    taxonomy: Taxonomy,
    runner: BatchRunner,
    mode: str,
    image_for: ImageFor = _no_image,
) -> list[dict]:
    """Label every product with Sonnet alone, then add attributes in `mode`.

    With one annotator, an answer is settled as it stands and nothing is arbitrated.
    """
    labeler = Labeler(taxonomy, runner, annotators=(ENRICHER,), image_for=image_for)
    rows = labeler.label(list(products), shortlists)
    extractor = AttributeExtractor(taxonomy, runner, config=ENRICHER, image_for=image_for)
    extractor.extract(list(products), rows, mode)
    use_mode(rows, mode)
    return rows


def enrichment_rows(rows: Sequence[dict]) -> list[dict]:
    """What the search index reads: a path and attributes per labeled product."""
    return [
        {
            "parent_asin": row["parent_asin"],
            "path": row["path"],
            "attributes": row.get("attributes", {}),
        }
        for row in rows
        if row.get("path")
    ]


# Pricing


@dataclass(frozen=True)
class WalkStats:
    """How often an annotator walked, and how far, measured on the dry run."""

    rate: float
    steps: float
    measured: bool


# Used only before a dry run has measured walking: a pessimistic guess, and reported as one.
ASSUMED_WALKS = WalkStats(rate=0.5, steps=3.0, measured=False)


def walk_stats(agreement: Mapping | None, vendor: str = "anthropic") -> WalkStats:
    """The Anthropic annotator's walking on the dry run, the closest model to Sonnet there."""
    stats = (agreement or {}).get("per_vendor", {}).get(vendor)
    if not stats or stats.get("mean_walk_steps") is None:
        return ASSUMED_WALKS
    return WalkStats(stats["walk_rate"], stats["mean_walk_steps"], True)


@dataclass(frozen=True)
class OutputGuess:
    name: str
    tokens_per_call: int


OUTPUTS = (OutputGuess("expected", 150), OutputGuess("high", 500))


def _attribute_request(
    extractor: AttributeExtractor, mode: str, product: dict, path: str
) -> LLMRequest | None:
    if not extractor.taxonomy.allowed_values(path):
        return None
    request, _ = extractor.request_for(mode, product, path)
    return request


def price_enrichment(
    sample: Sequence[dict],
    shortlists: Mapping[str, list[str]],
    taxonomy: Taxonomy,
    count: Counter,
    walks: WalkStats,
    products_total: int,
    mode: str = EXACT,
    image_for: ImageFor = _no_image,
) -> dict:
    """Price enriching `products_total` products from token counts of a sample's real requests.

    Input tokens are counted exactly, image included. The shared system prompt is billed as a cache
    read after the first request, as the batches cache it. Walk steps follow the dry run's rate,
    and since the category is not known before the run, each product's attribute request is
    built for its top shortlisted category. Output tokens are a guess, given at two levels.
    """
    system = stable_prefix(taxonomy)
    system_tokens = count(_system_only(system))
    extractor = AttributeExtractor(taxonomy, None, config=ENRICHER, image_for=image_for)
    picks, walk_steps, attributes = [], [], []
    for product in sample:
        candidates = shortlists[product["parent_asin"]]
        image = image_for(product)
        picks.append(count(pick_request(ENRICHER, system, product, candidates, image)))
        roots = taxonomy.roots()
        walk_steps.append(count(walk_request(ENRICHER, system, product, None, roots, image)))
        request = _attribute_request(extractor, mode, product, sorted(candidates)[0])
        attributes.append(count(request) if request is not None else 0)

    price = price_for(ENRICHER)
    pick, walk, attribute = (statistics.mean(x) for x in (picks, walk_steps, attributes))
    with_attributes = sum(1 for x in attributes if x) / len(attributes)
    calls = 1 + walks.rate * walks.steps + with_attributes
    per_product = {}
    for guess in OUTPUTS:
        steps = walks.rate * walks.steps
        uncached = pick - system_tokens + steps * (walk - system_tokens)
        usage = Usage(
            input_tokens=int(uncached + with_attributes * attribute),
            cached_tokens=int(system_tokens * (1 + steps)),
            output_tokens=int(calls * guess.tokens_per_call),
        )
        per_product[guess.name] = cost_at(usage, price)
    return {
        "model": ENRICHER.model,
        "mode": mode,
        "sample": len(sample),
        "products": products_total,
        "mean_input_tokens": {
            "system_prompt": system_tokens,
            "pick": round(pick),
            "walk_step": round(walk),
            "attributes": round(attribute),
        },
        "share_with_attributes": with_attributes,
        "walks": asdict(walks),
        "calls_per_product": calls,
        "output_tokens_per_call": {g.name: g.tokens_per_call for g in OUTPUTS},
        "per_product": per_product,
        "total": {name: cost * products_total for name, cost in per_product.items()},
    }


def _system_only(system: str) -> LLMRequest:
    """A request whose count is, near enough, the cached system prompt alone."""
    return LLMRequest(ENRICHER, "pick", system, ".")

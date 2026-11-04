"""Label products with three annotators and an arbitrator.

Each annotator picks from a retrieved shortlist or says none. Only a none triggers a top-down
walk. Answers are accepted when at least two annotators answered and all who answered agree. Only
a real conflict goes to the arbitrator, who may abstain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from taxon.labeling.llm import LLMResult, ModelConfig, Status
from taxon.labeling.prompts import (
    arbitrate_request,
    parse_choice,
    pick_request,
    stable_prefix,
    walk_request,
)
from taxon.labeling.runner import BatchRunner
from taxon.taxonomy import Taxonomy

ANNOTATORS = (
    ModelConfig("anthropic", "claude-haiku-5-5", max_tokens=1_024, effort="low"),
    ModelConfig("openai", "gpt-6-luna", max_tokens=1_024, effort="low"),
    ModelConfig("google", "gemini-3.5-flash-lite", max_tokens=1_024, effort="low"),
)
# Sonnet at low effort is the cheap default. Move to Opus only if scoring against the gold set
# shows Sonnet is clearly worse.
ARBITRATOR = ModelConfig("anthropic", "claude-sonnet-5-5", max_tokens=2_048, effort="low")
MAX_WALK_STEPS = 10
MIN_AGREEING = 2

ImageFor = Callable[[dict], tuple | None]


@dataclass
class _Walk:
    product: dict
    config: ModelConfig
    current: str | None = None
    steps: int = 0


def _no_image(_: dict) -> None:
    return None


def _children_of(taxonomy: Taxonomy, current: str | None) -> list[str]:
    return taxonomy.children(current) if current else taxonomy.roots()


class Labeler:
    def __init__(
        self,
        taxonomy: Taxonomy,
        runner: BatchRunner,
        annotators: Sequence[ModelConfig] = ANNOTATORS,
        arbitrator: ModelConfig = ARBITRATOR,
        image_for: ImageFor = _no_image,
    ):
        self.taxonomy = taxonomy
        self.runner = runner
        self.annotators = annotators
        self.arbitrator = arbitrator
        self.image_for = image_for
        self.system = stable_prefix(taxonomy)

    def _pick_round(self, products: list[dict], candidates: dict[str, list[str]], rows: dict):
        requests = {}
        for product in products:
            image = self.image_for(product)
            for config in self.annotators:
                request = pick_request(
                    config, self.system, product, candidates[product["parent_asin"]], image
                )
                requests[(product["parent_asin"], config.vendor)] = (request, product, config)
        results = self.runner.run(r for r, _, _ in requests.values())

        walks = []
        for (asin, vendor), (request, product, config) in requests.items():
            result = results[request.key]
            _add_cost(rows[asin], vendor, result)
            options = sorted(candidates[asin])
            entry = rows[asin]["annotators"][vendor]
            entry["status"] = result.status.value
            if result.status is not Status.OK:
                continue
            choice = parse_choice(result.text, len(options))
            if choice:
                entry["path"], entry["via"] = options[choice - 1], "pick"
            else:
                entry["invalid"] = choice is None
                walks.append(_Walk(product, config))
        return walks

    def _walk_rounds(self, walks: list[_Walk], rows: dict) -> None:
        """Advance every open walk one level per round, batched together."""
        while walks:
            requests = []
            for walk in walks:
                children = _children_of(self.taxonomy, walk.current)
                image = self.image_for(walk.product)
                request = walk_request(
                    walk.config, self.system, walk.product, walk.current, children, image
                )
                requests.append((walk, children, request))
            results = self.runner.run(r for _, _, r in requests)

            still_walking = []
            for walk, children, request in requests:
                asin, vendor = walk.product["parent_asin"], walk.config.vendor
                result = results[request.key]
                _add_cost(rows[asin], vendor, result)
                entry = rows[asin]["annotators"][vendor]
                entry["walk_steps"] = entry.get("walk_steps", 0) + 1
                walk.steps += 1
                choice = parse_choice(result.text, len(children))
                if result.status is Status.OK and choice:
                    walk.current = children[choice - 1]
                    if self.taxonomy.children(walk.current) and walk.steps < MAX_WALK_STEPS:
                        still_walking.append(walk)
                        continue
                if result.status is not Status.OK:
                    entry["status"] = result.status.value
                entry["path"], entry["via"] = walk.current, "walk"
            walks = still_walking

    def _arbitrate(self, products: list[dict], rows: dict) -> None:
        requests = {}
        for product in products:
            row = rows[product["parent_asin"]]
            answered = [a["path"] for a in row["annotators"].values() if a["path"]]
            proposals = sorted(set(answered))
            row["proposals"] = proposals
            if not proposals:
                row["source"] = "no_proposal"
            elif len(proposals) == 1 and len(answered) == len(self.annotators):
                row["path"], row["source"] = proposals[0], "unanimous"
            elif len(proposals) == 1 and len(answered) >= MIN_AGREEING:
                row["path"], row["source"] = proposals[0], "agreed_with_abstention"
            else:
                image = self.image_for(product)
                requests[product["parent_asin"]] = arbitrate_request(
                    self.arbitrator, self.system, product, proposals, image
                )
        results = self.runner.run(requests.values())
        for asin, request in requests.items():
            row = rows[asin]
            result = results[request.key]
            _add_cost(row, self.arbitrator.vendor, result)
            row["arbitrator_cost"] = result.cost
            row["arbitrator_status"] = result.status.value
            choice = parse_choice(result.text, len(row["proposals"]))
            if result.status is Status.OK and choice:
                row["path"], row["source"] = row["proposals"][choice - 1], "arbitrated"
            else:
                row["source"] = "abstained"

    def label(self, products: list[dict], candidates: dict[str, list[str]]) -> list[dict]:
        rows = {
            p["parent_asin"]: {
                "parent_asin": p["parent_asin"],
                "domain": p["domain"],
                "path": None,
                "source": None,
                "annotators": {
                    c.vendor: {"path": None, "via": None, "status": None} for c in self.annotators
                },
                "cost": {},
                "tokens": {},
            }
            for p in products
        }
        walks = self._pick_round(products, candidates, rows)
        self._walk_rounds(walks, rows)
        self._arbitrate(products, rows)
        return [rows[p["parent_asin"]] for p in products]


def _add_cost(row: dict, vendor: str, result: LLMResult) -> None:
    row["cost"][vendor] = row["cost"].get(vendor, 0.0) + result.cost
    tokens = row["tokens"].setdefault(vendor, {"input": 0, "cached": 0, "written": 0, "output": 0})
    tokens["input"] += result.usage.input_tokens
    tokens["cached"] += result.usage.cached_tokens
    tokens["written"] += result.usage.cache_write_tokens
    tokens["output"] += result.usage.output_tokens

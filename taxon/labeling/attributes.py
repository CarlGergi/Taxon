"""The attribute pass: a second, separate call after a product's category is settled.

One cheap annotator, with the image, fills in that category's attributes. It never touches the
category calls, so their cached responses are reused as they are, and adding this later costs only
the new calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from taxon.labeling.llm import ModelConfig, Status
from taxon.labeling.prompts import attribute_request, parse_attributes
from taxon.labeling.runner import BatchRunner
from taxon.taxonomy import Taxonomy

ATTRIBUTE_ANNOTATOR = ModelConfig("anthropic", "claude-haiku-5-5", max_tokens=1_024, effort="low")
# Categories settled by agreement or by the arbitrator. Abstentions get no attributes.
SETTLED = frozenset({"unanimous", "agreed_with_abstention", "arbitrated"})


def _no_image(_product: dict) -> None:
    return None


class AttributeExtractor:
    def __init__(
        self,
        taxonomy: Taxonomy,
        runner: BatchRunner,
        config: ModelConfig = ATTRIBUTE_ANNOTATOR,
        image_for: Callable[[dict], tuple | None] = _no_image,
    ):
        self.taxonomy = taxonomy
        self.runner = runner
        self.config = config
        self.image_for = image_for

    def extract(self, products: Sequence[dict], rows: Sequence[dict]) -> None:
        """Fill `attributes` on each row whose category is settled. Costs are kept apart from
        the category costs, so either can be projected alone."""
        by_asin = {p["parent_asin"]: p for p in products}
        requests = {}
        for row in rows:
            if row.get("source") not in SETTLED or not row.get("path"):
                row["attribute_status"] = "not_settled"
                continue
            handles = sorted(self.taxonomy.allowed_values(row["path"]))
            if not handles:
                row["attributes"], row["attribute_status"] = {}, "no_attributes"
                continue
            product = by_asin[row["parent_asin"]]
            requests[row["parent_asin"]] = attribute_request(
                self.config, product, row["path"], handles, self.image_for(product)
            )
        results = self.runner.run(requests.values())
        for row in rows:
            request = requests.get(row["parent_asin"])
            if request is None:
                continue
            result = results[request.key]
            row["attribute_cost"] = result.cost
            row["attribute_tokens"] = {
                "input": result.usage.input_tokens,
                "cached": result.usage.cached_tokens,
                "written": result.usage.cache_write_tokens,
                "output": result.usage.output_tokens,
            }
            parsed = None
            if result.status is Status.OK:
                parsed = parse_attributes(result.text, self.taxonomy.allowed_values(row["path"]))
            if parsed is None:
                row["attributes"] = {}
                row["attribute_status"] = (
                    result.status.value if result.status is not Status.OK else "invalid"
                )
                continue
            row["attributes"], row["attribute_unmatched"] = parsed
            row["attribute_status"] = "ok"

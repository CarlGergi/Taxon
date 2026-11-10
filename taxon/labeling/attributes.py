"""The attribute pass: a second, separate call after a product's category is settled.

One cheap annotator, with the image, fills in that category's attributes. It never touches the
category calls, so their cached responses are reused as they are, and adding this later costs only
the new calls.

Two versions exist, scored against hand labels to keep the better one. EXACT names the attributes
and keeps answers that match an allowed value exactly. SHORTLIST shows up to 20 allowed values per
attribute, retrieved for the listing, and the model picks one or none.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from taxon.labeling.llm import ModelConfig, Status
from taxon.labeling.prompts import (
    attribute_request,
    attribute_shortlist_request,
    parse_attribute_choices,
    parse_attributes,
)
from taxon.labeling.runner import BatchRunner
from taxon.labeling.values import ValueShortlister
from taxon.taxonomy import Taxonomy

ATTRIBUTE_ANNOTATOR = ModelConfig("anthropic", "claude-haiku-5-5", max_tokens=1_024, effort="low")
# Categories settled by agreement or by the arbitrator. Abstentions get no attributes.
SETTLED = frozenset({"unanimous", "agreed_with_abstention", "arbitrated"})
EXACT, SHORTLIST = "exact", "shortlist"
MODES = (EXACT, SHORTLIST)


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
        self.shortlister = ValueShortlister(taxonomy)

    def _request(self, mode: str, product: dict, path: str):
        image = self.image_for(product)
        if mode == EXACT:
            handles = sorted(self.taxonomy.allowed_values(path))
            return attribute_request(self.config, product, path, handles, image), None
        options = self.shortlister.shortlist(path, product)
        return attribute_shortlist_request(self.config, product, path, options, image), options

    def extract(self, products: Sequence[dict], rows: Sequence[dict], mode: str) -> None:
        """Record this mode's attributes on each row whose category is settled, under
        row["attribute_runs"][mode]. Costs stay apart from the category costs."""
        by_asin = {p["parent_asin"]: p for p in products}
        requests = {}
        for row in rows:
            runs = row.setdefault("attribute_runs", {})
            if row.get("source") not in SETTLED or not row.get("path"):
                runs[mode] = {"status": "not_settled"}
                continue
            if not self.taxonomy.allowed_values(row["path"]):
                runs[mode] = {"status": "no_attributes", "attributes": {}}
                continue
            product = by_asin[row["parent_asin"]]
            requests[row["parent_asin"]] = self._request(mode, product, row["path"])
        results = self.runner.run(request for request, _ in requests.values())
        for row in rows:
            if row["parent_asin"] not in requests:
                continue
            request, options = requests[row["parent_asin"]]
            result = results[request.key]
            run = {
                "cost": result.cost,
                "tokens": {
                    "input": result.usage.input_tokens,
                    "cached": result.usage.cached_tokens,
                    "written": result.usage.cache_write_tokens,
                    "output": result.usage.output_tokens,
                },
                "attributes": {},
            }
            parsed = None
            if result.status is Status.OK:
                if mode == EXACT:
                    allowed = self.taxonomy.allowed_values(row["path"])
                    parsed = parse_attributes(result.text, allowed)
                else:
                    parsed = parse_attribute_choices(result.text, options)
            if parsed is None:
                run["status"] = result.status.value if result.status is not Status.OK else "invalid"
            else:
                run["attributes"], run["unmatched"] = parsed
                run["status"] = "ok"
            row["attribute_runs"][mode] = run


def use_mode(rows: Sequence[dict], mode: str) -> None:
    """Make `mode`'s attributes the ones training reads."""
    for row in rows:
        run = row.get("attribute_runs", {}).get(mode, {})
        if "attributes" in run:
            row["attributes"], row["attribute_mode"] = run["attributes"], mode

"""GRPO rewards, computed exactly from the taxonomy. No reward model.

Each takes a model's reply and what was asked, and returns a number. They are kept separate so
training can log each one, and are added with weights in GRPOConfig.

Reward hacking to guard against: stopping at a shallow, safe category to collect partial credit.
Path credit is weighted by depth, so shallow levels are worth little, and stopping above the gold
depth costs a penalty on top.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from eval.fields import parse_fields
from taxon.taxonomy import Taxonomy, split_path
from taxon.training.examples import NONE
from taxon.training.fields import CATEGORY

DEPTH_PENALTY = 0.5


@dataclass(frozen=True)
class RewardWeights:
    compliance: float = 1.0
    valid_node: float = 1.0
    path_match: float = 2.0
    allowed_values: float = 1.0
    depth_penalty: float = 1.0

    def as_list(self) -> list[float]:
        return [
            self.compliance,
            self.valid_node,
            self.path_match,
            self.allowed_values,
            self.depth_penalty,
        ]


def predicted_category(text: str) -> str | None:
    fields = parse_fields(text)
    value = fields.get(CATEGORY) if fields else None
    return value if isinstance(value, str) else None


def field_compliance(text: str, requested: Sequence[str]) -> float:
    """1 if the reply is a JSON object with exactly the requested fields, else 0."""
    fields = parse_fields(text)
    return float(fields is not None and set(fields) == set(requested))


def valid_node(text: str, requested: Sequence[str], taxonomy: Taxonomy) -> float:
    """1 if a requested category is a real taxonomy path, and 0 when category was not asked."""
    if CATEGORY not in requested:
        return 0.0
    return float(predicted_category(text) in taxonomy)


def path_match(predicted: str | None, gold: str) -> float:
    """Depth-weighted credit for the shared prefix. Level i weighs i, so deep levels count most.

    The denominator covers the deeper of the two paths, so going past the gold path also costs.
    """
    if predicted is None:
        return 0.0
    mine, theirs = split_path(predicted), split_path(gold)
    shared = 0
    for a, b in zip(mine, theirs, strict=False):
        if a != b:
            break
        shared += 1
    earned = sum(range(1, shared + 1))
    possible = sum(range(1, max(len(mine), len(theirs)) + 1))
    return earned / possible


def depth_penalty(predicted: str | None, gold: str, weight: float = DEPTH_PENALTY) -> float:
    """Negative when the reply stops above the gold depth, in proportion to the levels missed."""
    if predicted is None:
        return 0.0
    mine, theirs = len(split_path(predicted)), len(split_path(gold))
    return -weight * (theirs - mine) / theirs if mine < theirs else 0.0


def allowed_values(text: str, requested: Sequence[str], taxonomy: Taxonomy, path: str) -> float:
    """Share of the attribute values given that the category allows. Null answers are fine.

    Values are checked against the predicted category when it is valid, else the gold one.
    """
    fields = parse_fields(text) or {}
    predicted = predicted_category(text)
    category = predicted if predicted in taxonomy else path
    allowed = taxonomy.allowed_values(category) if category in taxonomy else {}
    given = [
        (name, value)
        for name, value in fields.items()
        if name != CATEGORY and name in requested and value is not None
    ]
    if not given:
        return 1.0 if fields else 0.0
    return sum(isinstance(v, str) and v in allowed.get(n, ()) for n, v in given) / len(given)


def score(
    text: str, requested: Sequence[str], gold: Mapping, taxonomy: Taxonomy
) -> dict[str, float]:
    """Every reward for one reply. Category rewards apply only when category was requested."""
    asked_category = CATEGORY in requested
    predicted = predicted_category(text) if asked_category else None
    if asked_category and gold[CATEGORY] == NONE:
        # A shortlist without the labeled path: "none" is the right pick, and any path is wrong.
        right = float(predicted == NONE)
        return {
            "compliance": field_compliance(text, requested),
            "valid_node": right or float(predicted in taxonomy),
            "path_match": right,
            "allowed_values": allowed_values(text, requested, taxonomy, ""),
            "depth_penalty": 0.0,
        }
    valid = predicted in taxonomy
    return {
        "compliance": field_compliance(text, requested),
        "valid_node": valid_node(text, requested, taxonomy),
        "path_match": path_match(predicted, gold[CATEGORY]) if valid else 0.0,
        "allowed_values": allowed_values(text, requested, taxonomy, gold[CATEGORY]),
        "depth_penalty": depth_penalty(predicted, gold[CATEGORY]) if valid else 0.0,
    }

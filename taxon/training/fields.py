"""Selective field extraction: each example asks for a random subset of a product's fields.

A model trained on one fixed output cannot be asked for less at inference time, and extra requested
fields could change its other answers. Sampling the schema per example teaches it to follow whatever
is requested.
"""

from __future__ import annotations

import json
import random
from collections.abc import Collection, Iterable, Mapping, Sequence

CATEGORY = "category"
# How often an example also asks for an attribute the category cannot have, whose answer is null.
# Without these a model never learns to say a field does not apply. A value missing from a listing
# is not used this way, since that does not show the product lacks it.
NULL_FIELD_RATE = 0.25


def ordered(fields: Iterable[str]) -> list[str]:
    """The one order fields are ever listed in: category first, then alphabetical."""
    return sorted(set(fields), key=lambda name: (name != CATEGORY, name))


def sample_fields(known: Collection[str], rng: random.Random) -> list[str]:
    """A non-empty random subset, with every size equally likely."""
    pool = ordered(known)
    return ordered(rng.sample(pool, rng.randint(1, len(pool))))


def inapplicable_field(
    vocabulary: Sequence[str], allowed: Collection[str], taken: Collection[str], rng: random.Random
) -> str | None:
    """An attribute from `vocabulary` that this category does not have, if there is one."""
    options = [name for name in vocabulary if name not in allowed and name not in taken]
    return rng.choice(options) if options else None


def target_text(requested: Sequence[str], values: Mapping[str, str]) -> str:
    """The exact reply to learn: the requested fields in order, null where there is no value."""
    return json.dumps({name: values.get(name) for name in requested}, ensure_ascii=False)

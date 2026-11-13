"""Does an enriched catalog say only what its listings support?

Three free checks on a catalog exactly as the agent searched it:

- Hand labels. On the hand-labeled products, each specific value (not "Other") on an attribute the
  labeler was shown matches the hand label, contradicts it, or sits where the labeler left it
  blank. The labeling
  page asked for "only what the listing shows", but it also made attributes optional, so a blank
  is not proof: the share of values on blanks is an upper bound on what the listing leaves out.
- Country of origin. The product itself cannot reveal where it was made, so a listing that never
  names the country does not support it. The check reads the listing's text, not its image, and
  accepts loose spellings of the United States, so it undercounts unsupported origins.
- "Other". How many values are the taxonomy's catch-all, which says little about a product.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from eval.metrics import bootstrap_interval, wilson
from taxon.bench.catalog import build_catalog
from taxon.pins import DATA_DIR, ROOT
from taxon.taxonomy import Taxonomy

ORIGIN = "country-of-origin"
CATCH_ALL = "Other"
# Ways a listing names the United States. Generous on purpose: a false match only makes the
# unsupported count smaller.
_UNITED_STATES = re.compile(
    r"(?<![A-Za-z])(?:USA|U\.S\.A\.?|U\.S\.|United States|America|American)(?![A-Za-z])"
)


def _listing_text(row: Mapping) -> str:
    details = " ".join(f"{k} {v}" for k, v in (row.get("details") or {}).items())
    parts = [row.get("title"), row.get("store"), " ".join(row.get("features") or [])]
    return " ".join(str(p) for p in [*parts, row.get("description"), details] if p)


def names_country(row: Mapping, country: str) -> bool:
    """Whether a listing's text names a country, as the taxonomy spells it."""
    text = _listing_text(row)
    if country == "United States":
        return bool(_UNITED_STATES.search(text))
    pattern = rf"(?<![A-Za-z]){re.escape(country)}(?![A-Za-z])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def hand_label_support(
    rows: Iterable[Mapping], gold: Mapping[str, Mapping], taxonomy: Taxonomy
) -> dict:
    """A catalog's attribute values on hand-labeled products, against the hand labels."""
    per_product: list[tuple[int, int, int]] = []
    outside = 0
    for row in rows:
        label = gold.get(row["id"])
        if label is None or not row.get("category"):
            continue
        offered = dict(taxonomy.category(label["path"]).attribute_ids)
        match = contradict = blank = 0
        for handle, given in (row.get("attributes") or {}).items():
            values = [v for v in given if v != CATCH_ALL]
            if not values:
                continue
            if handle not in offered:
                outside += 1
                continue
            hand = label["attributes"].get(handle)
            if hand is None:
                blank += 1
            elif hand in values:
                match += 1
            else:
                contradict += 1
        per_product.append((match, contradict, blank))
    match, contradict, blank = (sum(p[i] for p in per_product) for i in range(3))
    shown = match + contradict + blank

    def blank_share(indices: list[int]) -> float | None:
        n = sum(sum(per_product[i]) for i in indices)
        return sum(per_product[i][2] for i in indices) / n if n else None

    return {
        "products": len(per_product),
        "values_on_shown_attributes": shown,
        "values_on_other_attributes": outside,
        "match": match,
        "contradict": contradict,
        "blank_in_hand_label": blank,
        # Values on one product share its listing, so the interval resamples products.
        "blank_share": blank / shown if shown else None,
        "blank_share_ci95": bootstrap_interval(blank_share, len(per_product)) if shown else None,
        "contradict_share_of_hand_labeled": contradict / (match + contradict)
        if match + contradict
        else None,
        "contradict_share_ci95": wilson(contradict, match + contradict),
    }


def origin_support(rows: Iterable[Mapping]) -> dict:
    """Specific countries of origin a catalog gives, and how many its listings name."""
    tagged = named = 0
    for row in rows:
        for country in (row.get("attributes") or {}).get(ORIGIN, []):
            if country == CATCH_ALL:
                continue
            tagged += 1
            named += names_country(row, country)
    unsupported = tagged - named
    return {
        "specific_countries": tagged,
        "named_in_listing": named,
        "not_named": unsupported,
        "not_named_share": unsupported / tagged if tagged else None,
        "not_named_share_ci95": wilson(unsupported, tagged),
    }


def catch_all_share(rows: Iterable[Mapping]) -> dict:
    values = [v for row in rows for vs in (row.get("attributes") or {}).values() for v in vs]
    other = sum(v == CATCH_ALL for v in values)
    return {
        "values": len(values),
        "other": other,
        "other_share": other / len(values) if values else None,
    }


def support_report(
    products: Sequence[Mapping],
    enrichments: Mapping[str, Path],
    gold: Mapping[str, Mapping],
    taxonomy: Taxonomy,
) -> dict:
    """Every check for each enriched catalog, built exactly as the agent searched it."""
    report = {}
    for name, path in enrichments.items():
        rows, _ = build_catalog(products, path, taxonomy)
        report[name] = {
            "hand_labels": hand_label_support(rows, gold, taxonomy),
            "country_of_origin": origin_support(rows),
            "catch_all": catch_all_share(rows),
        }
    return report


def enrichment_file(report_path: Path) -> Path | None:
    """The enrichments a saved enrichment report points at, wherever the repository now lives."""
    if not report_path.exists():
        return None
    given = Path(json.loads(report_path.read_text(encoding="utf-8"))["enrichments"])
    for candidate in (given, ROOT / given, DATA_DIR / "labeling" / given.name):
        if candidate.exists():
            return candidate
    return None

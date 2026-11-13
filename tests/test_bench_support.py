"""Checking whether an enriched catalog says only what its listings support."""

import json

import pytest

from taxon.bench import support

GOLD = {
    "A": {"path": "Apparel > Bags > Totes", "attributes": {"color": "Red"}},
    "B": {"path": "Apparel > Bags > Totes", "attributes": {"color": "Red"}},
}


def row(asin, attributes, title="A tote", category="Apparel > Bags > Totes", **extra):
    return {"id": asin, "title": title, "category": category, "attributes": attributes} | extra


def test_values_match_contradict_or_sit_on_a_blank_hand_label(taxonomy):
    rows = [
        row("A", {"color": ["Red"], "material": ["Cotton"], "strap-color": ["Blue"]}),
        row("B", {"color": ["Blue"], "material": ["Other"]}),
        row("C", {"color": ["Red"]}),
        row("A2", {}, category=None),
    ]
    report = support.hand_label_support(rows, GOLD, taxonomy)
    assert report["products"] == 2
    assert (report["match"], report["contradict"], report["blank_in_hand_label"]) == (1, 1, 1)
    # Strap color is not an attribute of totes, and "Other" is counted apart.
    assert report["values_on_other_attributes"] == 1
    assert report["values_on_shown_attributes"] == 3
    assert report["blank_share"] == pytest.approx(1 / 3)
    assert report["contradict_share_of_hand_labeled"] == 0.5


def test_a_country_counts_only_if_the_listing_names_it():
    assert support.names_country({"title": "Tea, Made in USA"}, "United States")
    assert support.names_country({"title": "Made in the U.S. by hand"}, "United States")
    assert support.names_country({"title": "Made in U.S.A."}, "United States")
    assert not support.names_country({"title": "USAGE notes"}, "United States")
    assert support.names_country({"details": {"Country of Origin": "france"}}, "France")
    assert not support.names_country({"title": "French press"}, "France")
    assert not support.names_country(
        {"title": "Wormseed extract", "store": "Botanicals"}, "United States"
    )
    rows = [
        row("A", {"country-of-origin": ["United States"]}, title="Made in the U.S. from oak"),
        row("B", {"country-of-origin": ["United States"]}, title="Wormseed extract"),
        row("C", {"country-of-origin": ["Other"]}),
    ]
    report = support.origin_support(rows)
    assert (report["specific_countries"], report["named_in_listing"], report["not_named"]) == (
        2,
        1,
        1,
    )
    assert report["not_named_share"] == 0.5


def test_the_catch_all_share_counts_every_value():
    rows = [row("A", {"color": ["Red", "Other"], "material": ["Other"]}), row("B", {})]
    assert support.catch_all_share(rows) == {"values": 3, "other": 2, "other_share": 2 / 3}


def test_an_enrichment_report_s_file_is_found_where_the_repository_now_lives(tmp_path, monkeypatch):
    monkeypatch.setattr(support, "DATA_DIR", tmp_path)
    (tmp_path / "labeling").mkdir()
    moved = tmp_path / "labeling" / "enrichments_sonnet.jsonl"
    moved.write_text("")
    report = tmp_path / "enrichment.json"
    report.write_text(
        json.dumps({"enrichments": "/elsewhere/data/labeling/enrichments_sonnet.jsonl"})
    )
    assert support.enrichment_file(report) == moved
    assert support.enrichment_file(tmp_path / "missing.json") is None


def test_the_report_checks_each_catalog_as_the_agent_searched_it(taxonomy, tmp_path):
    products = [
        {"parent_asin": "A", "title": "Red tote", "features": [], "details": {}, "price": 10.0},
        {"parent_asin": "B", "title": "Blue tote", "features": [], "details": {}, "price": 10.0},
    ]
    path = tmp_path / "enrichments.jsonl"
    rows = [
        {"parent_asin": "A", "path": "Apparel > Bags > Totes", "attributes": {"color": "Red"}},
        {"parent_asin": "B", "path": "Apparel > Bags > Totes", "attributes": {"color": "Green"}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = support.support_report(products, {"model": path}, GOLD, taxonomy)
    # Green is not an allowed color, so the catalog drops it before anyone can search on it.
    assert report["model"]["hand_labels"]["match"] == 1
    assert report["model"]["hand_labels"]["values_on_shown_attributes"] == 1
    assert report["model"]["catch_all"]["values"] == 1

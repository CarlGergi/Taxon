import copy

from taxon.bench.catalog import build_catalog, enriched_catalog, raw_catalog, raw_listing
from taxon.products import write_jsonl

PRODUCT = {
    "parent_asin": "B1",
    "domain": "Home_and_Kitchen",
    "main_category": "Amazon Home",
    "categories": ["Home & Kitchen", "Kitchen & Dining", "Knives"],
    "image_url": "https://m.media-amazon.com/images/I/x.jpg",
    "title": "Red chef knife",
    "store": "Acme",
    "features": ["Sharp"],
    "description": "A knife.",
    "details": {"Color": "Red", "Item Weight": 1.5, "Sizes": ["S", "M"]},
    "price": 19.99,
}


def test_a_raw_listing_holds_only_what_a_merchant_wrote():
    assert set(raw_listing(PRODUCT)) == {
        "id",
        "title",
        "store",
        "features",
        "description",
        "details",
        "price",
    }


def test_amazon_category_fields_cannot_reach_a_raw_listing():
    text = str(raw_listing(PRODUCT))
    assert "Kitchen & Dining" not in text and "Amazon Home" not in text
    assert "Home_and_Kitchen" not in text


def test_details_become_text_whatever_their_type():
    details = raw_listing(PRODUCT)["details"]
    assert details == {"Color": "Red", "Item Weight": "1.5", "Sizes": '["S", "M"]'}


def test_a_price_that_is_not_a_plain_number_counts_as_absent():
    for bad in [None, "from $5", True, 0, -3]:
        assert raw_listing({**PRODUCT, "price": bad})["price"] is None


def test_missing_optional_fields_get_empty_values():
    listing = raw_listing({"parent_asin": "B2", "title": "Thing"})
    assert listing["store"] is None
    assert (listing["features"], listing["description"], listing["details"]) == ([], "", {})


def test_the_raw_catalog_is_sorted_by_id():
    rows = raw_catalog([{**PRODUCT, "parent_asin": a} for a in ["C", "A", "B"]])
    assert [row["id"] for row in rows] == ["A", "B", "C"]


def raw_rows():
    return raw_catalog([{**PRODUCT, "parent_asin": a} for a in ["A", "B", "C", "D"]])


def enrichments():
    return {
        "A": {"path": "Apparel > Bags > Totes", "attributes": {"color": "Red"}},
        "B": {"path": "Apparel > Made Up", "attributes": {"color": "Red"}},
        "C": {
            "path": "Apparel > Bags > Totes",
            "attributes": {"color": ["Red", "Mauve"], "x": "y"},
        },
    }


def test_enrichment_adds_fields_and_changes_nothing_else(taxonomy):
    raw = raw_rows()
    rows, _ = enriched_catalog(raw, enrichments(), taxonomy)
    for before, after in zip(raw, rows, strict=True):
        assert {k: after[k] for k in before} == before
        assert set(after) - set(before) <= {"category", "attributes"}


def test_enrichment_leaves_the_raw_catalog_untouched(taxonomy):
    raw = raw_rows()
    snapshot = copy.deepcopy(raw)
    enriched_catalog(raw, enrichments(), taxonomy)
    assert raw == snapshot


def test_enrichment_keeps_valid_output_as_lists(taxonomy):
    rows, _ = enriched_catalog(raw_rows(), enrichments(), taxonomy)
    assert rows[0]["category"] == "Apparel > Bags > Totes"
    assert rows[0]["attributes"] == {"color": ["Red"]}


def test_an_invalid_path_leaves_the_listing_raw_and_is_counted(taxonomy):
    rows, report = enriched_catalog(raw_rows(), enrichments(), taxonomy)
    assert "category" not in rows[1] and "attributes" not in rows[1]
    assert report["invalid_path"] == 1


def test_values_the_taxonomy_does_not_allow_are_dropped_and_counted(taxonomy):
    rows, report = enriched_catalog(raw_rows(), enrichments(), taxonomy)
    assert rows[2]["attributes"] == {"color": ["Red"]}
    assert report["dropped_values"] == 2


def test_a_product_with_no_enrichment_stays_raw_and_is_counted(taxonomy):
    rows, report = enriched_catalog(raw_rows(), enrichments(), taxonomy)
    assert "category" not in rows[3]
    assert report == {"enriched": 2, "not_enriched": 1, "invalid_path": 1, "dropped_values": 2}


def test_build_catalog_reads_enrichments_from_a_file(tmp_path, taxonomy):
    path = tmp_path / "enrichments.jsonl"
    write_jsonl(path, [{"parent_asin": "B1", "path": "Home > Kitchen", "attributes": {}}])
    rows, report = build_catalog([PRODUCT], path, taxonomy)
    assert rows[0]["category"] == "Home > Kitchen"
    assert report["enriched"] == 1


def test_build_catalog_without_enrichments_is_the_raw_catalog():
    assert build_catalog([PRODUCT], None, None) == (raw_catalog([PRODUCT]), {})

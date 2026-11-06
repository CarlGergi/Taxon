import pytest
from conftest import category

from taxon.silver import (
    DOMAIN_VERTICALS,
    SilverLabeler,
    coverage,
    label_products,
    stated_attributes,
    tokens,
)
from taxon.taxonomy import Taxonomy


@pytest.fixture
def labeler(monkeypatch, taxonomy):
    monkeypatch.setitem(DOMAIN_VERTICALS, "Toy_Apparel", ("Apparel",))
    monkeypatch.setitem(DOMAIN_VERTICALS, "Toy_Home", ("Home",))
    monkeypatch.setitem(DOMAIN_VERTICALS, "Toy_Both", ("Apparel", "Home"))
    return SilverLabeler(taxonomy)


def test_tokens_normalize_case_plurals_accents_and_stopwords():
    assert tokens("Bags & Totes") == {"bag", "tote"}
    assert tokens("Cherries") == {"cherry"}
    assert tokens("Watches and Brushes") == {"watch", "brush"}
    assert tokens("Piñatas") == {"pinata"}
    assert tokens("Glass") == {"glass"}
    assert tokens("the of and") == frozenset()


@pytest.mark.parametrize(
    ("one", "many"),
    [
        ("knife", "knives"),
        ("shelf", "shelves"),
        ("glove", "gloves"),
        ("sleeve", "sleeves"),
        ("hoodie", "hoodies"),
        ("cookie", "cookies"),
        ("cherry", "cherries"),
        ("woman", "women"),
        ("brush", "brushes"),
        ("box", "boxes"),
        ("bag", "bags"),
    ],
)
def test_a_word_and_its_plural_share_one_form(one, many):
    assert tokens(one) == tokens(many)


def test_matches_the_deepest_breadcrumb_entry(labeler):
    assert labeler.label("Toy_Apparel", ["Apparel", "Bags", "Totes"]) == "Apparel > Bags > Totes"


def test_backs_off_when_the_deepest_entry_has_no_match(labeler):
    assert labeler.label("Toy_Apparel", ["Apparel", "Bags", "Zebras"]) == "Apparel > Bags"


def test_the_domain_restricts_the_match_to_its_verticals(labeler):
    assert labeler.label("Toy_Home", ["Bags"]) == "Home > Bags"
    assert labeler.label("Toy_Apparel", ["Bags"]) == "Apparel > Bags"
    assert labeler.label("Toy_Home", ["Totes"]) is None


def test_breadcrumb_context_breaks_a_tie_between_parents(labeler):
    assert labeler.label("Toy_Both", ["Home", "Bags"]) == "Home > Bags"
    assert labeler.label("Toy_Both", ["Apparel", "Bags"]) == "Apparel > Bags"


def test_a_weak_overlap_abstains(labeler):
    assert labeler.label("Toy_Apparel", ["Bags Hats Shoes Belts"]) is None


def test_an_empty_breadcrumb_abstains(labeler):
    assert labeler.label("Toy_Apparel", []) is None


def test_a_breadcrumb_of_only_stopwords_abstains(labeler):
    assert labeler.label("Toy_Apparel", ["the of"]) is None


def test_every_label_is_a_valid_path(labeler, taxonomy):
    crumbs = [["Bags"], ["Apparel"], ["Bag Straps"], ["Totes"], ["Bags", "Totes"]]
    for crumb in crumbs:
        label = labeler.label("Toy_Apparel", crumb)
        assert label is None or label in taxonomy


def test_ties_resolve_the_same_way_every_time(monkeypatch):
    monkeypatch.setitem(DOMAIN_VERTICALS, "D", ("Z",))
    release = {
        "version": "t",
        "verticals": [
            {
                "name": "Z",
                "prefix": "z",
                "categories": [
                    category("z", "Z", children=["z-1", "z-2"]),
                    category("z-1", "Z > P", parent="z", children=["z-1-1"]),
                    category("z-1-1", "Z > P > Mugs", parent="z-1"),
                    category("z-2", "Z > Mugs", parent="z"),
                ],
            }
        ],
        "attributes": [],
    }
    labeler = SilverLabeler(Taxonomy.from_release(release))
    assert labeler.label("D", ["Mugs"]) == "Z > Mugs"


def test_label_products_and_coverage(monkeypatch, taxonomy):
    monkeypatch.setitem(DOMAIN_VERTICALS, "Toy_Apparel", ("Apparel",))
    products = [
        {"parent_asin": "A", "domain": "Toy_Apparel", "categories": ["Bags"]},
        {"parent_asin": "B", "domain": "Toy_Apparel", "categories": []},
    ]
    labels = label_products(products, taxonomy)
    assert labels == [
        {"parent_asin": "A", "path": "Apparel > Bags"},
        {"parent_asin": "B", "path": None},
    ]
    summary = coverage(products, labels)
    assert summary["labeled"] == 1
    assert summary["abstained"] == 1
    assert summary["coverage"] == 0.5
    assert summary["distinct_paths"] == 1
    assert summary["per_domain_coverage"] == {"Toy_Apparel": 0.5}


def test_every_sampled_domain_has_verticals_that_exist():
    from taxon.pins import AMAZON_FILE_SIZES

    assert set(DOMAIN_VERTICALS) >= set(AMAZON_FILE_SIZES)


def _parts_taxonomy():
    return Taxonomy.from_release(
        {
            "version": "t",
            "verticals": [
                {
                    "name": "Home",
                    "prefix": "h",
                    "categories": [
                        category("h", "Home", children=["h-1", "h-2"]),
                        category("h-1", "Home > Kitchen", parent="h", children=["h-1-1"]),
                        category("h-1-1", "Home > Kitchen > Dishwasher Parts", parent="h-1"),
                        category("h-2", "Home > Plants", parent="h", children=["h-2-1"]),
                        category("h-2-1", "Home > Plants > Bulbs And Roots", parent="h-2"),
                    ],
                }
            ],
            "attributes": [],
        }
    )


def test_a_generic_breadcrumb_does_not_match_a_more_specific_node(monkeypatch):
    monkeypatch.setitem(DOMAIN_VERTICALS, "D", ("Home",))
    labeler = SilverLabeler(_parts_taxonomy())
    # "Parts" overlaps "Dishwasher Parts" enough on its own, but nothing here says dishwasher.
    assert labeler.label("D", ["Parts Parts"]) is None
    assert labeler.label("D", ["Appliance", "Parts"]) is None


def test_the_breadcrumb_can_supply_the_specific_word(monkeypatch):
    monkeypatch.setitem(DOMAIN_VERTICALS, "D", ("Home",))
    labeler = SilverLabeler(_parts_taxonomy())
    label = labeler.label("D", ["Dishwasher", "Dishwasher Parts"])
    assert label == "Home > Kitchen > Dishwasher Parts"


def test_a_multi_item_breadcrumb_does_not_justify_a_deep_node_for_one_item(monkeypatch):
    monkeypatch.setitem(DOMAIN_VERTICALS, "D", ("Home",))
    labeler = SilverLabeler(_parts_taxonomy())
    assert labeler.label("D", ["Bulbs Roots Seeds"]) is None


def test_a_multi_item_breadcrumb_may_still_match_a_shallow_node(monkeypatch):
    monkeypatch.setitem(DOMAIN_VERTICALS, "D", ("Home",))
    labeler = SilverLabeler(_parts_taxonomy())
    assert labeler.label("D", ["Home Garden"]) is None
    assert labeler.label("D", ["Home Goods Plants"]) is None
    assert labeler.label("D", ["Plants", "Home Plants"]) == "Home > Plants"


def test_stated_attributes_keep_exact_matches_in_the_taxonomy_spelling(taxonomy):
    details = {"Color": " red ", "Material": "COTTON", "Brand": "Acme"}
    found = stated_attributes(details, "Apparel > Bags > Totes", taxonomy)
    assert found == {"color": "Red", "material": "Cotton"}


def test_stated_attributes_skip_values_the_taxonomy_does_not_allow(taxonomy):
    assert stated_attributes({"Color": "Reddish"}, "Apparel > Bags > Totes", taxonomy) == {}


def test_stated_attributes_skip_attributes_the_category_does_not_have(taxonomy):
    assert stated_attributes({"Color": "Red"}, "Home > Kitchen", taxonomy) == {}


def test_stated_attributes_match_a_handle_written_with_spaces(taxonomy):
    found = stated_attributes({"Strap Color": "Blue"}, "Apparel > Bags", taxonomy)
    assert found == {"strap-color": "Blue"}


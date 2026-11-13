import pytest

from taxon.taxonomy import SEPARATOR, Taxonomy

COLOR = "gid://shopify/TaxonomyAttribute/1"
MATERIAL = "gid://shopify/TaxonomyAttribute/2"


def category(cid, full_name, parent=None, children=(), attributes=()):
    return {
        "id": cid,
        "level": full_name.count(SEPARATOR),
        "name": full_name.split(SEPARATOR)[-1],
        "full_name": full_name,
        "parent_id": parent,
        "attributes": [
            {"id": a, "name": h, "handle": h, "description": "", "extended": h != "color"}
            for h, a in attributes
        ],
        "children": [{"id": c, "name": c} for c in children],
        "ancestors": [],
    }


def toy_release():
    """A small tree in the release format, with the awkward cases on purpose:
    "Bags" under two parents, and "Bags" and "Bag Straps" sharing a string prefix."""
    return {
        "version": "toy-1",
        "verticals": [
            {
                "name": "Apparel",
                "prefix": "ap",
                "categories": [
                    category("ap", "Apparel", children=["ap-1", "ap-2"]),
                    category(
                        "ap-1",
                        "Apparel > Bags",
                        parent="ap",
                        children=["ap-1-1"],
                        attributes=[("color", COLOR), ("strap-color", COLOR)],
                    ),
                    category(
                        "ap-1-1",
                        "Apparel > Bags > Totes",
                        parent="ap-1",
                        attributes=[("color", COLOR), ("material", MATERIAL)],
                    ),
                    category("ap-2", "Apparel > Bag Straps", parent="ap"),
                ],
            },
            {
                "name": "Home",
                "prefix": "ho",
                "categories": [
                    category("ho", "Home", children=["ho-1", "ho-2"]),
                    category("ho-1", "Home > Bags", parent="ho"),
                    category("ho-2", "Home > Kitchen", parent="ho", children=["ho-2-1"]),
                    category("ho-2-1", "Home > Kitchen > Knives", parent="ho-2"),
                ],
            },
        ],
        "attributes": [
            {
                "id": COLOR,
                "name": "Color",
                "handle": "color",
                "description": "",
                "extended_attributes": [{"name": "Strap color", "handle": "strap-color"}],
                "values": [
                    {"id": "v1", "name": "Red", "handle": "red"},
                    {"id": "v2", "name": "Blue", "handle": "blue"},
                ],
            },
            {
                "id": MATERIAL,
                "name": "Material",
                "handle": "material",
                "description": "",
                "extended_attributes": [],
                "values": [{"id": "v3", "name": "Cotton", "handle": "cotton"}],
            },
        ],
    }


@pytest.fixture
def release():
    return toy_release()


@pytest.fixture
def taxonomy():
    return Taxonomy.from_release(toy_release())

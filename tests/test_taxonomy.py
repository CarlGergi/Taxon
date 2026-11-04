import gzip
import json

import pytest
from conftest import COLOR, category

from taxon.taxonomy import Taxonomy, TaxonomyError, path_prefixes, split_path


def test_loads_every_node_of_the_tree(taxonomy):
    assert taxonomy.version == "toy-1"
    assert len(taxonomy) == 8
    assert taxonomy.roots() == ["Apparel", "Home"]
    assert taxonomy.paths() == sorted(taxonomy.paths())


def test_internal_nodes_are_valid_paths_but_not_leaves(taxonomy):
    assert "Apparel > Bags" in taxonomy
    assert "Apparel > Bags" not in taxonomy.leaves()
    assert "Apparel > Bags > Totes" in taxonomy.leaves()
    assert taxonomy.leaves() == [
        "Apparel > Bag Straps",
        "Apparel > Bags > Totes",
        "Home > Bags",
        "Home > Kitchen > Knives",
    ]


def test_a_node_name_is_only_valid_under_its_own_parent(taxonomy):
    assert "Home > Bags" in taxonomy
    assert "Home > Totes" not in taxonomy
    assert "Bags" not in taxonomy


def test_category_properties(taxonomy):
    totes = taxonomy.category("Apparel > Bags > Totes")
    assert (totes.depth, totes.vertical, totes.is_leaf) == (3, "Apparel", True)
    assert taxonomy.children("Apparel") == ["Apparel > Bag Straps", "Apparel > Bags"]
    assert taxonomy.children("Home > Bags") == []


def test_path_helpers():
    assert split_path("A > B > C") == ("A", "B", "C")
    assert path_prefixes("A > B > C") == ["A", "A > B", "A > B > C"]
    assert path_prefixes("A") == ["A"]


def test_allowed_values_resolve_extended_attributes_to_the_base_list(taxonomy):
    allowed = taxonomy.allowed_values("Apparel > Bags")
    assert allowed == {"color": {"Red", "Blue"}, "strap-color": {"Red", "Blue"}}
    assert taxonomy.allowed_values("Apparel > Bags > Totes")["material"] == {"Cotton"}
    assert taxonomy.allowed_values("Home > Bags") == {}


def test_load_reads_plain_and_gzipped_files(tmp_path, release):
    plain = tmp_path / "taxonomy.json"
    plain.write_text(json.dumps(release), encoding="utf-8")
    packed = tmp_path / "taxonomy.json.gz"
    with gzip.open(packed, "wt", encoding="utf-8") as f:
        json.dump(release, f)
    assert Taxonomy.load(plain).paths() == Taxonomy.load(packed).paths()


def _categories(release):
    return release["verticals"][0]["categories"]


def _orphan(release):
    root, bags, *_ = _categories(release)
    root["children"] = [c for c in root["children"] if c["id"] != bags["id"]]
    bags["parent_id"] = "nope"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda r: _categories(r).append(_categories(r)[0]), "duplicate category id"),
        (lambda r: _orphan(r), "unknown parent"),
        (lambda r: _categories(r)[1].update(full_name="Home > Bags"), "duplicate category path"),
        (lambda r: _categories(r)[1].update(full_name="Other > Bags"), "does not extend parent"),
        (lambda r: _categories(r)[1].update(level=5), "level disagrees"),
        (lambda r: _categories(r)[0].update(children=[{"id": "ap-1"}]), "does not list child"),
        (lambda r: _categories(r)[0]["children"].append({"id": "ho"}), "non-child"),
        (lambda r: r["attributes"].pop(0), "unknown attribute"),
        (lambda r: r["attributes"].append(r["attributes"][0]), "duplicate attribute id"),
    ],
)
def test_rejects_a_corrupt_release(release, corrupt, message):
    corrupt(release)
    with pytest.raises(TaxonomyError, match=message):
        Taxonomy.from_release(release)


def test_rejects_duplicate_attribute_handles(release):
    _categories(release)[3] = category(
        "ap-2", "Apparel > Bag Straps", parent="ap", attributes=[("color", COLOR), ("color", COLOR)]
    )
    with pytest.raises(TaxonomyError, match="duplicate attribute handle"):
        Taxonomy.from_release(release)

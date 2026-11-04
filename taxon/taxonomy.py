"""Load and validate a Shopify product taxonomy release."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

SEPARATOR = " > "


class TaxonomyError(ValueError):
    """The release file violates a structural invariant."""


def split_path(path: str) -> tuple[str, ...]:
    return tuple(path.split(SEPARATOR))


def path_prefixes(path: str) -> list[str]:
    """Every ancestor path of `path`, shallowest first, ending with `path` itself."""
    parts = split_path(path)
    return [SEPARATOR.join(parts[: i + 1]) for i in range(len(parts))]


@dataclass(frozen=True)
class Attribute:
    id: str
    name: str
    handle: str
    values: frozenset[str]


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    path: str
    parent_id: str | None
    child_ids: tuple[str, ...]
    # Attribute handle as used by this category -> id of the base attribute holding its values.
    # An extended attribute such as "cover-color" points at the values of base "color".
    attribute_ids: tuple[tuple[str, str], ...]

    @property
    def depth(self) -> int:
        return self.path.count(SEPARATOR) + 1

    @property
    def vertical(self) -> str:
        return split_path(self.path)[0]

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids


class Taxonomy:
    def __init__(self, version: str, categories: list[Category], attributes: list[Attribute]):
        self.version = version
        self.attributes = {a.id: a for a in attributes}
        self.categories_by_id: dict[str, Category] = {}
        self.categories_by_path: dict[str, Category] = {}
        for category in categories:
            if category.id in self.categories_by_id:
                raise TaxonomyError(f"duplicate category id: {category.id}")
            if category.path in self.categories_by_path:
                raise TaxonomyError(f"duplicate category path: {category.path}")
            self.categories_by_id[category.id] = category
            self.categories_by_path[category.path] = category
        if len(self.attributes) != len(attributes):
            raise TaxonomyError("duplicate attribute id")
        self._validate()

    @classmethod
    def from_release(cls, raw: dict) -> Taxonomy:
        categories = []
        for vertical in raw["verticals"]:
            for item in vertical["categories"]:
                if item["level"] != item["full_name"].count(SEPARATOR):
                    raise TaxonomyError(f"level disagrees with path: {item['full_name']}")
                categories.append(
                    Category(
                        id=item["id"],
                        name=item["name"],
                        path=item["full_name"],
                        parent_id=item["parent_id"],
                        child_ids=tuple(child["id"] for child in item["children"]),
                        attribute_ids=tuple((a["handle"], a["id"]) for a in item["attributes"]),
                    )
                )
        attributes = [
            Attribute(
                id=a["id"],
                name=a["name"],
                handle=a["handle"],
                values=frozenset(v["name"] for v in a["values"]),
            )
            for a in raw["attributes"]
        ]
        return cls(raw["version"], categories, attributes)

    @classmethod
    def load(cls, path: Path) -> Taxonomy:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            return cls.from_release(json.load(f))

    def _validate(self) -> None:
        for category in self.categories_by_id.values():
            parent = self.categories_by_id.get(category.parent_id)
            if category.parent_id is None:
                expected = category.name
            elif parent is None:
                raise TaxonomyError(f"unknown parent of {category.path}: {category.parent_id}")
            else:
                expected = parent.path + SEPARATOR + category.name
                if category.id not in parent.child_ids:
                    raise TaxonomyError(f"{parent.path} does not list child {category.path}")
            if category.path != expected:
                raise TaxonomyError(f"path {category.path!r} does not extend parent: {expected!r}")
            for child_id in category.child_ids:
                child = self.categories_by_id.get(child_id)
                if child is None or child.parent_id != category.id:
                    raise TaxonomyError(f"{category.path} lists a non-child: {child_id}")
            handles = [handle for handle, _ in category.attribute_ids]
            if len(set(handles)) != len(handles):
                raise TaxonomyError(f"duplicate attribute handle in {category.path}")
            for handle, attribute_id in category.attribute_ids:
                if attribute_id not in self.attributes:
                    raise TaxonomyError(f"{category.path} uses unknown attribute: {handle}")

    def __len__(self) -> int:
        return len(self.categories_by_path)

    def __contains__(self, path: object) -> bool:
        return path in self.categories_by_path

    def paths(self) -> list[str]:
        """Every valid category path, internal nodes included, in sorted order."""
        return sorted(self.categories_by_path)

    def category(self, path: str) -> Category:
        return self.categories_by_path[path]

    def children(self, path: str) -> list[str]:
        ids = self.categories_by_path[path].child_ids
        return sorted(self.categories_by_id[child_id].path for child_id in ids)

    def leaves(self) -> list[str]:
        return sorted(p for p, c in self.categories_by_path.items() if c.is_leaf)

    def roots(self) -> list[str]:
        return sorted(p for p, c in self.categories_by_path.items() if c.parent_id is None)

    def allowed_values(self, path: str) -> dict[str, frozenset[str]]:
        """The closed set of values for each attribute a category accepts, keyed by handle."""
        return {
            handle: self.attributes[attribute_id].values
            for handle, attribute_id in self.categories_by_path[path].attribute_ids
        }

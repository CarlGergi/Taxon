"""Hand labels, kept as an append-only log so that no answer is ever lost or overwritten."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from taxon.taxonomy import Taxonomy


class InvalidLabel(ValueError):
    """The label names something outside the pool or the taxonomy."""


class GoldStore:
    def __init__(self, path: Path, queue: Sequence[str], taxonomy: Taxonomy):
        self.path = path
        self.queue = list(queue)
        self.taxonomy = taxonomy
        self._pool = set(queue)
        self._order: list[str] = []
        self._decisions: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line))

    def _apply(self, event: dict) -> None:
        asin = event["asin"]
        if event["event"] == "undo":
            self._decisions.pop(asin, None)
            self._order.remove(asin)
            return
        if asin in self._decisions:
            self._order.remove(asin)
        self._decisions[asin] = event
        self._order.append(asin)

    def _append(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._apply(event)

    def _check_asin(self, asin: str) -> None:
        if asin not in self._pool:
            raise InvalidLabel(f"{asin} is not in the gold pool")

    def label(self, asin: str, path: str, attributes: Mapping[str, str]) -> None:
        self._check_asin(asin)
        if path not in self.taxonomy:
            raise InvalidLabel(f"not a category: {path}")
        allowed = self.taxonomy.allowed_values(path)
        for handle, value in attributes.items():
            if handle not in allowed:
                raise InvalidLabel(f"{path} has no attribute {handle}")
            if value not in allowed[handle]:
                raise InvalidLabel(f"{value!r} is not an allowed value of {handle}")
        self._append({"event": "label", "asin": asin, "path": path, "attributes": dict(attributes)})

    def skip(self, asin: str) -> None:
        """Record that this product could not be judged. It does not count toward the target."""
        self._check_asin(asin)
        self._append({"event": "skip", "asin": asin})

    def undo(self) -> str | None:
        """Forget the most recent decision and return its product, or None if there is none."""
        if not self._order:
            return None
        asin = self._order[-1]
        self._append({"event": "undo", "asin": asin})
        return asin

    def labels(self) -> dict[str, dict]:
        """Current labels by product, skipped items excluded."""
        return {
            asin: {"path": event["path"], "attributes": event["attributes"]}
            for asin, event in self._decisions.items()
            if event["event"] == "label"
        }

    @property
    def n_labeled(self) -> int:
        return sum(e["event"] == "label" for e in self._decisions.values())

    @property
    def n_skipped(self) -> int:
        return sum(e["event"] == "skip" for e in self._decisions.values())

    def next_asin(self) -> str | None:
        return next((a for a in self.queue if a not in self._decisions), None)

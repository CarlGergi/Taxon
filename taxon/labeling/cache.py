"""Every final model response, kept on disk so nothing is paid for twice."""

from __future__ import annotations

import json
import os
from pathlib import Path

from taxon.labeling.llm import LLMResult, Status


class ResponseCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        return LLMResult.from_json(json.loads(path.read_text(encoding="utf-8")))

    def put(self, result: LLMResult) -> None:
        """A failed request has no response worth keeping, and must stay retryable."""
        if result.status is Status.FAILED:
            return
        path = self._path(result.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        partial.write_text(json.dumps(result.to_json(), sort_keys=True), encoding="utf-8")
        partial.replace(path)

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*.json"))

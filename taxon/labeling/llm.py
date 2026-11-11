"""The vendor-neutral request and result types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import Protocol

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"],
    "additionalProperties": False,
}
_CHOICE_JSON = json.dumps(CHOICE_SCHEMA, sort_keys=True)


class Status(StrEnum):
    OK = "ok"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    FAILED = "failed"


@dataclass(frozen=True)
class ModelConfig:
    vendor: str
    model: str
    max_tokens: int
    effort: str


@dataclass(frozen=True)
class LLMRequest:
    config: ModelConfig
    purpose: str
    system: str
    user: str
    image_path: str | None = None
    image_sha256: str | None = None
    # The reply's JSON schema, kept as text so a request stays hashable. Requests that pick an
    # option keep the original schema, so their cache keys are unchanged.
    schema_json: str = _CHOICE_JSON

    @property
    def schema(self) -> dict:
        return json.loads(self.schema_json)

    @cached_property
    def key(self) -> str:
        """Identifies the request by everything that affects the answer, so a repeat is free."""
        identity = [
            self.config.vendor,
            self.config.model,
            self.config.max_tokens,
            self.config.effort,
            self.schema,
            self.system,
            self.user,
            self.image_sha256,
        ]
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass
class Usage:
    """Token counts normalized across vendors. The three input fields never overlap."""

    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResult:
    key: str
    status: Status
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    cost: float = 0.0

    def to_json(self) -> dict:
        return {**asdict(self), "status": self.status.value}

    @classmethod
    def from_json(cls, data: dict) -> LLMResult:
        return cls(
            key=data["key"],
            status=Status(data["status"]),
            text=data["text"],
            usage=Usage(**data["usage"]),
            error=data["error"],
            cost=data["cost"],
        )


class Unreachable(ConnectionError):
    """The vendor could not be reached. The batch may well still be running on its side."""


class BatchClient(Protocol):
    def submit(self, requests: list[LLMRequest]) -> str:
        """Start a batch and return its id."""

    def poll(self, batch_id: str) -> bool:
        """True once the batch has finished, in whatever final state.

        Raises Unreachable when the vendor cannot be reached, so the caller can ask again later.
        """

    def results(self, batch_id: str) -> list[LLMResult]:
        """The results available for a finished batch. Missing keys count as failed."""

"""An in-memory vendor, for tests and for exercising the pipeline without keys or spending."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

from taxon.labeling.llm import LLMRequest, LLMResult, Status, Usage


def _usage_for(request: LLMRequest, text: str) -> Usage:
    return Usage(
        input_tokens=(len(request.system) + len(request.user)) // 4,
        output_tokens=len(text) // 4 + 1,
    )


def answer(request: LLMRequest, choice: int) -> LLMResult:
    text = json.dumps({"choice": choice})
    return LLMResult(request.key, Status.OK, text, _usage_for(request, text))


def demo_attributes(request: LLMRequest) -> LLMResult:
    """Says Red for a color now and then, and null for everything else."""
    seed = hashlib.sha256(request.user.encode()).digest()
    values = {
        handle: "Red" if handle == "color" and seed[0] < 128 else None
        for handle in request.schema["properties"]
    }
    text = json.dumps(values)
    return LLMResult(request.key, Status.OK, text, _usage_for(request, text))


def demo_judge(request: LLMRequest) -> LLMResult:
    """Mostly yes, sometimes no, now and then unsure."""
    seed = hashlib.sha256(request.user.encode()).digest()
    answers = {
        name: "yes" if seed[i] < 200 else "no" if seed[i] < 240 else "unsure"
        for i, name in enumerate(request.schema["properties"])
    }
    text = json.dumps(answers)
    return LLMResult(request.key, Status.OK, text, _usage_for(request, text))


def demo_responder(request: LLMRequest) -> LLMResult:
    """Deterministic and roughly plausible: usually option 1, sometimes another, sometimes none."""
    if request.purpose == "attributes":
        return demo_attributes(request)
    if request.purpose == "judge":
        return demo_judge(request)
    numbered = re.findall(r"^(\d+)\. ", request.user, flags=re.MULTILINE)
    n_options = len(numbered)
    seed = hashlib.sha256(f"{request.user}|{request.config.vendor}".encode()).digest()
    roll = seed[0] / 255
    if request.purpose == "arbitrate":
        choice = 1 if roll < 0.9 else 0
    elif roll < 0.72:
        choice = 1
    elif roll < 0.92 and n_options > 1:
        choice = 1 + seed[1] % n_options
    else:
        choice = 0
    return answer(request, choice if n_options else 0)


class FakeBatchClient:
    """Finishes each batch after `polls_until_done` polls, recording what it was sent."""

    def __init__(
        self,
        vendor: str,
        responder: Callable[[LLMRequest], LLMResult] = demo_responder,
        polls_until_done: int = 0,
    ):
        self.vendor = vendor
        self.responder = responder
        self.polls_until_done = polls_until_done
        self.submitted: list[list[LLMRequest]] = []
        self._batches: dict[str, list[LLMResult]] = {}
        self._polls: dict[str, int] = {}

    def submit(self, requests: list[LLMRequest]) -> str:
        batch_id = f"fake-{self.vendor}-{len(self.submitted)}"
        self.submitted.append(list(requests))
        self._batches[batch_id] = [self.responder(r) for r in requests]
        self._polls[batch_id] = 0
        return batch_id

    def poll(self, batch_id: str) -> bool:
        self._polls[batch_id] += 1
        return self._polls[batch_id] > self.polls_until_done

    def results(self, batch_id: str) -> list[LLMResult]:
        return self._batches[batch_id]

    @property
    def requests_sent(self) -> int:
        return sum(len(batch) for batch in self.submitted)

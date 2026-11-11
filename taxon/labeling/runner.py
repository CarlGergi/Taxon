"""Run requests through vendor batch APIs: cache first, then budget-checked chunks."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from taxon.labeling.budget import Budget, BudgetExceeded
from taxon.labeling.cache import ResponseCache
from taxon.labeling.llm import (
    BatchClient,
    LLMRequest,
    LLMResult,
    ModelConfig,
    Status,
    Unreachable,
)
from taxon.labeling.pricing import cost_of, worst_case_cost

MAX_REQUESTS_PER_BATCH = 1_000
MAX_BYTES_PER_BATCH = 100_000_000
MAX_ATTEMPTS = 3
POLL_SECONDS = 30.0
# A dropped connection, or a laptop asleep, should not end a run that has batches out. An hour of
# failed polls in a row does; the open batches are journaled, so the next run picks them up.
MAX_UNREACHABLE_POLLS = 120


def _request_bytes(request: LLMRequest) -> int:
    size = len(request.system) + len(request.user)
    if request.image_path is not None:
        size += os.path.getsize(request.image_path) * 4 // 3
    return size


def split_into_batches(
    requests: Iterable[LLMRequest],
    max_requests: int = MAX_REQUESTS_PER_BATCH,
    max_bytes: int = MAX_BYTES_PER_BATCH,
) -> list[list[LLMRequest]]:
    """Group by model, since a batch may target only one, then cut by count and size."""
    by_model: dict[ModelConfig, list[LLMRequest]] = defaultdict(list)
    for request in requests:
        by_model[request.config].append(request)
    batches = []
    for rows in by_model.values():
        current: list[LLMRequest] = []
        size = 0
        for request in rows:
            request_size = _request_bytes(request)
            if current and (len(current) >= max_requests or size + request_size > max_bytes):
                batches.append(current)
                current, size = [], 0
            current.append(request)
            size += request_size
        if current:
            batches.append(current)
    return batches


class BatchRunner:
    def __init__(
        self,
        clients: Mapping[str, BatchClient],
        cache: ResponseCache,
        budget: Budget,
        journal: Path,
        tracer=None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ):
        self.clients = clients
        self.cache = cache
        self.budget = budget
        self.journal = journal
        self.tracer = tracer
        self.sleep = sleep
        self.poll_seconds = poll_seconds

    def _journal_path(self, batch_id: str) -> Path:
        safe = batch_id.replace("/", "_")
        return self.journal / f"{safe}.json"

    def _open_batches(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.journal.glob("*"))]

    def _submit(self, vendor: str, batch: list[LLMRequest], worst_case: float) -> dict:
        batch_id = self.clients[vendor].submit(batch)
        # Written straight after submission, so a crash cannot lose a batch that is being paid for.
        entry = {
            "id": batch_id,
            "vendor": vendor,
            "config": vars(batch[0].config),
            "keys": [r.key for r in batch],
        }
        self.journal.mkdir(parents=True, exist_ok=True)
        path = self._journal_path(batch_id)
        partial = path.with_name(f"{path.name}.partial")
        partial.write_text(json.dumps(entry), encoding="utf-8")
        partial.replace(path)
        self.budget.reserve(vendor, batch_id, worst_case)
        return entry

    def _affordable(self, requests: list[LLMRequest]) -> tuple[list[LLMRequest], float]:
        vendor = requests[0].config.vendor
        remaining = self.budget.remaining(vendor)
        taken: list[LLMRequest] = []
        total = 0.0
        for request in requests:
            cost = worst_case_cost(request)
            if total + cost > remaining:
                break
            taken.append(request)
            total += cost
        return taken, total

    def _collect(self, entry: dict) -> dict[str, LLMResult]:
        vendor = entry["vendor"]
        config = ModelConfig(**entry["config"])
        got = {r.key: r for r in self.clients[vendor].results(entry["id"])}
        total = 0.0
        results = {}
        for key in entry["keys"]:
            result = got.get(key) or LLMResult(key, Status.FAILED, error="missing from batch")
            result.cost = cost_of(result.usage, config)
            total += result.cost
            self.cache.put(result)
            results[key] = result
        self.budget.settle(vendor, entry["id"], total)
        self._journal_path(entry["id"]).unlink()
        return results

    def run(self, requests: Iterable[LLMRequest]) -> dict[str, LLMResult]:
        """Return a final result per request key.

        Requests that keep failing come back with status FAILED. Raises BudgetExceeded once
        everything affordable has finished; those responses stay cached for the next run.
        """
        results: dict[str, LLMResult] = {}
        pending: dict[str, LLMRequest] = {}
        by_key: dict[str, LLMRequest] = {}
        for request in requests:
            key = request.key
            by_key[key] = request
            if key in results or key in pending:
                continue
            hit = self.cache.get(key)
            if hit is not None:
                results[key] = hit
            else:
                pending[key] = request

        open_batches = self._open_batches()
        for entry in open_batches:
            for key in entry["keys"]:
                pending.pop(key, None)

        attempts: Counter[str] = Counter()
        blocked: dict[str, int] = {}
        unreachable = 0
        while pending or open_batches:
            blocked = {}
            for batch in split_into_batches(pending.values()):
                vendor = batch[0].config.vendor
                affordable, worst_case = self._affordable(batch)
                if len(affordable) < len(batch):
                    blocked[vendor] = blocked.get(vendor, 0) + len(batch) - len(affordable)
                if affordable:
                    open_batches.append(self._submit(vendor, affordable, worst_case))
                    for request in affordable:
                        del pending[request.key]
            if not open_batches:
                break

            still_open = []
            for entry in open_batches:
                try:
                    done = self.clients[entry["vendor"]].poll(entry["id"])
                except Unreachable:
                    unreachable += 1
                    if unreachable >= MAX_UNREACHABLE_POLLS:
                        raise
                    still_open.append(entry)
                    continue
                unreachable = 0
                if not done:
                    still_open.append(entry)
                    continue
                for key, result in self._collect(entry).items():
                    attempts[key] += 1
                    request = by_key.get(key)
                    if result.status is Status.FAILED and attempts[key] < MAX_ATTEMPTS and request:
                        pending[key] = request
                        continue
                    if request is not None:
                        results[key] = result
                        if self.tracer is not None:
                            self.tracer.record(request, result)
            if len(still_open) == len(open_batches):
                self.sleep(self.poll_seconds)
            open_batches = still_open

        if pending:
            vendor = next(iter(blocked), pending[next(iter(pending))].config.vendor)
            raise BudgetExceeded(
                vendor, self.budget.spent(vendor), self.budget.caps[vendor], len(pending)
            )
        return results

"""Hard spending caps per vendor, enforced before anything is submitted.

A batch reserves its worst-case cost when submitted and settles to its real cost when collected.
Both events are appended to a ledger on disk, so money committed before a crash still counts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

CAPS = {"anthropic": 20.0, "openai": 3.0}


class BudgetExceeded(RuntimeError):
    """Submitting more would pass a vendor's cap. Nothing past the cap was sent."""

    def __init__(self, vendor: str, spent: float, cap: float, unsent: int):
        self.vendor = vendor
        self.spent = spent
        self.cap = cap
        self.unsent = unsent
        super().__init__(
            f"{vendor}: stopped before passing the ${cap:.2f} cap. "
            f"${spent:.2f} is spent or committed and {unsent} requests were not sent."
        )


class Budget:
    def __init__(self, ledger: Path, caps: Mapping[str, float] = CAPS):
        self.ledger = ledger
        self.caps = dict(caps)
        self._settled: dict[str, float] = {}
        self._reserved: dict[str, tuple[str, float]] = {}
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line))

    def _apply(self, event: dict) -> None:
        if event["event"] == "reserve":
            self._reserved[event["batch"]] = (event["vendor"], event["amount"])
        else:
            vendor = event["vendor"]
            self._reserved.pop(event.get("batch"), None)
            self._settled[vendor] = self._settled.get(vendor, 0.0) + event["amount"]

    def _append(self, event: dict) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._apply(event)

    def spent(self, vendor: str) -> float:
        """Settled cost plus what open batches could still cost."""
        reserved = sum(amount for v, amount in self._reserved.values() if v == vendor)
        return self._settled.get(vendor, 0.0) + reserved

    def settled(self, vendor: str) -> float:
        return self._settled.get(vendor, 0.0)

    def remaining(self, vendor: str) -> float:
        return self.caps[vendor] - self.spent(vendor)

    def reserve(self, vendor: str, batch: str, amount: float) -> None:
        self._append({"event": "reserve", "vendor": vendor, "batch": batch, "amount": amount})

    def settle(self, vendor: str, batch: str, amount: float) -> None:
        self._append({"event": "settle", "vendor": vendor, "batch": batch, "amount": amount})

    def charge(self, vendor: str, amount: float, reason: str) -> None:
        """Record a cost outside any batch, such as cache storage."""
        self._append({"event": "charge", "vendor": vendor, "amount": amount, "reason": reason})


def limited_caps(ledger: Path, limit: float | None, caps: Mapping[str, float] = CAPS) -> dict:
    """Caps for one run that may spend at most `limit` dollars per vendor, on top of what is spent.

    The overall caps still apply, whichever is lower. The runner submits only what fits under the
    cap at its worst case, so the run can never pass the limit; it stops instead.
    """
    if limit is None:
        return dict(caps)
    if limit < 0:
        raise ValueError("a spending limit cannot be negative")
    budget = Budget(ledger, caps)
    return {vendor: min(cap, budget.spent(vendor) + limit) for vendor, cap in caps.items()}

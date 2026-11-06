"""Commands for pooled judging. Judge calls share the labeling cache and spending ledger, so they
count toward the same caps."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from taxon.bench.judge import (
    JUDGE,
    UNSURE,
    agreement,
    estimate_cost,
    hand_review,
    judge_pairs,
    needs_call,
    pool,
    validation_sample,
)
from taxon.bench.tasks import Task
from taxon.labeling.budget import CAPS, Budget
from taxon.labeling.cache import ResponseCache
from taxon.labeling.env import load_secrets
from taxon.labeling.fakes import FakeBatchClient
from taxon.labeling.images import ImageStore
from taxon.labeling.runner import BatchRunner
from taxon.labeling.vendors import build_clients
from taxon.pins import DATA_DIR, SEED
from taxon.products import read_jsonl, write_jsonl

JUDGING_DIR = DATA_DIR / "bench" / "judging"


class Judging:
    """Where judgments live, and the labeling workspace whose cache and ledger they share."""

    def __init__(self, root: Path, labeling: Path):
        self.root = root
        self.judged = root / "judgments.jsonl"
        self.hand = root / "hand.jsonl"
        self.agreement = root / "agreement.json"
        self.labeling = labeling

    def load(self, path: Path) -> list[dict]:
        return read_jsonl(path) if path.exists() else []

    def merge(self, path: Path, rows: Sequence[dict]) -> list[dict]:
        merged = {(r["task"], r["asin"]): r for r in self.load(path)}
        merged.update({(r["task"], r["asin"]): r for r in rows})
        ordered = [merged[k] for k in sorted(merged)]
        write_jsonl(path, ordered)
        return ordered


def _runner(space: Judging, fake: bool) -> BatchRunner:
    budget = Budget(space.labeling / "ledger.jsonl", CAPS)
    if fake:
        clients = {v: FakeBatchClient(v) for v in CAPS}
    else:
        clients = build_clients(load_secrets(), charge=budget.charge)
    return BatchRunner(
        clients,
        ResponseCache(space.labeling / "responses"),
        budget,
        space.labeling / "open_batches",
    )


def _remaining(space: Judging) -> float:
    return Budget(space.labeling / "ledger.jsonl", CAPS).remaining(JUDGE.vendor)


def _image_for(fake: bool, products: Sequence[Mapping]):
    if fake:
        return lambda _product: None
    store = ImageStore(DATA_DIR / "images")
    print(store.ensure_all(list(products)))
    return store.image_for


def _refuse_over_cap(estimate: dict, space: Judging) -> None:
    if estimate["worst_case"] > _remaining(space):
        raise SystemExit(
            f"Judging could cost up to ${estimate['worst_case']:.2f}, more than the "
            f"${_remaining(space):.2f} left under the {JUDGE.vendor} cap."
        )


def run(
    command: str,
    space: Judging,
    tasks: Sequence[Task],
    products: Mapping[str, Mapping],
    gold_ids: set[str],
    runs_path: Path | None,
    fake: bool,
    read=input,
    write=print,
) -> dict:
    by_id = {t.id: t for t in tasks}
    pairs = pool(tasks, read_jsonl(runs_path), gold_ids) if runs_path else []
    # Only pairs the judge decides can measure it, so the sample leaves out what price settles.
    sample = validation_sample([p for p in pairs if needs_call(p, by_id, products)], SEED)
    judged = space.load(space.judged)
    hand = space.load(space.hand)

    if command == "judge-price":
        report = {
            "validation_sample": estimate_cost(sample, by_id, products, with_images=not fake),
            "whole_pool": estimate_cost(pairs, by_id, products, with_images=not fake),
            "left_under_cap": _remaining(space),
        }
    elif command == "judge-sample":
        estimate = estimate_cost(sample, by_id, products, with_images=not fake)
        _refuse_over_cap(estimate, space)
        image_for = _image_for(fake, [products[p["asin"]] for p in sample])
        new = judge_pairs(sample, by_id, products, _runner(space, fake), image_for)
        judged = space.merge(space.judged, new)
        calls = [j for j in new if j["by"] == "judge"]
        per_call = sum(j["cost"] for j in calls) / len(calls) if calls else 0.0
        left = sum(
            1
            for p in pairs
            if needs_call(p, by_id, products)
            and (p["task"], p["asin"]) not in {(j["task"], j["asin"]) for j in judged}
        )
        report = {
            "judged": len(new),
            "measured_per_call": per_call,
            "rest_of_pool_calls": left,
            "rest_of_pool_estimate": per_call * left,
            "left_under_cap": _remaining(space),
        }
    elif command == "judge-review":
        unsure = [j for j in judged if j["verdict"] == UNSURE]
        done = {(h["task"], h["asin"]): h["verdict"] for h in hand}
        # The validation sample first, so agreement can be measured, then what the judge left open.
        queue = sample + [u for u in unsure if u not in sample]
        new = hand_review(queue, by_id, products, done, read, write)
        hand = space.merge(space.hand, new)
        report = {"new": len(new), "hand_judged": len(hand)}
    elif command == "judge-agreement":
        keys = {(p["task"], p["asin"]) for p in sample}
        report = agreement([j for j in judged if (j["task"], j["asin"]) in keys], hand)
        space.root.mkdir(parents=True, exist_ok=True)
        space.agreement.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif command == "judge-pool":
        keys = {(p["task"], p["asin"]) for p in sample}
        trust = agreement([j for j in judged if (j["task"], j["asin"]) in keys], hand)
        if not trust["trusted"]:
            raise SystemExit(
                f"The judge is not trusted yet: it needs agreement of at least {trust['bar']:.0%} "
                f"on {len(sample)} hand-judged pairs. Run judge-sample, judge-review, "
                "then judge-agreement."
            )
        seen = {(j["task"], j["asin"]) for j in judged}
        todo = [p for p in pairs if (p["task"], p["asin"]) not in seen]
        _refuse_over_cap(estimate_cost(todo, by_id, products, with_images=not fake), space)
        image_for = _image_for(fake, [products[p["asin"]] for p in todo])
        new = judge_pairs(todo, by_id, products, _runner(space, fake), image_for)
        judged = space.merge(space.judged, new)
        verdicts = Counter(j["verdict"] for j in judged)
        open_ = [j for j in judged if j["verdict"] == UNSURE]
        handled = {(h["task"], h["asin"]) for h in hand}
        report = {
            "pairs": len(judged),
            "verdicts": dict(sorted(verdicts.items())),
            "unsure_awaiting_hand_review": sum(
                (j["task"], j["asin"]) not in handled for j in open_
            ),
        }
    else:
        raise ValueError(command)
    return report

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
from taxon.labeling.budget import CAPS, Budget, limited_caps
from taxon.labeling.cache import ResponseCache
from taxon.labeling.env import load_secrets
from taxon.labeling.fakes import FakeBatchClient
from taxon.labeling.images import ImageStore
from taxon.labeling.runner import BatchRunner
from taxon.labeling.vendors import OpenAIDirectClient, build_clients
from taxon.pins import DATA_DIR, SEED
from taxon.products import read_jsonl, write_jsonl

JUDGING_DIR = DATA_DIR / "bench" / "judging"
# Sent one by one instead of in a batch, a judge call costs this many times the batch price.
DIRECT_PRICE_FACTOR = OpenAIDirectClient.price_factor


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


def _runner(
    space: Judging, fake: bool, direct: bool = False, limit: float | None = None
) -> BatchRunner:
    """Judge calls go in batches, or one by one at full price with `direct` when a batch stalls;
    `limit` caps what this run may spend, on top of the overall cap."""
    ledger = space.labeling / "ledger.jsonl"
    budget = Budget(ledger, limited_caps(ledger, limit))
    if fake:
        clients = {v: FakeBatchClient(v) for v in CAPS}
    else:
        clients = build_clients(
            load_secrets(), CAPS, charge=budget.charge, direct=[JUDGE.vendor] if direct else ()
        )
    return BatchRunner(
        clients,
        ResponseCache(space.labeling / "responses"),
        budget,
        space.labeling / "open_batches",
    )


def _remaining(space: Judging, limit: float | None = None) -> float:
    ledger = space.labeling / "ledger.jsonl"
    return Budget(ledger, limited_caps(ledger, limit)).remaining(JUDGE.vendor)


def _image_for(fake: bool, products: Sequence[Mapping]):
    if fake:
        return lambda _product: None
    store = ImageStore(DATA_DIR / "images")
    print(store.ensure_all(list(products)))
    return store.image_for


def _priced(estimate: dict, factor: float) -> dict:
    """An estimate at batch prices, scaled to how the calls will be sent."""
    return {
        **estimate,
        "estimate": estimate["estimate"] * factor,
        "worst_case": estimate["worst_case"] * factor,
        "price_factor": factor,
    }


def _keys(rows: Sequence[Mapping]) -> set[tuple[str, str]]:
    return {(r["task"], r["asin"]) for r in rows}


def validation_report(
    space: Judging,
    tasks: Sequence[Task],
    products: Mapping[str, Mapping],
    gold_ids: set[str],
    rows: Sequence[Mapping],
) -> dict:
    """How the judge did on its validation sample, and what it said across the pool, for the
    benchmark report: rebuilt from the saved judgments, with no calls."""
    by_id = {t.id: t for t in tasks}
    pairs = pool(tasks, rows, gold_ids, products)
    callable_pairs = [p for p in pairs if needs_call(p, by_id, products)]
    judged, hand = space.load(space.judged), space.load(space.hand)

    def agreement_on(sample: Sequence[dict]) -> dict:
        keys = _keys(sample)
        return agreement([j for j in judged if (j["task"], j["asin"]) in keys], hand)

    sample = validation_sample(callable_pairs, SEED, judged)
    in_pool = _keys(pairs)
    verdicts = Counter(j["verdict"] for j in judged if (j["task"], j["asin"]) in in_pool)
    return {
        "pool": len(pairs),
        "sample": len(sample),
        "agreement": agreement_on(sample),
        # The first pairs drawn, before the sample grew for an unsure judge.
        "first_sample": agreement_on(validation_sample(callable_pairs, SEED)),
        "verdicts": dict(sorted(verdicts.items())),
    }


def _refuse_over_cap(estimate: dict, space: Judging, limit: float | None = None) -> None:
    left = _remaining(space, limit)
    if estimate["worst_case"] > left:
        under = "this run's limit" if limit is not None else f"the {JUDGE.vendor} cap"
        raise SystemExit(
            f"Judging could cost up to ${estimate['worst_case']:.2f}, more than the "
            f"${left:.2f} left under {under}."
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
    direct: bool = False,
    limit: float | None = None,
) -> dict:
    by_id = {t.id: t for t in tasks}
    factor = DIRECT_PRICE_FACTOR if direct else 1.0

    def priced(some: Sequence[dict]) -> dict:
        return _priced(estimate_cost(some, by_id, products, with_images=not fake), factor)

    pairs = pool(tasks, read_jsonl(runs_path), gold_ids, products) if runs_path else []
    # Only pairs the judge decides can measure it, so the sample leaves out what price settles.
    callable_pairs = [p for p in pairs if needs_call(p, by_id, products)]
    judged = space.load(space.judged)
    hand = space.load(space.hand)
    sample = validation_sample(callable_pairs, SEED, judged)

    if command == "judge-price":
        report = {
            "validation_sample": priced(sample),
            "whole_pool": priced(pairs),
            "left_under_cap": _remaining(space),
        }
    elif command == "judge-sample":
        # Judge what the sample lacks; if the judge was unsure too often, the sample grows by the
        # next pair in its order, which is judged in turn. One runner holds this command's limit.
        new: list[dict] = []
        runner = None
        while todo := [p for p in sample if (p["task"], p["asin"]) not in _keys(judged)]:
            _refuse_over_cap(priced(todo), space, limit)
            runner = runner or _runner(space, fake, direct, limit)
            image_for = _image_for(fake, [products[p["asin"]] for p in todo])
            judged_now = judge_pairs(todo, by_id, products, runner, image_for)
            judged = space.merge(space.judged, judged_now)
            new += judged_now
            sample = validation_sample(callable_pairs, SEED, judged)
        calls = [j for j in new if j["by"] == "judge"]
        per_call = sum(j["cost"] for j in calls) / len(calls) if calls else 0.0
        left = sum(
            1
            for p in pairs
            if needs_call(p, by_id, products)
            and (p["task"], p["asin"]) not in {(j["task"], j["asin"]) for j in judged}
        )
        in_sample = _keys(sample)
        report = {
            "judged": len(new),
            "sample": len(sample),
            "judge_answered_in_sample": sum(
                j["verdict"] != UNSURE for j in judged if (j["task"], j["asin"]) in in_sample
            ),
            "measured_per_call": per_call,
            "rest_of_pool_calls": left,
            "rest_of_pool_estimate": per_call * left,
            "left_under_cap": _remaining(space),
        }
    elif command == "judge-review":
        unsure = [j for j in judged if j["verdict"] == UNSURE]
        done = {(h["task"], h["asin"]): h["verdict"] for h in hand}
        # The validation sample first, so agreement can be measured, then what the judge left open.
        in_sample = _keys(sample)
        queue = sample + [u for u in unsure if (u["task"], u["asin"]) not in in_sample]
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
        _refuse_over_cap(priced(todo), space, limit)
        image_for = _image_for(fake, [products[p["asin"]] for p in todo])
        new = judge_pairs(todo, by_id, products, _runner(space, fake, direct, limit), image_for)
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

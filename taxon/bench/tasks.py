"""CatalogBench tasks: shopping requests with hard constraints, drawn only from the gold set.

Constraints and acceptable products come from hand labels and listing prices, never from any
model. Each request is then written by hand in a shopper's words, and each task is hand-checked.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from taxon.products import read_jsonl, seeded_hash, write_jsonl
from taxon.taxonomy import Taxonomy, path_prefixes, split_path

FIRST_TASK_COUNT = 100
FINAL_TASK_COUNT = 300
MIN_CONSTRAINTS = 2
MAX_CONSTRAINTS = 4
MAX_ACCEPTABLE = 10
_PRICE_HEADROOM = 1.25
_PRICE_ROUNDING = 5.0
_MIN_CATEGORY_DEPTH = 2
_MULTIWORD = 2


class TaskError(ValueError):
    """A task breaks the benchmark's rules."""


@dataclass(frozen=True)
class Constraint:
    kind: str
    path: str | None = None
    handle: str | None = None
    value: str | None = None
    amount: float | None = None

    def signature(self) -> tuple:
        return (self.kind, self.path, self.handle, self.value, self.amount)


@dataclass(frozen=True)
class GoldItem:
    asin: str
    path: str
    attributes: Mapping[str, str]
    price: float | None


@dataclass
class Task:
    id: str
    constraints: list[Constraint]
    acceptable: list[str]
    request: str = ""
    checked: bool = False
    notes: str = field(default="")

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "request": self.request,
            "checked": self.checked,
            "notes": self.notes,
            "acceptable": self.acceptable,
            "constraints": [
                {k: v for k, v in asdict(c).items() if v is not None} for c in self.constraints
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> Task:
        return cls(
            id=data["id"],
            constraints=[Constraint(**c) for c in data["constraints"]],
            acceptable=list(data["acceptable"]),
            request=data["request"],
            checked=data["checked"],
            notes=data.get("notes", ""),
        )


def _signature(constraints: Iterable[Constraint]) -> tuple[str, ...]:
    return tuple(sorted(str(c.signature()) for c in constraints))


def numeric_price(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def gold_items(labels: Mapping[str, dict], products: Mapping[str, dict]) -> list[GoldItem]:
    """Join hand labels with listing prices. Prices that are not plain numbers count as absent."""
    return [
        GoldItem(asin, row["path"], row["attributes"], numeric_price(products[asin].get("price")))
        for asin, row in sorted(labels.items())
    ]


def satisfies(item: GoldItem, constraint: Constraint) -> bool:
    if constraint.kind == "category":
        return item.path == constraint.path or item.path.startswith(f"{constraint.path} > ")
    if constraint.kind == "attribute":
        return item.attributes.get(constraint.handle) == constraint.value
    if constraint.kind == "max_price":
        return item.price is not None and item.price <= constraint.amount
    raise TaskError(f"unknown constraint kind: {constraint.kind}")


def acceptable_products(items: Iterable[GoldItem], constraints: Sequence[Constraint]) -> list[str]:
    return sorted(i.asin for i in items if all(satisfies(i, c) for c in constraints))


def _category_options(path: str) -> list[str]:
    """The category levels a shopper might mean, most specific first."""
    prefixes = path_prefixes(path)
    return list(reversed(prefixes[_MIN_CATEGORY_DEPTH - 1 :])) or [path]


def _extras(item: GoldItem, seed: int) -> list[Constraint]:
    attributes = sorted(
        item.attributes.items(), key=lambda kv: seeded_hash(seed, f"{item.asin}|{kv[0]}")
    )
    extras = [Constraint("attribute", handle=h, value=v) for h, v in attributes]
    if item.price is not None:
        ceiling = math.ceil(item.price * _PRICE_HEADROOM / _PRICE_ROUNDING) * _PRICE_ROUNDING
        extras.insert(1, Constraint("max_price", amount=ceiling))
    return extras


def draft_tasks(items: Sequence[GoldItem], n: int, seed: int) -> list[Task]:
    """Propose tasks with 2 to 4 constraints and a small acceptable set, requests left blank.

    Each draft starts from one labeled product, so at least that product is always acceptable.
    Sizes cycle so that the set holds a spread of 2, 3 and 4 constraints where the labels allow.
    """
    order = sorted(items, key=lambda i: seeded_hash(seed, f"task|{i.asin}"))
    tasks: list[Task] = []
    seen: set[tuple] = set()
    sizes = list(range(MIN_CONSTRAINTS, MAX_CONSTRAINTS + 1))
    for category_level in range(10):
        for item in order:
            if len(tasks) >= n:
                return tasks
            options = _category_options(item.path)
            if category_level >= len(options):
                continue
            category = Constraint("category", path=options[category_level])
            extras = _extras(item, seed)
            # Start from the size the spread calls for. If that matches too many products, a
            # tighter or looser task from the same product may still work.
            for offset in range(len(sizes)):
                size = sizes[(len(tasks) + offset) % len(sizes)]
                if len(extras) < size - 1:
                    continue
                constraints = [category, *extras[: size - 1]]
                signature = _signature(constraints)
                acceptable = acceptable_products(items, constraints)
                if signature not in seen and 1 <= len(acceptable) <= MAX_ACCEPTABLE:
                    seen.add(signature)
                    tasks.append(Task(f"task-{len(tasks) + 1:04d}", constraints, acceptable))
                    break
    return tasks


def lint_request(request: str, constraints: Sequence[Constraint]) -> list[str]:
    """Problems with a request's wording. Quoting taxonomy names would favor enriched catalogs."""
    problems = []
    lowered = request.casefold()
    if not request.strip():
        problems.append("the request is empty")
    if ">" in request:
        problems.append("the request quotes a taxonomy path")
    for constraint in constraints:
        if constraint.kind == "category":
            name = split_path(constraint.path)[-1]
            if len(name.split()) >= _MULTIWORD and name.casefold() in lowered:
                problems.append(f"the request quotes the category name {name!r}")
        if constraint.kind == "attribute" and "-" in constraint.handle:
            if constraint.handle.casefold() in lowered:
                problems.append(f"the request quotes the attribute handle {constraint.handle!r}")
    return problems


def validate_task(
    task: Task,
    items: Sequence[GoldItem],
    taxonomy: Taxonomy,
    training_ids: Collection[str],
    final: bool = True,
) -> None:
    """Raise TaskError unless the task follows the rules. `final` also requires the hand work."""
    if not MIN_CONSTRAINTS <= len(task.constraints) <= MAX_CONSTRAINTS:
        raise TaskError(f"{task.id}: needs 2 to 4 constraints, has {len(task.constraints)}")
    categories = [c for c in task.constraints if c.kind == "category"]
    if len(categories) != 1:
        raise TaskError(f"{task.id}: needs exactly one category constraint")
    if categories[0].path not in taxonomy:
        raise TaskError(f"{task.id}: unknown category {categories[0].path}")
    signatures = [c.signature() for c in task.constraints]
    if len(set(signatures)) != len(signatures):
        raise TaskError(f"{task.id}: repeats a constraint")
    if not task.acceptable:
        raise TaskError(f"{task.id}: has no acceptable products")

    gold = {i.asin for i in items}
    outside = set(task.acceptable) - gold
    if outside:
        raise TaskError(f"{task.id}: acceptable products outside the gold set: {sorted(outside)}")
    leaked = set(task.acceptable) & set(training_ids)
    if leaked:
        raise TaskError(f"{task.id}: acceptable products in training data: {sorted(leaked)}")
    # Hand-checking may drop a product from the set, but can never add one the rules exclude.
    extra = set(task.acceptable) - set(acceptable_products(items, task.constraints))
    if extra:
        raise TaskError(f"{task.id}: acceptable products break its constraints: {sorted(extra)}")

    if final:
        problems = lint_request(task.request, task.constraints)
        if problems:
            raise TaskError(f"{task.id}: {'; '.join(problems)}")
        if not task.checked:
            raise TaskError(f"{task.id}: has not been hand-checked")


def check_tasks(
    tasks: Sequence[Task],
    items: Sequence[GoldItem],
    taxonomy: Taxonomy,
    training_ids: Collection[str],
    need: int = FIRST_TASK_COUNT,
) -> dict:
    """Validate a whole task set and summarize it."""
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise TaskError("task ids are not unique")
    if len(tasks) < need:
        raise TaskError(f"need {need} tasks, have {len(tasks)}")
    requests = [t.request for t in tasks]
    if len(set(requests)) != len(requests):
        raise TaskError("two tasks share a request")
    for task in tasks:
        validate_task(task, items, taxonomy, training_ids)
    return {
        "tasks": len(tasks),
        "constraints": {
            str(n): sum(len(t.constraints) == n for t in tasks)
            for n in range(MIN_CONSTRAINTS, MAX_CONSTRAINTS + 1)
        },
        "mean_acceptable": sum(len(t.acceptable) for t in tasks) / len(tasks),
        "sha256": fingerprint(tasks),
    }


def fingerprint(tasks: Sequence[Task]) -> str:
    """Identifies a task set, so results can show it did not change after they were seen."""
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda t: t.id):
        digest.update(repr(sorted(task.to_json().items())).encode())
    return digest.hexdigest()


def save_tasks(path: Path, tasks: Sequence[Task]) -> None:
    write_jsonl(path, [t.to_json() for t in tasks])


def load_tasks(path: Path) -> list[Task]:
    return [Task.from_json(row) for row in read_jsonl(path)]


def merge_drafts(existing: Sequence[Task], drafts: Sequence[Task]) -> list[Task]:
    """Add new drafts without touching written tasks, so growing the set loses no hand work."""
    have = {_signature(t.constraints) for t in existing}
    merged = list(existing)
    for draft in drafts:
        signature = _signature(draft.constraints)
        if signature in have:
            continue
        have.add(signature)
        draft.id = f"task-{len(merged) + 1:04d}"
        merged.append(draft)
    return merged


def describe(constraint: Constraint) -> str:
    if constraint.kind == "category":
        return f"category within {constraint.path}"
    if constraint.kind == "attribute":
        return f"{constraint.handle} is {constraint.value}"
    return f"costs at most ${constraint.amount:.2f}"


def review(
    tasks: Sequence[Task],
    titles: Mapping[str, str],
    read,
    write,
) -> int:
    """Walk through written tasks that are not yet checked. Returns how many were approved."""
    approved = 0
    for task in tasks:
        if task.checked or not task.request.strip():
            continue
        write(f"\n{task.id}: {task.request}")
        for constraint in task.constraints:
            write(f"  must: {describe(constraint)}")
        kept = []
        for asin in task.acceptable:
            answer = read(f"  acceptable? {titles.get(asin, asin)[:80]} [Y/n/q] ").strip().lower()
            if answer == "q":
                return approved
            if answer != "n":
                kept.append(asin)
        if not kept:
            write("  no acceptable products left, so this task stays unchecked")
            continue
        task.acceptable = kept
        task.checked = True
        approved += 1
    return approved

"""Draft each task's request in a shopper's own words, for a person to check and edit.

The constraints and acceptable products still come only from hand labels and listing prices. A
model only proposes the wording, which a person then reads, edits if needed, and approves in the
review; a task is never run until then. The drafter is OpenAI's model, not Claude: the agent and
the frontier-enriched catalog are both Claude, and a request worded in Claude's own habits could
favor one of them. The drafter is never shown a product, only the requirements, so a request
cannot name the product it is meant to find.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from taxon.bench.tasks import Constraint, Task, lint_request
from taxon.labeling.budget import CAPS, Budget
from taxon.labeling.cache import ResponseCache
from taxon.labeling.env import load_secrets
from taxon.labeling.fakes import FakeBatchClient
from taxon.labeling.llm import LLMRequest, LLMResult, ModelConfig, Status, Usage
from taxon.labeling.runner import BatchRunner
from taxon.labeling.vendors import build_clients
from taxon.pins import DATA_DIR
from taxon.taxonomy import split_path

DRAFTER = ModelConfig("openai", "gpt-6-luna", max_tokens=1_024, effort="low")

INSTRUCTIONS = """\
You write shopping requests for a benchmark. Each task lists the requirements a product must meet. \
Write the one request a real shopper might type to a shopping assistant to find such a product.

- State every requirement, and nothing more. Never invent a brand, size, use or other detail.
- Use everyday words. Do not copy a category name word for word when a shopper would say it \
another way, and never write a category path or the ">" character.
- Give a price limit as a shopper would, such as "under $30" or "no more than $30".
- One or two sentences, in the first person, with no greeting.

Reply only with a JSON object: {"request": "..."}.
"""

SCHEMA = {
    "type": "object",
    "properties": {"request": {"type": "string"}},
    "required": ["request"],
    "additionalProperties": False,
}


def requirement(constraint: Constraint) -> str:
    """A requirement in plain words, as the drafter sees it."""
    if constraint.kind == "category":
        parts = split_path(constraint.path)
        within = f", a kind of {parts[-2]}" if len(parts) > 1 else ""
        return f"It is a product of this type: {parts[-1]}{within}."
    if constraint.kind == "attribute":
        name = constraint.handle.replace("-", " ")
        return f"Its {name} is {constraint.value}."
    return f"It costs at most ${constraint.amount:.2f}."


def draft_request(task: Task) -> LLMRequest:
    user = "Requirements:\n" + "\n".join(f"- {requirement(c)}" for c in task.constraints)
    return LLMRequest(DRAFTER, "request", INSTRUCTIONS, user, schema_json=json.dumps(SCHEMA))


def parse_request(text: str) -> str | None:
    try:
        request = json.loads(text)["request"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(request, str) or not request.strip():
        return None
    return " ".join(request.split())


def draft_requests(tasks: Sequence[Task], runner: BatchRunner) -> dict:
    """Fill in a drafted request on every task that has none and is not yet checked.

    Written or checked tasks are never touched, so drafting again loses no hand work.
    """
    blank = [t for t in tasks if not t.request.strip() and not t.checked]
    requests = {t.id: draft_request(t) for t in blank}
    results = runner.run(requests.values())
    drafted, failed, flagged = 0, [], []
    for task in blank:
        result = results[requests[task.id].key]
        text = parse_request(result.text) if result.status is Status.OK else None
        if text is None:
            failed.append(task.id)
            continue
        task.request = text
        task.drafted_by = DRAFTER.model
        drafted += 1
        if lint_request(text, task.constraints):
            flagged.append(task.id)
    return {"drafted": drafted, "failed": failed, "wording_to_fix": flagged}


def fake_request(request: LLMRequest) -> LLMResult:
    """A plain drafted request built from the requirements, for runs that spend nothing."""
    lines = [line[2:] for line in request.user.splitlines() if line.startswith("- ")]
    text = "I want something where: " + " ".join(lines).replace("It ", "it ")
    usage = Usage(input_tokens=len(request.user) // 4, output_tokens=len(text) // 4)
    return LLMResult(request.key, Status.OK, json.dumps({"request": text}), usage)


def runner(fake: bool) -> BatchRunner:
    """The labeling workspace's cache and ledger, so the OpenAI cap covers this too. A hundred
    short requests go one by one rather than waiting on a batch, for about a cent."""
    root = DATA_DIR / ("labeling_fake" if fake else "labeling")
    budget = Budget(root / "ledger.jsonl", CAPS)
    if fake:
        clients = {v: FakeBatchClient(v, fake_request) for v in CAPS}
    else:
        clients = build_clients(load_secrets(), CAPS, charge=budget.charge, direct=["openai"])
    return BatchRunner(clients, ResponseCache(root / "responses"), budget, root / "open_batches")

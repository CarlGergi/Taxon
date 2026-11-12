"""Drafting shopper requests for a person to check, with a fake model."""

import json

import pytest

from taxon.bench.drafting import (
    DRAFTER,
    INSTRUCTIONS,
    draft_request,
    draft_requests,
    fake_request,
    parse_request,
    requirement,
)
from taxon.bench.tasks import Constraint, Task, review
from taxon.labeling.budget import Budget
from taxon.labeling.cache import ResponseCache
from taxon.labeling.fakes import FakeBatchClient
from taxon.labeling.llm import LLMResult, Status
from taxon.labeling.runner import BatchRunner

TOTES = "Apparel > Bags > Totes"
CATEGORY = Constraint("category", path=TOTES)
RED = Constraint("attribute", handle="color", value="Red")
FABRIC = Constraint("attribute", handle="bag-material", value="Canvas")
CHEAP = Constraint("max_price", amount=30.0)


def task(task_id="task-0001", request="", checked=False, constraints=(CATEGORY, RED, CHEAP)):
    return Task(task_id, list(constraints), ["G1"], request, checked)


def runner_with(tmp_path, responder=fake_request):
    client = FakeBatchClient("openai", responder)
    return (
        BatchRunner(
            {"openai": client},
            ResponseCache(tmp_path / "cache"),
            Budget(tmp_path / "ledger.jsonl", {"openai": 1.0}),
            tmp_path / "open",
            sleep=lambda _: None,
        ),
        client,
    )


def replying(text):
    return lambda r: LLMResult(r.key, Status.OK, json.dumps({"request": text}))


# What the drafter sees


def test_requirements_read_plainly_and_never_as_a_path():
    assert requirement(CATEGORY) == "It is a product of this type: Totes, a kind of Bags."
    assert requirement(RED) == "Its color is Red."
    assert requirement(FABRIC) == "Its bag material is Canvas."
    assert requirement(CHEAP) == "It costs at most $30.00."
    assert ">" not in requirement(CATEGORY)


def test_the_drafter_sees_only_requirements_never_a_product():
    request = draft_request(task())
    assert request.config == DRAFTER
    assert request.image_path is None
    assert "G1" not in request.user
    assert request.user.count("\n- ") == 3


def test_the_drafter_is_not_claude():
    assert DRAFTER.vendor == "openai"


def test_the_instructions_forbid_invented_details_and_paths():
    assert "Never invent" in INSTRUCTIONS
    assert '">"' in INSTRUCTIONS


def test_a_reply_is_read_or_rejected():
    assert parse_request('{"request": "  I need   a red tote "}') == "I need a red tote"
    assert parse_request('{"request": ""}') is None
    assert parse_request('{"request": 3}') is None
    assert parse_request("a red tote") is None


# Drafting


def test_drafting_fills_blank_unchecked_tasks_only(tmp_path):
    blank = task("task-0001")
    written = task("task-0002", request="My own words")
    checked = task("task-0003", checked=True)
    runner, client = runner_with(tmp_path, replying("I want a red tote bag under $30."))
    report = draft_requests([blank, written, checked], runner)
    assert report == {"drafted": 1, "failed": [], "wording_to_fix": []}
    assert blank.request == "I want a red tote bag under $30."
    assert blank.drafted_by == DRAFTER.model and blank.checked is False
    assert written.request == "My own words" and written.drafted_by == ""
    assert checked.request == ""
    assert client.requests_sent == 1


def test_a_draft_that_breaks_the_wording_rules_is_kept_but_flagged(tmp_path):
    runner, _ = runner_with(tmp_path, replying("Apparel > Bags > Totes in red under $30"))
    blank = task()
    report = draft_requests([blank], runner)
    assert report["wording_to_fix"] == [blank.id]
    assert blank.request


def test_a_failed_draft_leaves_the_request_blank(tmp_path):
    runner, _ = runner_with(tmp_path, lambda r: LLMResult(r.key, Status.OK, "not json"))
    blank = task()
    report = draft_requests([blank], runner)
    assert report["failed"] == [blank.id]
    assert blank.request == "" and blank.drafted_by == ""


def test_drafting_twice_is_free_from_the_cache(tmp_path):
    runner, client = runner_with(tmp_path)
    draft_requests([task()], runner)
    again = task()
    draft_requests([again], runner)
    assert client.requests_sent == 1
    assert again.request


def test_the_fake_drafter_states_every_requirement():
    text = json.loads(fake_request(draft_request(task())).text)["request"]
    assert "Totes" in text and "Red" in text and "$30.00" in text


# Checking and editing


def test_keeping_a_drafted_request_approves_it_unedited():
    drafted = task(request="I want a red tote under $30.")
    drafted.drafted_by = DRAFTER.model
    lines = []
    answers = iter(["", "y"])
    assert review([drafted], {"G1": "Tote"}, lambda _: next(answers), lines.append) == 1
    assert drafted.checked and not drafted.edited
    assert any("drafted by gpt-6-luna" in line for line in lines)


def test_typing_a_new_request_replaces_the_draft_and_marks_it_edited():
    drafted = task(request="I want a red tote under $30.")
    answers = iter(["Looking for a red canvas-free tote, max $30", "y"])
    assert review([drafted], {}, lambda _: next(answers), lambda _: None) == 1
    assert drafted.request == "Looking for a red canvas-free tote, max $30"
    assert drafted.edited is True


def test_a_request_that_still_breaks_the_rules_is_not_approved():
    drafted = task(request="Apparel > Bags > Totes, red, under $30")
    lines = []
    answers = iter([""])
    assert review([drafted], {}, lambda _: next(answers), lines.append) == 0
    assert drafted.checked is False
    assert any("problem" in line for line in lines)


def test_fixing_the_wording_in_the_review_lets_it_through():
    drafted = task(request="Apparel > Bags > Totes, red, under $30")
    answers = iter(["A red tote bag for under $30", "y"])
    assert review([drafted], {}, lambda _: next(answers), lambda _: None) == 1


@pytest.mark.parametrize("field", ["drafted_by", "edited"])
def test_the_drafting_record_survives_saving(field):
    original = task(request="A red tote")
    original.drafted_by, original.edited = DRAFTER.model, True
    assert getattr(Task.from_json(original.to_json()), field) == getattr(original, field)


def test_old_task_files_without_the_record_still_load():
    data = task(request="A red tote").to_json()
    del data["drafted_by"], data["edited"]
    loaded = Task.from_json(data)
    assert loaded.drafted_by == "" and loaded.edited is False

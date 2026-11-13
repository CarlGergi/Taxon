import json
from pathlib import Path

import pytest

import taxon.bench.cli
import taxon.bench.tasks
from taxon.bench import cli
from taxon.bench.tasks import (
    Constraint,
    GoldItem,
    Task,
    TaskError,
    acceptable_products,
    check_tasks,
    describe,
    draft_tasks,
    fingerprint,
    gold_items,
    lint_request,
    load_tasks,
    merge_drafts,
    review,
    satisfies,
    save_tasks,
    validate_task,
)
from taxon.products import write_jsonl

TOTES = "Apparel > Bags > Totes"
BAGS = "Apparel > Bags"
KNIVES = "Home > Kitchen > Knives"
KITCHEN = "Home > Kitchen"
CAT_BAGS = Constraint("category", path=BAGS)
CAT_KITCHEN = Constraint("category", path=KITCHEN)
RED = Constraint("attribute", handle="color", value="Red")
CHEAP = Constraint("max_price", amount=20.0)


def items():
    return [
        GoldItem("G1", TOTES, {"color": "Red"}, 15.0),
        GoldItem("G2", TOTES, {"color": "Blue"}, 15.0),
        GoldItem("G3", BAGS, {"color": "Red"}, 40.0),
        GoldItem("G4", KNIVES, {"material": "Cotton"}, None),
        GoldItem("G5", KNIVES, {}, 12.0),
    ]


def task(constraints=(CAT_BAGS, RED), acceptable=("G1", "G3"), request="A red bag", checked=True):
    return Task("task-0001", list(constraints), list(acceptable), request, checked)


def test_gold_items_keep_only_plain_numeric_prices():
    labels = {
        "A": {"path": TOTES, "attributes": {}},
        "B": {"path": TOTES, "attributes": {}},
        "C": {"path": TOTES, "attributes": {}},
        "D": {"path": TOTES, "attributes": {}},
    }
    products = {"A": {"price": 9.5}, "B": {"price": "from 3.99"}, "C": {"price": None}, "D": {}}
    prices = {i.asin: i.price for i in gold_items(labels, products)}
    assert prices == {"A": 9.5, "B": None, "C": None, "D": None}


def test_a_category_constraint_covers_its_subtree_but_not_lookalikes():
    assert satisfies(items()[0], CAT_BAGS)
    assert satisfies(items()[2], CAT_BAGS)
    assert not satisfies(items()[3], CAT_BAGS)
    similar = GoldItem("X", "Apparel > Bags And More", {}, None)
    assert not satisfies(similar, CAT_BAGS)


def test_attribute_and_price_constraints():
    assert satisfies(items()[0], RED)
    assert not satisfies(items()[1], RED)
    assert satisfies(items()[0], CHEAP)
    assert not satisfies(items()[2], CHEAP)
    assert not satisfies(items()[3], CHEAP)


def test_an_unknown_constraint_kind_is_an_error():
    with pytest.raises(TaskError, match="unknown"):
        satisfies(items()[0], Constraint("vibe"))


def test_acceptable_products_meet_every_constraint():
    assert acceptable_products(items(), [CAT_BAGS, RED]) == ["G1", "G3"]
    assert acceptable_products(items(), [CAT_BAGS, RED, CHEAP]) == ["G1"]


def many_items(n=120):
    rows = []
    paths = [TOTES, KNIVES, BAGS, KITCHEN]
    for i in range(n):
        attributes = {"color": "Red" if i % 2 else "Blue"} if i % 3 else {}
        if i % 5 == 0:
            attributes["material"] = "Cotton"
        rows.append(GoldItem(f"G{i}", paths[i % 4], attributes, 5.0 + i if i % 4 else None))
    return rows


def test_drafts_follow_the_size_and_category_rules(taxonomy):
    drafts = draft_tasks(many_items(), 40, seed=1)
    assert drafts
    for draft in drafts:
        validate_task(draft, many_items(), taxonomy, training_ids=set(), final=False)
        assert draft.request == ""
        assert draft.checked is False


def test_drafts_are_deterministic_and_unique():
    first = draft_tasks(many_items(), 40, seed=1)
    second = draft_tasks(many_items(), 40, seed=1)
    assert [t.to_json() for t in first] == [t.to_json() for t in second]
    signatures = {tuple(sorted(c.signature() for c in t.constraints)) for t in first}
    assert len(signatures) == len(first)


def test_drafts_never_exceed_the_request_and_ids_are_unique():
    drafts = draft_tasks(many_items(), 7, seed=1)
    assert len(drafts) <= 7
    assert len({t.id for t in drafts}) == len(drafts)


def test_drafts_only_use_what_the_gold_labels_contain():
    allowed = {"G1", "G2", "G3"}
    rows = [i for i in items() if i.asin in allowed]
    for draft in draft_tasks(rows, 20, seed=1):
        assert set(draft.acceptable) <= allowed


def test_a_broad_match_tightens_rather_than_giving_up():
    rows = [GoldItem(f"G{i}", TOTES, {"color": "Red"}, 10.0 + i) for i in range(12)]
    drafts = draft_tasks(rows, 3, seed=1)
    assert drafts
    assert all(len(t.acceptable) <= 10 for t in drafts)
    assert all(len(t.constraints) >= 3 for t in drafts)


def test_a_product_with_no_attribute_or_price_yields_no_task():
    assert draft_tasks([GoldItem("A", TOTES, {}, None)], 5, seed=1) == []


def test_drafts_span_constraint_counts_when_labels_allow():
    counts = {len(t.constraints) for t in draft_tasks(many_items(), 40, seed=1)}
    assert counts >= {2, 3}


# Validation


def test_a_complete_task_passes(taxonomy):
    validate_task(task(), items(), taxonomy, {"TRAIN"})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"constraints": (CAT_BAGS,)}, "2 to 4"),
        ({"constraints": (CAT_BAGS, RED, CHEAP, CHEAP, CHEAP)}, "2 to 4"),
        ({"constraints": (RED, CHEAP)}, "one category"),
        ({"constraints": (CAT_BAGS, CAT_KITCHEN)}, "one category"),
        ({"constraints": (Constraint("category", path="Nope"), RED)}, "unknown category"),
        ({"constraints": (CAT_BAGS, RED, RED)}, "repeats"),
        ({"acceptable": ()}, "no acceptable"),
        ({"acceptable": ("G1", "NOT-GOLD")}, "outside the gold set"),
        ({"acceptable": ("G2",)}, "break its constraints"),
        ({"request": ""}, "empty"),
        ({"checked": False}, "hand-checked"),
    ],
)
def test_invalid_tasks_are_rejected(taxonomy, changes, message):
    with pytest.raises(TaskError, match=message):
        validate_task(task(**changes), items(), taxonomy, set())


def test_a_task_built_on_training_data_is_rejected(taxonomy):
    with pytest.raises(TaskError, match="in training data"):
        validate_task(task(), items(), taxonomy, {"G1"})


def test_hand_checking_may_narrow_the_acceptable_set(taxonomy):
    validate_task(task(acceptable=("G1",)), items(), taxonomy, set())


def test_a_draft_needs_no_hand_work_yet(taxonomy):
    validate_task(task(request="", checked=False), items(), taxonomy, set(), final=False)


# Fairness of wording


def test_a_shopper_phrasing_passes_lint():
    assert lint_request("a red tote under $20", [CAT_BAGS, RED]) == []


def test_lint_rejects_quoted_paths_and_multiword_category_names():
    leaf = Constraint("category", path="Home > Kitchen > Food Storage Bags")
    assert lint_request("Home > Kitchen please", [leaf])
    assert lint_request("I need food storage bags", [leaf])
    assert lint_request("something to keep leftovers in", [leaf]) == []


def test_lint_allows_a_single_word_name_no_one_could_avoid():
    assert lint_request("a set of knives", [Constraint("category", path=KNIVES)]) == []


def test_lint_rejects_attribute_handles():
    constraint = Constraint("attribute", handle="strap-color", value="Red")
    assert lint_request("with strap-color red", [constraint])


# Task sets


def test_check_summarizes_a_valid_set(taxonomy):
    second = Task("task-0002", [CAT_BAGS, CHEAP], ["G1", "G2"], "A cheap bag", True)
    report = check_tasks([task(), second], items(), taxonomy, set(), need=2)
    assert report["tasks"] == 2
    assert report["constraints"] == {"2": 2, "3": 0, "4": 0}
    assert report["mean_acceptable"] == 2
    assert len(report["sha256"]) == 64


def test_check_requires_enough_tasks(taxonomy):
    with pytest.raises(TaskError, match="need 100 hand-checked tasks, have 1"):
        check_tasks([task()], items(), taxonomy, set())


def test_check_runs_only_hand_checked_tasks_and_lists_the_rest(taxonomy):
    unverified = Task("task-0002", [CAT_BAGS, CHEAP], ["G9"], "A cheap bag", False)
    report = check_tasks([task(), unverified], items(), taxonomy, set(), need=1)
    assert report["tasks"] == 1
    assert report["left_unchecked"] == ["task-0002"]
    with pytest.raises(TaskError, match="need 2 hand-checked tasks, have 1"):
        check_tasks([task(), unverified], items(), taxonomy, set(), need=2)


def test_check_rejects_repeated_ids_and_requests(taxonomy):
    with pytest.raises(TaskError, match="ids are not unique"):
        check_tasks([task(), task()], items(), taxonomy, set(), need=2)
    other = Task("task-0002", [CAT_BAGS, CHEAP], ["G1"], "A red bag", True)
    with pytest.raises(TaskError, match="share a request"):
        check_tasks([task(), other], items(), taxonomy, set(), need=2)


def test_any_change_alters_the_fingerprint():
    changed = task(request="A crimson bag")
    assert fingerprint([task()]) == fingerprint([task()])
    assert fingerprint([task()]) != fingerprint([changed])


def test_tasks_round_trip_through_a_file(tmp_path):
    path = tmp_path / "tasks.jsonl"
    save_tasks(path, [task()])
    assert load_tasks(path)[0].to_json() == task().to_json()


def test_growing_the_set_keeps_written_tasks_untouched():
    written = task(request="My own words")
    drafts = [
        Task("task-0001", [CAT_BAGS, RED], ["G1"]),
        Task("task-0002", [CAT_KITCHEN, CHEAP], ["G5"]),
    ]
    merged = merge_drafts([written], drafts)
    assert merged[0].request == "My own words"
    assert len(merged) == 2
    assert merged[1].id == "task-0002"


def test_review_narrows_approves_and_saves_progress():
    tasks = [task(checked=False)]
    answers = iter(["", "y", "n"])
    assert review(tasks, {"G1": "Tote"}, lambda _: next(answers), lambda _: None) == 1
    assert tasks[0].acceptable == ["G1"]
    assert tasks[0].checked is True


def test_review_can_stop_part_way():
    tasks = [task(checked=False)]
    assert review(tasks, {}, lambda _: "q", lambda _: None) == 0
    assert tasks[0].checked is False


def test_review_leaves_a_blank_request_unchecked_and_rejecting_all_too():
    blank = task(request="", checked=False)
    rejected = task(checked=False)
    answers = iter(["", "", "n", "n", "n"])
    assert review([blank, rejected], {}, lambda _: next(answers), lambda _: None) == 0
    assert blank.checked is False and rejected.checked is False


def test_describe_reads_plainly():
    assert describe(CHEAP) == "costs at most $20.00"
    assert "color is Red" in describe(RED)
    assert BAGS in describe(CAT_BAGS)


@pytest.mark.parametrize("module", [taxon.bench.tasks, taxon.bench.cli])
def test_tasks_are_built_without_any_model_output(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("taxon.labeling", "training_labels", "shortlist", "baseline"):
        assert forbidden not in source


@pytest.fixture
def workspace(tmp_path, monkeypatch, taxonomy):
    monkeypatch.setattr(cli, "fetch", lambda _: None)
    monkeypatch.setattr(cli.Taxonomy, "load", lambda _: taxonomy)
    derived = tmp_path / "derived"
    products = [
        {"parent_asin": f"G{i}", "title": f"T{i}", "price": 10.0 + i, "domain": "d"}
        for i in range(12)
    ] + [{"parent_asin": "TRAIN", "title": "T", "price": 1.0, "domain": "d"}]
    write_jsonl(derived / "products.jsonl", products)
    write_jsonl(derived / "gold_queue.jsonl", [{"parent_asin": f"G{i}"} for i in range(12)])
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {"event": "label", "asin": f"G{i}", "path": TOTES, "attributes": {"color": "Red"}}
            for i in range(12)
        ],
    )
    monkeypatch.setattr(cli, "drafting_runner", lambda fake: fake_drafting_runner(tmp_path))
    return {"tasks_path": tmp_path / "tasks.jsonl", "labels_path": labels, "derived": derived}


def fake_drafting_runner(root):
    from taxon.bench.drafting import fake_request
    from taxon.labeling.budget import Budget
    from taxon.labeling.cache import ResponseCache
    from taxon.labeling.fakes import FakeBatchClient
    from taxon.labeling.runner import BatchRunner

    return BatchRunner(
        {"openai": FakeBatchClient("openai", fake_request)},
        ResponseCache(root / "cache"),
        Budget(root / "ledger.jsonl", {"openai": 1.0}),
        root / "open",
        sleep=lambda _: None,
    )


def test_draft_then_check_from_the_command_line(workspace, capsys):
    assert cli.main(["draft", "--n", "4"], **workspace) == 0
    assert "hand-labeled" in capsys.readouterr().out
    tasks = load_tasks(workspace["tasks_path"])
    assert tasks
    assert all("TRAIN" not in t.acceptable for t in tasks)
    for i, t in enumerate(tasks):
        t.request = f"A bag for trip {i}"
        t.checked = True
    save_tasks(workspace["tasks_path"], tasks)
    assert cli.main(["check", "--need", str(len(tasks))], **workspace) == 0
    assert json.loads(capsys.readouterr().out)["tasks"] == len(tasks)


def test_drafting_tasks_also_drafts_their_requests(workspace, capsys):
    assert cli.main(["draft", "--n", "3"], **workspace) == 0
    out = capsys.readouterr().out
    assert "tasks-review" in out
    tasks = load_tasks(workspace["tasks_path"])
    assert all(t.request and t.drafted_by == "gpt-6-luna" for t in tasks)
    assert not any(t.checked for t in tasks)


def test_requests_can_be_left_blank(workspace):
    assert cli.main(["draft", "--n", "3", "--no-requests"], **workspace) == 0
    assert all(t.request == "" for t in load_tasks(workspace["tasks_path"]))


def test_drafting_again_keeps_hand_written_requests(workspace):
    cli.main(["draft", "--n", "2"], **workspace)
    tasks = load_tasks(workspace["tasks_path"])
    tasks[0].request = "Hand written"
    save_tasks(workspace["tasks_path"], tasks)
    cli.main(["draft", "--n", "4"], **workspace)
    assert load_tasks(workspace["tasks_path"])[0].request == "Hand written"


def test_too_few_labels_says_so_rather_than_inventing_tasks(workspace, capsys):
    cli.main(["draft", "--n", "500"], **workspace)
    assert "Label more products" in capsys.readouterr().out


def test_each_draft_records_the_product_it_started_from(tmp_path):
    from taxon.bench.tasks import GoldItem, draft_tasks, load_tasks, save_tasks

    items = [
        GoldItem("A1", "Home > Kitchen > Knives", {"color": "black", "material": "steel"}, 20.0),
        GoldItem("A2", "Home > Kitchen > Pans", {"color": "red", "material": "iron"}, 40.0),
    ]
    tasks = draft_tasks(items, n=4, seed=1)
    assert tasks and all(t.source in {"A1", "A2"} and t.source in t.acceptable for t in tasks)
    save_tasks(tmp_path / "tasks.jsonl", tasks)
    assert [t.source for t in load_tasks(tmp_path / "tasks.jsonl")] == [t.source for t in tasks]

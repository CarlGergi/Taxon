import json

import pytest

from taxon.bench import cli, judging
from taxon.bench.judge import (
    AGREEMENT_BAR,
    FAILS,
    JUDGE,
    MEETS,
    MIN_ANSWERED,
    UNSURE,
    acceptable_with_pool,
    agreement,
    estimate_cost,
    hand_review,
    judge_pairs,
    judge_request,
    judge_schema,
    parse_verdict,
    pool,
    price_verdict,
    validation_sample,
)
from taxon.bench.tasks import Constraint, Task, save_tasks
from taxon.labeling.budget import Budget
from taxon.labeling.cache import ResponseCache
from taxon.labeling.fakes import FakeBatchClient
from taxon.labeling.llm import LLMResult, Status, Usage
from taxon.labeling.pipeline import ARBITRATOR
from taxon.labeling.runner import BatchRunner
from taxon.products import read_jsonl, write_jsonl

TOTES = "Apparel > Bags > Totes"
CATEGORY = Constraint("category", path="Apparel > Bags")
RED = Constraint("attribute", handle="color", value="Red")
CHEAP = Constraint("max_price", amount=30.0)
TASK = Task("t1", [CATEGORY, RED, CHEAP], ["G1"], "a red bag under $30", True)
PRICE_ONLY = Task("t2", [CHEAP, Constraint("max_price", amount=50.0)], ["G2"], "cheap", True)


def product(asin, price=20.0, **extra):
    return {
        "parent_asin": asin,
        "title": f"Canvas tote {asin}",
        "features": [],
        "description": "A red bag",
        "details": {"Color": "Red"},
        "price": price,
        "domain": "d",
    } | extra


def test_the_judge_is_not_from_the_arbitrator_s_family():
    """The arbitrator's labels shape the Taxon catalog, so a judge of its family could favor it."""
    assert JUDGE.vendor != ARBITRATOR.vendor
    assert JUDGE.vendor == "openai"


def test_the_schema_wants_one_answer_per_requirement():
    schema = judge_schema(2)
    assert schema["required"] == ["r1", "r2"]
    assert schema["properties"]["r1"]["enum"] == ["yes", "no", "unsure"]


def test_price_is_checked_in_code():
    assert price_verdict(TASK, product("X", 40.0)) == FAILS
    assert price_verdict(TASK, product("X", None)) == FAILS
    assert price_verdict(TASK, product("X", "about $5")) == FAILS
    assert price_verdict(TASK, product("X", 20.0)) is None
    assert price_verdict(PRICE_ONLY, product("X", 20.0)) == MEETS


def test_the_judge_sees_the_raw_listing_and_only_the_other_requirements():
    enriched = product("X", category="Apparel > Bags > Totes", attributes={"color": ["Red"]})
    request = judge_request(TASK, enriched, None)
    assert request.purpose == "judge"
    assert request.config == JUDGE
    assert "r1. category within Apparel > Bags\nr2. color is Red" in request.user
    assert "r3" not in request.user
    assert "Apparel > Bags > Totes" not in request.user
    assert request.schema == judge_schema(2)


@pytest.mark.parametrize(
    "text,verdict",
    [
        ('{"r1": "yes", "r2": "yes"}', MEETS),
        ('{"r1": "yes", "r2": "no"}', FAILS),
        ('{"r1": "unsure", "r2": "no"}', FAILS),
        ('{"r1": "yes", "r2": "unsure"}', UNSURE),
        ('{"r1": "yes"}', UNSURE),
        ('{"r1": "maybe", "r2": "yes"}', UNSURE),
        ("junk", UNSURE),
    ],
)
def test_parse_verdict(text, verdict):
    assert parse_verdict(text, 2)[0] == verdict


def runs():
    return [
        {"condition": "raw", "run": 1, "task": "t1", "returned": ["G1", "X", "Y"]},
        {"condition": "raw", "run": 2, "task": "t1", "returned": ["X"]},
        {"condition": "model", "run": 1, "task": "t1", "returned": ["X", "Z"]},
        {"condition": "model", "run": 1, "task": "t2", "returned": ["Y"]},
    ]


def test_the_pool_holds_each_non_gold_pair_once_with_its_conditions():
    pairs = pool([TASK, PRICE_ONLY], runs(), {"G1", "G2"})
    assert [(p["task"], p["asin"]) for p in pairs] == [
        ("t1", "X"),
        ("t1", "Y"),
        ("t1", "Z"),
        ("t2", "Y"),
    ]
    assert pairs[0]["conditions"] == ["model", "raw"]


def test_a_run_for_an_unknown_task_is_refused():
    with pytest.raises(ValueError, match="unknown task"):
        pool([TASK], [{"condition": "raw", "run": 1, "task": "nope", "returned": []}], set())


def test_the_validation_sample_is_seeded_and_order_independent():
    pairs = [{"task": "t1", "asin": f"P{i}", "conditions": ["raw"]} for i in range(200)]
    sample = validation_sample(pairs, 7)
    assert len(sample) == 50
    assert sample == validation_sample(list(reversed(pairs)), 7)
    assert sample != validation_sample(pairs, 8)


def test_the_estimate_counts_only_pairs_that_need_the_judge():
    products = {"X": product("X"), "Y": product("Y", 99.0)}
    pairs = [{"task": "t1", "asin": "X"}, {"task": "t1", "asin": "Y"}]
    text_only = estimate_cost(pairs, {"t1": TASK}, products, with_images=False)
    with_images = estimate_cost(pairs, {"t1": TASK}, products, with_images=True)
    assert (text_only["judge_calls"], text_only["decided_by_price"]) == (1, 1)
    assert 0 < text_only["estimate"] < with_images["estimate"] <= with_images["worst_case"]


def fake_runner(tmp_path, responder=None):
    clients = {JUDGE.vendor: FakeBatchClient(JUDGE.vendor, *(responder,) if responder else ())}
    budget = Budget(tmp_path / "ledger.jsonl", {JUDGE.vendor: 5.0})
    return BatchRunner(clients, ResponseCache(tmp_path / "c"), budget, tmp_path / "o"), clients


def no_image(_product):
    return None


def test_judging_calls_only_when_price_does_not_settle_it(tmp_path):
    runner, clients = fake_runner(tmp_path)
    products = {"X": product("X"), "Y": product("Y", 99.0)}
    pairs = [{"task": "t1", "asin": "X"}, {"task": "t1", "asin": "Y"}]
    judged = judge_pairs(pairs, {"t1": TASK}, products, runner, no_image)
    assert clients[JUDGE.vendor].requests_sent == 1
    assert [(j["asin"], j["by"]) for j in judged] == [("X", "judge"), ("Y", "price")]
    assert judged[1]["verdict"] == FAILS


def test_a_failed_judge_call_is_unsure(tmp_path):
    def refuse(request):
        return LLMResult(request.key, Status.REFUSED, "", Usage())

    runner, _ = fake_runner(tmp_path, refuse)
    judged = judge_pairs(
        [{"task": "t1", "asin": "X"}], {"t1": TASK}, {"X": product("X")}, runner, no_image
    )
    assert judged[0]["verdict"] == UNSURE


def test_hand_review_never_shows_the_judge_and_records_answers():
    pairs = [{"task": "t1", "asin": a} for a in "ABCD"]
    products = {a: product(a) for a in "ABCD"}
    shown = []
    answers = iter(["y", "n", "s", "q"])
    new = hand_review(
        pairs, {"t1": TASK}, products, {}, lambda _prompt: next(answers), shown.append
    )
    assert [(h["asin"], h["verdict"]) for h in new] == [("A", MEETS), ("B", FAILS)]
    text = "\n".join(shown)
    assert "judge" not in text.lower() and "verdict" not in text.lower()
    assert "must: color is Red" in text


def test_hand_review_skips_what_is_already_judged():
    asked = []
    hand_review(
        [{"task": "t1", "asin": "A"}],
        {"t1": TASK},
        {"A": product("A")},
        {("t1", "A"): MEETS},
        lambda p: asked.append(p) or "y",
        lambda _t: None,
    )
    assert asked == []


def judgments(verdicts, condition="raw"):
    return [
        {"task": "t1", "asin": f"P{i}", "verdict": v, "by": "judge", "conditions": [condition]}
        for i, v in enumerate(verdicts)
    ]


def hands(verdicts):
    return [
        {"task": "t1", "asin": f"P{i}", "verdict": v, "by": "hand"} for i, v in enumerate(verdicts)
    ]


def test_agreement_leaves_out_unsure_answers_and_reports_kappa():
    report = agreement(
        judgments([MEETS, MEETS, FAILS, UNSURE]), hands([MEETS, FAILS, FAILS, MEETS])
    )
    assert (report["hand_judged"], report["answered"], report["judge_unsure"]) == (4, 3, 1)
    assert report["agreement"] == pytest.approx(2 / 3)
    assert report["confusion"]["judge_meets_hand_fails"] == 1
    assert report["kappa"] is not None
    assert report["per_condition"]["raw"]["answered"] == 3


def test_the_judge_is_trusted_only_with_high_agreement_on_enough_pairs():
    good = [MEETS, FAILS] * 25
    assert agreement(judgments(good), hands(good))["trusted"] is True
    few = good[: MIN_ANSWERED - 1]
    assert agreement(judgments(few), hands(few))["trusted"] is False
    flipped = good[:]
    # Six of fifty wrong is 88%, under the 90% bar.
    for i in range(6):
        flipped[i] = FAILS if good[i] == MEETS else MEETS
    assert agreement(judgments(flipped), hands(good))["trusted"] is False
    assert AGREEMENT_BAR == 0.90


def test_agreement_is_reported_per_condition():
    judged = judgments([MEETS] * 4, "raw") + [
        {"task": "t1", "asin": "Q", "verdict": MEETS, "by": "judge", "conditions": ["model"]}
    ]
    hand = hands([MEETS] * 4) + [{"task": "t1", "asin": "Q", "verdict": FAILS, "by": "hand"}]
    per = agreement(judged, hand)["per_condition"]
    assert per["raw"]["agreement"] == 1.0
    assert per["model"]["agreement"] == 0.0


def test_acceptable_products_add_pooled_ones_with_hand_overriding_the_judge():
    judged = [
        {"task": "t1", "asin": "A", "verdict": MEETS, "by": "judge"},
        {"task": "t1", "asin": "B", "verdict": MEETS, "by": "judge"},
        {"task": "t1", "asin": "C", "verdict": UNSURE, "by": "judge"},
        {"task": "t1", "asin": "D", "verdict": UNSURE, "by": "judge"},
        {"task": "t2", "asin": "E", "verdict": MEETS, "by": "judge"},
    ]
    hand = [
        {"task": "t1", "asin": "B", "verdict": FAILS, "by": "hand"},
        {"task": "t1", "asin": "C", "verdict": MEETS, "by": "hand"},
    ]
    accepted, unresolved = acceptable_with_pool(TASK, judged, hand)
    assert accepted == {"G1", "A", "C"}
    assert unresolved == ["D"]


@pytest.fixture
def bench(tmp_path, monkeypatch, taxonomy):
    monkeypatch.setattr(cli, "fetch", lambda _: None)
    monkeypatch.setattr(cli.Taxonomy, "load", lambda _: taxonomy)
    derived = tmp_path / "derived"
    products = [product(f"G{i}") for i in range(3)] + [
        product(f"N{i:02}", 20.0 if i % 4 else 99.0) for i in range(80)
    ]
    write_jsonl(derived / "products.jsonl", products)
    write_jsonl(derived / "gold_queue.jsonl", [{"parent_asin": f"G{i}"} for i in range(3)])
    labels = tmp_path / "gold.jsonl"
    write_jsonl(
        labels,
        [{"event": "label", "asin": f"G{i}", "path": TOTES, "attributes": {}} for i in range(3)],
    )
    tasks = tmp_path / "tasks.jsonl"
    save_tasks(tasks, [Task("t1", [CATEGORY, CHEAP], ["G0"], "a bag under $30", True)])
    runs_path = tmp_path / "runs.jsonl"
    write_jsonl(
        runs_path,
        [
            {
                "condition": c,
                "run": 1,
                "task": "t1",
                "returned": [f"N{i:02}" for i in range(80)] + ["G0"],
            }
            for c in ("raw", "model")
        ],
    )
    return {
        "tasks_path": tasks,
        "labels_path": labels,
        "derived": derived,
        "judging_dir": tmp_path / "judging",
        "labeling_dir": tmp_path / "labeling",
    }, runs_path


def judge_cmd(bench, command, read=input):
    paths, runs_path = bench
    return cli.main([command, "--runs", str(runs_path), "--fake"], read=read, **paths)


def output(capsys):
    text = capsys.readouterr().out
    return json.loads(text[text.index("{") :])


def test_pricing_spends_nothing(bench, capsys):
    assert judge_cmd(bench, "judge-price") == 0
    report = output(capsys)
    assert report["whole_pool"]["pairs"] == 80
    assert report["whole_pool"]["decided_by_price"] == 20
    assert report["validation_sample"]["pairs"] == 50
    assert not (bench[0]["labeling_dir"] / "ledger.jsonl").exists()


def test_the_pool_is_refused_until_the_judge_is_trusted(bench):
    with pytest.raises(SystemExit, match="not trusted"):
        judge_cmd(bench, "judge-pool")


def answer_like_the_judge(bench, agree=True):
    judged = {
        (j["task"], j["asin"]): j["verdict"]
        for j in read_jsonl(bench[0]["judging_dir"] / "judgments.jsonl")
    }
    current = {}

    def read(prompt):
        verdict = judged.get(current["key"], MEETS)
        if verdict == UNSURE:
            return "y"
        return "y" if (verdict == MEETS) == agree else "n"

    def write(text):
        if text.startswith("  product: Canvas tote "):
            current["key"] = ("t1", text.rsplit(" ", 1)[-1])

    return read, write


def test_the_full_flow_trusts_an_agreeing_judge_then_judges_the_pool(bench, capsys, monkeypatch):
    assert judge_cmd(bench, "judge-sample") == 0
    sampled = output(capsys)
    assert sampled["judged"] == 50
    read, write = answer_like_the_judge(bench, agree=True)
    monkeypatch.setattr(judging, "hand_review", _with_writer(judging.hand_review, write))
    assert judge_cmd(bench, "judge-review", read) == 0
    capsys.readouterr()
    assert judge_cmd(bench, "judge-agreement") == 0
    assert output(capsys)["trusted"] is True
    assert judge_cmd(bench, "judge-pool") == 0
    report = output(capsys)
    assert report["pairs"] == 80
    assert sum(report["verdicts"].values()) == 80


def test_a_disagreeing_judge_is_not_used(bench, capsys, monkeypatch):
    assert judge_cmd(bench, "judge-sample") == 0
    read, write = answer_like_the_judge(bench, agree=False)
    monkeypatch.setattr(judging, "hand_review", _with_writer(judging.hand_review, write))
    assert judge_cmd(bench, "judge-review", read) == 0
    assert judge_cmd(bench, "judge-agreement") == 0
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not trusted"):
        judge_cmd(bench, "judge-pool")


def test_judging_stops_before_passing_the_cap(bench, monkeypatch):
    monkeypatch.setattr(judging, "_remaining", lambda _space: 0.0001)
    with pytest.raises(SystemExit, match="more than"):
        judge_cmd(bench, "judge-sample")


def _with_writer(review, write):
    def wrapped(pairs, tasks, products, done, read, _write):
        return review(pairs, tasks, products, done, read, write)

    return wrapped

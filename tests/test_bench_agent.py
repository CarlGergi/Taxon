"""The shopping agent, against a scripted model and both a fake and the real catalog."""

import asyncio
import json

import pytest

from taxon.bench import agent, search
from taxon.bench.agent import (
    AGENT,
    MAX_MODEL_CALLS,
    MAX_RETURNED,
    NUDGE,
    SUBMIT,
    SUBMIT_TOOL,
    McpCatalog,
    RunLog,
    Scenario,
    price_plan,
    request_params,
    run_condition,
    run_cost,
    runnable,
    shop,
    submitted_ids,
    worst_case_cost,
)
from taxon.bench.catalog import benchmark_ids, raw_catalog
from taxon.bench.fakes import (
    FAKE_TOOL,
    FakeCatalog,
    ScriptedModel,
    demo_reply,
    message,
    tool_use,
)
from taxon.bench.tasks import Constraint, Task
from taxon.labeling.budget import Budget, BudgetExceeded
from taxon.labeling.pricing import cost_at, interactive_price_for
from taxon.products import write_jsonl

LISTINGS = [
    {"id": "A", "title": "Red canvas tote", "price": 20.0},
    {"id": "B", "title": "Blue canvas tote", "price": 40.0},
    {"id": "C", "title": "Chef knife", "price": 15.0},
]
CHEAP = Constraint("max_price", amount=30.0)


def task(task_id="t1", request="red canvas tote", checked=True):
    return Task(task_id, [CHEAP, Constraint("max_price", amount=50.0)], ["A"], request, checked)


@pytest.fixture
def budget(tmp_path):
    return Budget(tmp_path / "ledger.jsonl", {"anthropic": 5.0})


def run(coroutine):
    return asyncio.run(coroutine)


def shop_with(script, budget, catalog=None, request="red canvas tote"):
    model = ScriptedModel(script)
    outcome = run(shop(request, catalog or FakeCatalog(LISTINGS), model, budget, "k"))
    return outcome, model


def submitting(ids):
    return message([tool_use(SUBMIT, {"product_ids": ids})], "tool_use")


# One run


def test_the_demo_agent_searches_then_submits_what_it_found(budget):
    outcome, model = shop_with(demo_reply, budget)
    assert outcome.stopped == "submitted"
    assert outcome.returned == ["A", "B"]
    assert (outcome.searches, outcome.model_calls) == (1, 2)
    assert len(model.calls) == 2


def test_cost_is_priced_interactively_and_settled_on_the_ledger(budget, tmp_path):
    outcome, _ = shop_with(demo_reply, budget)
    price = interactive_price_for(AGENT)
    assert outcome.cost == pytest.approx(cost_at(outcome.usage, price))
    assert outcome.cost > 0
    reloaded = Budget(tmp_path / "ledger.jsonl", {"anthropic": 5.0})
    assert reloaded.spent("anthropic") == pytest.approx(outcome.cost)
    assert reloaded.settled("anthropic") == pytest.approx(outcome.cost)


def test_cache_reads_and_writes_count_at_their_own_prices(budget):
    reply = message(
        [tool_use(SUBMIT, {"product_ids": []})], "tool_use", 100, 10, cached=1000, written=200
    )
    outcome, _ = shop_with(lambda p, c: reply, budget)
    assert outcome.usage.cached_tokens == 1000
    assert outcome.usage.cache_write_tokens == 200
    expected = (100 * 0.10 + 1000 * 0.01 + 200 * 0.125 + 10 * 0.50) / 1_000_000
    assert outcome.cost == pytest.approx(expected)


def test_search_results_go_back_as_one_tool_result_per_call(budget):
    def script(params, call):
        if call == 0:
            uses = [
                tool_use(search.TOOL, {"query": "tote"}, "u1"),
                tool_use(search.TOOL, {"query": "knife"}, "u2"),
            ]
            return message(uses, "tool_use")
        return submitting(["C"])

    outcome, model = shop_with(script, budget)
    results = model.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["u1", "u2"]
    assert json.loads(results[1]["content"])["results"][0]["id"] == "C"
    assert not any(r.get("is_error") for r in results)
    assert outcome.searches == 2


def test_a_refused_search_and_an_unknown_tool_come_back_as_errors(budget):
    def script(params, call):
        if call == 0:
            uses = [
                tool_use(search.TOOL, {"query": 7}, "u1"),
                tool_use("checkout", {}, "u2"),
            ]
            return message(uses, "tool_use")
        return submitting([])

    outcome, model = shop_with(script, budget)
    results = model.calls[1]["messages"][-1]["content"]
    assert all(r["is_error"] for r in results)
    assert "Unknown tool" in results[1]["content"]
    assert outcome.searches == 1


def test_an_answer_in_prose_gets_one_nudge_to_submit(budget):
    def script(params, call):
        if call == 0:
            return message([{"type": "text", "text": "Product A fits."}], "end_turn")
        return submitting(["A"])

    outcome, model = shop_with(script, budget)
    assert model.calls[1]["messages"][-1] == {"role": "user", "content": NUDGE}
    assert outcome.returned == ["A"]


def test_an_agent_that_never_submits_stops_at_the_call_limit(budget):
    def script(params, call):
        return message([tool_use(search.TOOL, {"query": "tote"}, f"u{call}")], "tool_use")

    outcome, model = shop_with(script, budget)
    assert outcome.stopped == "call_limit"
    assert outcome.returned == []
    assert len(model.calls) == outcome.model_calls == MAX_MODEL_CALLS


@pytest.mark.parametrize("stop, label", [("refusal", "refused"), ("max_tokens", "max_tokens")])
def test_a_refusal_or_a_cut_off_reply_ends_the_run_with_nothing(budget, stop, label):
    reply = message([tool_use(SUBMIT, {"product_ids": ["A"]})], stop)
    outcome, _ = shop_with(lambda p, c: reply, budget)
    assert (outcome.stopped, outcome.returned) == (label, [])


def test_the_answer_is_cleaned_and_capped():
    many = [f"P{i}" for i in range(MAX_RETURNED + 5)]
    assert submitted_ids({"product_ids": [" A", "A", 3, "", "B"]}) == ["A", "B"]
    assert submitted_ids({"product_ids": many}) == many[:MAX_RETURNED]
    assert submitted_ids({"product_ids": "A"}) == []
    assert submitted_ids({}) == []


def test_the_transcript_keeps_every_turn_as_plain_data(budget):
    outcome, _ = shop_with(demo_reply, budget)
    roles = [m["role"] for m in outcome.transcript]
    assert roles == ["user", "assistant", "user", "assistant"]
    json.dumps(outcome.transcript)


# What is sent


def test_every_request_holds_the_model_the_prompt_and_the_tools_fixed(budget):
    _, model = shop_with(demo_reply, budget)
    for params in model.calls:
        assert params["model"] == "claude-haiku-5-5"
        assert params["system"] == agent.SYSTEM
        assert params["tools"] == [FAKE_TOOL, SUBMIT_TOOL]
        assert params["thinking"] == {"type": "disabled"}
        assert params["output_config"] == {"effort": "low"}
        assert "temperature" not in params
        assert params["cache_control"] == {"type": "ephemeral"}


def test_the_prompt_treats_listings_as_data():
    assert "not instructions" in agent.SYSTEM


def test_the_submit_tool_is_strict():
    schema = SUBMIT_TOOL["input_schema"]
    assert SUBMIT_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["product_ids"]


def test_the_worst_case_grows_with_the_conversation():
    short = request_params([FAKE_TOOL], [{"role": "user", "content": "a"}])
    long = request_params([FAKE_TOOL], [{"role": "user", "content": "a" * 10_000}])
    assert worst_case_cost(long) > worst_case_cost(short) > 0


# The cap


def test_a_call_that_could_pass_the_cap_is_never_sent(tmp_path):
    tight = Budget(tmp_path / "l.jsonl", {"anthropic": 1e-6})
    model = ScriptedModel(demo_reply)
    with pytest.raises(BudgetExceeded) as caught:
        run(shop("tote", FakeCatalog(LISTINGS), model, tight, "k"))
    assert caught.value.vendor == "anthropic"
    assert model.calls == []


def test_a_failed_call_keeps_its_worst_case_on_the_ledger(budget):
    class Broken:
        async def create(self, **params):
            raise ConnectionError("dropped")

    with pytest.raises(ConnectionError):
        run(shop("tote", FakeCatalog(LISTINGS), Broken(), budget, "k"))
    assert budget.settled("anthropic") > 0
    assert budget.spent("anthropic") == budget.settled("anthropic")


# Many runs


def test_a_condition_logs_each_run_and_a_rerun_skips_them(budget, tmp_path):
    log = RunLog(tmp_path / "agent")
    tasks = [task("t1"), task("t2", "chef knife")]
    catalog = FakeCatalog(LISTINGS)
    model = ScriptedModel(demo_reply)
    assert run(run_condition("raw", tasks, 3, catalog, model, budget, log, report=print)) == 6
    rows = log.rows()
    assert sorted((r["run"], r["task"]) for r in rows) == [
        (r, t) for r in (1, 2, 3) for t in ("t1", "t2")
    ]
    assert {r["condition"] for r in rows} == {"raw"}
    knife = next(r for r in rows if r["task"] == "t2")
    assert knife["returned"] == ["C"]
    assert (tmp_path / "agent" / "transcripts" / "raw" / "2" / "t1.json").exists()
    calls = len(model.calls)
    assert run(run_condition("raw", tasks, 3, catalog, model, budget, log)) == 0
    assert len(model.calls) == calls


def test_the_log_rows_are_what_pooled_judging_reads(budget, tmp_path):
    log = RunLog(tmp_path / "agent")
    run(
        run_condition(
            "raw", [task()], 1, FakeCatalog(LISTINGS), ScriptedModel(demo_reply), budget, log
        )
    )
    row = log.rows()[0]
    assert {"condition", "run", "task", "returned"} <= set(row)
    assert row["model"] == AGENT.model


def test_only_checked_tasks_with_a_request_are_shopped():
    ready = runnable([task("t1"), task("t2", checked=False), task("t3", request=" ")])
    assert [t.id for t in ready] == ["t1"]
    with pytest.raises(ValueError, match="hand-checked"):
        runnable([task("t2", checked=False)])


def test_a_condition_stops_cleanly_at_the_cap_and_keeps_finished_runs(tmp_path):
    first = Budget(tmp_path / "l.jsonl", {"anthropic": 5.0})
    log = RunLog(tmp_path / "agent")
    catalog = FakeCatalog(LISTINGS)
    run(run_condition("raw", [task("t1")], 1, catalog, ScriptedModel(demo_reply), first, log))
    tight = Budget(tmp_path / "l.jsonl", {"anthropic": first.spent("anthropic") + 1e-6})
    with pytest.raises(BudgetExceeded):
        run(
            run_condition(
                "raw", [task("t1"), task("t2")], 1, catalog, ScriptedModel(demo_reply), tight, log
            )
        )
    assert [r["task"] for r in log.rows()] == ["t1"]


# The benchmark catalog


def test_the_catalog_holds_all_gold_and_distractors_from_outside_training():
    ids = [f"P{i:03d}" for i in range(100)]
    gold = ids[:10]
    trained = ids[10:40]
    chosen = benchmark_ids(ids, gold, trained, seed=1, distractors=20)
    assert len(chosen) == 30
    assert set(gold) <= set(chosen)
    assert not set(trained) & set(chosen)
    assert chosen == benchmark_ids(list(reversed(ids)), gold, trained, seed=1, distractors=20)
    assert chosen != benchmark_ids(ids, gold, trained, seed=2, distractors=20)


def test_the_catalog_refuses_when_there_are_too_few_distractors():
    with pytest.raises(ValueError, match="distractors"):
        benchmark_ids(["A", "B", "C"], ["A"], ["B"], seed=1, distractors=5)


def test_the_planned_training_set_is_the_first_labeling_run():
    products = [{"parent_asin": f"P{i}", "domain": "d"} for i in range(3_000)]
    queue = [f"P{i}" for i in range(500)]
    planned = agent.planned_training_ids(products, queue)
    assert len(planned) == 2_000
    assert not set(planned) & set(queue)


# Pricing


def test_a_run_costs_more_with_more_searches_and_no_cache():
    cheap = Scenario("cheap", 1, 50, True, 1.0)
    dear = Scenario("dear", 6, 400, False, 1.5)
    assert 0 < run_cost(1_000, 800, cheap) < run_cost(1_000, 800, dear)


def test_caching_makes_later_calls_cheaper():
    cached = Scenario("cached", 4, 100, True, 1.0)
    uncached = Scenario("uncached", 4, 100, False, 1.0)
    assert run_cost(2_000, 1_000, cached) < run_cost(2_000, 1_000, uncached)


def test_a_plan_scales_with_tasks_conditions_and_runs():
    report = price_plan(FAKE_TOOL, 900.0, (100, 300), 3, 3, spent=1.0, cap=20.0)
    small, large = report["plans"]["100"], report["plans"]["300"]
    assert small["agent_runs"] == 900 and large["agent_runs"] == 2_700
    assert large["expected"] == pytest.approx(3 * small["expected"])
    assert small["high"] > small["expected"]
    assert report["per_run"]["expected"] * 900 == pytest.approx(small["expected"])
    assert report["anthropic_cap"] == 20.0


# The real catalog over MCP


@pytest.fixture(scope="module")
def raw_index(tmp_path_factory):
    try:
        binary = search.build_binary()
    except search.RustMissing as error:
        pytest.skip(str(error))
    root = tmp_path_factory.mktemp("agent_catalog")
    products = [
        {"parent_asin": row["id"], "title": row["title"], "price": row["price"]} for row in LISTINGS
    ]
    write_jsonl(root / "raw.jsonl", raw_catalog(products))
    search.build_index(root / "raw.jsonl", root / "raw", binary)
    return root / "raw"


def test_the_agent_shops_the_real_engine_over_mcp(raw_index, budget):
    async def go():
        async with McpCatalog(raw_index) as catalog:
            model = ScriptedModel(demo_reply)
            outcome = await shop("chef knife", catalog, model, budget, "k")
            return catalog.tool, outcome

    tool, outcome = run(go())
    assert tool["name"] == search.TOOL
    assert "query" in tool["input_schema"]["properties"]
    assert outcome.returned[0] == "C"


def test_the_real_tool_schema_is_what_the_plan_is_priced_with(raw_index):
    async def go():
        async with McpCatalog(raw_index) as catalog:
            return catalog.tool, await agent.measure_replies(catalog, ["tote", "knife"])

    tool, reply_tokens = run(go())
    assert reply_tokens > 0
    assert price_plan(tool, reply_tokens, (100,), 3, 3, 0.0, 20.0)["plans"]["100"]["expected"] > 0


def test_a_catalog_that_fails_to_start_raises_instead_of_hanging(tmp_path, raw_index):
    async def go():
        async with McpCatalog(tmp_path / "missing"):
            pass

    with pytest.raises(BaseException):  # noqa: B017 - the MCP client's own error types vary
        run(go())


def test_a_fake_condition_runs_end_to_end_on_the_real_engine(raw_index, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "agent_dir", lambda fake: tmp_path / ("fake" if fake else "real"))
    report = agent.shop_condition("raw", [task("t1", "chef knife")], 2, raw_index, fake=True)
    assert report["fake"] is True
    assert report["new_runs"] == report["runs_logged"] == 2
    assert report["cost_logged"] > 0
    assert not (tmp_path / "real").exists()
    rows = RunLog(tmp_path / "fake").rows()
    assert all(r["returned"][0] == "C" for r in rows)


def test_pricing_the_agent_reads_the_real_engine_and_spends_nothing(
    raw_index, tmp_path, monkeypatch
):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(agent, "ledger_path", lambda fake: ledger)
    report = agent.price_agent(raw_index, ["Chef knife", "Red canvas tote"], (100, 300), 3)
    assert report["plans"]["100"]["agent_runs"] == 900
    assert report["anthropic_spent"] == 0.0
    assert not ledger.exists()


def test_three_runs_when_they_fit_two_when_only_two_do_and_none_below_that():
    assert agent.runs_that_fit(1.0, 3.0, 3) == 3
    assert agent.runs_that_fit(1.0, 2.9, 3) == 2
    assert agent.runs_that_fit(1.0, 1.9, 3) == 0


def test_a_plan_says_how_many_runs_the_remaining_budget_covers():
    roomy = price_plan(FAKE_TOOL, 900.0, (100,), 3, 3, spent=0.0, cap=20.0)
    assert roomy["plans"]["100"]["runs_that_fit"] == 3
    per_round = roomy["plans"]["100"]["expected"] / 3
    tight = price_plan(FAKE_TOOL, 900.0, (100,), 3, 3, spent=20.0 - 2.5 * per_round, cap=20.0)
    assert tight["plans"]["100"]["runs_that_fit"] == 2

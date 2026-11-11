import json
import os

import pytest

from taxon.labeling.budget import CAPS, Budget, BudgetExceeded
from taxon.labeling.cache import ResponseCache
from taxon.labeling.env import MissingKey, Secrets, load_secrets
from taxon.labeling.fakes import FakeBatchClient, answer
from taxon.labeling.llm import (
    CHOICE_SCHEMA,
    LLMRequest,
    LLMResult,
    ModelConfig,
    Status,
    Unreachable,
    Usage,
)
from taxon.labeling.pricing import PRICES, cost_of, price_for, worst_case_cost
from taxon.labeling.runner import MAX_UNREACHABLE_POLLS, BatchRunner, split_into_batches
from taxon.pins import ROOT

HAIKU = ModelConfig("anthropic", "claude-haiku-5-5", 1024, "low")
OPUS = ModelConfig("anthropic", "claude-opus-5-5", 4096, "medium")
SONNET = ModelConfig("anthropic", "claude-sonnet-5-5", 2048, "low")
LUNA = ModelConfig("openai", "gpt-6-luna", 1024, "low")


def request(text="hello", config=HAIKU, **extra):
    return LLMRequest(config, "pick", "system", text, **extra)


@pytest.fixture
def world(tmp_path):
    def build(caps=CAPS, **client_args):
        clients = {
            v: FakeBatchClient(v, lambda r: answer(r, 1), **client_args)
            for v in ("anthropic", "openai", "google")
        }
        runner = BatchRunner(
            clients,
            ResponseCache(tmp_path / "cache"),
            Budget(tmp_path / "ledger.jsonl", caps),
            tmp_path / "journal",
            sleep=lambda _: None,
        )
        return runner, clients

    return build


# Keys


def test_a_value_in_the_env_file_wins_over_the_environment(tmp_path):
    path = tmp_path / ".env"
    path.write_text("ANTHROPIC_API_KEY=from-file\nOPENAI_API_KEY=\n", encoding="utf-8")
    secrets = load_secrets(path, {"ANTHROPIC_API_KEY": "from-env", "OPENAI_API_KEY": "env-2"})
    assert secrets.get("ANTHROPIC_API_KEY") == "from-file"
    assert secrets.get("OPENAI_API_KEY") == "env-2"


def test_loading_keys_leaves_the_environment_alone(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=abc\n", encoding="utf-8")
    load_secrets(path, {})
    assert "abc" not in os.environ.values()


def test_a_missing_file_is_fine_and_a_missing_key_names_only_itself():
    secrets = load_secrets(ROOT / "does-not-exist", {})
    with pytest.raises(MissingKey, match="OPENAI_API_KEY"):
        secrets.require("OPENAI_API_KEY")


def test_secrets_never_show_values():
    secrets = Secrets({"OPENAI_API_KEY": "sk-very-secret"})
    assert "sk-very-secret" not in repr(secrets)
    assert "sk-very-secret" not in str(secrets)
    assert secrets.redact("bad key sk-very-secret given") == "bad key <OPENAI_API_KEY> given"


def test_the_env_template_has_only_empty_values_and_env_is_ignored():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(line.endswith("=") for line in lines)
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# Requests


def test_the_key_changes_with_anything_that_affects_the_answer():
    base = request()
    assert base.key == request().key
    assert base.key != request("other").key
    assert base.key != request(config=LUNA).key
    assert base.key != request(image_sha256="abc").key
    assert base.key != LLMRequest(HAIKU, "pick", "changed", "hello").key


def test_the_key_can_serve_as_a_batch_custom_id():
    assert len(request().key) == 64
    assert request().key.isalnum()


def test_the_schema_requires_the_choice():
    assert CHOICE_SCHEMA["required"] == ["choice"]
    assert CHOICE_SCHEMA["additionalProperties"] is False


# Pricing


def test_cost_applies_each_token_class_at_its_own_rate():
    usage = Usage(
        input_tokens=1_000_000,
        cached_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost_of(usage, HAIKU) == pytest.approx(0.05 + 0.005 + 0.10 + 0.25)
    assert cost_of(usage, OPUS) == pytest.approx(2.00 + 0.10 + 4.00 + 10.00)
    assert cost_of(usage, SONNET) == pytest.approx(1.00 + 0.05 + 2.00 + 5.00)


def test_batch_prices_are_half_the_published_list_prices():
    assert PRICES[("anthropic", "claude-haiku-5-5")].input == pytest.approx(0.10 / 2)
    assert PRICES[("anthropic", "claude-opus-5-5")].output == pytest.approx(20.0 / 2)
    assert PRICES[("openai", "gpt-6-luna")].output == pytest.approx(0.50 / 2)
    assert PRICES[("google", "gemini-3.5-flash-lite")].input == pytest.approx(0.30 / 2)


def test_an_unknown_model_raises_rather_than_counting_as_free():
    with pytest.raises(KeyError, match="no price"):
        price_for(ModelConfig("openai", "mystery", 10, "low"))


def test_worst_case_is_an_upper_bound_on_a_realistic_response():
    r = request("x" * 8_000)
    real = Usage(input_tokens=2_000, output_tokens=1_024)
    assert worst_case_cost(r) > cost_of(real, HAIKU)


def test_worst_case_grows_with_an_image():
    assert worst_case_cost(request(image_path="a.jpg")) > worst_case_cost(request())


# Cache


def test_cache_round_trips_a_result(tmp_path):
    cache = ResponseCache(tmp_path)
    result = LLMResult("ab" + "0" * 62, Status.OK, "{}", Usage(1, 2, 3, 4), cost=0.5)
    cache.put(result)
    assert cache.get(result.key) == result
    assert len(cache) == 1


def test_a_miss_returns_none(tmp_path):
    assert ResponseCache(tmp_path).get("0" * 64) is None


@pytest.mark.parametrize("status", [Status.REFUSED, Status.TRUNCATED])
def test_paid_for_non_answers_are_cached(tmp_path, status):
    cache = ResponseCache(tmp_path)
    cache.put(LLMResult("a" * 64, status))
    assert cache.get("a" * 64).status is status


def test_failures_are_not_cached_so_they_can_be_retried(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put(LLMResult("a" * 64, Status.FAILED))
    assert cache.get("a" * 64) is None


# Budget


def test_caps_are_the_agreed_amounts():
    assert CAPS == {"anthropic": 20.0, "openai": 3.0}


def test_a_reservation_counts_until_it_settles_to_the_real_cost(tmp_path):
    budget = Budget(tmp_path / "l.jsonl")
    budget.reserve("openai", "b1", 2.0)
    assert budget.spent("openai") == 2.0
    budget.settle("openai", "b1", 0.5)
    assert budget.spent("openai") == 0.5
    assert budget.remaining("openai") == 2.5


def test_the_ledger_survives_a_restart_including_open_reservations(tmp_path):
    path = tmp_path / "l.jsonl"
    budget = Budget(path)
    budget.settle("google", "b1", 3.0)
    budget.reserve("google", "b2", 4.0)
    budget.charge("google", 0.5, "cache storage")
    assert Budget(path).spent("google") == 7.5
    assert Budget(path).settled("google") == 3.5


# Runner


def test_splits_by_model_count_and_size():
    rows = [request(str(i)) for i in range(5)] + [request("a", config=LUNA)]
    batches = split_into_batches(rows, max_requests=2)
    assert [len(b) for b in batches] == [2, 2, 1, 1]
    assert all(len({r.config for r in b}) == 1 for b in batches)
    assert len(split_into_batches([request("x" * 100), request("y" * 100)], max_bytes=120)) == 2


def test_results_come_back_by_key_with_their_cost(world):
    runner, clients = world()
    r = request()
    results = runner.run([r])
    assert results[r.key].status is Status.OK
    assert results[r.key].cost > 0
    assert clients["anthropic"].requests_sent == 1


def test_a_repeat_run_is_served_from_cache_without_calling_the_vendor(world):
    runner, clients = world()
    rows = [request(str(i)) for i in range(3)]
    first = runner.run(rows)
    second = runner.run(rows)
    assert clients["anthropic"].requests_sent == 3
    assert {k: v.text for k, v in first.items()} == {k: v.text for k, v in second.items()}


def test_identical_requests_are_sent_once(world):
    runner, clients = world()
    runner.run([request(), request()])
    assert clients["anthropic"].requests_sent == 1


def test_only_uncached_requests_are_sent(world):
    runner, clients = world()
    runner.run([request("a")])
    runner.run([request("a"), request("b")])
    assert clients["anthropic"].requests_sent == 2


def test_real_cost_is_settled_into_the_ledger(world, tmp_path):
    runner, _ = world()
    r = request()
    result = runner.run([r])[r.key]
    assert runner.budget.spent("anthropic") == pytest.approx(result.cost)
    assert not list((tmp_path / "journal").glob("*"))


def test_polls_until_a_batch_finishes(world):
    runner, clients = world(polls_until_done=3)
    r = request()
    assert runner.run([r])[r.key].status is Status.OK
    assert clients["anthropic"]._polls["fake-anthropic-0"] == 4


def test_a_dropped_connection_while_polling_does_not_end_the_run(world):
    runner, clients = world(unreachable_polls=5)
    r = request()
    assert runner.run([r])[r.key].status is Status.OK
    assert clients["anthropic"].requests_sent == 1


def test_polling_gives_up_after_an_hour_unreachable_and_a_rerun_collects(world, tmp_path):
    runner, clients = world(unreachable_polls=MAX_UNREACHABLE_POLLS)
    r = request()
    with pytest.raises(Unreachable):
        runner.run([r])
    assert list((tmp_path / "journal").glob("*"))
    assert runner.run([r])[r.key].status is Status.OK
    assert clients["anthropic"].requests_sent == 1


def test_stops_before_a_cap_would_be_passed(world):
    r = request("x" * 1_000)
    runner, clients = world(caps={"anthropic": worst_case_cost(r) / 2})
    with pytest.raises(BudgetExceeded) as caught:
        runner.run([r])
    assert caught.value.vendor == "anthropic"
    assert caught.value.unsent == 1
    assert clients["anthropic"].requests_sent == 0


def test_sends_what_fits_then_stops_and_keeps_those_responses(world, tmp_path):
    rows = [request(f"{i}" * 100) for i in range(10)]
    # Room for three at worst case on the first pass, then more as reservations settle lower.
    cap = worst_case_cost(rows[0]) * 3.5
    runner, clients = world(caps={"anthropic": cap})
    try:
        runner.run(rows)
    except BudgetExceeded:
        pass
    sent = clients["anthropic"].requests_sent
    assert sent >= 3
    assert runner.budget.spent("anthropic") <= cap
    assert len(ResponseCache(tmp_path / "cache")) == sent


def test_spending_never_passes_the_cap_even_when_every_request_costs_its_worst_case(tmp_path):
    def expensive(r):
        usage = Usage(input_tokens=100, output_tokens=r.config.max_tokens)
        return LLMResult(r.key, Status.OK, '{"choice": 1}', usage)

    client = FakeBatchClient("anthropic", expensive)
    rows = [request(f"{i}") for i in range(50)]
    cap = worst_case_cost(rows[0]) * 7.5
    runner = BatchRunner(
        {"anthropic": client},
        ResponseCache(tmp_path / "c"),
        Budget(tmp_path / "l", {"anthropic": cap}),
        tmp_path / "j",
        sleep=lambda _: None,
    )
    with pytest.raises(BudgetExceeded):
        runner.run(rows)
    assert runner.budget.spent("anthropic") <= cap
    assert client.requests_sent < 50


def test_a_crash_does_not_resubmit_a_batch_already_paid_for(tmp_path):
    class Crash(Exception):
        pass

    client = FakeBatchClient("anthropic", lambda r: answer(r, 1), polls_until_done=99)

    def crash(_):
        raise Crash

    def build(sleep):
        return BatchRunner(
            {"anthropic": client},
            ResponseCache(tmp_path / "c"),
            Budget(tmp_path / "l"),
            tmp_path / "j",
            sleep=sleep,
        )

    r = request()
    with pytest.raises(Crash):
        build(crash).run([r])
    assert client.requests_sent == 1
    assert build(crash).budget.spent("anthropic") > 0

    client.polls_until_done = 0
    client._polls["fake-anthropic-0"] = 0
    resumed = build(lambda _: None)
    assert resumed.run([r])[r.key].status is Status.OK
    assert client.requests_sent == 1


def test_a_resumed_batch_outside_this_run_is_still_collected_and_cached(tmp_path):
    client = FakeBatchClient("anthropic", lambda r: answer(r, 1), polls_until_done=99)

    class Crash(Exception):
        pass

    def crash(_):
        raise Crash

    def build(sleep=lambda _: None):
        return BatchRunner(
            {"anthropic": client},
            ResponseCache(tmp_path / "c"),
            Budget(tmp_path / "l"),
            tmp_path / "j",
            sleep=sleep,
        )

    orphan = request("orphan")
    with pytest.raises(Crash):
        build(crash).run([orphan])
    client.polls_until_done = 0
    client._polls["fake-anthropic-0"] = 0
    other = request("other")
    assert set(build().run([other])) == {other.key}
    assert ResponseCache(tmp_path / "c").get(orphan.key) is not None


def test_a_failed_request_is_retried_then_reported(world, tmp_path):
    calls = []

    def flaky(r):
        calls.append(r.key)
        return LLMResult(r.key, Status.FAILED, error="server error")

    runner, clients = world()
    clients["anthropic"].responder = flaky
    r = request()
    assert runner.run([r])[r.key].status is Status.FAILED
    assert len(calls) == 3
    assert ResponseCache(tmp_path / "cache").get(r.key) is None


def test_a_retry_that_succeeds_is_used(world):
    seen = []

    def flaky(r):
        seen.append(r.key)
        return LLMResult(r.key, Status.FAILED) if len(seen) == 1 else answer(r, 1)

    runner, clients = world()
    clients["anthropic"].responder = flaky
    r = request()
    assert runner.run([r])[r.key].status is Status.OK


def test_results_missing_from_a_batch_count_as_failed(world):
    runner, clients = world()
    clients["anthropic"].results = lambda _: []
    r = request()
    assert runner.run([r])[r.key].status is Status.FAILED


def test_the_tracer_sees_each_collected_result_once(world):
    seen = []

    class Tracer:
        def record(self, request, result):
            seen.append((request.purpose, result.status))

    runner, _ = world()
    runner.tracer = Tracer()
    r = request()
    runner.run([r])
    runner.run([r])
    assert seen == [("pick", Status.OK)]


def test_nothing_on_disk_contains_a_key(world, tmp_path):
    runner, _ = world()
    secrets = Secrets({"ANTHROPIC_API_KEY": "sk-super-secret"})
    runner.run([request()])
    assert "sk-super-secret" not in repr(secrets)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "sk-super-secret" not in path.read_text(encoding="utf-8")


def test_journal_entries_are_valid_json_while_a_batch_is_open(tmp_path):
    client = FakeBatchClient("anthropic", polls_until_done=99)

    class Crash(Exception):
        pass

    def crash(_):
        raise Crash

    runner = BatchRunner(
        {"anthropic": client},
        ResponseCache(tmp_path / "c"),
        Budget(tmp_path / "l"),
        tmp_path / "j",
        sleep=crash,
    )
    with pytest.raises(Crash):
        runner.run([request()])
    entry = json.loads(next((tmp_path / "j").glob("*")).read_text(encoding="utf-8"))
    assert entry["vendor"] == "anthropic"
    assert entry["keys"] == [request().key]

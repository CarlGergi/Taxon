import hashlib
import json

import pytest

from taxon.labeling import cli
from taxon.labeling.attributes import (
    ATTRIBUTE_ANNOTATOR,
    EXACT,
    MODES,
    SETTLED,
    SHORTLIST,
    AttributeExtractor,
    use_mode,
)
from taxon.labeling.budget import Budget
from taxon.labeling.cache import ResponseCache
from taxon.labeling.env import Secrets
from taxon.labeling.fakes import FakeBatchClient, demo_responder
from taxon.labeling.llm import CHOICE_SCHEMA, LLMResult, Status, Usage
from taxon.labeling.pipeline import ANNOTATORS, Labeler
from taxon.labeling.prompts import (
    attribute_request,
    attribute_schema,
    attribute_shortlist_request,
    parse_attribute_choices,
    parse_attributes,
    pick_request,
    shortlist_schema,
)
from taxon.labeling.report import attribute_projection, attribute_report
from taxon.labeling.runner import BatchRunner
from taxon.labeling.values import ValueShortlister, listing_text
from taxon.labeling.vendors import AnthropicBatchClient, OpenAIBatchClient
from taxon.products import read_jsonl, write_jsonl
from taxon.reproduce import attribute_pass_score
from taxon.taxonomy import Taxonomy
from taxon.training.examples import LabelSource, load_labeled
from tests.conftest import COLOR, toy_release

TOTES = "Apparel > Bags > Totes"
KITCHEN = "Home > Kitchen"


def product(asin="A1", **extra):
    return {
        "parent_asin": asin,
        "domain": "d",
        "title": "Canvas tote",
        "features": [],
        "description": "A bag",
        "details": {"Color": "Blue"},
    } | extra


def request(path=TOTES, image=None):
    return attribute_request(ATTRIBUTE_ANNOTATOR, product(), path, ["color", "material"], image)


def test_the_schema_asks_for_every_attribute_and_allows_null():
    schema = attribute_schema(["color", "material"])
    assert schema["required"] == ["color", "material"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["color"] == {"type": ["string", "null"]}


def test_the_request_names_the_category_and_attributes_but_no_value_lists():
    r = request()
    assert r.purpose == "attributes"
    assert f"Category: {TOTES}" in r.user
    assert "- color\n- material" in r.user
    assert "Cotton" not in r.user
    assert r.schema == attribute_schema(["color", "material"])


def test_the_image_reaches_the_request():
    r = request(image=("x.jpg", "abc"))
    assert (r.image_path, r.image_sha256) == ("x.jpg", "abc")
    assert r.key != request().key


def test_category_requests_keep_the_cache_keys_they_had(taxonomy):
    """The schema became a field, so cached category responses must still be found."""
    r = pick_request(ANNOTATORS[0], "system", product(), [TOTES], None)
    identity = [
        r.config.vendor,
        r.config.model,
        r.config.max_tokens,
        r.config.effort,
        CHOICE_SCHEMA,
        r.system,
        r.user,
        r.image_sha256,
    ]
    old_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert r.key == old_key


def test_vendors_send_each_request_its_own_schema():
    r = request()
    assert AnthropicBatchClient.params(r)["output_config"]["format"]["schema"] == r.schema
    response_format = OpenAIBatchClient.body(r)["response_format"]["json_schema"]
    assert (response_format["name"], response_format["schema"]) == ("attributes", r.schema)


def test_answers_take_the_taxonomy_spelling_and_the_rest_are_counted(taxonomy):
    allowed = taxonomy.allowed_values(TOTES)
    text = json.dumps({"color": " red ", "material": "Hemp", "made-up": "x"})
    assert parse_attributes(text, allowed) == ({"color": "Red"}, ["material=Hemp"])


def test_null_and_empty_answers_are_dropped_quietly(taxonomy):
    allowed = taxonomy.allowed_values(TOTES)
    text = json.dumps({"color": None, "material": " "})
    assert parse_attributes(text, allowed) == ({}, [])


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '"Red"'])
def test_an_unusable_reply_is_none(text, taxonomy):
    assert parse_attributes(text, taxonomy.allowed_values(TOTES)) is None


def world(tmp_path, taxonomy, responder=demo_responder):
    clients = {v: FakeBatchClient(v, responder) for v in ("anthropic", "openai", "google")}
    budget = Budget(tmp_path / "ledger.jsonl", {"anthropic": 5, "openai": 5, "google": 5})
    runner = BatchRunner(clients, ResponseCache(tmp_path / "cache"), budget, tmp_path / "open")
    return AttributeExtractor(taxonomy, runner), clients


def row(asin, path, source):
    return {"parent_asin": asin, "path": path, "source": source, "cost": {}}


def replying(values, status=Status.OK):
    def respond(request):
        if request.purpose != "attributes":
            return demo_responder(request)
        text = values if isinstance(values, str) else json.dumps(values)
        return LLMResult(request.key, status, text, Usage(input_tokens=100, output_tokens=10))

    return respond


def test_only_settled_categories_get_an_attribute_call(tmp_path, taxonomy):
    extractor, clients = world(tmp_path, taxonomy, replying({"color": "Red", "material": None}))
    rows = [
        row("A", TOTES, "unanimous"),
        row("B", TOTES, "arbitrated"),
        row("C", TOTES, "abstained"),
        row("D", None, "no_proposal"),
    ]
    extractor.extract([product(a, title=f"Tote {a}") for a in "ABCD"], rows, EXACT)
    assert clients["anthropic"].requests_sent == 2
    runs = [r["attribute_runs"][EXACT] for r in rows]
    assert [run["status"] for run in runs] == ["ok", "ok", "not_settled", "not_settled"]
    assert runs[0]["attributes"] == {"color": "Red"}
    assert "attributes" not in runs[2]
    assert {"unanimous", "agreed_with_abstention", "arbitrated"} == SETTLED


def test_a_category_without_attributes_needs_no_call(tmp_path, taxonomy):
    extractor, clients = world(tmp_path, taxonomy)
    rows = [row("A", KITCHEN, "unanimous")]
    extractor.extract([product("A")], rows, SHORTLIST)
    assert clients["anthropic"].requests_sent == 0
    assert rows[0]["attribute_runs"][SHORTLIST] == {"status": "no_attributes", "attributes": {}}


def test_attribute_costs_stay_apart_from_category_costs(tmp_path, taxonomy):
    extractor, _ = world(tmp_path, taxonomy, replying({"color": "Red", "material": None}))
    rows = [row("A", TOTES, "unanimous")]
    extractor.extract([product("A")], rows, EXACT)
    assert rows[0]["cost"] == {}
    assert rows[0]["attribute_runs"][EXACT]["cost"] > 0
    assert rows[0]["attribute_runs"][EXACT]["tokens"]["input"] == 100


@pytest.mark.parametrize(
    "values,status,expected",
    [("not json", Status.OK, "invalid"), ("{}", Status.REFUSED, "refused")],
)
def test_a_failed_reply_leaves_no_attributes(tmp_path, taxonomy, values, status, expected):
    extractor, _ = world(tmp_path, taxonomy, replying(values, status))
    rows = [row("A", TOTES, "unanimous")]
    extractor.extract([product("A")], rows, EXACT)
    assert rows[0]["attribute_runs"][EXACT]["attributes"] == {}
    assert rows[0]["attribute_runs"][EXACT]["status"] == expected


def test_adding_attributes_later_does_not_repeat_category_calls(tmp_path, taxonomy):
    extractor, clients = world(tmp_path, taxonomy)
    labeler = Labeler(taxonomy, extractor.runner)
    products = [product(f"P{i}") for i in range(6)]
    shortlists = {p["parent_asin"]: [TOTES, KITCHEN] for p in products}
    labeler.label(products, shortlists)
    before = {v: c.requests_sent for v, c in clients.items()}
    rows = labeler.label(products, shortlists)
    assert {v: c.requests_sent for v, c in clients.items()} == before
    for mode in MODES:
        extractor.extract(products, rows, mode)
    sent = [r for batch in clients["anthropic"].submitted for r in batch]
    new = sent[before["anthropic"] :]
    assert all(r.purpose == "attributes" for r in new)
    assert clients["openai"].requests_sent == before["openai"]


def test_a_second_attribute_pass_is_free(tmp_path, taxonomy):
    extractor, clients = world(tmp_path, taxonomy)
    rows = [row("A", TOTES, "unanimous")]
    extractor.extract([product("A")], rows, SHORTLIST)
    sent = clients["anthropic"].requests_sent
    extractor.extract([product("A")], [row("A", TOTES, "unanimous")], SHORTLIST)
    assert clients["anthropic"].requests_sent == sent


def priced(n, cost):
    return [{"attribute_runs": {EXACT: {"cost": cost}}, "cost": {}} for _ in range(n)]


def test_the_projection_counts_categories_attributes_and_spending_against_the_cap():
    category = {"anthropic": {"planned": {"upper_95": 10.0}}}
    rows = priced(9, 0.001) + [{"cost": {}}]
    entry = attribute_projection(
        rows, {"planned": 1000}, {"anthropic": 2.0}, category, "anthropic", "planned", EXACT
    )
    assert entry["per_product"] == pytest.approx(0.0009)
    assert entry["called_for"] == 9
    assert entry["planned"]["estimate"] == pytest.approx(0.9)
    total = entry["everything_planned"]["upper_95"]
    assert total == pytest.approx(2.0 + 10.0 + entry["planned"]["upper_95"])
    assert entry["everything_planned"]["fits_cap"] is True
    over = attribute_projection(
        rows, {"planned": 1000}, {"anthropic": 9.5}, category, "anthropic", "planned", EXACT
    )
    assert over["everything_planned"]["fits_cap"] is False
    extra = attribute_projection(
        rows, {"planned": 1000}, {"anthropic": 2.0}, category, "anthropic", "planned", EXACT, 8.0
    )
    assert extra["everything_planned"]["upper_95"] == pytest.approx(total + 8.0)
    assert extra["everything_planned"]["fits_cap"] is False


def test_the_report_counts_unmatched_answers():
    runs = [
        {
            "cost": 0.1,
            "status": "ok",
            "attributes": {"color": "Red"},
            "unmatched": ["material=Hemp"],
        },
        {"cost": 0.1, "status": "ok", "attributes": {}},
        {"status": "not_settled"},
    ]
    rows = [{"attribute_runs": {SHORTLIST: run}} for run in runs]
    report = attribute_report(rows, SHORTLIST)
    assert report["called"] == 2
    assert report["values_per_product"] == 0.5
    assert report["unmatched_share"] == 0.5
    assert report["unmatched_examples"] == ["material=Hemp"]
    assert report["statuses"] == {"not_settled": 1, "ok": 2}


def test_hand_labels_score_the_pass_beside_the_details_alone(taxonomy):
    def runs(exact, shortlist):
        return {EXACT: {"attributes": exact}, SHORTLIST: {"attributes": shortlist}}

    rows = {
        "A": {
            "path": TOTES,
            "attribute_runs": runs({"color": "Red"}, {"color": "Red", "material": "Cotton"}),
        },
        "B": {"path": TOTES, "attribute_runs": runs({}, {})},
    }
    gold = {
        "A": {"path": TOTES, "attributes": {"color": "Red", "material": "Cotton"}},
        "B": {"path": KITCHEN, "attributes": {"color": "Blue"}},
        "C": {"path": TOTES, "attributes": {}},
    }
    products = {a: product(a) for a in "ABC"}
    score = attribute_pass_score(rows, gold, products, taxonomy)
    assert score["products"] == 2
    assert score["same_category_as_hand_label"] == 1
    assert score["versions"][EXACT]["accuracy"] == pytest.approx(1 / 3)
    assert score["versions"][SHORTLIST]["accuracy"] == pytest.approx(2 / 3)
    assert score["better"] == SHORTLIST
    assert score["details_only"]["values"] == 3


def test_a_tie_keeps_the_exact_version(taxonomy):
    same = {EXACT: {"attributes": {"color": "Red"}}, SHORTLIST: {"attributes": {"color": "Red"}}}
    rows = {"A": {"path": TOTES, "attribute_runs": same}}
    gold = {"A": {"path": TOTES, "attributes": {"color": "Red"}}}
    assert attribute_pass_score(rows, gold, {"A": product("A")}, taxonomy)["better"] == EXACT


def test_no_hand_labeled_attributes_means_no_score(taxonomy):
    rows = {"A": {"path": TOTES, "attribute_runs": {EXACT: {"attributes": {"color": "Red"}}}}}
    gold = {"A": {"path": TOTES, "attributes": {}}}
    assert attribute_pass_score(rows, gold, {}, taxonomy) is None


def test_training_prefers_the_pass_and_falls_back_to_the_details(tmp_path, taxonomy):
    products = [product(f"P{i:02}", details={"Color": "Blue"}) for i in range(40)]
    rows = [
        {"parent_asin": p["parent_asin"], "path": TOTES, "attributes": {"material": "Cotton"}}
        if i % 2 == 0
        else {"parent_asin": p["parent_asin"], "path": TOTES}
        for i, p in enumerate(products)
    ]
    write_jsonl(tmp_path / "labels.jsonl", rows)
    source = LabelSource("pipeline", tmp_path / "labels.jsonl", False)
    items = load_labeled(source, products, [], taxonomy, 7, "train")
    for item in items:
        if int(item.asin[1:]) % 2 == 0:
            assert item.attributes == {"material": "Cotton"}
        else:
            assert item.attributes == {"color": "Blue"}


@pytest.fixture
def inputs(monkeypatch, taxonomy):
    products = [product(f"A{i}") for i in range(30)]
    gold = ["A0", "A1", "A2"]
    shortlists = {p["parent_asin"]: [TOTES, KITCHEN] for p in products}
    monkeypatch.setattr(cli, "_load_inputs", lambda: (taxonomy, products, gold, shortlists))
    monkeypatch.setattr(cli, "FIRST_RUN_PRODUCTS", 10)
    monkeypatch.setattr(cli, "load_secrets", lambda: Secrets({}))
    return products, gold


def run(tmp_path, *argv):
    return cli.main(list(argv), root=tmp_path)


def gold_eval(tmp_path, monkeypatch, scored):
    path = tmp_path / "gold_eval.json"
    score = {"attribute_pass": {"products": 5, "better": SHORTLIST}}
    path.write_text(json.dumps(score if scored else {}))
    original = cli.Workspace.__init__

    def init(self, root):
        original(self, root)
        self.gold_eval = path

    monkeypatch.setattr(cli.Workspace, "__init__", init)


def test_the_dry_run_prices_the_attribute_pass(tmp_path, inputs, capsys):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    out = capsys.readouterr().out
    assert "attribute pass, exact (anthropic)" in out
    assert "attribute pass, shortlist (anthropic)" in out
    report = json.loads((tmp_path / "results" / "labeling_dry_run.json").read_text())
    assert set(report["attribute_cost"]) == set(MODES)
    for entry in report["attribute_cost"].values():
        assert {"gold_pool", "first_run", "everything_planned"} <= set(entry)
    assert set(report["attributes"]) == set(MODES)


def test_the_first_run_stays_category_only(tmp_path, inputs):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--first", "--fake") == 0
    rows = read_jsonl(tmp_path / "labels.jsonl")
    assert not any("attributes" in r or "attribute_cost" in r for r in rows)


def test_the_gold_pool_gets_attributes_so_the_pass_can_be_scored(tmp_path, inputs):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--gold-pool", "--fake") == 0
    rows = read_jsonl(tmp_path / "gold_pool_labels.jsonl")
    assert all(set(r["attribute_runs"]) == set(MODES) for r in rows)


def test_the_attribute_stage_needs_a_dry_run(tmp_path, inputs, monkeypatch):
    gold_eval(tmp_path, monkeypatch, scored=True)
    with pytest.raises(SystemExit, match="dry run first"):
        run(tmp_path, "run", "--attributes", "--fake")


def test_the_attribute_stage_needs_a_score_against_hand_labels(tmp_path, inputs, monkeypatch):
    gold_eval(tmp_path, monkeypatch, scored=False)
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--first", "--fake") == 0
    with pytest.raises(SystemExit, match="against hand labels"):
        run(tmp_path, "run", "--attributes", "--fake")


def test_the_attribute_stage_needs_an_estimate_within_the_cap(tmp_path, inputs, monkeypatch):
    gold_eval(tmp_path, monkeypatch, scored=True)
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    path = tmp_path / "results" / "labeling_dry_run.json"
    report = json.loads(path.read_text())
    report["attribute_cost"][SHORTLIST]["everything_planned"]["fits_cap"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="attribute pass would pass its cap"):
        run(tmp_path, "run", "--attributes", "--fake")


def test_the_attribute_stage_adds_attributes_to_training_labels(tmp_path, inputs, monkeypatch):
    products, gold = inputs
    gold_eval(tmp_path, monkeypatch, scored=True)
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--first", "--fake") == 0
    assert run(tmp_path, "run", "--attributes", "--fake") == 0
    rows = read_jsonl(tmp_path / "labels.jsonl")
    settled = [r for r in rows if r["source"] in SETTLED]
    assert settled and all("attributes" in r for r in settled)
    assert all(r["attribute_mode"] == SHORTLIST for r in settled)
    assert not any(EXACT in r["attribute_runs"] for r in rows)
    exported = read_jsonl(tmp_path / "training_labels.jsonl")
    assert any("attributes" in r for r in exported)
    assert not {r["parent_asin"] for r in exported} & set(gold)


def shortlist_taxonomy():
    release = toy_release()
    color = next(a for a in release["attributes"] if a["id"] == COLOR)
    for i, name in enumerate(["Navy", "Navy blue", "Cotton candy", "Teal", "Sky blue", "Black"]):
        color["values"].append({"id": f"c{i}", "name": name, "handle": name.lower()})
    return Taxonomy.from_release(release)


def test_the_shortlist_reaches_a_value_an_exact_match_would_miss():
    listing = product(title="Canvas tote", details={"Colour": "navy blue shade"})
    options = ValueShortlister(shortlist_taxonomy(), size=3).shortlist(TOTES, listing)
    assert set(options) == {"color", "material"}
    assert {"Navy", "Navy blue"} <= set(options["color"])
    assert len(options["color"]) == 3


def test_the_shortlist_is_deterministic_and_fills_up_with_alphabetical_ties():
    shortlister = ValueShortlister(shortlist_taxonomy(), size=20)
    first = shortlister.shortlist(TOTES, product(title="zzz"))
    assert first == shortlister.shortlist(TOTES, product(title="zzz"))
    assert len(first["color"]) == 8


def test_the_listing_text_puts_details_before_the_description():
    text = listing_text(product(description="LATER", details={"Color": "Teal"}))
    assert text.index("Teal") < text.index("LATER")


def options():
    return {"color": ["Navy", "Navy blue"], "material": ["Cotton"]}


def test_the_shortlist_request_numbers_each_attribute_s_options():
    r = attribute_shortlist_request(ATTRIBUTE_ANNOTATOR, product(), TOTES, options(), None)
    assert "color:\n1. Navy\n2. Navy blue" in r.user
    assert "material:\n1. Cotton" in r.user
    assert r.schema == shortlist_schema(["color", "material"])
    assert r.key != request().key


@pytest.mark.parametrize(
    "answer,kept,unmatched",
    [
        ({"color": 2, "material": 0}, {"color": "Navy blue"}, []),
        ({"color": 3, "material": 1}, {"material": "Cotton"}, ["color=3"]),
        ({"color": "Navy", "material": True}, {}, ["color=Navy", "material=True"]),
        ({"made-up": 1}, {}, []),
    ],
)
def test_choices_map_to_values_and_bad_numbers_are_counted(answer, kept, unmatched):
    assert parse_attribute_choices(json.dumps(answer), options()) == (kept, unmatched)


def test_an_unusable_choice_reply_is_none():
    assert parse_attribute_choices("junk", options()) is None


def test_the_shortlist_version_records_its_own_run(tmp_path, taxonomy):
    def pick_first(request):
        if request.purpose != "attributes":
            return demo_responder(request)
        text = json.dumps({h: 1 for h in request.schema["properties"]})
        return LLMResult(request.key, Status.OK, text, Usage(input_tokens=10, output_tokens=5))

    extractor, _ = world(tmp_path, taxonomy, pick_first)
    rows = [row("A", TOTES, "unanimous")]
    extractor.extract([product("A")], rows, SHORTLIST)
    run = rows[0]["attribute_runs"][SHORTLIST]
    assert run["status"] == "ok"
    assert set(run["attributes"]) == {"color", "material"}
    assert "attributes" not in rows[0]
    use_mode(rows, SHORTLIST)
    assert rows[0]["attributes"] == run["attributes"]
    assert rows[0]["attribute_mode"] == SHORTLIST

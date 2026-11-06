"""The real binary, spoken to by a real MCP client over stdio."""

import asyncio
import json
import subprocess

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from taxon.bench import cli, search
from taxon.bench.catalog import enriched_catalog, raw_catalog
from taxon.products import read_jsonl, write_jsonl
from taxon.reproduce import search_report
from taxon.taxonomy import Taxonomy
from tests.conftest import toy_release


def product(asin, title, price=None, **extra):
    return {"parent_asin": asin, "title": title, "price": price, "domain": "D", **extra}


PRODUCTS = [
    product("A", "Red canvas tote", 20.0, description="A roomy bag"),
    product("B", "Blue canvas tote", 40.0),
    product("C", "Chef knife", 15.0),
    product("D", "Unpriced tote"),
]
ENRICHMENTS = {
    "A": {"path": "Apparel > Bags > Totes", "attributes": {"color": "Red"}},
    "B": {"path": "Apparel > Bags > Totes", "attributes": {"color": "Blue"}},
    "C": {"path": "Home > Kitchen > Knives", "attributes": {}},
}


@pytest.fixture(scope="module")
def binary():
    try:
        return search.build_binary()
    except search.RustMissing as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def indexes(binary, tmp_path_factory):
    root = tmp_path_factory.mktemp("catalogs")

    def build(name, rows):
        path = root / f"{name}.jsonl"
        write_jsonl(path, rows)
        search.build_index(path, root / name, binary)
        return root / name

    raw = raw_catalog(PRODUCTS)
    enriched, _ = enriched_catalog(raw, ENRICHMENTS, Taxonomy.from_release(toy_release()))
    return {"raw": build("raw", raw), "enriched": build("enriched", enriched)}


async def _handshake(index, binary):
    async with stdio_client(search.server_parameters(index, binary)) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            tools = await session.list_tools()
            return info, tools.tools


def handshake(index, binary):
    return asyncio.run(_handshake(index, binary))


def ask(index, binary, *requests):
    return search.search_many(index, requests, binary)


def ids(response):
    return [hit["id"] for hit in response["results"]]


def test_the_server_offers_exactly_one_tool(indexes, binary):
    _, tools = handshake(indexes["raw"], binary)
    assert [tool.name for tool in tools] == [search.TOOL]


def test_the_agent_sees_the_same_tool_on_every_catalog(indexes, binary):
    raw_info, raw_tools = handshake(indexes["raw"], binary)
    enriched_info, enriched_tools = handshake(indexes["enriched"], binary)
    assert raw_tools[0].model_dump() == enriched_tools[0].model_dump()
    assert raw_info.server_info == enriched_info.server_info


def test_nothing_the_agent_sees_names_the_condition(indexes, binary):
    info, tools = handshake(indexes["enriched"], binary)
    shown = json.dumps([info.server_info.model_dump(), tools[0].model_dump()], default=str).lower()
    response = ask(indexes["enriched"], binary, {"query": "tote"})[0]
    for word in ("raw", "enriched", "taxon", "frontier"):
        assert word not in shown
        assert word not in json.dumps(response).lower()


def test_the_tool_schema_has_the_agreed_arguments(indexes, binary):
    _, tools = handshake(indexes["raw"], binary)
    arguments = set(tools[0].input_schema["properties"])
    assert arguments == {"query", "category", "attributes", "min_price", "max_price", "limit"}


def test_keywords_work_over_the_protocol(indexes, binary):
    response = ask(indexes["raw"], binary, {"query": "chef knife"})[0]
    assert ids(response) == ["C"]
    assert response["results"][0]["title"] == "Chef knife"


def test_a_raw_catalog_answers_with_price_filters_and_reports_the_others(indexes, binary):
    response = ask(
        indexes["raw"],
        binary,
        {"query": "tote", "max_price": 30, "category": "Apparel", "attributes": {"color": "Red"}},
    )[0]
    assert ids(response) == ["A"]
    assert response["ignored_filters"] == ["category", "attributes"]


def test_an_enriched_catalog_applies_every_filter(indexes, binary):
    response = ask(
        indexes["enriched"],
        binary,
        {"query": "tote", "category": "Apparel > Bags", "attributes": {"color": "Blue"}},
    )[0]
    assert ids(response) == ["B"]
    assert response["results"][0]["category"] == "Apparel > Bags > Totes"
    assert response["results"][0]["attributes"] == {"color": ["Blue"]}
    assert "ignored_filters" not in response


def test_a_listing_reads_the_same_on_both_catalogs(indexes, binary):
    raw = ask(indexes["raw"], binary, {"query": "chef knife"})[0]["results"][0]
    enriched = ask(indexes["enriched"], binary, {"query": "chef knife"})[0]["results"][0]
    assert {k: enriched[k] for k in raw} == raw
    assert set(enriched) - set(raw) == {"category"}


def test_a_refused_request_comes_back_as_an_error_not_a_crash(indexes, binary):
    bad, good = ask(indexes["raw"], binary, {"query": "!!!"}, {"query": "tote"})
    assert "keywords" in bad["error"]
    assert ids(good)


def test_unknown_arguments_are_refused(indexes, binary):
    async def call():
        params = search.server_parameters(indexes["raw"], binary)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(search.TOOL, {"query": "tote", "sort": "price"})

    try:
        result = asyncio.run(call())
    except Exception as error:  # noqa: BLE001
        assert "sort" in str(error)
    else:
        assert result.is_error


def test_serving_a_missing_index_fails_with_a_message(binary, tmp_path):
    done = subprocess.run(
        [str(binary), "serve", "--index", str(tmp_path / "nope")], capture_output=True, text=True
    )
    assert done.returncode != 0
    assert "no finished index" in done.stderr
    assert done.stdout == ""


def test_self_retrieval_finds_each_product_by_its_title(indexes, binary):
    rows = raw_catalog(PRODUCTS)
    report = search.self_retrieval(rows, indexes["raw"], seed=1, binary=binary)
    assert report == {"sample": 4, "found_first": 1.0, "found_in_top_10": 1.0}


def test_the_reproduce_step_indexes_the_raw_catalog_and_checks_it(binary, tmp_path):
    report = search_report(PRODUCTS, tmp_path)
    assert report["index"]["documents"] == 4
    assert report["with_price"] == 3
    assert report["self_retrieval"]["found_in_top_10"] == 1.0
    assert len(read_jsonl(tmp_path / "catalogs" / "raw.jsonl")) == 4


def test_the_reproduce_step_gives_the_same_report_twice(binary, tmp_path):
    assert search_report(PRODUCTS, tmp_path / "a") == search_report(PRODUCTS, tmp_path / "b")


def test_the_command_builds_and_searches_a_catalog(binary, tmp_path, capsys):
    write_jsonl(tmp_path / "products.jsonl", PRODUCTS)
    argv = ["search-index"]
    assert cli.main(argv, tmp_path / "t", tmp_path / "l", tmp_path) == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 4
    argv = ["search", '{"query": "chef knife"}']
    assert cli.main(argv, tmp_path / "t", tmp_path / "l", tmp_path) == 0
    assert ids(json.loads(capsys.readouterr().out)) == ["C"]


def test_the_command_builds_an_enriched_catalog_under_its_own_name(
    binary, tmp_path, capsys, monkeypatch, taxonomy
):
    monkeypatch.setattr(cli, "fetch", lambda _pin: "unused")
    monkeypatch.setattr(cli.Taxonomy, "load", staticmethod(lambda _path: taxonomy))
    write_jsonl(tmp_path / "products.jsonl", PRODUCTS)
    enrichments = tmp_path / "enrichments.jsonl"
    write_jsonl(enrichments, [{"parent_asin": k, **v} for k, v in ENRICHMENTS.items()])
    argv = ["search-index", "--name", "model", "--enrichments", str(enrichments)]
    assert cli.main(argv, tmp_path / "t", tmp_path / "l", tmp_path) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["has_category"] is True
    assert report["enriched"] == 3
    assert (tmp_path / "search" / "model" / "manifest.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["search-index", "--enrichments", "e.jsonl"],
        ["search-index", "--name", "model"],
    ],
)
def test_the_command_refuses_a_mismatched_name_and_enrichments(argv, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(argv, tmp_path / "t", tmp_path / "l", tmp_path)


def test_a_missing_toolchain_is_reported_clearly(monkeypatch):
    monkeypatch.setattr(search.shutil, "which", lambda _name: None)

    def no_rustup(*_args, **_kwargs):
        raise OSError("no rustup")

    monkeypatch.setattr(search.subprocess, "run", no_rustup)
    with pytest.raises(search.RustMissing, match="rustup"):
        search.build_binary()

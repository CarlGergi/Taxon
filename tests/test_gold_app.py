import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import taxon.gold.app
import taxon.gold.server
from taxon.gold.app import GoldApp, PathSearch
from taxon.gold.server import STATIC, make_handler
from taxon.gold.store import GoldStore, InvalidLabel

QUEUE = ["G1", "G2", "G3"]
BAGS = "Apparel > Bags"
TOTES = "Apparel > Bags > Totes"
KNIVES = "Home > Kitchen > Knives"


def product(asin):
    return {
        "parent_asin": asin,
        "title": f"Title {asin}",
        "description": "Desc",
        "features": ["Feat"],
        "details": {"Color": "red", "Material": "Unknown stuff"},
        "price": 9.5,
        "store": "Acme",
        "image_url": "https://m.media-amazon.com/x.jpg",
        "categories": ["AMAZON-CRUMB"],
        "main_category": "AMAZON-MAIN",
        "domain": "AMAZON-DOMAIN",
    }


@pytest.fixture
def store(tmp_path, taxonomy):
    return GoldStore(tmp_path / "gold" / "labels.jsonl", QUEUE, taxonomy)


@pytest.fixture
def app(store, taxonomy):
    products = {a: product(a) for a in QUEUE}
    return GoldApp(products, {"G1": [TOTES, KNIVES]}, taxonomy, store, target=2)


# Store


def test_labels_survive_a_restart(tmp_path, taxonomy, store):
    store.label("G1", TOTES, {"color": "Red"})
    store.skip("G2")
    again = GoldStore(store.path, QUEUE, taxonomy)
    assert again.labels() == {"G1": {"path": TOTES, "attributes": {"color": "Red"}}}
    assert again.n_skipped == 1
    assert again.next_asin() == "G3"


def test_a_product_outside_the_pool_is_refused_so_gold_cannot_leave_it(store):
    with pytest.raises(InvalidLabel, match="not in the gold pool"):
        store.label("TRAINING-ITEM", TOTES, {})
    with pytest.raises(InvalidLabel, match="not in the gold pool"):
        store.skip("TRAINING-ITEM")
    assert not store.path.exists()


def test_an_invalid_category_is_refused(store):
    with pytest.raises(InvalidLabel, match="not a category"):
        store.label("G1", "Home > Totes", {})


def test_an_attribute_outside_the_category_or_its_values_is_refused(store):
    with pytest.raises(InvalidLabel, match="no attribute"):
        store.label("G1", KNIVES, {"color": "Red"})
    with pytest.raises(InvalidLabel, match="not an allowed value"):
        store.label("G1", TOTES, {"color": "Plaid"})


def test_undo_forgets_the_latest_decision_and_keeps_the_log(store, taxonomy):
    store.label("G1", TOTES, {})
    store.skip("G2")
    assert store.undo() == "G2"
    assert store.next_asin() == "G2"
    assert store.undo() == "G1"
    assert store.undo() is None
    lines = store.path.read_text().splitlines()
    assert len(lines) == 4
    assert GoldStore(store.path, QUEUE, taxonomy).labels() == {}


def test_relabeling_replaces_the_answer_and_undo_steps_back_through_order(store):
    store.label("G1", TOTES, {})
    store.label("G2", KNIVES, {})
    store.label("G1", BAGS, {})
    assert store.labels()["G1"]["path"] == BAGS
    assert store.n_labeled == 2
    assert store.undo() == "G1"
    assert store.undo() == "G2"


# Search


def test_search_finds_by_word_and_prefix(taxonomy):
    search = PathSearch(taxonomy.paths())
    assert search.search("knives")[0] == KNIVES
    assert search.search("knif")[0] == KNIVES
    assert search.search("KNIFE")[0] == KNIVES


def test_search_requires_every_word(taxonomy):
    search = PathSearch(taxonomy.paths())
    assert search.search("home bag") == ["Home > Bags"]
    assert search.search("home totes") == []


def test_search_favors_the_node_own_name_then_the_shallower_path(taxonomy):
    search = PathSearch(taxonomy.paths())
    results = search.search("bags")
    assert results[0] in {BAGS, "Home > Bags"}
    assert results.index(BAGS) < results.index(TOTES)


def test_blank_or_stopword_queries_return_nothing(taxonomy):
    search = PathSearch(taxonomy.paths())
    assert search.search("") == []
    assert search.search("   ") == []
    assert search.search("the") == []


def test_search_respects_its_limit(taxonomy):
    assert len(PathSearch(taxonomy.paths()).search("apparel", limit=2)) == 2


# Page state


def test_the_item_shows_the_listing_and_baseline_suggestions_only(app):
    item = app.state()["item"]
    assert item["parent_asin"] == "G1"
    assert item["suggestions"] == [TOTES, KNIVES]
    assert set(item) == {
        "parent_asin",
        "title",
        "description",
        "features",
        "details",
        "price",
        "store",
        "image_url",
        "suggestions",
    }
    assert "AMAZON" not in json.dumps(item)


def test_annotator_output_cannot_reach_the_page(app, tmp_path):
    secret_path = "Home > Kitchen > ANNOTATOR-ANSWER"
    (tmp_path / "labels.jsonl").write_text(json.dumps({"parent_asin": "G1", "path": secret_path}))
    assert "ANNOTATOR" not in json.dumps(app.state())
    assert "ANNOTATOR" not in json.dumps(app.node(TOTES, "G1"))


@pytest.mark.parametrize(
    "module", [taxon.gold.app, taxon.gold.server, taxon.gold.store]
)
def test_the_page_code_never_loads_annotation_modules_or_files(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "taxon.labeling" not in source
    assert "training_labels" not in source
    assert '"labeling"' not in source


def test_static_files_never_mention_annotators():
    for name in ("app.js", "index.html"):
        text = (STATIC / name).read_text(encoding="utf-8").lower()
        for word in ("annotator", "arbitrat", "unanimous", "/labeling"):
            assert word not in text


def test_the_script_never_builds_markup_from_strings():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for unsafe in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert unsafe not in script


def test_state_moves_on_and_stops_at_the_target(app):
    assert app.state()["labeled"] == 0
    state = app.label("G1", TOTES, {})
    assert (state["labeled"], state["item"]["parent_asin"]) == (1, "G2")
    state = app.skip("G2")
    assert (state["skipped"], state["item"]["parent_asin"]) == (1, "G3")
    state = app.label("G3", KNIVES, {})
    assert state["labeled"] == 2
    assert state["item"] is None


def test_raising_the_target_continues_where_it_stopped(store, taxonomy):
    products = {a: product(a) for a in QUEUE}
    first = GoldApp(products, {}, taxonomy, store, target=1)
    first.label("G1", TOTES, {})
    assert first.state()["item"] is None
    grown = GoldApp(products, {}, taxonomy, store, target=2)
    assert grown.state()["item"]["parent_asin"] == "G2"


def test_undo_brings_the_item_back(app):
    app.label("G1", TOTES, {})
    assert app.undo()["item"]["parent_asin"] == "G1"


def test_blank_attributes_are_dropped(app):
    app.label("G1", TOTES, {"color": "", "material": "Cotton"})
    assert app.store.labels()["G1"]["attributes"] == {"material": "Cotton"}


def test_a_node_lists_navigation_and_attributes(app):
    node = app.node(BAGS)
    assert node["parent"] == "Apparel"
    assert node["children"] == [TOTES]
    assert [a["handle"] for a in node["attributes"]] == ["color", "strap-color"]
    assert node["attributes"][1]["name"] == "Strap color"
    assert app.node("Apparel")["parent"] is None


def test_attributes_prefill_only_from_exact_matches_in_the_listing(app):
    node = app.node(TOTES, "G1")
    prefill = {a["handle"]: a["prefill"] for a in node["attributes"]}
    assert prefill == {"color": "Red", "material": ""}


def test_values_rank_prefix_matches_first_and_are_limited(app):
    assert app.values(TOTES, "color", "re") == ["Red"]
    assert app.values(TOTES, "color", "")[:2] == ["Blue", "Red"]
    assert app.values(BAGS, "strap-color", "BLU") == ["Blue"]
    assert app.values(TOTES, "color", "zzz") == []


# HTTP


@pytest.fixture
def base(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def call(url, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    all_headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(url, data=data, headers={**all_headers, **(headers or {})})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def test_the_page_and_assets_are_served_with_a_strict_policy(base):
    status, html, headers = call(f"{base}/")
    assert status == 200
    assert b"Gold labels" in html
    assert "script-src" not in headers["Content-Security-Policy"]
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert call(f"{base}/app.js")[0] == 200
    assert call(f"{base}/style.css")[0] == 200


def test_a_full_round_trip_over_http(base):
    _, body, _ = call(f"{base}/api/state")
    assert json.loads(body)["item"]["parent_asin"] == "G1"
    assert json.loads(call(f"{base}/api/search?q=knife")[1])[0] == KNIVES
    node = json.loads(call(f"{base}/api/node?path={TOTES.replace(' ', '%20')}&asin=G1")[1])
    assert node["path"] == TOTES
    status, body, _ = call(
        f"{base}/api/label", {"asin": "G1", "path": TOTES, "attributes": {"color": "Red"}}
    )
    assert status == 200
    assert json.loads(body)["labeled"] == 1
    assert json.loads(call(f"{base}/api/skip", {"asin": "G2"})[1])["skipped"] == 1
    assert json.loads(call(f"{base}/api/undo", {})[1])["skipped"] == 0


def test_a_bad_label_gets_a_readable_error(base):
    status, body, _ = call(f"{base}/api/label", {"asin": "G1", "path": "Nope", "attributes": {}})
    assert status == 400
    assert "not a category" in json.loads(body)["error"]


def test_another_site_cannot_post_labels(base, app):
    status, _, _ = call(
        f"{base}/api/label",
        {"asin": "G1", "path": TOTES, "attributes": {}},
        {"Origin": "https://evil.example"},
    )
    assert status == 403
    assert app.store.n_labeled == 0


def test_a_rebinding_host_header_is_refused(base):
    assert call(f"{base}/api/state", headers={"Host": "evil.example"})[0] == 403


def test_a_form_post_is_refused(base):
    status, _, _ = call(
        f"{base}/api/skip", {"asin": "G1"}, {"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert status == 403


def test_unknown_routes_are_not_found(base):
    assert call(f"{base}/api/nothing")[0] == 404
    assert call(f"{base}/../../etc/passwd")[0] == 404
    assert call(f"{base}/api/node?path=Nope")[0] == 404


def test_the_server_only_binds_to_the_local_machine():
    assert taxon.gold.server.HOST == "127.0.0.1"


def test_the_page_hides_the_form_once_the_target_is_reached():
    # main has its own display rule, which would otherwise override the hidden attribute and
    # leave the form on screen with no product behind it.
    static = Path(__file__).resolve().parents[1] / "taxon" / "gold" / "static"
    assert "[hidden] { display: none !important; }" in (static / "style.css").read_text()
    script = (static / "app.js").read_text()
    for name in ("save", "skip"):
        assert f"async function {name}() {{\n  if (!state || !state.item) return;" in script

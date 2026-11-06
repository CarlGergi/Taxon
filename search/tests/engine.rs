use std::collections::BTreeMap;
use std::io::Cursor;

use taxon_search::catalog::{Listing, parse_listings};
use taxon_search::engine::{Engine, SearchRequest, build_index};
use tempfile::TempDir;

fn listing(id: &str, title: &str) -> Listing {
    Listing {
        id: id.into(),
        title: title.into(),
        store: None,
        features: vec![],
        description: String::new(),
        details: BTreeMap::new(),
        price: None,
        category: None,
        attributes: BTreeMap::new(),
    }
}

fn enriched(
    id: &str,
    title: &str,
    path: &str,
    attributes: &[(&str, &[&str])],
    price: f64,
) -> Listing {
    let mut item = listing(id, title);
    item.category = Some(path.into());
    item.price = Some(price);
    item.attributes = attributes
        .iter()
        .map(|(h, vs)| (h.to_string(), vs.iter().map(|v| v.to_string()).collect()))
        .collect();
    item
}

fn engine(listings: Vec<Listing>) -> (TempDir, Engine) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("index");
    build_index(listings, &path).unwrap();
    let opened = Engine::open(&path).unwrap();
    (dir, opened)
}

fn ids(engine: &Engine, request: &SearchRequest) -> Vec<String> {
    engine
        .search(request)
        .unwrap()
        .results
        .into_iter()
        .map(|h| h.id)
        .collect()
}

fn keywords(text: &str) -> SearchRequest {
    SearchRequest {
        query: Some(text.into()),
        ..Default::default()
    }
}

fn in_category(path: &str) -> SearchRequest {
    SearchRequest {
        category: Some(path.into()),
        ..Default::default()
    }
}

fn with_attributes(pairs: &[(&str, &str)]) -> SearchRequest {
    let map = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    SearchRequest {
        attributes: Some(map),
        ..Default::default()
    }
}

fn kitchen() -> Vec<Listing> {
    vec![
        enriched(
            "A",
            "Red chef knife",
            "Home > Kitchen > Knives",
            &[("color", &["Red"])],
            20.0,
        ),
        enriched(
            "B",
            "Blue chef knife",
            "Home > Kitchen > Knives",
            &[("color", &["Blue"])],
            40.0,
        ),
        enriched(
            "C",
            "Paring knife set",
            "Home > Kitchen",
            &[("color", &["Red", "Black"])],
            60.0,
        ),
        enriched(
            "D",
            "Garden hose",
            "Home > Kitchenette",
            &[("color", &["Red"])],
            15.0,
        ),
        enriched(
            "E",
            "Red raincoat",
            "Apparel/Outdoor > Coats",
            &[("cover-color", &["Red"])],
            80.0,
        ),
    ]
}

#[test]
fn a_title_match_ranks_above_a_description_match() {
    let mut in_description = listing("A", "Kitchen tool");
    in_description.description = "A useful spatula for any kitchen".into();
    let in_title = listing("B", "Silicone spatula");
    let (_dir, engine) = engine(vec![in_description, in_title]);
    assert_eq!(ids(&engine, &keywords("spatula")), ["B", "A"]);
}

#[test]
fn every_text_field_is_searchable() {
    let mut item = listing("A", "Thing");
    item.store = Some("Acme".into());
    item.features = vec!["waterproof".into()];
    item.description = "handmade".into();
    item.details = BTreeMap::from([("Material".to_string(), "bamboo".to_string())]);
    let (_dir, engine) = engine(vec![item, listing("B", "Other")]);
    for word in ["acme", "waterproof", "handmade", "material", "bamboo"] {
        assert_eq!(ids(&engine, &keywords(word)), ["A"], "{word}");
    }
}

#[test]
fn plurals_and_case_match_through_stemming() {
    let (_dir, engine) = engine(vec![
        listing("A", "Chef Knife Handle"),
        listing("B", "Hose"),
    ]);
    assert_eq!(ids(&engine, &keywords("HANDLES")), ["A"]);
}

#[test]
fn query_text_is_never_parsed_as_syntax() {
    let (_dir, engine) = engine(vec![listing("A", "Red knife"), listing("B", "Hose")]);
    for text in [
        "title:red",
        "red AND (",
        "\"red",
        "red^9",
        "price:[1 TO 2]",
        "NOT red",
    ] {
        assert!(engine.search(&keywords(text)).is_ok(), "{text}");
    }
    assert!(
        engine
            .search(&keywords("*"))
            .unwrap_err()
            .0
            .contains("keywords")
    );
    assert_eq!(ids(&engine, &keywords("NOT red")), ["A"]);
}

#[test]
fn a_category_filter_takes_the_node_and_its_descendants_but_not_siblings() {
    let (_dir, engine) = engine(kitchen());
    assert_eq!(
        ids(&engine, &in_category("Home > Kitchen")),
        ["A", "B", "C"]
    );
    assert_eq!(
        ids(&engine, &in_category("Home > Kitchen > Knives")),
        ["A", "B"]
    );
    assert_eq!(ids(&engine, &in_category("Home")), ["A", "B", "C", "D"]);
}

#[test]
fn a_category_filter_ignores_case_and_spacing() {
    let (_dir, engine) = engine(kitchen());
    assert_eq!(
        ids(&engine, &in_category("home  >  KITCHEN > knives")),
        ["A", "B"]
    );
}

#[test]
fn a_slash_in_a_category_name_is_not_a_level() {
    let (_dir, engine) = engine(kitchen());
    assert_eq!(ids(&engine, &in_category("Apparel/Outdoor")), ["E"]);
    assert!(ids(&engine, &in_category("Apparel")).is_empty());
}

#[test]
fn an_unknown_category_returns_nothing_rather_than_everything() {
    let (_dir, engine) = engine(kitchen());
    let response = engine.search(&in_category("Home > Nowhere")).unwrap();
    assert_eq!(response.total_matches, 0);
    assert!(response.ignored_filters.is_empty());
}

#[test]
fn attribute_filters_are_exact_and_all_must_hold() {
    let (_dir, engine) = engine(kitchen());
    assert_eq!(
        ids(&engine, &with_attributes(&[("color", "red")])),
        ["A", "C", "D"]
    );
    assert_eq!(
        ids(&engine, &with_attributes(&[("color", "Bl")])),
        Vec::<String>::new()
    );
    assert_eq!(
        ids(&engine, &with_attributes(&[("cover-color", "RED")])),
        ["E"]
    );
    assert!(
        ids(
            &engine,
            &with_attributes(&[("color", "Red"), ("cover-color", "Red")])
        )
        .is_empty()
    );
}

#[test]
fn any_value_of_a_multivalued_attribute_matches() {
    let (_dir, engine) = engine(kitchen());
    assert_eq!(ids(&engine, &with_attributes(&[("color", "Black")])), ["C"]);
}

#[test]
fn a_value_is_not_mistaken_for_a_handle() {
    let (_dir, engine) = engine(kitchen());
    assert!(ids(&engine, &with_attributes(&[("red", "color")])).is_empty());
}

#[test]
fn price_bounds_are_inclusive_and_leave_out_unpriced_products() {
    let mut items = kitchen();
    items.push(listing("F", "Unpriced knife"));
    let (_dir, engine) = engine(items);
    let range = |low, high| SearchRequest {
        min_price: low,
        max_price: high,
        ..Default::default()
    };
    assert_eq!(ids(&engine, &range(Some(20.0), Some(40.0))), ["A", "B"]);
    assert_eq!(ids(&engine, &range(None, Some(15.0))), ["D"]);
    assert_eq!(ids(&engine, &range(Some(60.0), None)), ["C", "E"]);
    assert!(!ids(&engine, &range(Some(0.0), None)).contains(&"F".to_string()));
}

#[test]
fn a_filter_narrows_results_without_changing_their_order() {
    let (_dir, engine) = engine(kitchen());
    let open = ids(&engine, &keywords("red knife"));
    let mut filtered = keywords("red knife");
    filtered.max_price = Some(1000.0);
    assert_eq!(ids(&engine, &filtered), open);
    filtered.category = Some("Home".into());
    let expected: Vec<String> = open.iter().filter(|id| *id != "E").cloned().collect();
    assert_eq!(ids(&engine, &filtered), expected);
}

#[test]
fn enriched_text_is_searchable_by_keyword() {
    let (_dir, engine) = engine(kitchen());
    assert_eq!(ids(&engine, &keywords("coats")), ["E"]);
}

fn raw() -> Vec<Listing> {
    let mut item = listing("A", "Red chef knife");
    item.price = Some(20.0);
    vec![item, listing("B", "Blue chef knife")]
}

#[test]
fn a_raw_catalog_reports_filters_it_cannot_apply_and_still_answers() {
    let (_dir, engine) = engine(raw());
    let mut request = keywords("knife");
    request.category = Some("Home > Kitchen".into());
    request.attributes = Some(BTreeMap::from([("color".to_string(), "Red".to_string())]));
    let response = engine.search(&request).unwrap();
    assert_eq!(response.ignored_filters, ["category", "attributes"]);
    assert_eq!(response.total_matches, 2);
}

#[test]
fn a_raw_catalog_still_filters_on_price() {
    let (_dir, engine) = engine(raw());
    let mut request = keywords("knife");
    request.max_price = Some(25.0);
    let response = engine.search(&request).unwrap();
    assert!(response.ignored_filters.is_empty());
    assert_eq!(response.total_matches, 1);
}

#[test]
fn a_raw_catalog_needs_keywords_when_its_only_filter_is_unavailable() {
    let (_dir, engine) = engine(raw());
    let error = engine.search(&in_category("Home")).unwrap_err();
    assert!(error.0.contains("no data for category"), "{error}");
}

#[test]
fn a_raw_result_has_no_empty_enrichment_fields() {
    let (_dir, engine) = engine(raw());
    let response = engine.search(&keywords("red")).unwrap();
    let json = serde_json::to_value(&response).unwrap();
    let first = json["results"][0].as_object().unwrap();
    assert!(!first.contains_key("category") && !first.contains_key("attributes"));
    assert!(!json.as_object().unwrap().contains_key("ignored_filters"));
}

#[test]
fn bad_requests_are_refused_with_a_reason() {
    let (_dir, engine) = engine(kitchen());
    let refused = |request: SearchRequest| engine.search(&request).unwrap_err().0;
    assert!(refused(SearchRequest::default()).contains("keywords"));
    assert!(refused(keywords("!!!")).contains("keywords"));
    assert!(refused(in_category("  ")).contains("empty"));
    assert!(refused(with_attributes(&[("color", " ")])).contains("empty"));
    let mut request = keywords("knife");
    request.limit = Some(0);
    assert!(refused(request).contains("limit"));
    let mut request = keywords("knife");
    request.min_price = Some(9.0);
    request.max_price = Some(1.0);
    assert!(refused(request).contains("above"));
    let mut request = keywords("knife");
    request.max_price = Some(-1.0);
    assert!(refused(request).contains("non-negative"));
}

#[test]
fn the_limit_defaults_to_ten_and_stops_at_fifty() {
    let items = (0..60)
        .map(|i| listing(&format!("{i:03}"), "Same knife"))
        .collect();
    let (_dir, engine) = engine(items);
    let response = engine.search(&keywords("knife")).unwrap();
    assert_eq!((response.results.len(), response.total_matches), (10, 60));
    let mut request = keywords("knife");
    request.limit = Some(500);
    assert_eq!(engine.search(&request).unwrap().results.len(), 50);
}

#[test]
fn results_do_not_depend_on_input_order_and_ties_fall_back_to_id() {
    let items: Vec<Listing> = (0..40)
        .map(|i| listing(&format!("{i:02}"), "Same knife"))
        .collect();
    let mut reversed = items.clone();
    reversed.reverse();
    let (_a, forward) = engine(items);
    let (_b, backward) = engine(reversed);
    let mut request = keywords("knife");
    request.limit = Some(5);
    let expected = ["00", "01", "02", "03", "04"];
    assert_eq!(ids(&forward, &request), expected);
    assert_eq!(ids(&backward, &request), expected);
}

#[test]
fn long_text_is_clipped_on_character_boundaries() {
    let mut item = listing("A", "Knife");
    item.description = "é".repeat(1000);
    item.features = vec!["ü".repeat(1000); 9];
    item.details = (0..30)
        .map(|i| (format!("k{i:02}"), "ß".repeat(1000)))
        .collect();
    let (_dir, engine) = engine(vec![item]);
    let hit = engine.search(&keywords("knife")).unwrap().results.remove(0);
    assert_eq!(hit.description.chars().count(), 303);
    assert_eq!(hit.features.len(), 5);
    assert_eq!(hit.features[0].chars().count(), 163);
    assert_eq!(hit.details.len(), 20);
    assert_eq!(hit.details["k00"].chars().count(), 123);
}

#[test]
fn the_manifest_says_what_the_catalog_holds() {
    let (_a, raw_engine) = engine(raw());
    let manifest = raw_engine.manifest();
    assert_eq!(manifest.documents, 2);
    assert!(!manifest.has_category && !manifest.has_attributes);
    let (_b, full) = engine(kitchen());
    assert!(full.manifest().has_category && full.manifest().has_attributes);
}

#[test]
fn building_replaces_an_earlier_index() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("index");
    build_index(raw(), &path).unwrap();
    build_index(kitchen(), &path).unwrap();
    assert_eq!(Engine::open(&path).unwrap().manifest().documents, 5);
}

#[test]
fn building_refuses_a_folder_that_is_not_an_index() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("precious.txt"), "keep me").unwrap();
    assert!(build_index(raw(), dir.path()).is_err());
    assert!(dir.path().join("precious.txt").exists());
}

#[test]
fn opening_refuses_an_unfinished_index() {
    let dir = TempDir::new().unwrap();
    assert!(Engine::open(dir.path()).is_err());
}

fn parse(text: &str) -> anyhow::Result<Vec<Listing>> {
    parse_listings(Cursor::new(text))
}

#[test]
fn the_catalog_parser_reads_lines_and_skips_blanks() {
    let listings =
        parse("{\"id\": \"A\", \"title\": \"x\"}\n\n{\"id\": \"B\", \"title\": \"y\"}\n");
    assert_eq!(listings.unwrap().len(), 2);
}

#[test]
fn the_catalog_parser_names_the_bad_line() {
    let error = parse("{\"id\": \"A\", \"title\": \"x\"}\nnot json\n").unwrap_err();
    assert!(format!("{error:#}").contains("line 2"), "{error:#}");
}

#[test]
fn the_catalog_parser_refuses_bad_listings() {
    let duplicate = "{\"id\": \"A\", \"title\": \"x\"}\n{\"id\": \"A\", \"title\": \"y\"}\n";
    assert!(
        parse(duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    assert!(parse("{\"id\": \"\", \"title\": \"x\"}\n").is_err());
    assert!(parse("{\"id\": \"A\", \"title\": \"x\", \"price\": -1}\n").is_err());
}

#[test]
fn the_catalog_parser_refuses_fields_that_could_leak_a_label() {
    let leaky = "{\"id\": \"A\", \"title\": \"x\", \"main_category\": \"Kitchen\"}\n";
    assert!(parse(leaky).is_err());
}

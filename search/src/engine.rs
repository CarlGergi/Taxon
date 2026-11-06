//! The tantivy index behind `search_catalog`.
//!
//! Results must not depend on how an index happened to be built, because a benchmark compares
//! runs. Listings are added in id order by one thread with merging off, so equal scores always
//! fall back to id order.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::ops::Bound;
use std::path::Path;

use anyhow::{Context, Result, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tantivy::collector::{Count, TopDocs};
use tantivy::indexer::NoMergePolicy;
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, ConstScoreQuery, Occur, Query, RangeQuery, TermQuery,
};
use tantivy::schema::{
    FAST, FacetOptions, Field, INDEXED, IndexRecordOption, STORED, Schema, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::catalog::Listing;

const MANIFEST: &str = "manifest.json";
const FORMAT: u32 = 1;
const WRITER_MEMORY: usize = 200_000_000;

const DEFAULT_LIMIT: u32 = 10;
const MAX_LIMIT: u32 = 50;
const MAX_QUERY_TERMS: usize = 64;

// What one result can show, so a long listing cannot flood an agent's context.
const DESCRIPTION_CHARS: usize = 300;
const MAX_FEATURES: usize = 5;
const FEATURE_CHARS: usize = 160;
const MAX_DETAILS: usize = 20;
const DETAIL_CHARS: usize = 120;

const SEPARATOR: &str = " > ";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub documents: usize,
    pub has_category: bool,
    pub has_attributes: bool,
}

/// The arguments of `search_catalog`.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    /// Keywords, matched against each listing's text. Plain words only, not a query language.
    #[serde(default)]
    pub query: Option<String>,
    /// A category path such as "Home & Garden > Kitchen & Dining". Matches that category and
    /// everything under it.
    #[serde(default)]
    pub category: Option<String>,
    /// Attribute values a product must have, by attribute handle, such as {"color": "Red"}.
    #[serde(default)]
    pub attributes: Option<BTreeMap<String, String>>,
    /// Lowest price, inclusive. Products with no listed price are left out.
    #[serde(default)]
    pub min_price: Option<f64>,
    /// Highest price, inclusive. Products with no listed price are left out.
    #[serde(default)]
    pub max_price: Option<f64>,
    /// How many results to return, from 1 to 50. The default is 10.
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct Hit {
    pub id: String,
    pub title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct SearchResponse {
    /// How many products matched, of which `results` holds the best.
    pub total_matches: usize,
    pub results: Vec<Hit>,
    /// Filters this catalog has no data for, which therefore did not narrow the results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored_filters: Vec<String>,
}

/// A request that cannot be run, reported back to the caller rather than treated as a failure.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestError(pub String);

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RequestError {}

#[derive(Clone, Copy)]
struct Fields {
    id: Field,
    listing: Field,
    price: Field,
    category: Field,
    attribute: Field,
    // Each with the weight of a match there.
    text: [(Field, f32); 6],
}

fn schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let text = || {
        TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("en_stem")
                .set_index_option(IndexRecordOption::WithFreqs),
        )
    };
    let id = builder.add_text_field("id", tantivy::schema::STRING | STORED);
    let listing = builder.add_text_field("listing", STORED);
    let price = builder.add_f64_field("price", INDEXED | FAST);
    let category = builder.add_facet_field("category", FacetOptions::default());
    let attribute = builder.add_facet_field("attribute", FacetOptions::default());
    let title = builder.add_text_field("title", text());
    let enrichment = builder.add_text_field("enrichment", text());
    let features = builder.add_text_field("features", text());
    let store = builder.add_text_field("store", text());
    let description = builder.add_text_field("description", text());
    let details = builder.add_text_field("details", text());
    let fields = Fields {
        id,
        listing,
        price,
        category,
        attribute,
        text: [
            (title, 3.0),
            (enrichment, 2.0),
            (features, 1.5),
            (store, 1.0),
            (description, 1.0),
            (details, 1.0),
        ],
    };
    (builder.build(), fields)
}

fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn category_facet(path: &str) -> tantivy::schema::Facet {
    tantivy::schema::Facet::from_path(path.split(SEPARATOR).map(normalized))
}

fn attribute_facet(handle: &str, value: &str) -> tantivy::schema::Facet {
    tantivy::schema::Facet::from_path([normalized(handle), normalized(value)])
}

fn clipped(text: &str, limit: usize) -> String {
    match text.char_indices().nth(limit) {
        Some((end, _)) => format!("{}...", &text[..end]),
        None => text.to_string(),
    }
}

fn hit(listing: Listing) -> Hit {
    Hit {
        id: listing.id,
        title: listing.title,
        store: listing.store,
        price: listing.price,
        category: listing.category,
        attributes: listing.attributes,
        features: listing
            .features
            .iter()
            .take(MAX_FEATURES)
            .map(|f| clipped(f, FEATURE_CHARS))
            .collect(),
        description: clipped(&listing.description, DESCRIPTION_CHARS),
        details: listing
            .details
            .iter()
            .take(MAX_DETAILS)
            .map(|(k, v)| (k.clone(), clipped(v, DETAIL_CHARS)))
            .collect(),
    }
}

fn add_listing(writer: &IndexWriter, fields: &Fields, listing: &Listing) -> Result<()> {
    let mut doc = TantivyDocument::default();
    doc.add_text(fields.id, &listing.id);
    doc.add_text(fields.listing, serde_json::to_string(listing)?);
    if let Some(price) = listing.price {
        doc.add_f64(fields.price, price);
    }
    if let Some(path) = &listing.category {
        doc.add_facet(fields.category, category_facet(path));
    }
    for (handle, values) in &listing.attributes {
        for value in values {
            doc.add_facet(fields.attribute, attribute_facet(handle, value));
        }
    }
    let details: Vec<String> = listing
        .details
        .iter()
        .map(|(k, v)| format!("{k} {v}"))
        .collect();
    let enrichment: Vec<&str> = listing
        .category
        .iter()
        .map(String::as_str)
        .chain(listing.attributes.values().flatten().map(String::as_str))
        .collect();
    let [title, enriched, features, store, description, detail] = fields.text.map(|(f, _)| f);
    doc.add_text(title, &listing.title);
    doc.add_text(enriched, enrichment.join(" "));
    doc.add_text(features, listing.features.join(" "));
    doc.add_text(store, listing.store.as_deref().unwrap_or(""));
    doc.add_text(description, &listing.description);
    doc.add_text(detail, details.join(" "));
    writer.add_document(doc)?;
    Ok(())
}

/// Build an index at `dir`, replacing an earlier index there but nothing else.
pub fn build_index(mut listings: Vec<Listing>, dir: &Path) -> Result<Manifest> {
    if dir.exists() {
        let empty = fs::read_dir(dir)?.next().is_none();
        if !empty && !dir.join(MANIFEST).exists() {
            bail!("{} holds something other than an index", dir.display());
        }
        fs::remove_dir_all(dir)?;
    }
    fs::create_dir_all(dir)?;
    listings.sort_by(|a, b| a.id.cmp(&b.id));

    let (schema, fields) = schema();
    let index = Index::create_in_dir(dir, schema)?;
    let mut writer: IndexWriter = index.writer_with_num_threads(1, WRITER_MEMORY)?;
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for listing in &listings {
        add_listing(&writer, &fields, listing)?;
    }
    writer.commit()?;
    writer.wait_merging_threads()?;

    let manifest = Manifest {
        format: FORMAT,
        documents: listings.len(),
        has_category: listings.iter().any(|l| l.category.is_some()),
        has_attributes: listings.iter().any(|l| !l.attributes.is_empty()),
    };
    // Written last, so an interrupted build leaves something open() refuses.
    fs::write(dir.join(MANIFEST), serde_json::to_string_pretty(&manifest)?)?;
    Ok(manifest)
}

pub struct Engine {
    index: Index,
    reader: IndexReader,
    fields: Fields,
    manifest: Manifest,
}

impl Engine {
    pub fn open(dir: &Path) -> Result<Engine> {
        let path = dir.join(MANIFEST);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("no finished index at {}", dir.display()))?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        if manifest.format != FORMAT {
            bail!(
                "index format {} is not {FORMAT}; rebuild it",
                manifest.format
            );
        }
        let index = Index::open_in_dir(dir)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(Engine {
            index,
            reader,
            fields: schema().1,
            manifest,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn terms(&self, text: &str) -> Result<Vec<String>> {
        let mut analyzer = self.index.tokenizer_for_field(self.fields.text[0].0)?;
        let mut stream = analyzer.token_stream(text);
        let mut seen = HashSet::new();
        let mut terms = Vec::new();
        while stream.advance() {
            let term = stream.token().text.clone();
            if seen.insert(term.clone()) {
                terms.push(term);
            }
        }
        terms.truncate(MAX_QUERY_TERMS);
        Ok(terms)
    }

    fn keyword_query(&self, terms: &[String]) -> Box<dyn Query> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for term in terms {
            for (field, weight) in self.fields.text {
                let term = Term::from_field_text(field, term);
                let query = TermQuery::new(term, IndexRecordOption::WithFreqs);
                clauses.push((
                    Occur::Should,
                    Box::new(BoostQuery::new(Box::new(query), weight)),
                ));
            }
        }
        Box::new(BooleanQuery::new(clauses))
    }

    pub fn search(&self, request: &SearchRequest) -> Result<SearchResponse, RequestError> {
        self.run(request)
            .map_err(|error| match error.downcast::<RequestError>() {
                Ok(request_error) => request_error,
                Err(other) => RequestError(format!("search failed: {other}")),
            })
    }

    fn run(&self, request: &SearchRequest) -> Result<SearchResponse> {
        let limit = request.limit.unwrap_or(DEFAULT_LIMIT);
        if limit == 0 {
            return Err(RequestError("limit must be at least 1".into()).into());
        }
        let limit = limit.min(MAX_LIMIT) as usize;
        let mut filters: Vec<Box<dyn Query>> = Vec::new();
        let mut ignored = Vec::new();

        if let Some(path) = &request.category {
            if normalized(path).is_empty() {
                return Err(RequestError("category is empty".into()).into());
            }
            if self.manifest.has_category {
                let term = Term::from_facet(self.fields.category, &category_facet(path));
                filters.push(Box::new(TermQuery::new(term, IndexRecordOption::Basic)));
            } else {
                ignored.push("category".to_string());
            }
        }
        if let Some(attributes) = request.attributes.as_ref().filter(|a| !a.is_empty()) {
            if attributes
                .iter()
                .any(|(k, v)| normalized(k).is_empty() || normalized(v).is_empty())
            {
                return Err(
                    RequestError("attribute names and values cannot be empty".into()).into(),
                );
            }
            if self.manifest.has_attributes {
                for (handle, value) in attributes {
                    let facet = attribute_facet(handle, value);
                    let term = Term::from_facet(self.fields.attribute, &facet);
                    filters.push(Box::new(TermQuery::new(term, IndexRecordOption::Basic)));
                }
            } else {
                ignored.push("attributes".to_string());
            }
        }
        if let Some(range) = self.price_filter(request)? {
            filters.push(range);
        }

        let terms = self.terms(request.query.as_deref().unwrap_or(""))?;
        if terms.is_empty() && filters.is_empty() {
            return Err(RequestError(if ignored.is_empty() {
                "give keywords or at least one filter".into()
            } else {
                format!(
                    "this catalog has no data for {}, so give keywords",
                    ignored.join(" or ")
                )
            })
            .into());
        }
        let base: Box<dyn Query> = if terms.is_empty() {
            Box::new(AllQuery)
        } else {
            self.keyword_query(&terms)
        };
        let mut clauses = vec![(Occur::Must, base)];
        // A filter only narrows the results. It must not change how they rank.
        for filter in filters {
            clauses.push((Occur::Must, Box::new(ConstScoreQuery::new(filter, 0.0))));
        }
        let query = BooleanQuery::new(clauses);

        let searcher = self.reader.searcher();
        let (top, total_matches) = searcher.search(
            &query,
            &(TopDocs::with_limit(limit).order_by_score(), Count),
        )?;
        let mut results = Vec::with_capacity(top.len());
        for (_, address) in top {
            let doc: TantivyDocument = searcher.doc(address)?;
            let stored = doc
                .get_first(self.fields.listing)
                .and_then(|v| v.as_str())
                .context("a document has no stored listing")?;
            results.push(hit(serde_json::from_str(stored)?));
        }
        Ok(SearchResponse {
            total_matches,
            results,
            ignored_filters: ignored,
        })
    }

    fn price_filter(&self, request: &SearchRequest) -> Result<Option<Box<dyn Query>>> {
        let (low, high) = (request.min_price, request.max_price);
        for price in [low, high].into_iter().flatten() {
            if !price.is_finite() || price < 0.0 {
                return Err(RequestError("prices must be non-negative numbers".into()).into());
            }
        }
        if low.zip(high).is_some_and(|(l, h)| l > h) {
            return Err(RequestError("min_price is above max_price".into()).into());
        }
        if low.is_none() && high.is_none() {
            return Ok(None);
        }
        let bound = |price: Option<f64>| match price {
            Some(p) => Bound::Included(Term::from_field_f64(self.fields.price, p)),
            None => Bound::Unbounded,
        };
        Ok(Some(Box::new(RangeQuery::new(bound(low), bound(high)))))
    }
}

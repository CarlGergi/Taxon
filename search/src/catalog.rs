//! The catalog file: one JSON listing per line, the same shape for raw and enriched catalogs.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Unknown fields are refused, so a field that would leak a label cannot slip into the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Listing {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub details: BTreeMap<String, String>,
    #[serde(default)]
    pub price: Option<f64>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, Vec<String>>,
}

pub fn parse_listings(reader: impl BufRead) -> Result<Vec<Listing>> {
    let mut listings = Vec::new();
    let mut seen = HashSet::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let number = i + 1;
        let listing: Listing =
            serde_json::from_str(&line).with_context(|| format!("line {number}"))?;
        if listing.id.is_empty() {
            bail!("line {number}: empty id");
        }
        if listing.price.is_some_and(|p| !p.is_finite() || p < 0.0) {
            bail!("line {number}: price must be a non-negative number");
        }
        if !seen.insert(listing.id.clone()) {
            bail!("line {number}: duplicate id {}", listing.id);
        }
        listings.push(listing);
    }
    Ok(listings)
}

pub fn read_catalog(path: &Path) -> Result<Vec<Listing>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    parse_listings(BufReader::new(file)).with_context(|| format!("reading {}", path.display()))
}

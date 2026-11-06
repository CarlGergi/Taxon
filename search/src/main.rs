use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use taxon_search::catalog::read_catalog;
use taxon_search::engine::{Engine, SearchRequest, build_index};
use taxon_search::server::serve_stdio;

#[derive(Parser)]
#[command(about = "Keyword search with filters over a product catalog")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build an index from a catalog file.
    Build {
        #[arg(long)]
        catalog: PathBuf,
        #[arg(long)]
        index: PathBuf,
    },
    /// Serve an index as an MCP server over stdio.
    Serve {
        #[arg(long)]
        index: PathBuf,
    },
    /// Run one search_catalog request, given as JSON, and print the response.
    Query {
        #[arg(long)]
        index: PathBuf,
        request: String,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Build { catalog, index } => {
            let manifest = build_index(read_catalog(&catalog)?, &index)?;
            println!("{}", serde_json::to_string(&manifest)?);
        }
        Command::Serve { index } => {
            let engine = Engine::open(&index)?;
            tokio::runtime::Runtime::new()?.block_on(serve_stdio(engine))?;
        }
        Command::Query { index, request } => {
            let request: SearchRequest = serde_json::from_str(&request)?;
            let response = Engine::open(&index)?.search(&request)?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
    }
    Ok(())
}

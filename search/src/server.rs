//! The local MCP server. It exposes one tool, with the same schema for every catalog, so an agent's
//! tools do not change between benchmark conditions.

use std::sync::Arc;

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::{ServiceExt, tool, tool_handler, tool_router};

use crate::engine::{Engine, SearchRequest, SearchResponse};

const DESCRIPTION: &str = "Search the product catalog. Keywords match listing text, and optional \
filters narrow the results by category, attribute values and price. A catalog with no category or \
attribute data lists those filters in ignored_filters instead of applying them.";

#[derive(Clone)]
pub struct CatalogServer {
    engine: Arc<Engine>,
}

#[tool_router]
impl CatalogServer {
    pub fn new(engine: Engine) -> Self {
        CatalogServer {
            engine: Arc::new(engine),
        }
    }

    #[tool(name = "search_catalog", description = DESCRIPTION)]
    fn search_catalog(
        &self,
        Parameters(request): Parameters<SearchRequest>,
    ) -> Result<Json<SearchResponse>, String> {
        self.engine
            .search(&request)
            .map(Json)
            .map_err(|error| error.0)
    }
}

#[tool_handler(name = "catalog", version = "0.1.0")]
impl rmcp::ServerHandler for CatalogServer {}

/// Serve over stdin and stdout until the client disconnects. Nothing else may write to stdout.
pub async fn serve_stdio(engine: Engine) -> anyhow::Result<()> {
    let service = CatalogServer::new(engine)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}

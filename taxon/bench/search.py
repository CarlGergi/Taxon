"""Build the Rust search engine and talk to it the way an agent will: over MCP."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from taxon.pins import ROOT
from taxon.products import seeded_hash, write_jsonl

BINARY = ROOT / "target" / "release" / "taxon-search"
TOOL = "search_catalog"
RAW = "raw"
SELF_RETRIEVAL_SAMPLE = 500
SELF_RETRIEVAL_TOP = 10


class RustMissing(RuntimeError):
    """No Rust toolchain was found."""


def _cargo() -> str:
    found = shutil.which("cargo")
    if found:
        return found
    # Homebrew's rustup keeps cargo off PATH.
    try:
        done = subprocess.run(
            ["rustup", "which", "cargo"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RustMissing("Rust is not installed; get it from https://rustup.rs") from error
    return done.stdout.strip()


def build_binary() -> Path:
    cargo = _cargo()
    env = {**os.environ, "PATH": f"{Path(cargo).parent}{os.pathsep}{os.environ['PATH']}"}
    command = [cargo, "build", "--release", "--locked", "--quiet", "-p", "taxon-search"]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return BINARY


def catalog_path(name: str, derived: Path) -> Path:
    return derived / "catalogs" / f"{name}.jsonl"


def index_dir(name: str, derived: Path) -> Path:
    return derived / "search" / name


def build_index(catalog: Path, index: Path, binary: Path = BINARY) -> dict:
    command = [str(binary), "build", "--catalog", str(catalog), "--index", str(index)]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def index_catalog(rows: Sequence[dict], name: str, derived: Path) -> dict:
    """Write a catalog under `name` and index it, returning what the index holds."""
    path = catalog_path(name, derived)
    write_jsonl(path, rows)
    index = index_dir(name, derived)
    index.parent.mkdir(parents=True, exist_ok=True)
    return build_index(path, index, build_binary())


def server_parameters(index: Path, binary: Path = BINARY) -> StdioServerParameters:
    return StdioServerParameters(command=str(binary), args=["serve", "--index", str(index)])


def _read(result) -> dict:
    if result.is_error:
        return {"error": " ".join(block.text for block in result.content)}
    return result.structured_content


async def _search_all(index: Path, requests: Sequence[dict], binary: Path) -> list[dict]:
    async with stdio_client(server_parameters(index, binary)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return [_read(await session.call_tool(TOOL, request)) for request in requests]


def search_many(index: Path, requests: Sequence[dict], binary: Path = BINARY) -> list[dict]:
    """Responses to `requests`, in one session. A refused request comes back as {"error": ...}."""
    return asyncio.run(_search_all(index, requests, binary))


def self_retrieval(rows: Sequence[dict], index: Path, seed: int, binary: Path = BINARY) -> dict:
    """How often searching a product's exact title finds that product.

    This checks that the engine and the catalog work together. It says nothing about how well an
    agent will shop, and involves no labels.
    """
    ordered = sorted(rows, key=lambda row: seeded_hash(seed, f"search|{row['id']}"))
    sample = ordered[:SELF_RETRIEVAL_SAMPLE]
    requests = [{"query": row["title"], "limit": SELF_RETRIEVAL_TOP} for row in sample]
    responses = search_many(index, requests, binary)
    first = top = 0
    for row, response in zip(sample, responses, strict=True):
        found = [hit["id"] for hit in response["results"]]
        first += found[:1] == [row["id"]]
        top += row["id"] in found
    return {
        "sample": len(sample),
        "found_first": first / len(sample),
        f"found_in_top_{SELF_RETRIEVAL_TOP}": top / len(sample),
    }

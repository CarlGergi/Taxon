"""Serve the labeling page on the local machine only."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from taxon.fetch import fetch
from taxon.gold.app import GoldApp
from taxon.gold.pool import FIRST_TARGET
from taxon.gold.store import GoldStore, InvalidLabel
from taxon.pins import DATA_DIR, DERIVED_DIR, TAXONOMY
from taxon.products import read_jsonl
from taxon.taxonomy import Taxonomy

STATIC = Path(__file__).parent / "static"
LABELS_PATH = DATA_DIR / "gold" / "labels.jsonl"
HOST = "127.0.0.1"
PORT = 8765
_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
# Listing text is untrusted. Scripts load only from here, and images only over https.
_CSP = "default-src 'self'; img-src https:; object-src 'none'; base-uri 'none'"


def make_handler(app: GoldApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", _CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _local_request(self) -> bool:
            """Refuse other sites reaching this server through a browser."""
            allowed = {f"{HOST}:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                return False
            origin = self.headers.get("Origin")
            return origin is None or urlparse(origin).netloc in allowed

        def do_GET(self) -> None:
            if not self._local_request():
                return self._json({"error": "forbidden"}, 403)
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in _FILES:
                name, content_type = _FILES[url.path]
                return self._send(200, (STATIC / name).read_bytes(), content_type)
            try:
                if url.path == "/api/state":
                    return self._json(app.state())
                if url.path == "/api/search":
                    return self._json(app.search(query.get("q", "")))
                if url.path == "/api/node":
                    return self._json(app.node(query["path"], query.get("asin")))
                if url.path == "/api/values":
                    found = app.values(query["path"], query["handle"], query.get("q", ""))
                    return self._json(found)
            except KeyError as missing:
                return self._json({"error": f"not found: {missing}"}, 404)
            return self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if not self._local_request() or self.headers.get("Content-Type") != "application/json":
                return self._json({"error": "forbidden"}, 403)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/label":
                    return self._json(app.label(body["asin"], body["path"], body["attributes"]))
                if self.path == "/api/skip":
                    return self._json(app.skip(body["asin"]))
                if self.path == "/api/undo":
                    return self._json(app.undo())
            except (InvalidLabel, KeyError, json.JSONDecodeError) as error:
                return self._json({"error": str(error)}, 400)
            return self._json({"error": "not found"}, 404)

    return Handler


def load_app(labels: Path, target: int, derived: Path = DERIVED_DIR) -> GoldApp:
    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    queue = [row["parent_asin"] for row in read_jsonl(derived / "gold_queue.jsonl")]
    pool = set(queue)
    products = {
        p["parent_asin"]: p
        for p in read_jsonl(derived / "products.jsonl")
        if p["parent_asin"] in pool
    }
    suggestions = {
        row["parent_asin"]: row["suggestions"]
        for row in read_jsonl(derived / "gold_suggestions.jsonl")
    }
    return GoldApp(products, suggestions, taxonomy, GoldStore(labels, queue, taxonomy), target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m taxon.gold", description=__doc__)
    parser.add_argument("--target", type=int, default=FIRST_TARGET)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args(argv)
    app = load_app(args.labels, args.target)
    server = ThreadingHTTPServer((HOST, args.port), make_handler(app))
    print(f"Label at http://{HOST}:{args.port}  ({app.store.n_labeled}/{args.target} done)")
    server.serve_forever()

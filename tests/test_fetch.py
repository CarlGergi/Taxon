import hashlib
import ssl
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from taxon.fetch import _TLS, cached_range, fetch, fetch_range
from taxon.pins import PinnedFile

BODY = b"0123456789" * 10


class Handler(BaseHTTPRequestHandler):
    throttle_next = 0
    reported_size = len(BODY)
    ignore_range = False

    def log_message(self, *args):
        pass

    def do_GET(self):
        if Handler.throttle_next:
            Handler.throttle_next -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        byte_range = self.headers.get("Range")
        if byte_range is None or Handler.ignore_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
            return
        start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
        chunk = BODY[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{Handler.reported_size}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def url():
    Handler.throttle_next = 0
    Handler.reported_size = len(BODY)
    Handler.ignore_range = False
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/file"
    server.shutdown()
    server.server_close()


def pin(url, sha=None):
    return PinnedFile(url, sha or hashlib.sha256(BODY).hexdigest(), "dir/file.bin")


def test_fetch_downloads_and_caches(tmp_path, url):
    path = fetch(pin(url), tmp_path)
    assert path.read_bytes() == BODY
    assert fetch(pin("http://127.0.0.1:9/never-called"), tmp_path) == path


def test_fetch_rejects_a_bad_checksum_and_leaves_nothing_behind(tmp_path, url):
    with pytest.raises(ValueError, match="checksum"):
        fetch(pin(url, sha="0" * 64), tmp_path)
    assert list((tmp_path / "dir").iterdir()) == []


def test_fetch_rejects_a_tampered_cache(tmp_path, url):
    path = fetch(pin(url), tmp_path)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cached file"):
        fetch(pin(url), tmp_path)


def test_fetch_retries_a_rate_limit(tmp_path, url):
    Handler.throttle_next = 2
    assert fetch(pin(url), tmp_path, backoff=0).read_bytes() == BODY


def test_fetch_range_reads_exact_bytes(url):
    assert fetch_range(url, 5, 10, len(BODY)) == BODY[5:15]


def test_fetch_range_retries_a_rate_limit(url):
    Handler.throttle_next = 3
    assert fetch_range(url, 0, 4, len(BODY), backoff=0) == BODY[:4]


def test_fetch_range_rejects_a_file_of_the_wrong_size(url):
    Handler.reported_size = 999
    with pytest.raises(ValueError, match="not the pinned one"):
        fetch_range(url, 0, 4, len(BODY))


def test_fetch_range_rejects_a_server_that_ignores_ranges(url):
    Handler.ignore_range = True
    with pytest.raises(ValueError, match="range response"):
        fetch_range(url, 0, 4, len(BODY))


def test_cached_range_serves_repeat_reads_from_disk(tmp_path, url):
    read = cached_range(url, len(BODY), tmp_path / "chunks")
    assert read(5, 10) == BODY[5:15]
    offline = cached_range("http://127.0.0.1:9/never-called", len(BODY), tmp_path / "chunks")
    assert offline(5, 10) == BODY[5:15]
    assert sorted(p.name for p in (tmp_path / "chunks").iterdir()) == ["5-10.bin"]


def test_tls_verification_stays_on():
    assert _TLS.verify_mode == ssl.CERT_REQUIRED
    assert _TLS.check_hostname is True
    assert _TLS.cert_store_stats()["x509_ca"] > 0


def test_a_certificate_failure_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise urllib.error.URLError(ssl.SSLCertVerificationError("bad certificate"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(urllib.error.URLError):
        fetch(pin("https://example.invalid/file"), tmp_path, backoff=0)
    assert len(calls) == 1


def test_a_dropped_connection_is_retried(tmp_path, monkeypatch, url):
    real = urllib.request.urlopen
    failures = [urllib.error.URLError(ConnectionResetError("reset"))]

    def flaky(*args, **kwargs):
        if failures:
            raise failures.pop()
        return real(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    assert fetch(pin(url), tmp_path, backoff=0).read_bytes() == BODY

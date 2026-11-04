"""Download pinned inputs, verifying checksums and retrying transient failures."""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

from taxon.pins import RAW_DIR, PinnedFile

_RETRYABLE = {408, 429, 500, 502, 503, 504}
_USER_AGENT = "taxon/0.1"
# Some Python builds ship without a CA store, so verification uses a pinned bundle everywhere.
_TLS = ssl.create_default_context(cafile=certifi.where())


def _open(url: str, headers: dict[str, str], attempts: int, backoff: float):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **headers})
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=120, context=_TLS)
        except urllib.error.HTTPError as error:
            if error.code not in _RETRYABLE or attempt == attempts - 1:
                raise
            delay = float(error.headers.get("Retry-After") or backoff * 2**attempt)
        except urllib.error.URLError as error:
            if isinstance(error.reason, ssl.SSLCertVerificationError) or attempt == attempts - 1:
                raise
            delay = backoff * 2**attempt
        time.sleep(delay)
    raise RuntimeError("unreachable")


def open_url(url: str, attempts: int = 3, backoff: float = 1.0):
    """Open a URL with the same TLS settings and retries as pinned downloads."""
    return _open(url, {}, attempts, backoff)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(pin: PinnedFile, root: Path = RAW_DIR, attempts: int = 5, backoff: float = 1.0) -> Path:
    """Return the verified local copy of `pin`, downloading it if needed."""
    target = root / pin.relative_path
    if target.exists():
        if sha256_of(target) != pin.sha256:
            raise ValueError(f"cached file fails its checksum, delete it and rerun: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + f".{os.getpid()}.partial")
    try:
        with _open(pin.url, {}, attempts, backoff) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        if sha256_of(partial) != pin.sha256:
            raise ValueError(f"download fails its checksum: {pin.url}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def fetch_range(
    url: str,
    offset: int,
    length: int,
    expected_size: int,
    attempts: int = 6,
    backoff: float = 1.0,
) -> bytes:
    """Read `length` bytes at `offset`, and confirm the remote file is the size we pinned."""
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    for attempt in range(attempts):
        try:
            with _open(url, headers, attempts, backoff) as response:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status != 206 or match is None:
                    raise ValueError(f"expected a range response from {url}: {content_range!r}")
                if int(match[1]) != offset or int(match[3]) != expected_size:
                    raise ValueError(f"remote file is not the pinned one: {content_range!r}")
                body = response.read()
            if len(body) != int(match[2]) - offset + 1:
                raise ConnectionError(f"short read from {url}")
            return body
        except (ConnectionError, TimeoutError, http.client.HTTPException, ssl.SSLError):
            if attempt == attempts - 1:
                raise
            time.sleep(backoff * 2**attempt)
    raise RuntimeError("unreachable")


def cached_range(url: str, expected_size: int, cache_dir: Path):
    """A range reader that keeps each chunk on disk, so reruns need no network."""

    def read(offset: int, length: int) -> bytes:
        path = cache_dir / f"{offset}-{length}.bin"
        if path.exists():
            return path.read_bytes()
        body = fetch_range(url, offset, length, expected_size)
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
        partial.write_bytes(body)
        partial.replace(path)
        return body

    return read

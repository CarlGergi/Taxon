"""Fetch product images once and keep small JPEG copies, so every request stays cheap."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from taxon.fetch import open_url

MAX_SIDE = 512
JPEG_QUALITY = 85
MAX_DOWNLOAD_BYTES = 20_000_000
ALLOWED_HOSTS = {"m.media-amazon.com", "images-na.ssl-images-amazon.com"}

Image.MAX_IMAGE_PIXELS = 50_000_000


def to_small_jpeg(data: bytes) -> bytes:
    """Re-encode as an RGB JPEG no larger than MAX_SIDE. Raises if `data` is not an image."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def download(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"not an allowed image host: {parsed.hostname}")
    with open_url(url) as response:
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("image too large")
    return data


class ImageStore:
    def __init__(self, root: Path, fetch: Callable[[str], bytes] = download):
        self.root = root
        self.fetch = fetch
        self.failures_path = root / "failures.json"
        self._failures: dict[str, str] = {}
        if self.failures_path.exists():
            self._failures = json.loads(self.failures_path.read_text(encoding="utf-8"))

    def _path(self, asin: str) -> Path:
        return self.root / f"{asin}.jpg"

    def _ensure(self, product: dict) -> None:
        asin = product["parent_asin"]
        path = self._path(asin)
        if path.exists() or asin in self._failures:
            return
        try:
            small = to_small_jpeg(self.fetch(product["image_url"]))
        except Exception as error:  # noqa: BLE001
            self._failures[asin] = type(error).__name__
            return
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        partial.write_bytes(small)
        partial.replace(path)

    def ensure_all(self, products: Sequence[dict], max_workers: int = 8) -> dict:
        """Fetch what is missing. A product whose image fails is simply labeled from its text."""
        self.root.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(self._ensure, products))
        self.failures_path.write_text(json.dumps(self._failures, sort_keys=True), encoding="utf-8")
        have = sum(self._path(p["parent_asin"]).exists() for p in products)
        return {"images": have, "without_image": len(products) - have}

    def image_for(self, product: dict) -> tuple[str, str] | None:
        path = self._path(product["parent_asin"])
        if not path.exists():
            return None
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

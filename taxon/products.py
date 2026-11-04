"""Sample products across Amazon Reviews 2023 domains.

The metadata files total about 100 GB, so rather than scanning them, each domain is
read through seeded byte ranges: one fixed-size chunk at a random offset inside each
of `n_chunks` equal strata of the file. Complete lines in those chunks form the
candidate pool. Candidates are ordered by a seeded hash of their id, so the outcome
depends only on the pinned file contents and the seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK_BYTES = 64 * 1024
N_CHUNKS = 48
SPLIT_FRACTIONS = (("train", 0.8), ("val", 0.1), ("test", 0.1))

# These details repeat Amazon's own category names and would leak the label.
_LEAKY_DETAILS = {"Best Sellers Rank"}

ReadRange = Callable[[int, int], bytes]


def seeded_hash(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def chunk_offsets(
    domain: str, size: int, seed: int, n_chunks: int = N_CHUNKS, chunk_bytes: int = CHUNK_BYTES
) -> list[int]:
    """One seeded offset per stratum, or just offset 0 when chunks would cover the file."""
    if size <= n_chunks * chunk_bytes:
        return [0]
    rng = random.Random(f"{seed}|{domain}")
    stratum = size // n_chunks
    return [i * stratum + rng.randrange(stratum - chunk_bytes) for i in range(n_chunks)]


def complete_lines(blob: bytes, offset: int, size: int) -> list[bytes]:
    """The whole lines in a chunk: drops a partial first line unless the chunk starts the
    file, and a partial last line unless it reaches the end."""
    lines = blob.split(b"\n")
    if offset > 0:
        lines = lines[1:]
    if offset + len(blob) < size:
        lines = lines[:-1]
    return [line for line in lines if line.strip()]


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _main_image(images: object) -> str | None:
    if not isinstance(images, list):
        return None
    main = [i for i in images if isinstance(i, dict) and i.get("variant") == "MAIN"]
    for image in [*main, *[i for i in images if isinstance(i, dict)]]:
        url = image.get("hi_res") or image.get("large")
        if url:
            return url
    return None


def normalize_record(raw: Mapping, domain: str) -> dict | None:
    """The product fields kept downstream, or None if it lacks an id, title or image."""
    asin = raw.get("parent_asin")
    title = _clean_text(raw.get("title") or "")
    image_url = _main_image(raw.get("images"))
    if not asin or not title or not image_url:
        return None
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
    return {
        "parent_asin": asin,
        "domain": domain,
        "title": title,
        "description": _clean_text(" ".join(map(str, raw.get("description") or []))),
        "features": [_clean_text(f) for f in raw.get("features") or []],
        "details": {k: v for k, v in details.items() if k not in _LEAKY_DETAILS},
        "store": raw.get("store"),
        "price": raw.get("price"),
        "image_url": image_url,
        "main_category": raw.get("main_category"),
        "categories": list(raw.get("categories") or []),
    }


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def read_candidates(
    domain: str, size: int, read_range: ReadRange, seed: int, max_workers: int = 8
) -> list[dict]:
    """Valid products from a domain's chunks, in file order."""
    offsets = chunk_offsets(domain, size, seed)
    length = size if offsets == [0] else CHUNK_BYTES

    def read(offset: int) -> list[bytes]:
        return complete_lines(read_range(offset, length), offset, size)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        chunks = list(pool.map(read, offsets))
    records = []
    for chunk in chunks:
        for line in chunk:
            record = normalize_record(json.loads(line), domain)
            if record is not None:
                records.append(record)
    return records


def quotas(total: int, domains: Iterable[str]) -> dict[str, int]:
    """Split `total` evenly, giving the remainder to the first domains in sorted order."""
    ordered = sorted(domains)
    base, extra = divmod(total, len(ordered))
    return {d: base + (i < extra) for i, d in enumerate(ordered)}


def select(candidates: Mapping[str, list[dict]], total: int, seed: int) -> list[dict]:
    """Take each domain's quota in seeded-hash order, skipping any product already taken
    by id or normalized title. Raises rather than under-sampling a domain."""
    chosen = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for domain, quota in quotas(total, candidates).items():
        ordered = sorted(candidates[domain], key=lambda r: seeded_hash(seed, r["parent_asin"]))
        taken = 0
        for record in ordered:
            title = normalized_title(record["title"])
            if taken == quota:
                break
            if record["parent_asin"] in seen_ids or title in seen_titles:
                continue
            seen_ids.add(record["parent_asin"])
            seen_titles.add(title)
            chosen.append(record)
            taken += 1
        if taken < quota:
            raise ValueError(f"{domain} has {taken} usable products, needs {quota}")
    return chosen


def assign_split(parent_asin: str, seed: int) -> str:
    position = int(seeded_hash(seed, f"split|{parent_asin}")[:12], 16) / 16**12
    cumulative = 0.0
    for name, fraction in SPLIT_FRACTIONS:
        cumulative += fraction
        if position < cumulative:
            return name
    return SPLIT_FRACTIONS[-1][0]


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize(products: list[dict]) -> dict:
    return {
        "products": len(products),
        "per_domain": dict(sorted(Counter(p["domain"] for p in products).items())),
        "with_breadcrumb": sum(bool(p["categories"]) for p in products),
        "with_description": sum(bool(p["description"]) for p in products),
        "with_features": sum(bool(p["features"]) for p in products),
    }

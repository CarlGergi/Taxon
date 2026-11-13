import json
from collections import Counter

import pytest

from taxon.products import (
    CHUNK_BYTES,
    N_CHUNKS,
    assign_split,
    chunk_offsets,
    complete_lines,
    normalize_record,
    normalized_title,
    quotas,
    read_candidates,
    read_jsonl,
    select,
    summarize,
    write_jsonl,
)

SEED = 7


def raw(asin, title="A title", **extra):
    record = {
        "parent_asin": asin,
        "title": title,
        "images": [{"variant": "MAIN", "hi_res": f"https://img/{asin}.jpg", "large": "x"}],
        "categories": ["Toys", "Plush"],
        "description": ["Soft", "and  warm"],
        "features": ["Machine washable"],
        "details": {"Color": "Red", "Best Sellers Rank": {"Toys": 5}},
        "main_category": "Toys",
        "store": "Acme",
        "price": 9.5,
    }
    return record | extra


def test_chunk_offsets_are_deterministic_and_stay_inside_their_strata():
    size = 10**9
    offsets = chunk_offsets("Books", size, SEED)
    assert offsets == chunk_offsets("Books", size, SEED)
    assert len(offsets) == N_CHUNKS
    stratum = size // N_CHUNKS
    for i, offset in enumerate(offsets):
        assert i * stratum <= offset
        assert offset + CHUNK_BYTES <= (i + 1) * stratum


def test_chunk_offsets_differ_by_domain_and_seed():
    size = 10**9
    assert chunk_offsets("Books", size, SEED) != chunk_offsets("Toys", size, SEED)
    assert chunk_offsets("Books", size, SEED) != chunk_offsets("Books", size, SEED + 1)


def test_a_small_file_is_read_whole():
    assert chunk_offsets("Tiny", N_CHUNKS * CHUNK_BYTES, SEED) == [0]


def test_complete_lines_trims_partial_lines_at_both_ends():
    blob = b'rtial"}\n{"a":1}\n{"b":2}\n{"par'
    assert complete_lines(blob, offset=100, size=1000) == [b'{"a":1}', b'{"b":2}']


def test_complete_lines_keeps_the_first_line_at_the_start_of_a_file():
    assert complete_lines(b'{"a":1}\n{"pa', offset=0, size=1000) == [b'{"a":1}']


def test_complete_lines_keeps_the_last_line_at_the_end_of_a_file():
    blob = b'x"}\n{"a":1}'
    assert complete_lines(blob, offset=1000 - len(blob), size=1000) == [b'{"a":1}']


def test_normalize_keeps_fields_and_removes_the_rank_leak():
    record = normalize_record(raw("A1"), "Toys")
    assert record["image_url"] == "https://img/A1.jpg"
    assert record["description"] == "Soft and warm"
    assert record["details"] == {"Color": "Red"}
    assert "Best Sellers Rank" not in json.dumps(record)


@pytest.mark.parametrize(
    "broken",
    [
        {"parent_asin": None},
        {"title": ""},
        {"title": None},
        {"title": "   "},
        {"images": []},
        {"images": [{"variant": "MAIN", "hi_res": None, "large": None}]},
    ],
)
def test_normalize_rejects_products_without_id_title_or_image(broken):
    assert normalize_record(raw("A1") | broken, "Toys") is None


def test_normalize_falls_back_to_large_then_any_variant():
    images = [{"variant": "PT01", "hi_res": None, "large": "https://img/l.jpg"}]
    assert normalize_record(raw("A1", images=images), "Toys")["image_url"] == "https://img/l.jpg"


def test_normalize_tolerates_missing_optional_fields():
    record = normalize_record(
        {"parent_asin": "A", "title": "T", "images": [{"large": "u"}], "details": None}, "Toys"
    )
    assert record["description"] == ""
    assert record["features"] == []
    assert record["details"] == {}
    assert record["categories"] == []


def _file_of(records):
    return "\n".join(json.dumps(r) for r in records).encode() + b"\n"


def test_read_candidates_on_a_small_file_reads_each_record_once():
    blob = _file_of([raw(f"A{i}") for i in range(50)])
    candidates = read_candidates("Toys", len(blob), lambda o, n: blob[o : o + n], SEED)
    assert [c["parent_asin"] for c in candidates] == [f"A{i}" for i in range(50)]


def test_read_candidates_on_a_large_file_requests_chunks_and_parses_them():
    line = json.dumps(raw("AX")).encode() + b"\n"
    blob = line * ((N_CHUNKS * CHUNK_BYTES * 2) // len(line))
    requested = []

    def read_range(offset, length):
        requested.append((offset, length))
        return blob[offset : offset + length]

    candidates = read_candidates("Toys", len(blob), read_range, SEED)
    assert len(requested) == N_CHUNKS
    assert all(length == CHUNK_BYTES for _, length in requested)
    assert candidates and all(c["parent_asin"] == "AX" for c in candidates)


def test_read_candidates_skips_invalid_records():
    blob = _file_of([raw("A1"), raw("A2", title=""), raw("A3", images=[])])
    candidates = read_candidates("Toys", len(blob), lambda o, n: blob[o : o + n], SEED)
    assert [c["parent_asin"] for c in candidates] == ["A1"]


def test_quotas_sum_to_total_and_differ_by_at_most_one():
    result = quotas(10_000, [f"d{i}" for i in range(28)])
    assert sum(result.values()) == 10_000
    assert max(result.values()) - min(result.values()) == 1


def _candidates(domain, n, prefix=None):
    prefix = prefix or domain
    return [
        normalize_record(raw(f"{prefix}{i}", title=f"{prefix} product {i}"), domain)
        for i in range(n)
    ]


def test_select_meets_every_quota_exactly():
    pool = {"a": _candidates("a", 50), "b": _candidates("b", 50), "c": _candidates("c", 50)}
    chosen = select(pool, 10, SEED)
    assert Counter(r["domain"] for r in chosen) == {"a": 4, "b": 3, "c": 3}


def test_select_is_deterministic_and_independent_of_input_order():
    pool = {"a": _candidates("a", 50)}
    flipped = {"a": list(reversed(_candidates("a", 50)))}
    assert select(pool, 10, SEED) == select(flipped, 10, SEED)


def test_select_depends_on_the_seed():
    pool = {"a": _candidates("a", 50)}
    assert select(pool, 10, SEED) != select(pool, 10, SEED + 1)


def test_select_removes_ids_shared_across_domains():
    shared = _candidates("a", 5, prefix="s")
    pool = {"a": shared, "b": [r | {"domain": "b"} for r in shared] + _candidates("b", 5)}
    chosen = select(pool, 10, SEED)
    ids = [r["parent_asin"] for r in chosen]
    assert len(ids) == len(set(ids))
    assert Counter(r["domain"] for r in chosen) == {"a": 5, "b": 5}


def test_select_removes_near_identical_titles():
    records = [normalize_record(raw(f"A{i}", title="Blue-Mug!"), "a") for i in range(3)]
    other = [
        normalize_record(raw("Z", title="blue mug"), "a"),
        normalize_record(raw("Y", title="Red Mug"), "a"),
    ]
    chosen = select({"a": records + other}, 2, SEED)
    assert {normalized_title(r["title"]) for r in chosen} == {"blue mug", "red mug"}


def test_select_raises_rather_than_under_sampling():
    with pytest.raises(ValueError, match="a has 3 usable products, needs 10"):
        select({"a": _candidates("a", 3)}, 10, SEED)


def test_assign_split_is_stable_and_near_the_target_fractions():
    assert assign_split("A1", SEED) == assign_split("A1", SEED)
    counts = Counter(assign_split(f"A{i}", SEED) for i in range(20_000))
    assert counts["train"] / 20_000 == pytest.approx(0.8, abs=0.02)
    assert counts["val"] / 20_000 == pytest.approx(0.1, abs=0.02)
    assert counts["test"] / 20_000 == pytest.approx(0.1, abs=0.02)


def test_jsonl_round_trip_keeps_non_ascii_text(tmp_path):
    rows = [{"title": "Piñata", "n": 1}]
    path = tmp_path / "sub" / "rows.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert "Piñata" in path.read_text(encoding="utf-8")


def test_summarize_counts_coverage():
    products = _candidates("a", 2) + [normalize_record(raw("Z", categories=[]), "b")]
    summary = summarize(products)
    assert summary["products"] == 3
    assert summary["per_domain"] == {"a": 2, "b": 1}
    assert summary["with_breadcrumb"] == 2

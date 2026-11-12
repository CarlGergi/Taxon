"""The quantization study's measuring, and the trie guide that every precision decodes with."""

import pytest

from taxon import quant
from taxon.rl.decoding import TrieGuide
from taxon.trie import TokenTrie

PATHS = ["Home", "Home > Kitchen", "Apparel > Bags"]
CODES = {"Home": [1], "Home > Kitchen": [1, 2], "Apparel > Bags": [3, 4]}
OPENING = [10, 11]
CLOSERS = [99]


def guide():
    trie = TokenTrie.build(PATHS, lambda text: CODES[text])
    return TrieGuide(OPENING, trie, CLOSERS)


# The guide


def test_the_guide_forces_the_opening_then_follows_the_trie():
    g = guide()
    assert g.allowed() == [10]
    g.advance(10)
    assert g.allowed() == [11]
    g.advance(11)
    assert sorted(g.allowed()) == [1, 3]


def test_a_path_may_close_only_where_one_ends():
    g = guide()
    for token in (10, 11, 3):
        g.advance(token)
    assert 99 not in g.allowed()
    g.advance(4)
    assert g.allowed() == [99]


def test_a_shorter_path_can_close_or_go_deeper():
    g = guide()
    for token in (10, 11, 1):
        g.advance(token)
    assert sorted(g.allowed()) == [2, 99]


def test_after_closing_the_reply_is_free():
    g = guide()
    for token in (10, 11, 1, 99):
        g.advance(token)
    assert g.allowed() is None


# Measuring


def test_latency_summary_gives_percentiles_throughput_and_hours():
    report = quant.latency_summary([1.0] * 19 + [3.0])
    assert report["replies"] == 20
    assert report["latency_p50_s"] == 1.0
    assert report["latency_p95_s"] > 1.0
    assert report["products_per_hour"] == pytest.approx(3_600 / 1.1)
    assert report["hours_per_million_products"] == pytest.approx(1.1 * 1_000_000 / 3_600)
    low, high = report["latency_p95_s_ci95"]
    assert low <= report["latency_p95_s"] <= high
    low, high = report["products_per_hour_ci95"]
    assert low <= report["products_per_hour"] <= high
    low, high = report["hours_per_million_products_ci95"]
    assert low <= report["hours_per_million_products"] <= high


def test_quantized_versions_are_compared_with_full_precision_product_by_product():
    golds = ["A", "B", "C", "D"]
    precisions = {
        "bf16": {"predictions": ["A", "B", "C", "x"]},
        "int8": {"predictions": ["A", "B", "C", "x"]},
        "int4": {"predictions": ["A", "x", "x", "x"]},
    }
    report = quant.against_full_precision(golds, precisions)
    assert set(report) == {"int8", "int4"}
    assert report["int8"]["difference"] == 0.0
    assert report["int4"]["difference"] == pytest.approx(-0.5)


def test_measuring_scores_each_level_and_skips_the_warm_up_in_timing():
    replies = {"a": '{"category": "Home > Kitchen"}', "b": '{"category": "Apparel"}'}
    rows = [{"id": "a", "image_path": None}, {"id": "b", "image_path": "/x.jpg"}]
    seen = []

    def generate(prompt, image, g):
        seen.append((prompt, image))
        return replies[prompt]

    ticks = iter([0.0, 2.0, 10.0, 14.0])
    report = quant.measure_rows(
        rows,
        ["Home > Kitchen", "Apparel > Bags"],
        lambda row: row["id"],
        generate,
        guide,
        PATHS,
        clock=lambda: next(ticks),
    )
    assert seen[0] == ("a", None) and len(seen) == 3
    assert report["accuracy"]["exact_leaf_accuracy"] == 0.5
    assert report["accuracy"]["levels"]
    assert report["latency_p50_s"] == 3.0
    assert report["predictions"] == ["Home > Kitchen", "Apparel"]
    assert report["accuracy"]["exact_leaf_accuracy_ci95"] is not None


def test_every_precision_is_named_with_its_bits():
    assert quant.PRECISIONS == {"bf16": None, "int8": 8, "int4": 4}


# The MLX processor, where MLX is installed


def test_the_mlx_processor_masks_all_but_allowed_tokens_and_tracks_the_reply():
    mx = pytest.importorskip("mlx.core")
    processor = quant.MlxTrie(guide())
    logits = mx.zeros((1, 100))
    first = processor(mx.array([5, 6]), logits)
    assert float(first[0, 10]) == 0 and float(first[0, 11]) == float("-inf")
    second = processor(mx.array([5, 6, 10]), logits)
    assert float(second[0, 11]) == 0 and float(second[0, 10]) == float("-inf")
    processor(mx.array([5, 6, 10, 11]), logits)
    free = quant.MlxTrie(guide())
    free.guide.phase = "free"
    assert free(mx.array([1]), logits) is logits

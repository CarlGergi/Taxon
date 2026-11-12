from collections import Counter

import pytest

from taxon.gold.pool import (
    FIRST_TARGET,
    GOLD_POOL_SIZE,
    GoldOverlap,
    assert_disjoint,
    gold_queue,
    stratified_order,
    training_products,
)


def products(domains=28, per_domain=357):
    return [
        {"parent_asin": f"D{d}-{i}", "domain": f"domain{d}"}
        for d in range(domains)
        for i in range(per_domain)
    ]


def test_the_queue_has_the_right_size_and_no_repeats():
    queue = gold_queue(products(), seed=1)
    assert len(queue) == GOLD_POOL_SIZE
    assert len(set(queue)) == GOLD_POOL_SIZE


def test_the_queue_is_deterministic_and_seed_dependent():
    assert gold_queue(products(), seed=1) == gold_queue(products(), seed=1)
    assert gold_queue(products(), seed=1) != gold_queue(products(), seed=2)


def test_the_queue_ignores_input_order():
    assert gold_queue(products(), seed=1) == gold_queue(list(reversed(products())), seed=1)


def test_growing_the_set_keeps_earlier_items():
    assert (
        gold_queue(products(), seed=1, size=FIRST_TARGET)
        == gold_queue(products(), seed=1)[:FIRST_TARGET]
    )


def test_the_first_target_is_balanced_across_domains():
    rows = {p["parent_asin"]: p["domain"] for p in products()}
    counts = Counter(rows[a] for a in gold_queue(products(), seed=1)[:FIRST_TARGET])
    assert len(counts) == 28
    assert max(counts.values()) - min(counts.values()) <= 1


def test_stratified_order_keeps_every_product_with_uneven_domains():
    rows = products(domains=2, per_domain=3) + [{"parent_asin": "X", "domain": "tiny"}]
    assert len(stratified_order(rows, seed=1, salt="s")) == 7


def test_training_products_exclude_the_pool():
    rows = products(domains=2, per_domain=5)
    queue = gold_queue(rows, seed=1, size=3)
    train = training_products(rows, queue)
    assert len(train) == 7
    assert not ({p["parent_asin"] for p in train} & set(queue))


def test_the_gold_set_never_overlaps_training_data():
    rows = products()
    queue = gold_queue(rows, seed=1)
    assert_disjoint((p["parent_asin"] for p in training_products(rows, queue)), queue)


def test_an_overlap_is_refused():
    with pytest.raises(GoldOverlap, match="1 gold products"):
        assert_disjoint(["a", "b"], ["b", "c"])

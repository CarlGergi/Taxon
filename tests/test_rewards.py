import json

import pytest

from taxon.rl.rewards import (
    RewardWeights,
    allowed_values,
    depth_penalty,
    field_compliance,
    path_match,
    score,
    valid_node,
)

TOTES = "Apparel > Bags > Totes"
BAGS = "Apparel > Bags"
KNIVES = "Home > Kitchen > Knives"


def reply(**fields):
    return json.dumps(fields)


@pytest.mark.parametrize(
    "predicted,gold,expected",
    [
        (TOTES, TOTES, 1.0),
        (BAGS, TOTES, 3 / 6),
        ("Apparel", TOTES, 1 / 6),
        ("Apparel > Bag Straps", TOTES, 1 / 6),
        (KNIVES, TOTES, 0.0),
        (TOTES, BAGS, 3 / 6),
        (None, TOTES, 0.0),
    ],
)
def test_path_match_weights_deep_levels_most(predicted, gold, expected):
    assert path_match(predicted, gold) == pytest.approx(expected)


def test_a_shared_name_at_another_level_earns_nothing():
    assert path_match("Home > Bags", BAGS) == 0.0


def test_the_penalty_applies_only_above_the_gold_depth():
    assert depth_penalty("Apparel", TOTES) == pytest.approx(-0.5 * 2 / 3)
    assert depth_penalty(BAGS, TOTES) == pytest.approx(-0.5 / 3)
    assert depth_penalty(TOTES, TOTES) == 0.0
    assert depth_penalty(TOTES, BAGS) == 0.0
    assert depth_penalty(None, TOTES) == 0.0


def category_reward(predicted, gold, taxonomy):
    weights = RewardWeights()
    parts = score(reply(category=predicted), ["category"], {"category": gold}, taxonomy)
    return weights.path_match * parts["path_match"] + weights.depth_penalty * parts["depth_penalty"]


def test_stopping_at_the_vertical_earns_nothing(taxonomy):
    assert category_reward("Apparel", TOTES, taxonomy) <= 0.0


def test_each_correct_level_deeper_earns_more(taxonomy):
    rewards = [category_reward(p, TOTES, taxonomy) for p in ("Apparel", BAGS, TOTES)]
    assert rewards == sorted(rewards)
    assert len(set(rewards)) == 3


def test_compliance_needs_exactly_the_requested_fields():
    assert field_compliance(reply(category=TOTES), ["category"]) == 1.0
    assert field_compliance(reply(category=TOTES, color="Red"), ["category"]) == 0.0
    assert field_compliance(reply(color=None), ["category", "color"]) == 0.0
    assert field_compliance("not json", ["category"]) == 0.0


def test_a_valid_node_must_be_a_real_path(taxonomy):
    assert valid_node(reply(category=TOTES), ["category"], taxonomy) == 1.0
    assert valid_node(reply(category="Apparel > Totes"), ["category"], taxonomy) == 0.0
    assert valid_node(reply(category=5), ["category"], taxonomy) == 0.0
    assert valid_node(reply(color="Red"), ["color"], taxonomy) == 0.0


def test_attribute_values_must_be_allowed_for_the_category(taxonomy):
    text = reply(category=TOTES, color="Red", material="Velvet")
    assert allowed_values(text, ["category", "color", "material"], taxonomy, TOTES) == 0.5


def test_null_attribute_answers_are_fine(taxonomy):
    assert allowed_values(reply(color=None), ["color"], taxonomy, TOTES) == 1.0
    assert allowed_values("junk", ["color"], taxonomy, TOTES) == 0.0


def test_values_are_checked_against_the_predicted_category_when_valid(taxonomy):
    # Knives allows no color, so a color given for a knife is not allowed.
    text = reply(category=KNIVES, color="Red")
    assert allowed_values(text, ["category", "color"], taxonomy, TOTES) == 0.0
    text = reply(category="Not > Real", color="Red")
    assert allowed_values(text, ["category", "color"], taxonomy, TOTES) == 1.0


def test_an_invalid_category_earns_no_path_credit(taxonomy):
    parts = score(
        reply(category="Apparel > Bags > Nope"), ["category"], {"category": TOTES}, taxonomy
    )
    assert parts["valid_node"] == 0.0
    assert parts["path_match"] == 0.0
    assert parts["depth_penalty"] == 0.0


def test_category_rewards_wait_until_category_is_asked(taxonomy):
    parts = score(reply(color="Red"), ["color"], {"category": TOTES}, taxonomy)
    assert parts == {
        "compliance": 1.0,
        "valid_node": 0.0,
        "path_match": 0.0,
        "allowed_values": 1.0,
        "depth_penalty": 0.0,
    }


def test_a_perfect_reply_earns_every_reward(taxonomy):
    parts = score(
        reply(category=TOTES, color="Red"), ["category", "color"], {"category": TOTES}, taxonomy
    )
    assert parts == {
        "compliance": 1.0,
        "valid_node": 1.0,
        "path_match": 1.0,
        "allowed_values": 1.0,
        "depth_penalty": 0.0,
    }


def test_weights_come_in_the_order_of_the_rewards():
    assert RewardWeights().as_list() == [1.0, 1.0, 2.0, 1.0, 1.0]

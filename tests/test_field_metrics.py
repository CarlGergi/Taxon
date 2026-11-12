import pytest

from eval.fields import attribute_accuracy, field_compliance, field_invariance, parse_fields


def row(product, requested, output):
    return {"product": product, "requested": requested, "output": output}


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"color": "Red"}', {"color": "Red"}),
        ('  {"color": null}\n', {"color": None}),
        ("not json", None),
        ('["color"]', None),
        ('{"color": "Red"} and more', None),
        ("", None),
    ],
)
def test_parse_fields_accepts_only_a_json_object(text, expected):
    assert parse_fields(text) == expected


def test_compliance_means_exactly_the_requested_fields():
    report = field_compliance(
        [
            row("A", ["category"], '{"category": "X"}'),
            row("B", ["category"], '{"category": "X", "color": "Red"}'),
            row("C", ["category", "color"], '{"category": "X"}'),
            row("D", ["category"], "not json"),
        ]
    )
    assert {k: v for k, v in report.items() if not k.endswith("_ci95")} == {
        "requests": 4,
        "compliance_rate": 0.25,
        "parse_rate": 0.75,
        "extra_field_rate": 0.25,
        "missing_field_rate": 0.25,
    }
    low, high = report["compliance_rate_ci95"]
    assert low < 0.25 < high


def test_a_null_answer_still_complies():
    report = field_compliance([row("A", ["color"], '{"color": null}')])
    assert report["compliance_rate"] == 1.0


def test_field_order_does_not_affect_compliance():
    report = field_compliance([row("A", ["a", "b"], '{"b": 1, "a": 2}')])
    assert report["compliance_rate"] == 1.0


def test_compliance_on_nothing_is_undefined_rather_than_perfect():
    assert field_compliance([])["compliance_rate"] is None


def test_invariance_compares_a_field_across_schemas():
    report = field_invariance(
        [
            row("A", ["category"], '{"category": "X"}'),
            row("A", ["category", "color"], '{"category": "X", "color": "Red"}'),
            row("A", ["color"], '{"color": "Blue"}'),
        ]
    )
    assert report["groups"] == 2
    assert report["invariance_rate"] == 0.5
    assert report["per_field"]["category"]["invariance_rate"] == 1.0
    assert report["per_field"]["color"]["invariance_rate"] == 0.0


def test_a_field_asked_once_says_nothing_about_invariance():
    report = field_invariance([row("A", ["category"], '{"category": "X"}')])
    assert report == {
        "groups": 0,
        "invariance_rate": None,
        "invariance_rate_ci95": None,
        "per_field": {},
    }


def test_products_are_compared_only_with_themselves():
    report = field_invariance(
        [row("A", ["color"], '{"color": "Red"}'), row("B", ["color"], '{"color": "Blue"}')]
    )
    assert report["groups"] == 0


def test_a_dropped_field_counts_as_a_changed_answer():
    report = field_invariance(
        [row("A", ["color"], '{"color": "Red"}'), row("A", ["color", "x"], '{"x": 1}')]
    )
    assert report["invariance_rate"] == 0.0


def test_unreadable_replies_are_never_consistent_with_each_other():
    report = field_invariance([row("A", ["color"], "junk"), row("A", ["color", "x"], "junk")])
    assert report["invariance_rate"] == 0.0


def test_the_same_null_answer_twice_is_consistent():
    report = field_invariance(
        [row("A", ["color"], '{"color": null}'), row("A", ["color", "x"], '{"color": null}')]
    )
    assert report["invariance_rate"] == 1.0


def test_attribute_accuracy_counts_each_labeled_value():
    report = attribute_accuracy(
        [{"color": "Red", "material": "Cotton"}, {"color": "Blue"}],
        [{"color": "Red", "material": "Wool", "extra": "x"}, None],
    )
    assert report["values"] == 3
    assert report["accuracy"] == pytest.approx(1 / 3)
    color = report["per_attribute"]["color"]
    assert (color["values"], color["accuracy"]) == (2, 0.5)
    assert color["accuracy_ci95"][0] < 0.5 < color["accuracy_ci95"][1]
    low, high = report["accuracy_ci95"]
    assert 0.0 <= low <= 1 / 3 <= high <= 1.0


def test_attribute_accuracy_resamples_products_not_values():
    # Two products: one with every value right, one with every value wrong. Resampling values
    # would give a narrow interval around 0.5; resampling products can draw both the same.
    golds = [{"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "2", "c": "3"}]
    report = attribute_accuracy(golds, [golds[0], {}], seed=3)
    assert report["accuracy_ci95"] == [0.0, 1.0]


def test_attribute_accuracy_with_no_labels_is_undefined():
    assert attribute_accuracy([{}], [{"color": "Red"}])["accuracy"] is None


def test_the_paired_attribute_difference_resamples_products():
    from eval.fields import paired_attribute_difference

    golds = [{"color": "Red", "size": "M"}, {"color": "Blue"}]
    report = paired_attribute_difference(golds, [golds[0], golds[1]], [{}, golds[1]], seed=1)
    assert report["values"] == 3
    assert report["difference"] == pytest.approx(2 / 3)
    low, high = report["ci95"]
    assert 0.0 <= low <= 2 / 3 <= high <= 1.0

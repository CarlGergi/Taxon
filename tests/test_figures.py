"""The README's figures are drawn from saved results."""

import json

from PIL import Image

from taxon import figures


def catalog(mean, low, high, runs):
    return {
        "over_tasks": {"success": {"mean": mean, "ci95": [low, high]}},
        "runs": [{"success": r} for r in runs],
    }


REPORT = {
    "catalogs": {
        "raw": catalog(0.4, 0.3, 0.5, [0.38, 0.4, 0.42]),
        "taxon": catalog(0.5, 0.4, 0.6, [0.5, 0.49, 0.51]),
    }
}


def test_success_rows_follow_the_catalog_order_and_skip_missing_ones():
    rows = figures.success_rows(REPORT)
    assert [r["catalog"] for r in rows] == ["raw", "taxon"]
    assert rows[1]["ci95"] == [0.4, 0.6]
    assert rows[0]["runs"] == [0.38, 0.4, 0.42]


def test_the_task_success_chart_is_a_white_png(tmp_path):
    path = figures.task_success(REPORT, tmp_path / "out" / "task-success.png")
    with Image.open(path) as image:
        assert image.size == (1400, 720)
        assert image.convert("RGB").getpixel((2, 2)) == (255, 255, 255)


def test_draw_all_draws_only_what_has_results(tmp_path):
    assert figures.draw_all(tmp_path, tmp_path / "images") == []
    (tmp_path / "catalogbench.json").write_text(json.dumps(REPORT))
    assert [p.name for p in figures.draw_all(tmp_path, tmp_path / "images")] == ["task-success.png"]

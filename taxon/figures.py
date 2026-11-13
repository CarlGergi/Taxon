"""The README's figures, drawn from the saved results so every image matches a number in results/.

One style for every figure: white background, one blue, gray ink, no baked-in title (the README's
caption names the figure). Run `make figures`; `make reproduce` redraws them when the results exist.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from taxon.pins import RESULTS_DIR, ROOT

IMAGES_DIR = ROOT / "docs" / "images"
BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
CATALOG_NAMES = {
    "raw": "Raw catalog",
    "sonnet": "Enriched by Sonnet",
    "taxon": "Enriched by Taxon",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def success_rows(report: Mapping) -> list[dict]:
    """Each catalog's task success: the mean over tasks, its 95% interval, and each run's rate."""
    rows = []
    for name in CATALOG_NAMES:
        catalog = report["catalogs"].get(name)
        if catalog is None:
            continue
        over = catalog["over_tasks"]["success"]
        rows.append(
            {
                "catalog": name,
                "label": CATALOG_NAMES[name],
                "mean": over["mean"],
                "ci95": over["ci95"],
                "runs": [run["success"] for run in catalog["runs"]],
            }
        )
    return rows


def task_success(report: Mapping, path: Path) -> Path:
    """Bars of task success per catalog with 95% intervals; a dot per run."""
    plt = _style()
    rows = success_rows(report)
    figure, axis = plt.subplots(figsize=(7, 3.6), dpi=200)
    xs = range(len(rows))
    means = [100 * r["mean"] for r in rows]
    axis.bar(xs, means, width=0.5, color=BLUE, zorder=2)
    low = [m - 100 * r["ci95"][0] for m, r in zip(means, rows, strict=True)]
    high = [100 * r["ci95"][1] - m for m, r in zip(means, rows, strict=True)]
    axis.errorbar(xs, means, yerr=[low, high], fmt="none", ecolor=INK, capsize=5, lw=1.2, zorder=3)
    for x, row in zip(xs, rows, strict=True):
        runs = [100 * v for v in row["runs"]]
        offsets = [x + 0.12 * (i - (len(runs) - 1) / 2) for i in range(len(runs))]
        axis.scatter(offsets, runs, s=18, color="white", edgecolor=INK, lw=1, zorder=4)
        axis.text(x + 0.3, 100 * row["mean"], f"{100 * row['mean']:.1f}%", va="center", color=INK)
    axis.set_xticks(list(xs), [r["label"] for r in rows])
    axis.set_ylabel("Tasks solved (%)")
    axis.set_ylim(0, 100)
    axis.set_xlim(-0.5, len(rows) - 0.5 + 0.3)
    axis.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return path


def draw_all(results: Path = RESULTS_DIR, images: Path = IMAGES_DIR) -> list[Path]:
    """Every figure whose results exist."""
    drawn = []
    bench = results / "catalogbench.json"
    if bench.exists():
        report = json.loads(bench.read_text(encoding="utf-8"))
        drawn.append(task_success(report, images / "task-success.png"))
    return drawn


def main() -> int:
    drawn = draw_all()
    for path in drawn:
        print(f"drew {path.relative_to(ROOT)}")
    if not drawn:
        print("No results to draw yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

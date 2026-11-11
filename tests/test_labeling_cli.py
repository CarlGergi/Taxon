import json

import pytest

from taxon.gold.pool import GoldOverlap
from taxon.labeling import cli
from taxon.labeling.budget import BudgetExceeded
from taxon.labeling.env import Secrets
from taxon.labeling.pipeline import ANNOTATORS, Labeler
from taxon.products import read_jsonl


def product(asin, domain="d"):
    return {
        "parent_asin": asin,
        "domain": domain,
        "title": f"Item {asin}",
        "features": [],
        "description": "",
        "details": {},
        "image_url": "https://m.media-amazon.com/x.jpg",
    }


@pytest.fixture
def inputs(monkeypatch, taxonomy):
    products = [product(f"A{i}", f"d{i % 3}") for i in range(30)]
    gold = ["A0", "A1", "A2"]
    shortlists = {p["parent_asin"]: taxonomy.paths()[:5] for p in products}
    monkeypatch.setattr(cli, "_load_inputs", lambda: (taxonomy, products, gold, shortlists))
    monkeypatch.setattr(cli, "FIRST_RUN_PRODUCTS", 10)
    monkeypatch.setattr(cli, "load_secrets", lambda: Secrets({}))
    return products, gold


def run(tmp_path, *argv):
    return cli.main(list(argv), root=tmp_path)


def test_gold_products_are_never_in_the_labeling_order(inputs):
    products, gold = inputs
    order = cli.labeling_order(products, gold)
    assert len(order) == 27
    assert not ({p["parent_asin"] for p in order} & set(gold))


def test_each_stage_extends_the_one_before(inputs):
    products, gold = inputs
    order = [p["parent_asin"] for p in cli.labeling_order(products, gold)]
    assert order == [p["parent_asin"] for p in cli.labeling_order(list(reversed(products)), gold)]


def test_a_fake_dry_run_prints_an_estimate_per_vendor(tmp_path, inputs, capsys):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    out = capsys.readouterr().out
    for vendor in ("anthropic", "openai"):
        assert f"{vendor}: $" in out
    assert "google" not in out
    assert "cap $20.00" in out
    assert "measured on this sample" in out
    assert "Sent to the arbitrator:" in out
    assert "First run, all vendors (" in out
    report = json.loads((tmp_path / "results" / "labeling_dry_run.json").read_text())
    assert report["sample"] == 6
    assert report["fake"] is True
    assert set(report["cost"]["openai"]) >= {
        "first_run",
        "first_run_and_gold_pool",
        "all_training",
        "cap",
        "spent",
    }
    assert "unanimous" in report["agreement"]["agreement"]


def test_a_run_refuses_without_a_dry_run(tmp_path, inputs):
    with pytest.raises(SystemExit, match="dry run first"):
        run(tmp_path, "run", "--first", "--fake")


def test_a_run_refuses_when_the_estimate_passes_a_cap(tmp_path, inputs):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    path = tmp_path / "results" / "labeling_dry_run.json"
    report = json.loads(path.read_text())
    report["cost"]["openai"]["first_run"]["fits_cap"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="passes the cap for: openai"):
        run(tmp_path, "run", "--first", "--fake")


def test_the_gold_pool_stage_also_needs_a_dry_run(tmp_path, inputs):
    with pytest.raises(SystemExit, match="dry run first"):
        run(tmp_path, "run", "--gold-pool", "--fake")


def test_the_first_stage_then_the_rest(tmp_path, inputs, capsys):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--first", "--fake") == 0
    assert len(read_jsonl(tmp_path / "labels.jsonl")) == 10
    assert (tmp_path / "results" / "labeling_10.json").exists()
    assert "Spent by vendor" in capsys.readouterr().out
    assert run(tmp_path, "run", "--rest", "--fake") == 0
    assert len(read_jsonl(tmp_path / "labels.jsonl")) == 27
    assert (tmp_path / "results" / "labeling_27.json").exists()


def test_exported_training_labels_exclude_gold_and_abstentions(tmp_path, inputs):
    products, gold = inputs
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--rest", "--fake") == 0
    rows = read_jsonl(tmp_path / "training_labels.jsonl")
    assert rows
    assert all(row["path"] for row in rows)
    assert not ({row["parent_asin"] for row in rows} & set(gold))


def test_export_refuses_a_gold_product(tmp_path):
    rows = [{"parent_asin": "G1", "path": "Home", "source": "unanimous"}]
    with pytest.raises(GoldOverlap):
        cli.export_training_labels(rows, ["G1"], tmp_path / "t.jsonl")
    assert not (tmp_path / "t.jsonl").exists()


def test_a_cap_stop_returns_its_own_exit_code_and_message(tmp_path, inputs, monkeypatch, capsys):
    def stop(self, products, shortlists):
        raise BudgetExceeded("openai", 69.5, 70.0, 12)

    monkeypatch.setattr(Labeler, "label", stop)
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == cli.EXIT_BUDGET
    out = capsys.readouterr().out
    assert "STOPPED" in out
    assert "$70.00" in out
    assert "12 requests" in out


def test_real_mode_without_keys_names_the_key_and_makes_no_calls(tmp_path, inputs, capsys):
    assert run(tmp_path, "dry-run", "--n", "6") == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert "_API_KEY" in out
    assert "is not set" in out


def test_fake_mode_never_builds_real_clients(tmp_path, inputs, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("real clients built in fake mode")

    monkeypatch.setattr(cli, "build_clients", forbidden)
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0


def test_a_fake_run_keeps_to_its_own_directory(monkeypatch, inputs, tmp_path):
    seen = []
    real = cli.Workspace

    def spy(root):
        seen.append(root)
        return real(tmp_path)

    monkeypatch.setattr(cli, "Workspace", spy)
    cli.main(["dry-run", "--n", "6", "--fake"])
    assert seen[0].name == "labeling_fake"


def test_nothing_written_contains_a_key(tmp_path, inputs, monkeypatch, capsys):
    secret = "sk-super-secret-value"
    monkeypatch.setattr(cli, "load_secrets", lambda: Secrets({"OPENAI_API_KEY": secret}))
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert secret not in capsys.readouterr().out
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")


def test_a_rerun_of_the_same_stage_spends_nothing_more(tmp_path, inputs):
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--first", "--fake") == 0
    before = (tmp_path / "ledger.jsonl").read_text()
    assert run(tmp_path, "run", "--first", "--fake") == 0
    assert (tmp_path / "ledger.jsonl").read_text() == before


def test_the_annotators_are_used_in_the_cli_labeler(tmp_path, taxonomy):
    labeler, extractor, runner, tracer = cli._build_labeler(
        taxonomy, cli.Workspace(tmp_path), True, Secrets({})
    )
    assert labeler.annotators == ANNOTATORS
    assert extractor.runner is runner
    assert tracer is None
    assert runner.budget.caps == cli.CAPS


def test_the_gold_pool_stage_keeps_its_rows_apart_from_training(tmp_path, inputs, capsys):
    products, gold = inputs
    assert run(tmp_path, "dry-run", "--n", "6", "--fake") == 0
    assert run(tmp_path, "run", "--rest", "--fake") == 0
    before = (tmp_path / "training_labels.jsonl").read_text()
    labels_before = (tmp_path / "labels.jsonl").read_text()
    assert run(tmp_path, "run", "--gold-pool", "--fake") == 0
    rows = read_jsonl(tmp_path / "gold_pool_labels.jsonl")
    assert {row["parent_asin"] for row in rows} == set(gold)
    assert (tmp_path / "training_labels.jsonl").read_text() == before
    assert (tmp_path / "labels.jsonl").read_text() == labels_before
    assert "kept apart" in capsys.readouterr().out

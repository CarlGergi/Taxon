import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from taxon import backup
from taxon.backup import create_backup, locally_excluded
from taxon.pins import ROOT


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "gold").mkdir(parents=True)
    (root / "data" / "gold" / "labels.jsonl").write_text('{"asin": "A"}\n')
    responses = root / "data" / "labeling" / "responses" / "ab"
    responses.mkdir(parents=True)
    (responses / "abc.json").write_text("{}")
    (root / "data" / "labeling" / "ledger.jsonl").write_text("{}\n")
    return root


def sources(root):
    labeling = root / "data" / "labeling"
    return [root / "data" / "gold", labeling / "responses", labeling / "ledger.jsonl"]


def names(path):
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def test_the_zip_holds_gold_labels_responses_and_the_ledger(repo, tmp_path):
    target = create_backup(tmp_path / "out", sources(repo), repo)
    assert names(target) == {
        "data/gold/labels.jsonl",
        "data/labeling/responses/ab/abc.json",
        "data/labeling/ledger.jsonl",
    }
    with zipfile.ZipFile(target) as archive:
        assert archive.read("data/gold/labels.jsonl") == b'{"asin": "A"}\n'


def test_the_name_is_dated_so_backups_do_not_overwrite_each_other(repo, tmp_path):
    first = create_backup(tmp_path / "out", sources(repo), repo, datetime(2026, 11, 3, 9, 0, 0))
    second = create_backup(tmp_path / "out", sources(repo), repo, datetime(2026, 11, 4, 9, 0, 0))
    assert first.name == "taxon-backup-20261103-090000.zip"
    assert first != second
    assert first.exists() and second.exists()


def test_a_destination_inside_the_repository_is_refused(repo):
    with pytest.raises(ValueError, match="outside the repository"):
        create_backup(repo / "backups", sources(repo), repo)
    assert not (repo / "backups").exists()


def test_a_destination_that_resolves_into_the_repository_is_refused(repo, tmp_path):
    link = tmp_path / "sneaky"
    link.symlink_to(repo)
    with pytest.raises(ValueError, match="outside the repository"):
        create_backup(link / "backups", sources(repo), repo)


def test_nothing_to_save_writes_nothing(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert create_backup(tmp_path / "out", [root / "data" / "gold"], root) is None
    assert not (tmp_path / "out").exists()


def test_keys_and_partial_files_are_never_included(repo, tmp_path):
    (repo / "data" / "gold" / ".env").write_text("OPENAI_API_KEY=secret")
    (repo / "data" / "gold" / "x.json.123.partial").write_text("half")
    target = create_backup(tmp_path / "out", sources(repo), repo)
    assert not any(".env" in n or "partial" in n for n in names(target))


def test_no_partial_file_is_left_behind(repo, tmp_path):
    create_backup(tmp_path / "out", sources(repo), repo)
    assert [p.suffix for p in (tmp_path / "out").iterdir()] == [".zip"]


def excluded_repo(tmp_path, lines):
    root = tmp_path / "repo"
    (root / ".git" / "info").mkdir(parents=True)
    (root / ".git" / "info" / "exclude").write_text(lines)
    return root


def test_locally_excluded_reads_plain_paths_and_skips_the_rest(tmp_path):
    root = excluded_repo(
        tmp_path, "# a comment\n\nprivate.md\n.scratch/\n/rooted.txt\n*.log\n!keep\n../outside\n"
    )
    assert locally_excluded(root) == [root / "private.md", root / ".scratch", root / "rooted.txt"]


def test_locally_excluded_is_empty_without_a_git_folder(tmp_path):
    assert locally_excluded(tmp_path) == []


def test_files_that_only_this_checkout_has_go_in_by_default(tmp_path):
    root = excluded_repo(tmp_path, "private.md\n.scratch/\nmissing.md\n")
    (root / "private.md").write_text("rules")
    (root / ".scratch").mkdir()
    (root / ".scratch" / "notes.md").write_text("notes")
    (root / ".scratch" / ".env").write_text("OPENAI_API_KEY=secret")
    target = create_backup(tmp_path / "out", root=root)
    assert names(target) == {"private.md", ".scratch/notes.md"}


def test_an_explicit_source_list_is_used_as_given(tmp_path):
    root = excluded_repo(tmp_path, "private.md\n")
    (root / "private.md").write_text("rules")
    assert create_backup(tmp_path / "out", [], root) is None


def test_the_default_sources_cover_gold_and_the_response_cache():
    relative = {str(p.relative_to(ROOT)) for p in backup.SOURCES}
    assert {"data/gold", "data/labeling/responses"} <= relative
    assert not any("labeling_fake" in s for s in relative)


def test_the_default_destination_is_outside_the_repository():
    assert not backup.DEFAULT_DIR.resolve().is_relative_to(ROOT.resolve())


def test_the_default_destination_is_icloud_drive():
    assert backup.DEFAULT_DIR.parent.name == "com~apple~CloudDocs"
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "BACKUP_DIR ?= $(HOME)/Library/Mobile Documents/com~apple~CloudDocs/" in makefile
    assert '--dir "$(BACKUP_DIR)"' in makefile


def test_a_missing_icloud_drive_stops_rather_than_faking_a_backup(monkeypatch, tmp_path, capsys):
    default = tmp_path / "Library" / "CloudDocs" / "taxon-backups"
    monkeypatch.setattr(backup, "DEFAULT_DIR", default)
    assert backup.main(["--dir", str(default)]) == 2
    assert "BACKUP_DIR" in capsys.readouterr().out
    assert not default.parent.exists()


def test_paid_responses_cannot_be_committed_by_accident():
    ignored = Path(ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in ignored
    assert ".env" in ignored


def test_the_command_says_when_there_is_nothing_yet(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(backup, "create_backup", lambda _destination: None)
    assert backup.main(["--dir", str(tmp_path / "out")]) == 0
    assert "Nothing to back up" in capsys.readouterr().out

"""Copy what git does not hold into a zip on iCloud Drive, so it survives losing this disk.

Hand labels and paid responses would cost time or money to replace. The spending ledger comes along
so that restored responses keep counting toward the caps, and so do hand-written benchmark tasks.
Files listed in .git/info/exclude are private to this checkout, so nothing else has a copy of them.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from taxon.pins import DATA_DIR, ROOT

DEFAULT_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/taxon-backups"
# Relative to the repository being backed up.
SOURCES = (
    DATA_DIR.relative_to(ROOT) / "gold",
    DATA_DIR.relative_to(ROOT) / "bench",
    DATA_DIR.relative_to(ROOT) / "labeling" / "responses",
    DATA_DIR.relative_to(ROOT) / "labeling" / "ledger.jsonl",
)
_NEVER = {".env"}


def locally_excluded(root: Path = ROOT) -> list[Path]:
    """The plain paths in .git/info/exclude. Glob patterns and paths outside `root` are skipped."""
    exclude = root / ".git" / "info" / "exclude"
    if not exclude.is_file():
        return []
    found = []
    for line in exclude.read_text(encoding="utf-8").splitlines():
        entry = line.strip().rstrip("/")
        if not entry or entry.startswith(("#", "!")) or any(c in entry for c in "*?["):
            continue
        path = root / entry.lstrip("/")
        if path.resolve().is_relative_to(root.resolve()):
            found.append(path)
    return found


def _files_under(sources: Iterable[Path]) -> list[Path]:
    found = []
    for source in sources:
        if source.is_file():
            found.append(source)
        elif source.is_dir():
            found.extend(p for p in source.rglob("*") if p.is_file())
    return sorted(p for p in found if p.name not in _NEVER and not p.name.endswith(".partial"))


def create_backup(
    destination: Path,
    sources: Iterable[Path] | None = None,
    root: Path = ROOT,
    now: datetime | None = None,
) -> Path | None:
    """Write a dated zip into `destination` and return it, or None if there is nothing to save.

    Refuses a destination inside the repository, where it could end up committed or be lost with
    the checkout. The archive is tested before it takes its final name.
    """
    destination = destination.expanduser().resolve()
    if destination.is_relative_to(root.resolve()):
        raise ValueError(f"a backup must go outside the repository, not in {destination}")
    defaults = [*(root / source for source in SOURCES), *locally_excluded(root)]
    files = _files_under(defaults if sources is None else sources)
    if not files:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = destination / f"taxon-backup-{stamp}.zip"
    partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.resolve().relative_to(root.resolve()))
        with zipfile.ZipFile(partial) as archive:
            broken = archive.testzip()
            if broken is not None:
                raise OSError(f"backup failed its check at {broken}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m taxon.backup", description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    if args.dir == DEFAULT_DIR and not DEFAULT_DIR.parent.is_dir():
        # Creating this path would look like a backup but never sync anywhere.
        print(f"iCloud Drive is not at {DEFAULT_DIR.parent}. Set BACKUP_DIR to choose a folder.")
        return 2
    target = create_backup(args.dir)
    if target is None:
        print("Nothing to back up yet.")
    else:
        with zipfile.ZipFile(target) as archive:
            print(f"Backed up {len(archive.namelist())} files to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

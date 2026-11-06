.PHONY: setup test lint reproduce clean backup gold label-fake label-dry-run label-first \
	label-gold-pool tasks-draft tasks-review tasks-check

TARGET ?= 200
TASKS ?= 100
BACKUP_DIR ?= $(HOME)/Library/Mobile Documents/com~apple~CloudDocs/taxon-backups

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

# Runs the offline tests, rebuilds every result from pinned sources, then checks the real
# artifacts. Downloads are cached in data/ and verified by checksum. This never spends money.
reproduce:
	uv run pytest -q
	uv run python -m taxon.reproduce
	uv run pytest -q -m data

clean:
	rm -rf results data/derived

# Zip what git does not hold into iCloud Drive, so it survives losing this disk.
backup:
	uv run python -m taxon.backup --dir "$(BACKUP_DIR)"

# Hand-label the gold set. Labels are kept in data/gold/ and survive make clean.
gold:
	uv run python -m taxon.gold --target $(TARGET)

# The whole labeling flow against in-memory vendors, with no keys and no spending.
label-fake:
	uv run python -m taxon.labeling dry-run --fake
	uv run python -m taxon.labeling run --first --fake

# These call the vendors and spend money, within the caps in taxon/labeling/budget.py.
label-dry-run:
	uv run python -m taxon.labeling dry-run

label-first:
	uv run python -m taxon.labeling run --first

label-gold-pool:
	uv run python -m taxon.labeling run --gold-pool

tasks-draft:
	uv run python -m taxon.bench draft --n $(TASKS)

tasks-review:
	uv run python -m taxon.bench review

tasks-check:
	uv run python -m taxon.bench check --need $(TASKS)

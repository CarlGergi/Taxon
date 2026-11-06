.PHONY: setup test lint reproduce clean backup gold label-fake label-dry-run label-first \
	label-gold-pool tasks-draft tasks-review tasks-check search-index search search-serve

TARGET ?= 200
TASKS ?= 100
BACKUP_DIR ?= $(HOME)/Library/Mobile Documents/com~apple~CloudDocs/taxon-backups
CATALOG ?= raw

# Homebrew's rustup keeps cargo off PATH, so ask rustup where this repository's toolchain is. The
# full path is needed because make 3.81 ignores its own PATH when it looks up a command.
CARGO := $(shell rustup which cargo 2>/dev/null)
ifeq ($(CARGO),)
CARGO := cargo
else
export PATH := $(dir $(CARGO)):$(PATH)
endif

setup:
	uv sync

test:
	uv run pytest -q
	$(CARGO) test --locked --quiet

lint:
	uv run ruff check .
	$(CARGO) fmt --all --check
	$(CARGO) clippy --locked --all-targets --quiet -- -D warnings

# Runs the offline tests, rebuilds every result from pinned sources, then checks the real
# artifacts. Downloads are cached in data/ and verified by checksum. This never spends money.
reproduce:
	uv run pytest -q
	$(CARGO) test --locked --quiet
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

# The catalog search engine. An enriched catalog takes CATALOG=name ENRICHMENTS=path.
search-index:
	uv run python -m taxon.bench search-index --name $(CATALOG) $(if $(ENRICHMENTS),--enrichments "$(ENRICHMENTS)")

# make search REQUEST='{"query": "red chef knife", "max_price": 50}'
search:
	uv run python -m taxon.bench search --name $(CATALOG) '$(REQUEST)'

# The MCP server, over stdio. Stdout belongs to the protocol, so make must stay quiet.
search-serve:
	@target/release/taxon-search serve --index data/derived/search/$(CATALOG)


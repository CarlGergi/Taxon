.PHONY: setup test lint reproduce clean backup gold label-fake label-dry-run label-first \
	label-gold-pool label-attributes tasks-draft tasks-review tasks-check search-index search search-serve train-tiny train-gpu \
	judge-price judge-sample judge-review judge-agreement judge-pool rl-tiny rl-gpu measure

TARGET ?= 200
TASKS ?= 100
BACKUP_DIR ?= $(HOME)/Library/Mobile Documents/com~apple~CloudDocs/taxon-backups
CATALOG ?= raw
RUN ?= qwen-lora
PRODUCTS ?= 2000
STEPS ?= 2000
# Sized from make measure on the M5 Max: about 16 s a step, so 1,600 steps fill an 8-hour night.
GRPO_STEPS ?= 1600
RUNS ?= data/bench/runs.jsonl

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

# Attributes for the labeled training products, once the pass has a score against hand labels.
label-attributes:
	uv run python -m taxon.labeling run --attributes

tasks-draft:
	uv run python -m taxon.bench draft --n $(TASKS)

tasks-review:
	uv run python -m taxon.bench review

tasks-check:
	uv run python -m taxon.bench check --need $(TASKS)

# Pooled judging of what agents return. RUNS is the agents' output: condition, run, task, returned.
# judge-price spends nothing. The rest spend within the caps, and judge-pool waits for agreement.
judge-price judge-sample judge-pool:
	uv run python -m taxon.bench $@ --runs "$(RUNS)"

judge-review judge-agreement:
	uv run python -m taxon.bench $@ --runs "$(RUNS)"

# The catalog search engine. An enriched catalog takes CATALOG=name ENRICHMENTS=path.
search-index:
	uv run python -m taxon.bench search-index --name $(CATALOG) $(if $(ENRICHMENTS),--enrichments "$(ENRICHMENTS)")

# make search REQUEST='{"query": "red chef knife", "max_price": 50}'
search:
	uv run python -m taxon.bench search --name $(CATALOG) '$(REQUEST)'

# The MCP server, over stdio. Stdout belongs to the protocol, so make must stay quiet.
search-serve:
	@target/release/taxon-search serve --index data/derived/search/$(CATALOG)

# Fine-tuning on CPU with tiny random weights and provisional labels. This only shows the pipeline
# runs: its numbers are not results.
train-tiny:
	uv run python -m taxon.training train
	uv run python -m taxon.training evaluate

# The real model on this machine's GPU (Apple or NVIDIA), with labels from the labeling pipeline.
train-gpu:
	uv run python -m taxon.training train --gpu --labels pipeline --name $(RUN) --products $(PRODUCTS) \
		--max-steps $(STEPS)

# GRPO on CPU with tiny random weights, on top of make train-tiny's adapter, then free against
# trie-constrained decoding. This only shows the pipeline runs: its numbers are not results.
rl-tiny:
	uv run python -m taxon.rl train --from tiny
	uv run python -m taxon.rl decode

# GRPO on the GPU, on top of a fine-tuning run, with pipeline labels. vLLM rollouts on NVIDIA,
# plain generation on Apple.
rl-gpu:
	uv run python -m taxon.rl train --gpu --labels pipeline --from $(RUN) --name $(RUN)-grpo \
		--products $(PRODUCTS) --max-steps $(GRPO_STEPS)

# A few real fine-tuning and GRPO steps on this machine's GPU, timed, to size overnight runs.
# Writes results/training_hardware.json. It uses provisional labels only for timing.
measure:
	uv run python -m taxon.training measure

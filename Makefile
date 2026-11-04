.PHONY: setup test lint reproduce clean

setup:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .

# Runs the offline tests, rebuilds every result from pinned sources, then checks
# the real artifacts. Downloads are cached in data/ and verified by checksum.
reproduce:
	uv run pytest -q
	uv run python -m taxon.reproduce
	uv run pytest -q -m data

clean:
	rm -rf results data/derived

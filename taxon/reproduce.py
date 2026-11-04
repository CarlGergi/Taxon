"""Rebuild every result from the pinned sources. Run through `make reproduce`."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from taxon.baseline import run_baseline
from taxon.fetch import cached_range, fetch
from taxon.pins import (
    AMAZON_FILE_SIZES,
    AMAZON_REVISION,
    DERIVED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    SEED,
    TAXONOMY,
    TOKENIZERS,
    amazon_url,
)
from taxon.products import read_candidates, select, summarize, write_jsonl
from taxon.silver import coverage, label_products
from taxon.taxonomy import Taxonomy
from taxon.trie import TokenTrie

N_PRODUCTS = 10_000


def write_result(name: str, payload: dict, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (results_dir / f"{name}.json").write_text(text, encoding="utf-8")
    print(f"wrote results/{name}.json", flush=True)


def build_trie(taxonomy: Taxonomy, tokenizer: Tokenizer) -> TokenTrie:
    return TokenTrie.build(
        taxonomy.paths(),
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
    )


def taxonomy_summary(taxonomy: Taxonomy) -> dict:
    depths = Counter(c.depth for c in taxonomy.categories_by_path.values())
    return {
        "version": taxonomy.version,
        "sha256": TAXONOMY.sha256,
        "categories": len(taxonomy),
        "leaves": len(taxonomy.leaves()),
        "verticals": len(taxonomy.roots()),
        "attributes": len(taxonomy.attributes),
        "categories_per_depth": {str(d): depths[d] for d in sorted(depths)},
    }


def sample_products() -> list[dict]:
    candidates = {}
    for domain, size in sorted(AMAZON_FILE_SIZES.items()):
        cache = RAW_DIR / "amazon" / AMAZON_REVISION / domain
        reader = cached_range(amazon_url(domain), size, cache)
        candidates[domain] = read_candidates(domain, size, reader, SEED)
        print(f"  {domain}: {len(candidates[domain])} candidates", flush=True)
    return select(candidates, N_PRODUCTS, SEED)


def main() -> None:
    taxonomy = Taxonomy.load(fetch(TAXONOMY))
    write_result("taxonomy", taxonomy_summary(taxonomy))

    for name, pin in TOKENIZERS.items():
        tokenizer = Tokenizer.from_file(str(fetch(pin)))
        stats = build_trie(taxonomy, tokenizer).stats()
        write_result(f"trie_{name}", {"tokenizer_sha256": pin.sha256, **stats})

    products = sample_products()
    products_path = DERIVED_DIR / "products.jsonl"
    write_jsonl(products_path, products)
    digest = hashlib.sha256(products_path.read_bytes()).hexdigest()
    write_result("sample", {"seed": SEED, "sha256": digest, **summarize(products)})

    labels = label_products(products, taxonomy)
    write_jsonl(DERIVED_DIR / "silver_labels.jsonl", labels)
    write_result("silver", coverage(products, labels))

    by_asin = {row["parent_asin"]: row["path"] for row in labels}
    write_result("baseline", run_baseline(products, by_asin, taxonomy, SEED, "silver"))


if __name__ == "__main__":
    main()

# Taxon

Catalog intelligence for AI shopping agents.

Taxon post-trains a small vision-language model to classify products into Shopify's open
product taxonomy and extract their attributes. CatalogBench, its agent benchmark, then measures
whether shopping agents find the right products more often on a Taxon-enriched catalog than on a
raw one.

## Status

Milestone 1 of 6 is complete: taxonomy loader, token trie, product sample, baseline, and metrics.

## Running it

You need [uv](https://docs.astral.sh/uv/). Every input is pinned to an exact version and checksum.

```bash
make setup
make test
make reproduce
```

`make test` runs offline. `make reproduce` downloads about 115 MB, rebuilds every number below
into `results/`, and takes a few minutes on a CPU.

## Results

### Label space

Shopify taxonomy release 2026-10: 64,292 categories, of which 50,427 are leaves, across
26 verticals, with 12,334 attributes.

### Token trie

Every category path, internal nodes included, as token sequences. A walk through the trie accepts
exactly the taxonomy, and each path decodes back to its original text. Qwen3.5-2B and Qwen3.5-4B
share one vocabulary, as do Qwen2.5-VL and Qwen3-VL.

| Tokenizer | States | Mean tokens per path | Longest path | Widest branch |
| --- | --- | --- | --- | --- |
| Qwen3.5 | 272,297 | 25.1 | 67 | 68 |
| Qwen2.5-VL | 274,434 | 25.2 | 67 | 68 |

### Product sample

10,000 products from Amazon Reviews 2023 item metadata, split evenly across 28 domains, with
duplicates removed by id and by normalized title. Each has a title and an image.

### Baseline

TF-IDF plus one-vs-rest logistic regression on title, features, description, and details.

**These are provisional numbers against silver labels, not a measure of real accuracy.** Labels
come from matching Amazon's own category breadcrumbs onto the taxonomy, so scores show agreement
with that mapping. They are replaced by annotator and hand-labeled gold evaluation in milestone 2.

Silver labels cover 5,126 of 10,000 products (51.3%) and 1,046 distinct paths. They
cover nothing in All_Beauty, Amazon_Fashion, CDs_and_Vinyl, Health_and_Personal_Care.

Test set of 544 products, regularization chosen on a separate validation split:

| Model | Exact leaf accuracy |
| --- | --- |
| TF-IDF + logistic regression | 48.3% |
| Always predict the most common label | 4.4% |

11.8% of test labels never appear in training, so a flat classifier cannot predict them.

| Level | Gold support | Micro precision | Micro recall | Macro precision | Macro recall |
| --- | --- | --- | --- | --- | --- |
| 1 | 544 | 75.0% | 75.0% | 72.3% | 59.9% |
| 2 | 495 | 72.0% | 65.9% | 47.2% | 35.8% |
| 3 | 367 | 67.1% | 45.0% | 30.6% | 26.3% |
| 4 | 251 | 63.6% | 32.7% | 23.8% | 20.4% |
| 5 | 144 | 69.1% | 26.4% | 19.7% | 18.1% |
| 6 | 48 | 73.9% | 35.4% | 29.7% | 28.2% |
| 7 | 9 | 100.0% | 66.7% | 66.7% | 66.7% |
| 8 | 4 | 100.0% | 50.0% | 50.0% | 50.0% |

Validation accuracy is still rising with more training data:

| Training examples | Exact leaf accuracy |
| --- | --- |
| 1,025 | 35.3% |
| 2,050 | 42.9% |
| 4,100 | 49.4% |

## Data and model licences

Amazon Reviews 2023 comes from the McAuley Lab. Qwen2.5-VL-3B-Instruct is under the Qwen Research
License, which is non-commercial. Qwen3.5-2B and Qwen3.5-4B are under Apache-2.0.

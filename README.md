# Taxon

Catalog intelligence for AI shopping agents.

Taxon post-trains a small vision-language model to classify products into Shopify's open
product taxonomy and extract their attributes. CatalogBench, its agent benchmark, then measures
whether shopping agents find the right products more often on a Taxon-enriched catalog than on a
raw one.

## Status

Milestone 1 is complete. Milestone 2's code is built and tested against fake vendors. The labeling
run, the gold set and the benchmark tasks are still to come, so nothing below measures real
annotator labels. The search side of milestone 5 is built and tested on the raw catalog. Milestone 3's
training pipeline is built and tested on CPU with tiny random weights, so no model has been trained
for real yet.

## Running it

You need [uv](https://docs.astral.sh/uv/) and [rustup](https://rustup.rs), which installs the Rust
version pinned in `rust-toolchain.toml`. Every input is pinned to an exact version and checksum.

```bash
make setup
make test
make reproduce
```

`make test` runs offline. `make reproduce` downloads about 115 MB, rebuilds every number below
into `results/`, and takes a few minutes on a CPU. It never calls a paid API and never reads a key.

## Labeling

Copy `.env.example` to `.env` and fill in the keys. Each annotator picks from a shortlist of up to
50 paths or says none, and only a none starts a walk down the tree. A stronger model settles real
disagreements or abstains, and is scored against the hand labels. Every response is cached on disk,
and hard caps of $20 for Anthropic, $10 for OpenAI and $10 for Google stop a run before they are
passed.

| Role | Model |
| --- | --- |
| Annotators | `claude-haiku-5-5`, `gpt-6-luna`, `gemini-3.5-flash-lite` |
| Arbitrator | `claude-sonnet-5-5` |

```bash
make label-fake
make label-dry-run
make label-first
make label-gold-pool
```

`make label-fake` runs the flow against in-memory vendors, with no keys and no spending. The others
spend money. The dry run prints the measured cost per vendor from 100 products, and later stages
refuse to start without it. `make label-first` labels 2,000 products. Whether to label more is
decided from the learning curve at milestone 3. The gold-pool stage exists only to score the
pipeline against hand labels, and its output never reaches training.

Attributes come from a separate second pass, never from the category calls. Once a product's
category is settled, by agreement or by the arbitrator, one cheap annotator with the image fills in
that category's attributes in one call. Value lists are too long to send (color alone allows over
22,000 values), so there are two versions:

- **Exact:** the model names each value, and only answers that match an allowed value exactly are
  kept.
- **Shortlist:** for each attribute, the 20 allowed values closest to the listing's text, by
  character n-grams, are listed, and the model picks one or none. So "navy blue" can still reach
  "Navy".

Both count answers that named no allowed value. The dry run prices both, and the gold-pool stage
runs both so each can be scored against hand-labeled attributes. `make label-attributes` then runs
only the version that scored better. Category responses are cached,
so adding the pass later pays only for the new calls. `make label-first` stays category-only.

Hand labels and paid responses are not in git. This zips them, along with anything listed in
`.git/info/exclude`, into iCloud Drive. Set `BACKUP_DIR` to use another folder. Keys are never
included.

```bash
make backup
```

## Gold set

```bash
make gold
make gold TARGET=500
```

This opens a local labeling page with keyboard shortcuts, a search box over every category path, and
the baseline's top 5 suggestions. It cannot see annotator or arbitrator output. Labels are kept in
`data/gold/`, which `make clean` leaves alone.

The pool of 500 products is set aside before any training and never overlaps it. The first
200 are balanced across all 28 domains.

## Benchmark tasks

```bash
make tasks-draft
make tasks-review
make tasks-check
```

Tasks are drafted from hand labels and listing prices only. Each request is written by hand, in a
shopper's words, and each task is hand-checked.

### Judging products outside the gold set

Agents can return correct products that have no hand label. Those are judged rather than left out,
with pooled judging: every non-gold product any condition returns for a task is collected once, and
checked against the task's constraints. Price is checked in code. Category and attribute
constraints go to an LLM judge that sees only the raw listing and image, never a catalog's
enrichment, so no condition is judged on its own data. The judge is `gpt-6-luna`, not Claude:
the arbitrator is Claude, and a judge from the same family could favor the Taxon catalog.

```bash
make judge-price
make judge-sample
make judge-review
make judge-agreement
make judge-pool
```

The judge is used only if it agrees with hand judgments on a seeded sample of 50 pairs: at least
90%, with no more than 5 of them unsure. Agreement is also reported per condition. Every pair the
judge is unsure of goes to hand review, and scoring waits for them. `make judge-price` estimates the
cost before anything is sent, and the other commands spend within the same caps as labeling.

## Catalog search

A Rust engine on tantivy, behind a local MCP server that exposes one tool, `search_catalog`. It takes
keywords plus optional category, attribute and price filters. The tool has the same schema on every
catalog, so the data is the only thing that changes between benchmark conditions.

```bash
make search-index
make search REQUEST='{"query": "red chef knife", "max_price": 50}'
make search-serve
```

The raw catalog holds listings without Amazon's category fields. Two choices keep the comparison
fair, and both make the raw baseline harder to beat:

- A category or attribute filter on the raw catalog is not applied. It is named in `ignored_filters`,
  and keyword results still come back. Returning nothing instead would cost the raw condition an extra
  tool call on every task.
- The price filter works on the raw catalog, because price is part of the listing.

An enriched catalog is the same listings plus `category` and `attributes`, keeping only values the
taxonomy allows:

```bash
make search-index CATALOG=model ENRICHMENTS=enrichments.jsonl
make search-serve CATALOG=model
```

## Fine-tuning

Each training example asks for a random subset of a product's fields, so the model learns to return
whatever is requested. Sometimes that includes an attribute the category cannot have, whose answer is
null. Two metrics check this, and neither needs labels:

- **Field compliance:** how often the model outputs only the requested fields.
- **Field invariance:** how often a field's answer stays the same when the rest of the requested schema
  changes.

```bash
make train-tiny
make train-gpu RUN=name PRODUCTS=2000 STEPS=2000
```

`make train-tiny` runs on CPU with tiny random weights and provisional labels. It shows the pipeline
works, and its numbers are not results. `make train-gpu` fine-tunes the real model with LoRA, needs a
GPU, and refuses provisional labels. Accuracy is only ever computed against hand labels, with
`python -m taxon.training evaluate --on gold`.

## Results

### Label space

Shopify taxonomy release 2026-10: 64,292 categories, of which 50,427 are leaves, across
26 verticals, with 12,334 attributes.

### Token trie

Every category path, internal nodes included, as token sequences for Qwen/Qwen3.5-2B. A walk through the
trie accepts exactly the taxonomy, and each path decodes back to its original text. Qwen3.5-4B shares
this vocabulary.

| States | Mean tokens per path | Longest path | Widest branch |
| --- | --- | --- | --- |
| 272,297 | 25.1 | 67 | 68 |

### Product sample

10,000 products from Amazon Reviews 2023 item metadata, split evenly across 28 domains, with
duplicates removed by id and by normalized title. Each has a title and an image.

### Shortlists

Share of 4,874 products whose provisional label is in the list. Close means an ancestor or
descendant of the label is in it.

| Method | Exact, top 30 | Exact, top 50 | Close, top 50 |
| --- | --- | --- | --- |
| TF-IDF similarity alone | 35.2% | 40.0% | 75.5% |
| Similarity, classifier and ancestors | 89.3% | 91.8% | 98.2% |

**The second row is flattering.** Its classifier learned from the same provisional labels it is scored
against, and knows only their categories. Recall on the gold set will be the real figure.

### Baseline

TF-IDF plus one-vs-rest logistic regression on title, features, description, and details.

**These are provisional numbers against silver labels, not a measure of real accuracy.** Labels
come from matching Amazon's own category breadcrumbs onto the taxonomy, so scores show agreement
with that mapping. They will be replaced by evaluation on the hand-labeled gold set.

Silver labels cover 5,126 of 10,000 products (51.3%) and 1,046 distinct paths. They
cover nothing in All_Beauty, Amazon_Fashion, CDs_and_Vinyl, Health_and_Personal_Care.

Test set of 523 products, regularization chosen on a separate validation split:

| Model | Exact leaf accuracy |
| --- | --- |
| TF-IDF + logistic regression | 48.9% |
| Always predict the most common label | 4.6% |

11.9% of test labels never appear in training, so a flat classifier cannot predict them.

| Level | Gold support | Micro precision | Micro recall | Macro precision | Macro recall |
| --- | --- | --- | --- | --- | --- |
| 1 | 523 | 75.7% | 75.7% | 72.5% | 60.5% |
| 2 | 477 | 72.0% | 67.5% | 46.6% | 36.8% |
| 3 | 352 | 66.1% | 46.0% | 30.4% | 26.9% |
| 4 | 240 | 62.2% | 32.9% | 23.6% | 20.6% |
| 5 | 139 | 69.8% | 26.6% | 20.0% | 19.0% |
| 6 | 47 | 70.8% | 36.2% | 29.9% | 28.9% |
| 7 | 9 | 100.0% | 66.7% | 66.7% | 66.7% |
| 8 | 4 | 100.0% | 50.0% | 50.0% | 50.0% |

Validation accuracy is still rising with more training data:

| Training examples | Exact leaf accuracy |
| --- | --- |
| 972 | 35.9% |
| 1,945 | 44.1% |
| 3,891 | 48.3% |

### Catalog search

Searching for a product's exact title over MCP, on the raw catalog of 10,000 products. This checks that
the engine and catalog work together, not how well an agent shops.

| Products searched | Found first | Found in top 10 |
| --- | --- | --- |
| 500 | 99.6% | 99.8% |

## Data and model licences

Amazon Reviews 2023 comes from the McAuley Lab. Qwen3.5-2B and Qwen3.5-4B are under Apache-2.0.

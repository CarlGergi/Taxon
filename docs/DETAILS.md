# Taxon in detail

How each part works, the commands that run it, and every result behind the
[README](../README.md).

## Labeling

Copy `.env.example` to `.env` and fill in the Anthropic and OpenAI keys. Each of two annotators
picks from a shortlist of up to 50 paths or says none, and only a none starts a walk down the tree.
A label is accepted when both annotators give the same answer. A stronger model settles every
disagreement, and every product only one annotator answered, or abstains. Every response is cached
on disk, and hard caps of $20 for Anthropic and $3 for OpenAI stop a run before they are passed.

There are two annotators, not three, to keep the cost down: a third vendor means a third bill, and
the arbitrator already covers what a third vote would decide. The price is that every disagreement
goes to the arbitrator, with no majority vote to fall back on. The labeler takes any number of
annotators, and the Gemini client is kept, unused, so a third can be added back.

| Measure | Result, with 95% interval |
| --- | --- |
| Products where both annotators gave the same category (of 2,000) | 75.0% [73.1, 76.9] |
| Products sent to the arbitrator (of 2,000) | 24.9% [23.1, 26.9] |
| Arbitrator accuracy on disputed products, against the hand labels | 57.1% [25.0, 84.2], 4 of 7 |

The first two come from the agreement report of the 2,000-product run (`results/labeling.json`).
The third is the `arbitration` entry of `results/gold_eval.json`: only 7 of the 50 hand-labeled
products were disputed, so it says little. On those 7 the arbitrator chose the same category as
`gpt-6-luna` every time.

| Role | Model |
| --- | --- |
| Annotators | `claude-haiku-5-5`, `gpt-6-luna` |
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
50, the verified set, cover all 28 domains, one or two products each.

## Benchmark tasks

```bash
make tasks-draft
make tasks-review
make tasks-check
```

Constraints and acceptable products come from hand labels and listing prices only. Each request,
in a shopper's words, is drafted by `gpt-6-luna` from the requirements alone, never a product, and
then read, edited where needed and approved by hand, along with the acceptable products. The drafter
is not Claude, because the agent and the frontier-enriched catalog both are. Each task records
whether its request was drafted and whether it was rewritten. A request that quotes a taxonomy path
or a multi-word category name is held back until it is reworded. Drafting 100 requests costs about
a cent; `FAKE=1` drafts plain stand-ins for free.

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

The judge is used only if it agrees with hand judgments on a seeded sample of pairs: at least 90% of
the pairs it answered, with at least 45 answered. The sample starts at 50. If the judge is unsure of
more than 5 of them, the sample grows by the next pairs in the same seeded order until it has
answered 45. Agreement is also reported per catalog. Every pair the judge is unsure of goes to hand
review, and scoring waits for them. `make judge-price` estimates the cost before anything is sent,
and the other commands spend within the same caps as labeling. Judge calls go to OpenAI's batch
API; `DIRECT=1` sends them one at a time at full price when a batch stalls, and `LIMIT=` caps what
one command may spend.

On this benchmark the judge was unsure of 6 of the first 50 pairs, one more than the rule allows,
so the sample grew to 51. It answered the 51st, which made 45 answered. It agreed with the hand
judgments on 42 of the 44 it answered in the first 50 (95.5%), and on 43 of the 45 in the full
sample: **95.6% [85.2, 98.8]**, with Cohen's kappa **0.81 [0.46, 1.00]**. Both disagreements were
lenient: the judge accepted a product the hand judgment rejected, never the reverse. Both products
had been returned on all three catalogs, so neither disagreement favors one catalog. Per catalog,
the judge agreed on 94.3% [81.4, 98.4] of the raw catalog's 35 answered pairs, 94.3% [81.4, 98.4]
of Sonnet's 35, and 93.9% [80.4, 98.3] of Taxon's 33.

Across the whole pool of 238 pairs, the judge said 190 meet their task, 17 fail and 31 are unsure.
The 31 unsure ones are decided by hand. The OpenAI batch stalled, so every judge call went
direct, at full price: $0.020 in all.

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

Every benchmark catalog holds the same 2,000 products: all 500 gold-queue products plus 1,500
distractors drawn with a fixed seed. Distractors never come from the products the model is trained
on, so no condition shops listings Taxon has already seen with their labels.

## Frontier enrichment

The benchmark's middle condition is the same 2,000 listings enriched by `claude-sonnet-5-5` alone,
with no fine-tuning. It goes through the same prompts as the labeling annotators (pick from the
shortlist, or walk down the tree) and the same attribute pass, in the version that scored better
on the hand labels, so the model is the only difference from the Taxon catalog. Its labels for the
gold products double as the zero-shot baseline of the classifier ladder.

```bash
make enrich-price
make enrich
make search-index CATALOG=sonnet ENRICHMENTS=data/labeling/enrichments_sonnet.jsonl
```

The Taxon catalog comes from the shortlist fine-tune (`qwen-lora-shortlist`) on this machine, in
two passes like Sonnet's. The first pass picks the category, decoded through a trie of the product's
50 shortlist candidates plus "none", so the answer is always one of them. The second asks for that
category's attributes and keeps only allowed values. It costs no API money: on an Apple M5 Max it
took 4.5 seconds a product, about 1,260 hours per million products.

```bash
make enrich-taxon RUN=qwen-lora-shortlist RUNS_DIR=data/training
make search-index CATALOG=taxon ENRICHMENTS=data/training/qwen-lora-shortlist/enrichments_taxon.jsonl
```

Taxon's catalog gives no category to 112 of the 2,000 products (5.6%). For each, the model answered
"none": no candidate on its shortlist fit. Sonnet's catalog gives a category to all 2,000. The
benchmark keeps the 112 as they are, with no special handling:

- They stay in the catalog as raw listings, with no `category` or `attributes`. Keywords still
  match their listing text, but not the category and attribute words that other listings in this
  catalog also match on.
- A search with a category or attribute filter never returns them. On Taxon's catalog, 208 of the
  agent's 715 searches (29%) used one.
- They are scored like any other product: returned, one counts only if it is a task's gold
  product or the judge accepts it.
- 27 of them are gold-queue products and 85 are distractors. They include every gold product of 3
  of the 92 tasks, so on those 3 tasks the agent can reach a gold product only by keywords.

`make enrich-price` spends nothing: it counts the input tokens of a sample's real requests, image
included, with Anthropic's free counting endpoint, and takes walking rates from the dry run.
`make enrich` waits for the labeling run, its scoring on the gold set, and a price that fits what
is left under the cap.

## Shopping agent

One agent shops each catalog: `claude-haiku-5-5` with a fixed prompt, the `search_catalog` tool and
a `submit_products` tool for its answer. Only the catalog behind the search tool changes between
conditions. Haiku takes no temperature setting, so thinking is off, effort is low, and each
condition runs three times so the spread between runs is measured. Each run may make at most 8
model calls. Search results reach the agent as merchant data, and the prompt says never to follow
instructions inside them.

```bash
make shop-price
make shop CONDITION=raw FAKE=1
make shop CONDITION=raw
```

`make shop-price` prices the runs from the real tool schema and the real size of search replies,
spending nothing, and says how many runs per catalog the remaining Anthropic budget covers: three
if they fit, otherwise two, never more than the cap. `FAKE=1` runs a scripted model against the
real engine. A real run reserves each call's worst-case cost against the cap before sending it,
logs every finished run with its transcript, and picks up where it stopped. Runs go one round at a
time across the catalogs (`RUNS=1` on each, then `RUNS=2`), so a stop never leaves one catalog
with more runs than another. Only hand-checked tasks run: 92 of the 100 drafted, three rounds each.

`make bench-score` turns the logged runs into the benchmark report: for each catalog, task success
(the first product returned meets every requirement), precision of everything returned, empty
answers, searches, model calls, tokens and dollars per task, each as a mean with its spread across
runs and a 95% interval over tasks, and the paired difference in success against the raw catalog
with a 95% interval. The 92 tasks come from 41 hand-labeled products, about two per product, and
two tasks from one product are not independent, so every interval resamples source products
rather than tasks (a cluster bootstrap). It also gives each catalog's enrichment cost per million
products. It refuses while any catalog is missing runs or any returned product is unjudged or
unresolved.

Every number measured on the gold set comes with a 95% interval: a Wilson interval for a share of
products, a bootstrap one for macro averages and latencies, and a paired bootstrap for the
difference between two systems on the same products. With 50 products, an accuracy near 70% has an
interval about 13 points either side, so differences smaller than about 15 points are reported
as unresolved rather than ranked.

Paid steps share the $20 Anthropic cap and run in order of importance: the labeling run, scoring
it on the gold set, Sonnet's enrichment, then the agent runs, re-priced from what has actually been
spent.

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
works, and its numbers are not results. `make train-gpu` fine-tunes the real model with LoRA on a GPU,
Apple's through MPS or NVIDIA's through CUDA, and refuses provisional labels. Accuracy is only ever computed against hand labels, with
`python -m taxon.training evaluate --on gold`.

## Reinforcement learning

GRPO with TRL follows fine-tuning, starting from the fine-tuned adapter. Its rewards are computed
exactly from the taxonomy, with no reward model:

- **Field compliance:** the reply holds exactly the requested fields.
- **Valid node:** the category is a real taxonomy path.
- **Path match:** credit for each level shared with the label, weighted by depth, so deep levels
  count most. Going past the labeled path costs too.
- **Allowed values:** attribute values are ones the category allows.
- **Depth penalty:** stopping above the labeled depth costs extra.

Depth weighting and the penalty together keep the model from retreating to shallow, safe
categories: stopping at the top level earns nothing net. Each run also plots predicted against
labeled depth over training.

Trie-constrained decoding makes every category valid by construction. When category is requested,
a logits processor opens the reply with it, then allows only tokens that continue a real taxonomy
path, and lets the value close only where a path ends.

```bash
make rl-tiny
make rl-gpu RUN=name PRODUCTS=2000 STEPS=500
```

`make rl-tiny` runs on CPU with tiny random weights and provisional labels, then compares free and
constrained decoding on validity and latency. Its numbers are not results. `make rl-gpu` uses the
real model on a GPU and refuses provisional labels. Rollouts use vLLM on NVIDIA and plain generation
on Apple, where vLLM does not run.

### Hardware

Training runs on an Apple M5 Max through PyTorch's MPS backend, in BF16. `make measure` times a few
real fine-tuning and GRPO steps there and sizes overnight runs from them. Every result file records
the machine and device it came from. The quantization study needs an NVIDIA GPU and runs later.

## Quantization

The final model at three precisions, on this Mac. The trained adapters are merged into the base
weights and MLX converts the result to bfloat16, 8-bit and 4-bit affine weight quantization (groups
of 64). MLX leaves some layers, such as the vision tower, at full precision, so the averages are
about 9.6 and 6.2 bits per weight. Each version answers the gold products the same way: category
alone, greedy, through the same token trie, one at a time after one untimed warm-up reply. The
report gives accuracy per taxonomy level, p50 and p95 latency, throughput, size, peak memory and
the local hours a million products would take. FP8 needs an NVIDIA GPU and is not measured.

```bash
make quant-prepare QUANT_RUN=data/training/qwen-lora-shortlist
make quant
```

## Detailed results

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

### Training labels against the hand labels

The fine-tuned model learns from the labeling pipeline's categories, so their accuracy bounds what
it can be taught. On the 50 hand-labeled products (Wilson 95% intervals):

| Measure | Pipeline labels |
| --- | --- |
| Exact category | 52.0% [38.5, 65.2] |
| Right at the top level | 94.0% [83.8, 97.9] |

| Level | Gold support | Precision | Recall |
| --- | --- | --- | --- |
| 1 | 50 | 94.0% [83.8, 97.9] | 94.0% [83.8, 97.9] |
| 2 | 49 | 92.0% [81.2, 96.8] | 93.9% [83.5, 97.9] |
| 3 | 46 | 88.6% [76.0, 95.0] | 84.8% [71.8, 92.4] |
| 4 | 41 | 71.4% [56.4, 82.8] | 73.2% [58.1, 84.3] |
| 5 | 24 | 65.2% [44.9, 81.2] | 62.5% [42.7, 78.8] |

**The training labels are noisy at the deepest levels.** They almost always have the right
department and usually the right aisle, but about half name the exact leaf wrong. Of the 24
misses, 17 are on the right path at the wrong depth (10 too deep, 7 too shallow) and 7 take a
different branch, 3 of them from the top. A model trained on them is taught that noise too. Levels below 5 have fewer than 10 products each
and are left out.

### Does training beat its labels?

Each trained model is asked for the category alone of the same 50 hand-labeled products, greedy,
once free and once constrained to real taxonomy paths by the token trie (`make ladder`, then
`make reproduce`). The difference against the pipeline labels is paired over products.

| Model | Decoding | Exact category | Top level | Invalid paths | Minus pipeline labels |
| --- | --- | --- | --- | --- | --- |
| Pipeline labels (the training data) | | 52.0% [38.5, 65.2] | 94.0% [83.8, 97.9] | 0.0% | |
| Fine-tuned | free | 12.0% [5.6, 23.8] | 90.0% [78.6, 95.7] | 56.0% [42.3, 68.8] | -40.0 [-58.0, -22.0] |
| Fine-tuned | constrained | 20.0% [11.2, 33.0] | 90.0% [78.6, 95.7] | 0.0% [0.0, 7.1] | -32.0 [-50.0, -14.0] |
| Fine-tuned on a shortlist | free | 48.0% [34.8, 61.5] | 88.0% [76.2, 94.4] | 0.0% [0.0, 7.4] | -4.0 [-14.0, 6.0] |
| Fine-tuned on a shortlist | constrained | 48.0% [34.8, 61.5] | 88.0% [76.2, 94.4] | 0.0% [0.0, 7.4] | -4.0 [-14.0, 6.0] |
| Fine-tuned on a shortlist, then GRPO | free | 48.0% [34.8, 61.5] | 86.0% [73.8, 93.0] | 0.0% [0.0, 7.7] | -4.0 [-14.0, 6.0] |
| Fine-tuned on a shortlist, then GRPO | constrained | 48.0% [34.8, 61.5] | 86.0% [73.8, 93.0] | 0.0% [0.0, 7.7] | -4.0 [-14.0, 6.0] |

**Fine-tuning on the open taxonomy does not beat its labels; fine-tuning on a shortlist matches
them.** The first model is asked to write a path from all 64,292 categories and is 32 points worse
than its labels even when constrained to real paths. It learns the department (90%) but mostly
picks a different branch below it, and free decoding invents a path that does not exist more than
half the time.

The second model is shown the same top-50 candidate paths the annotators were given and picks one,
or says "none". It scores 48.0%, 28 points [10, 44] above the first model constrained, and 4 points
below its labels with an interval that holds zero: as accurate as the labeling pipeline on these 50
products, at the cost of one local model call instead of two paid annotators and an arbitrator.
**Its ceiling is 76%**: the exact hand label is among the 50 candidates for 38 of the 50 products,
so no pick can do better. Its 26 misses are 9 too deep, 6 too shallow, 9 on another branch (4 of
them in another department), and 2 "none".
Free and constrained decoding agree because both pick from the list.

**GRPO on top of the shortlist model changed nothing measurable.** After 200 steps it gets the same
24 of the 50 products right, a difference of 0.0 [0.0, 0.0] from the model it started from, and
answers "none" for 4 products instead of 2. Its logged loss stayed near zero throughout, most
likely because the four sampled picks per product nearly always agreed, which leaves no
group-relative advantage to learn from. The run took 17 hours on this Mac, because it did not fit in
36 GB of memory and swapped, at up to about 5 minutes a step.

Why the first model fell short: asked about 50 of its own training products, it reproduces their
labels only 28% of the time, so it had not learned them, not merely failed to generalize. Each
product's category was a training target about once and a half in 2,000 steps. The shortlist turns
the task into a choice among 50, which it learns in the same budget:

| Model | Decoding | Exact on 50 training products | Top level |
| --- | --- | --- | --- |
| Fine-tuned | free | 18.0% [9.8, 30.8] | 88.0% [76.2, 94.4] |
| Fine-tuned | constrained | 28.0% [17.5, 41.7] | 88.0% [76.2, 94.4] |
| Fine-tuned on a shortlist | free | 92.0% [81.2, 96.8] | 94.0% [83.8, 97.9] |
| Fine-tuned on a shortlist | constrained | 92.0% [81.2, 96.8] | 94.0% [83.8, 97.9] |
| Fine-tuned on a shortlist, then GRPO | free | 92.0% [81.2, 96.8] | 94.0% [83.8, 97.9] |
| Fine-tuned on a shortlist, then GRPO | constrained | 92.0% [81.2, 96.8] | 94.0% [83.8, 97.9] |

The keyword baseline on the same products: 8.0% [3.2, 18.8] exact when trained on the silver
labels, 12.0% [5.6, 23.8] when trained on the pipeline labels. The paired difference, +4.0 points
[−6.0, +16.0], is too small to call. The category shortlist given to the annotators holds the exact
hand label for 76.0% of products in its top 50, and a close one (an ancestor or descendant) for
96.0%.

Attributes, on 66 hand-labeled values across 39 products, with intervals that resample products:

| Attribute pass | Values matched |
| --- | --- |
| Shortlist of allowed values | 53.0% [39.7, 66.1] |
| Exact attribute names only | 31.8% [17.7, 45.6] |
| Only what the listing's details state | 18.2% [8.7, 28.1] |

The shortlist version beats the exact one by 21.2 points [3.4, 41.5] on the same products, so it
is the version run on the 2,000 training products.

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

### Quantization results

The shortlist model, merged and converted with MLX, asked for the category alone of the 50
hand-labeled products: greedy, picked from its 50 candidates through the trie, one at a time after
one untimed warm-up reply, on one M5 Max. Intervals resample the 50 replies; the difference from
16-bit is paired over products. FP8 needs an NVIDIA GPU and was not measured.

| Precision | Exact category | Top level | Minus 16-bit | p50 latency | p95 latency | Products per hour | Size | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16-bit | 48.0% [34.8, 61.5] | 86.0% [73.8, 93.0] |  | 0.51 s [0.48, 0.56] | 0.73 s [0.65, 0.82] | 6,896 [6,490, 7,364] | 4.4 GB | 5.7 GB |
| 8-bit | 48.0% [34.8, 61.5] | 88.0% [76.2, 94.4] | 0.0 [0.0, 0.0] | 0.42 s [0.37, 0.44] | 0.59 s [0.53, 0.69] | 8,494 [7,995, 9,046] | 2.7 GB | 4.0 GB |
| 4-bit | 46.0% [33.0, 59.6] | 90.0% [78.6, 95.7] | -2.0 [-12.0, 6.0] | 0.34 s [0.31, 0.38] | 0.56 s [0.45, 0.65] | 10,177 [9,399, 10,961] | 1.7 GB | 3.2 GB |

These replies give the category alone. The benchmark's enrichment also asks for attributes, in
PyTorch, at 4.5 seconds a product.

## Unsupported attribute values

Taxon's attribute pass states values that its listings do not support, and the agent trusts them.
A real case from the benchmark: Taxon's catalog tags Sunflower Botanicals Wormseed Extract as made
in the United States, though nothing in its listing says where it is made. Sonnet's catalog leaves
its origin blank. On two tasks asking for a supplement made in the United States, the agent found
the extract through that tag, twice by filtering on it and once through a keyword match on it, and
returned it in 4 runs. By hand it fails both tasks, because the listing never shows the origin. It
was never the first product returned, so it lowers Taxon's precision, not its task success.

How often each catalog says more than its listings:

| Check | Taxon | Sonnet |
| --- | --- | --- |
| Specific countries of origin that the listing text never names | 18 of 32, 56.2% [39.3, 71.8] | 4 of 9, 44.4% [18.9, 73.3] |
| Values on verified products where the hand label is blank | 191 of 244, 78.3% [71.9, 83.5] | 120 of 168, 71.4% [64.1, 77.8] |
| Values on verified products that contradict the hand label | 9 of 53, 17.0% [9.2, 29.2] | 9 of 48, 18.8% [10.2, 31.9] |
| Values that are the catch-all "Other" | 3,908 of 14,733, 26.5% | 133 of 7,300, 1.8% |

- Country of origin is the clearest check, because a product cannot show where it was made: only
  the listing can say. The check reads the listing's text, not its photo, and accepts loose
  spellings of the United States, so it undercounts unsupported origins.
- The hand-label rows count specific values (not "Other") on attributes the labeler was shown, on
  the verified products each catalog gave a category (Taxon 48, Sonnet 50). The labeling page asked
  for only what the listing shows but left attributes optional, so a blank is not proof the listing
  leaves the value out: the blank share is an upper bound. Intervals resample products.
- Taxon states about twice as many values as Sonnet, and over a quarter of them are "Other", which
  tells a shopper little.

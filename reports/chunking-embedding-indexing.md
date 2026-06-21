# Report: Chunking, Embedding, and Vector Store Indexing

## 1. Sampling strategy

The full Task 1 output (`data/processed/filtered_complaints.csv`) is far larger than
needed to develop and validate a chunking/embedding pipeline, so Task 2 works
from a **stratified sample of 12,000 complaints** (the midpoint of the
10,000–15,000 range).

**Why stratified, and why proportional:** CFPB complaint volume is heavily
skewed toward certain products (Credit Card complaints substantially
outnumber Savings Account complaints, for instance). A simple random sample
of 12,000 rows would inherit that skew, but a _proportional stratified_
sample preserves each product's true share of the data while still letting
every category scale down together — so the sample stays representative of
real complaint volume rather than artificially equalizing categories (which
would misrepresent how common each issue actually is).

**How the allocation is computed** (`stratified_sample()`):

1. Compute each category's share of the full filtered dataset.
2. Multiply by the target sample size (12,000) to get a raw allocation per
   category.
3. Floor each allocation, then distribute the leftover slots (from
   rounding) to the categories with the largest fractional remainder —
   the _largest-remainder method_ — so the allocations sum to exactly
   12,000 instead of drifting off by a few rows due to independent rounding.
4. Sample without replacement within each category at a fixed
   `random_state=42` for reproducibility.
5. If a category has fewer rows available than its allocation (only
   possible for a very small/rare category), take all available rows for
   it and print a warning rather than erroring out.

## 2. Chunking approach

**Why chunk at all:** consumer narratives vary enormously in length (Task 1
EDA showed a right-skewed distribution with a meaningful tail of long
entries). Embedding an entire long narrative as one vector blurs together
multiple distinct issues a consumer raised, which hurts retrieval precision
— a question about one specific issue may fail to surface a narrative whose
single embedding is dominated by a different issue mentioned in the same
text. Chunking lets each embedded vector represent a more focused span of
text.

**Method:** `langchain_text_splitters.RecursiveCharacterTextSplitter`,
which tries to split on paragraph breaks first, then sentences, then words,
only falling back to a hard character cut if nothing else fits — this keeps
chunks from breaking mid-sentence whenever possible.

**Chunk size/overlap experimentation:** `compare_chunk_configs()` runs the
splitter over a probe sample at three settings and reports the resulting
chunk count and average chunk length:

| chunk_size | chunk_overlap | chunks/complaint (typical) |     avg chunk length |
| ---------: | ------------: | -------------------------: | -------------------: |
|        300 |            30 |                       ~2.0 | shorter, more chunks |
|        500 |            50 |                       ~1.4 |             balanced |
|        800 |            80 |                       ~1.0 | longer, fewer chunks |

(Exact numbers depend on the sample drawn each run — the script prints the
live comparison every time it executes.)

**Final choice: `chunk_size=500`, `chunk_overlap=50`.**

- 300 chars over-fragments shorter narratives (median complaint narrative
  length from Task 1 EDA was well under 500 characters), creating many
  near-duplicate chunks per complaint and inflating the index without
  adding retrieval value.
- 800 chars under-chunks the long tail, re-creating the "one vector, many
  topics" problem chunking is meant to solve.
- 500/50 keeps most short-to-medium narratives as a single, complete chunk
  while still splitting the longer tail into focused pieces, and the 10%
  overlap reduces the chance that a sentence relevant to a question gets
  cut in half at a chunk boundary.
- It also matches the chunk size/overlap of the **pre-built vector store**
  provided for Tasks 3–4, so chunk granularity stays consistent across the
  whole project regardless of which store is used downstream.

## 3. Embedding model choice

**Model: `all-MiniLM-L6-v2`** (384 dimensions) — matching the exact name used
in the assignment's pre-built vector store spec. Note this is functionally
identical to `sentence-transformers/all-MiniLM-L6-v2`: the `sentence-transformers`
library auto-prepends the `sentence-transformers` org name to any model
string with no `/` in it, so both forms resolve to the same Hugging Face
repo and load identical weights.

- It's the model already used to build the pre-built vector store supplied
  for Tasks 3–4, so using it here too means the sample-based index built in
  this task and the full pre-built index are embedded in the same vector
  space — directly comparable, and either can be swapped in for the RAG
  pipeline without re-embedding.
- It's small (~80MB) and fast enough to embed tens of thousands of chunks
  on CPU in a reasonable time, which matters since trainees aren't assumed
  to have GPU access for this task.
- It's trained specifically for semantic similarity / sentence-embedding
  tasks (contrastive training over paraphrase and NLI-style data), which is
  exactly the retrieval use case here, as opposed to a general-purpose
  language model embedding that isn't optimized for nearest-neighbor
  similarity search.
- It has a strong track record as a default baseline for semantic search
  and RAG retrieval, trading a small amount of accuracy (versus larger
  embedding models) for substantially lower compute cost and latency —
  a reasonable tradeoff for an internal tool answering complaint questions
  in seconds.

Embeddings are L2-normalized at encoding time (`normalize_embeddings=True`),
so cosine similarity and dot-product similarity searches in the vector
store are equivalent.

## 4. Vector store

**ChromaDB** (`chromadb.PersistentClient`), persisted to `vector_store/`,
for consistency with the pre-built store specified for Tasks 3–4 (also
ChromaDB). Each chunk is stored with:

- `ids`: `"{complaint_id}_{chunk_index}"` (unique per chunk, traceable back
  to the source complaint)
- `embeddings`: the 384-dim normalized vector
- `documents`: the chunk's cleaned text
- `metadatas`: `complaint_id`, `product_category`, `product`, `issue`,
  `sub_issue`, `company`, `state`, `date_received`, `chunk_index`,
  `total_chunks` — matching the metadata schema of the pre-built store, with
  numeric fields stored as actual `int`/`float` types (not strings) so they
  remain filterable in Chroma's `where` queries.

## How to reproduce

```bash
python src/build_pipeline.py
```

Optional flags (defaults shown):

```bash
python src/build_pipeline.py \
  --input data/filtered_complaints.csv \
  --sample-size 12000 \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --embedding-model all-MiniLM-L6-v2 \
  --vector-store-dir vector_store \
  --collection-name complaint_chunks
```

Outputs:

- `data/processed/sampled_complaints.csv` — the 12,000-row stratified sample
- `data/processed/chunks_metadata.csv` — every chunk with its metadata (for
  inspection/debugging without querying Chroma directly)
- `vector_store/` — the persisted ChromaDB collection

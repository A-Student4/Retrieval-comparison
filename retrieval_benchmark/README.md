# FastAPI Documentation Retrieval Benchmark

**Claim**: For FAQ-style retrieval over the FastAPI documentation corpus,
dense (LSA) retrieval has the highest Recall@5, but Hybrid (RRF) has the
highest MRR@5. The choice depends on whether you optimise for finding the
right page at all, or for ranking it first.

---

## Setup

```bash
make run
```

This runs `pip install -r requirements.txt`, then `python build_corpus.py`,
then `python evaluate.py`. All outputs go to `data/` and `results/`.

**Requirements**: Python 3.10+, internet access to raw.githubusercontent.com.

---

## Corpus

**Source**: FastAPI official documentation, fetched from GitHub raw content
(`raw.githubusercontent.com/tiangolo/fastapi/master/`), pinned to the master
branch at time of evaluation.

**Why FastAPI docs**: Three reasons. First, the vocabulary gap between how
developers phrase questions (Stack Overflow natural language) and how
documentation is written (technical API prose) creates genuine retrieval
challenge — the same concept often has multiple surface forms. Second, I
have enough domain knowledge to write and verify 20 labelled queries without
hesitation, which is the binding constraint on ground truth quality under a
two-day deadline. Third, the query set comes from real Stack Overflow
questions rather than synthetically constructed ones, removing the risk of
unconsciously writing queries that favour the configuration I expected to win.

**Corpus stats**:
- 56 documentation pages (tutorial, advanced, deployment, core)
- 2,095 retrieval chunks after paragraph-level chunking
- Chunk length: min 59 chars, mean 268 chars, max 1,482 chars

**Chunking strategy**: Paragraph-level split on double newlines, with short
paragraphs (<80 chars) merged into the next, long paragraphs (>600 chars)
split at sentence boundaries, and one-sentence lookahead overlap between
adjacent chunks. This preserves semantic units (a complete explanation, a code
example with its description) rather than splitting mid-concept. Fixed-size
character splitting was explicitly rejected: FastAPI docs mix short code
examples with long prose explanations, and character boundaries would
frequently split a code snippet from its explanation.

---

## Queries

20 real Stack Overflow questions (top-voted FastAPI questions, fetched from
the Stack Exchange API), stratified across four retrieval difficulty categories:

| Category | n | Description |
|---|---|---|
| Lexical | 5 | Query vocabulary overlaps heavily with doc vocabulary |
| Mixed | 5 | Requires both term matching and semantic bridging |
| Semantic | 5 | Query vocabulary diverges significantly from docs |
| Hard | 5 | Deliberate maximum vocabulary mismatch |

Ground truth: each query is labelled with 1-2 relevant `doc_id`s (the
documentation pages that answer the question). A retrieved chunk is relevant
if its source page is in the relevant set. Labels were verified by reading
both the question and the documentation page — not inferred algorithmically.

---

## Configurations

### BM25

`BM25Okapi` (rank-bm25 library) with whitespace+punctuation tokenisation and
lowercasing. Standard TF-IDF weighting with document length normalisation.

**Design choice**: Tokenizer preserves technical terms (`CORSMiddleware`,
`422`, `asyncio`) as single tokens rather than splitting on camelCase or
special characters. This matters because most meaningful FastAPI queries
contain specific API terms that should be matched exactly.

### Dense (LSA)

TF-IDF matrix → TruncatedSVD (300 dimensions) → L2-normalised dense vectors.
Cosine similarity via dot product on normalised vectors.

**Why LSA and not all-MiniLM-L6-v2**: HuggingFace was not reachable from this
environment. The originally planned model was `all-MiniLM-L6-v2` (384-dim
sentence transformer), which would produce stronger semantic generalisation
via subword tokenisation and pretraining across diverse corpora. LSA was
used as the dense alternative because it is architecturally distinct from
BM25 in the way that matters for this comparison: it produces dense vectors
in a latent semantic space, capturing co-occurrence patterns that pure term
matching misses. "Parallel" and "concurrent" cluster together in the LSA
space if they appear in similar document contexts; BM25 treats them as
unrelated tokens.

**Honest limitation of LSA vs neural embeddings**: LSA cannot generalise
across vocabulary gaps it has not seen in the corpus. A query like "purpose
of Uvicorn" fails under LSA because "purpose" doesn't co-occur strongly with
the `async` and `ASGI` terminology in the relevant deployment page. A
pretrained transformer would handle this through its pretraining signal. The
hard query failures (q19, q20) are where this limitation bites hardest.

**TF-IDF configuration**: `sublinear_tf=True` (better for technical docs where
term frequency within a chunk doesn't scale linearly with relevance),
`ngram_range=(1,2)` (captures compound technical terms like "background task",
"async def"), `min_df=2` (remove hapax legomena), `max_df=0.9` (remove
near-universal terms).

### Hybrid (RRF)

Reciprocal Rank Fusion combining BM25 and Dense rankings:

```
RRF(d) = 1/(k + rank_BM25(d)) + 1/(k + rank_Dense(d)), k=60
```

**Why RRF over score normalisation**: RRF requires no hyperparameter tuning
of the fusion weight. Score normalisation requires choosing a weight α for
the linear combination BM25 + α·Dense, which would need to be tuned on a
held-out set — we don't have one large enough to tune on without overfitting.
RRF's implicit weighting via rank position is robust to this and is the
standard choice for unsupervised hybrid retrieval.

**k=60** is the standard default from the original RRF paper (Cormack et al.,
2009). It controls how much weight rank-1 receives relative to rank-60 —
lower k = steeper drop-off.

---

---

## Results

```
Config       Recall@5      MRR@5     p95 (ms)
-----------------------------------------------------------------
BM25            0.400      0.360          6ms
Dense           0.450      0.313         30ms
Hybrid          0.400      0.396         35ms
Reranker        0.225      0.175         74ms  ← bonus attempt: worse than Hybrid
```

All configurations satisfy the p95 < 1,000ms constraint with large margin.

**Per-category Recall@5:**

```
Category            BM25       Dense      Hybrid    Reranker
------------------------------------------------------------
lexical            0.200       0.400       0.400       0.300
mixed              0.400       0.500       0.500       0.200
semantic           0.500       0.400       0.400       0.200
hard               0.500       0.500       0.300       0.200
```

---

## Analysis

### Dense wins on Recall@5

Dense (LSA) at 0.450 vs BM25 at 0.400. The most surprising finding is that
BM25 performs worst on **lexical** queries (0.200), where it is theoretically
strongest. Investigation of q3 ("How can I enable CORS in FastAPI") reveals
why: the tutorial__cors page ranks 7th under BM25, below tutorial__middleware
and deployment pages that mention CORS in passing. BM25's document length
normalisation creates noise when a key term (CORS) appears across many pages
at similar frequency. LSA's latent semantic space projects the CORS page
to a vector closest to the query's representation because of co-occurrence
with the specific CORS-explaining vocabulary on that page.

### Hybrid wins on MRR

Hybrid at 0.396 vs BM25 at 0.360 vs Dense at 0.313. When BM25 ranks the
correct page at position 1, RRF preserves that ranking by boosting any
document that both signals agree on. The combined signal reduces the noise
that causes Dense to sometimes rank correct results at position 3 or 4.

### A case where Hybrid hurts

**q16** ("FastAPI runs API calls in serial instead of parallel"): BM25
retrieves the `async` page at rank 3 (R@5=1.0) but Dense ranks it much
lower. RRF fuses rank-3 + rank-50+, pushing `async` below rank-5. This is
the canonical RRF failure mode: when one signal is correct and the other
strongly wrong, fusion hurts.

### The reranker: attempted, honest failure

The feature-based reranker regresses to 0.225 R@5 from 0.400 for Hybrid.
The failure mechanism is instructive.

The reranker applies six features to the top-20 Hybrid candidates: BM25/Dense
scores (normalised), query term coverage, term density, early-hit bonus, and
exact bigram match. It then re-ranks by weighted combination.

**Why it fails**: the coverage and density features are noisy proxies for
relevance in technical documentation. "How to add both file and JSON body in
a FastAPI POST request" produces high coverage scores against `tutorial__body`,
`tutorial__body-multiple-params`, and `tutorial__middleware` — all of which
contain "file", "JSON", "body", "POST" at high density. The correct answer
page (`tutorial__request-forms-and-files`) is concise and specific; it doesn't
repeat query terms as densely as pages that discuss body handling generally.
The reranker demotes the correct answer in favour of term-denser but less
relevant candidates.

BM25 corrects for this via IDF weighting and document length normalisation;
the raw coverage feature has no such correction. The reranker double-counts
lexical signals (BM25 + coverage are correlated), suppresses the Dense signal,
and introduces a length bias toward longer chunks that happen to contain query
terms.

**What would fix it**: a neural cross-encoder jointly encodes the full
(query, passage) pair — it doesn't count terms, it predicts relevance from
contextual interaction between query tokens and passage tokens. This is why
cross-encoders consistently outperform feature-based rerankers: the features
you hand-engineer are approximations of what attention learns from (query,
passage) pair supervision.

**The correct conclusion**: feature-based reranking over a hybrid first stage
is harder than it looks, and the naive feature set actively hurts. The
**recommended configuration is Hybrid (RRF) without reranking** for this
corpus and query distribution, with a cross-encoder reranker as the clear
next step once model downloads are available.

### Where every config still loses

**Hard queries with deep vocabulary mismatch or corpus gaps**:

- **q19** ("Pydantic enum does not get converted to string"): Answer is
  `use_enum_values=True`. The FastAPI docs don't deeply cover Pydantic enum
  internals — the answer is not well-represented in the corpus.
- **q20** ("read body as any valid json"): Answer is `Body()` with `Any` type.
  Vocabulary gap between "any valid json" and `Any` is too large for LSA
  without pretraining.
- **q11** ("Architecture Flask vs FastAPI"): "Architecture" appears nowhere
  in the docs in this framing. Answer lives in `async.md` but query vocabulary
  has zero overlap.

These are corpus and vocabulary-gap failures, not retrieval failures. A neural
model pretrained on broad technical text would close the vocabulary gap;
expanding the corpus to include Pydantic docs would close the corpus gap.

---

## What I'd do with another week

1. **Neural embeddings**: Deploy `all-MiniLM-L6-v2` with the asymmetric setup
   (RETRIEVAL_DOCUMENT at index time, RETRIEVAL_QUERY at search time). Expected:
   meaningful improvement on hard vocabulary-mismatch queries.

2. **Neural cross-encoder reranker**: `ms-marco-MiniLM-L6-v2` over Hybrid
   top-20 candidates. The feature-based reranker failed; a learned joint
   query-document scorer is the principled fix.

3. **Query expansion**: For BM25's CORS failure, expanding the query with
   related terms (middleware → CORS middleware) would likely fix the rank-7
   problem.

4. **Larger query set**: 20 queries is enough to distinguish configurations
   but too few for statistical significance. 100 queries would support p-value
   reporting on Recall@5 differences.

5. **Corpus expansion**: Add Pydantic docs and Starlette docs to close the
   corpus gaps that cause universal failures on q19 and others.

---

## File structure

```
retrieval_benchmark/
├── Makefile              # make run
├── README.md             # this file
├── requirements.txt
├── build_corpus.py       # fetch FastAPI docs, chunk, write data/corpus.jsonl
├── evaluate.py           # BM25 / Dense / Hybrid / Reranker + metrics
├── data/
│   ├── corpus.jsonl      # 2095 chunks (built by build_corpus.py)
│   └── queries.json      # 20 labelled queries with ground truth
└── results/
    └── results.json      # full per-query results
```

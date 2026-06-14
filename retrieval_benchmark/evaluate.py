"""
evaluate.py
Runs four retrieval configurations over the FastAPI docs corpus and reports
Recall@5, MRR, and p95 latency for each.

Configurations:
  1. BM25       — BM25Okapi lexical retrieval (rank_bm25)
  2. Dense      — LSA (TF-IDF + TruncatedSVD 300-dim) dense retrieval
  3. Hybrid     — Reciprocal Rank Fusion (RRF, k=60) combining BM25 + Dense
  4. Reranker   — Hybrid first stage + feature-based second-stage reranker

Usage: python evaluate.py
"""
import json
import math
import time
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ── constants ─────────────────────────────────────────────────────────────────

CORPUS_PATH   = Path("data/corpus.jsonl")
QUERIES_PATH  = Path("data/queries.json")
RESULTS_PATH  = Path("results/results.json")
K             = 5    # recall@K and MRR@K
RRF_K         = 60   # RRF constant — standard default, robust to rank noise
N_TIMING_RUNS = 10   # runs per query for latency measurement
LSA_DIMS      = 300  # latent semantic dimensions (standard for technical corpora)


# ── corpus loading ─────────────────────────────────────────────────────────────

def load_corpus() -> tuple[list[dict], list[str], list[str]]:
    """Returns (chunks, chunk_ids, texts)."""
    chunks = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts     = [c["text"]     for c in chunks]
    return chunks, chunk_ids, texts


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


# ── tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer with lowercasing.
    Keeps technical terms intact (e.g. CORSMiddleware, BM25, 422).
    """
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


# ── BM25 configuration ────────────────────────────────────────────────────────

class BM25Config:
    def __init__(self, texts: list[str]):
        print("Building BM25 index...")
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built over {len(texts)} chunks")

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns (chunk_idx, score) pairs sorted descending."""
        qtokens = tokenize(query)
        scores  = self.bm25.get_scores(qtokens)
        top_k   = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]


# ── Dense configuration ───────────────────────────────────────────────────────

class DenseConfig:
    """
    LSA (Latent Semantic Analysis) dense retrieval.

    Pipeline: TF-IDF → TruncatedSVD (300 dims) → L2-normalised dense vectors.
    Cosine similarity via dot product on normalised vectors.

    Design decision: LSA chosen as the dense configuration because:
      1. Network constraints in this environment block HuggingFace model downloads.
         In production the first-choice model would be all-MiniLM-L6-v2 or
         BAAI/bge-small-en-v1.5 for stronger semantic generalisation.
      2. LSA is architecturally distinct from BM25 in the way that matters for
         this comparison: it produces dense continuous vectors in a latent semantic
         space, capturing co-occurrence patterns that BM25's term-matching misses.
         'parallel' and 'concurrent' will cluster together in the LSA space if they
         appear in similar contexts across the corpus; BM25 treats them as unrelated.
      3. LSA is reproducible with no external dependencies and satisfies the p95
         latency constraint with room to spare (sub-10ms per query at this scale).

    The honest limitation vs neural embeddings: LSA cannot generalise across
    vocabulary gaps it hasn't seen in the corpus. A query using a term absent from
    all documents will produce a near-zero projection. Neural sentence-transformers
    handle this via subword tokenisation and pretraining on large corpora.

    n_components=300 is standard for LSA on technical corpora of this size.
    Higher dimensions capture more variance but with diminishing returns and
    slower SVD decomposition.
    """
    def __init__(self, texts: list[str]):
        print(f"Building LSA index (TF-IDF + TruncatedSVD, {LSA_DIMS} dims)...")
        t0 = time.time()

        # TF-IDF with sublinear TF scaling — better for technical docs
        # where term frequency within a chunk doesn't scale linearly with importance
        self.tfidf = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),      # unigrams + bigrams for technical compound terms
            min_df=2,                # ignore terms appearing in only 1 chunk
            max_df=0.9,              # ignore near-universal terms
        )
        X = self.tfidf.fit_transform(texts)
        print(f"  TF-IDF matrix: {X.shape} ({X.nnz} non-zeros)")

        # SVD projection to dense latent space
        self.svd = TruncatedSVD(n_components=LSA_DIMS, random_state=42)
        X_dense = self.svd.fit_transform(X)

        # L2-normalise for cosine similarity via dot product
        self.doc_embeddings = normalize(X_dense, norm="l2")

        elapsed = time.time() - t0
        print(f"  LSA index built in {elapsed:.1f}s, "
              f"explained variance: {self.svd.explained_variance_ratio_.sum():.3f}")

    def _embed_query(self, query: str) -> np.ndarray:
        """Project a query into the LSA space."""
        q_tfidf = self.tfidf.transform([query])
        q_lsa   = self.svd.transform(q_tfidf)
        return normalize(q_lsa, norm="l2")[0]

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns (chunk_idx, score) pairs sorted descending."""
        q_emb  = self._embed_query(query)
        scores = self.doc_embeddings @ q_emb   # cosine similarity
        top_k  = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_k]


# ── Hybrid (RRF) configuration ────────────────────────────────────────────────

class HybridConfig:
    """
    Reciprocal Rank Fusion combining BM25 and Dense rankings.

    RRF score for document d = Σ 1 / (k + rank_i(d))
    where k=60 (standard constant, robust to noise in low-ranked results).

    Design decision: RRF chosen over score normalisation and linear combination
    because RRF requires no hyperparameter tuning of the fusion weight — the
    relative weighting is implicit in rank position. This avoids overfitting
    to the query set while remaining deterministic and reproducible.
    """
    def __init__(self, bm25: BM25Config, dense: DenseConfig, n_docs: int):
        self.bm25   = bm25
        self.dense  = dense
        self.n_docs = n_docs

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns (chunk_idx, rrf_score) pairs sorted descending."""
        # Retrieve a larger candidate set to fuse over
        candidate_k = min(self.n_docs, max(100, k * 10))

        bm25_ranked  = self.bm25.retrieve(query,  candidate_k)
        dense_ranked = self.dense.retrieve(query, candidate_k)

        # Build rank dictionaries
        bm25_ranks  = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_ranked)}
        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_ranked)}

        # All candidate doc indices
        all_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        # RRF fusion
        rrf_scores = {}
        for idx in all_indices:
            r1 = bm25_ranks.get(idx,  candidate_k + 1)
            r2 = dense_ranks.get(idx, candidate_k + 1)
            rrf_scores[idx] = 1.0 / (RRF_K + r1) + 1.0 / (RRF_K + r2)

        top_k = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top_k


# ── Reranker configuration ────────────────────────────────────────────────────

class RerankerConfig:
    """
    Feature-based reranker: Hybrid first stage → feature scoring second stage.

    Takes the top-20 candidates from Hybrid (RRF) and re-scores each
    (query, passage) pair using features that first-stage retrieval computes
    only implicitly across the full corpus:

      1. BM25 score (normalised)         — first-stage lexical signal
      2. Dense score (normalised)        — first-stage semantic signal
      3. Query term coverage             — |Q ∩ D| / |Q| (did we find ALL terms?)
      4. Query term density              — term hits / passage length (specificity)
      5. Early-position bonus            — any query term in first 60 chars?
      6. Exact bigram match              — any 2-token query phrase appears verbatim?

    Features 3-6 are joint query-passage signals: they vary per (query, passage)
    pair, not just per passage. This is what distinguishes a reranker from a
    first-stage retriever — it conditions on the specific query when scoring each
    candidate, rather than scoring candidates independently against an indexed
    representation.

    Design note — why not a neural cross-encoder:
    The standard production reranker is a cross-encoder (e.g. ms-marco-MiniLM-L6-v2
    from HuggingFace), which encodes the full (query, passage) string jointly and
    produces a single relevance score. Cross-encoders consistently improve MRR by
    5-15 points over hybrid first stages on MS MARCO. They were unavailable here
    due to network restrictions on HuggingFace model downloads. The feature-based
    reranker is the pre-neural equivalent used in production ES/Solr deployments
    and is architecturally sound as a second stage.

    Candidate pool: top-20 from Hybrid. Expanding to top-50 improves recall
    ceiling at the cost of reranker latency; 20 is the standard starting point.
    """

    CANDIDATE_K = 20  # first-stage candidate pool size

    def __init__(self, hybrid: HybridConfig, bm25: BM25Config,
                 dense: DenseConfig, texts: list[str]):
        self.hybrid = hybrid
        self.bm25   = bm25
        self.dense  = dense
        self.texts  = texts
        print("Reranker ready (feature-based, Hybrid top-20 candidates)")

    @staticmethod
    def _norm_scores(scored: list[tuple[int, float]]) -> dict[int, float]:
        """Min-max normalise a scored list to [0, 1]."""
        if not scored:
            return {}
        vals = [s for _, s in scored]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {idx: 1.0 for idx, _ in scored}
        return {idx: (s - lo) / (hi - lo) for idx, s in scored}

    def _rerank_features(self, query: str, candidate_indices: list[int],
                         bm25_norm: dict, dense_norm: dict) -> list[tuple[int, float]]:
        """Compute feature scores for each (query, candidate) pair."""
        qtokens = set(tokenize(query))
        qbigrams = set()
        qtok_list = list(qtokens)
        for i in range(len(qtok_list) - 1):
            qbigrams.add(qtok_list[i] + " " + qtok_list[i + 1])

        scored = []
        for idx in candidate_indices:
            text = self.texts[idx]
            dtokens = tokenize(text)
            dset = set(dtokens)

            # Feature 1 & 2: normalised first-stage scores
            f_bm25  = bm25_norm.get(idx, 0.0)
            f_dense = dense_norm.get(idx, 0.0)

            # Feature 3: query term coverage — fraction of query tokens in passage
            f_coverage = len(qtokens & dset) / max(len(qtokens), 1)

            # Feature 4: query term density — query hits per 100 passage tokens
            hits = sum(1 for t in dtokens if t in qtokens)
            f_density = hits / max(len(dtokens), 1) * 100

            # Feature 5: early position bonus — any query term in first 60 chars
            f_early = float(any(t in tokenize(text[:60]) for t in qtokens))

            # Feature 6: exact bigram match in passage
            passage_text_lower = text.lower()
            f_bigram = float(any(bg in passage_text_lower for bg in qbigrams))

            # Weighted combination (weights tuned by intuition, not on held-out set)
            score = (
                0.25 * f_bm25     +
                0.25 * f_dense    +
                0.25 * f_coverage +
                0.10 * f_density  +
                0.10 * f_early    +
                0.05 * f_bigram
            )
            scored.append((idx, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Two-stage retrieval: Hybrid top-20 → feature reranker → top-k."""
        # Stage 1: Hybrid retrieval for candidate pool
        candidates = self.hybrid.retrieve(query, self.CANDIDATE_K)
        candidate_indices = [idx for idx, _ in candidates]

        # Get normalised first-stage scores for the candidate pool
        bm25_scored  = self.bm25.retrieve(query, self.CANDIDATE_K)
        dense_scored = self.dense.retrieve(query, self.CANDIDATE_K)
        bm25_norm    = self._norm_scores(bm25_scored)
        dense_norm   = self._norm_scores(dense_scored)

        # Stage 2: feature-based reranking
        reranked = self._rerank_features(query, candidate_indices,
                                         bm25_norm, dense_norm)
        return reranked[:k]


# ── relevance checking ────────────────────────────────────────────────────────

def is_relevant(chunk_id: str, query: dict) -> bool:
    """A chunk is relevant if it comes from any of the query's relevant docs."""
    doc_id = chunk_id.split("__chunk_")[0]
    return doc_id in query["relevant_docs"]


# ── metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[int], chunk_ids: list[str], query: dict, k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-k retrieved."""
    relevant_in_corpus = sum(
        1 for cid in chunk_ids
        if cid.split("__chunk_")[0] in query["relevant_docs"]
    )
    if relevant_in_corpus == 0:
        return 0.0
    # Count unique relevant doc_ids found in top-k
    found_docs = set()
    for idx in retrieved[:k]:
        cid = chunk_ids[idx]
        doc_id = cid.split("__chunk_")[0]
        if doc_id in query["relevant_docs"]:
            found_docs.add(doc_id)
    return len(found_docs) / len(query["relevant_docs"])


def mrr_at_k(retrieved: list[int], chunk_ids: list[str], query: dict, k: int) -> float:
    """MRR@K: 1/rank of the first relevant result."""
    for rank, idx in enumerate(retrieved[:k], start=1):
        if is_relevant(chunk_ids[idx], query):
            return 1.0 / rank
    return 0.0


def p95_latency(retrieve_fn, query_text: str, k: int, n_runs: int) -> float:
    """Measure p95 latency in milliseconds over n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        retrieve_fn(query_text, k)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    idx = math.ceil(0.95 * len(times)) - 1
    return times[max(0, idx)]


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate_config(
    name: str,
    retrieve_fn,
    queries: list[dict],
    chunk_ids: list[str],
    k: int,
) -> dict:
    """Run all queries through a retrieval config and collect metrics."""
    print(f"\nEvaluating {name}...")
    per_query = []

    for q in queries:
        # Timed retrieval
        t0 = time.perf_counter()
        results = retrieve_fn(q["text"], k)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_indices = [idx for idx, _ in results]
        rec  = recall_at_k(retrieved_indices, chunk_ids, q, k)
        mrr  = mrr_at_k(retrieved_indices, chunk_ids, q, k)

        # Measure p95 over multiple runs
        p95  = p95_latency(retrieve_fn, q["text"], k, N_TIMING_RUNS)

        top5_docs = list(dict.fromkeys([
            chunk_ids[idx].split("__chunk_")[0]
            for idx, _ in results[:5]
        ]))

        per_query.append({
            "qid":       q["qid"],
            "text":      q["text"],
            "category":  q["category"],
            "recall":    round(rec, 3),
            "mrr":       round(mrr, 3),
            "latency_ms": round(latency_ms, 2),
            "p95_ms":    round(p95, 2),
            "top5_docs": top5_docs,
            "relevant":  q["relevant_docs"],
        })

        status = "✓" if rec > 0 else "✗"
        print(f"  {status} {q['qid']:3} [{q['category']:8}]  "
              f"R@5={rec:.2f}  MRR={mrr:.2f}  p95={p95:.0f}ms  "
              f"| {q['text'][:55]}")

    # Aggregate
    recalls   = [r["recall"]  for r in per_query]
    mrrs      = [r["mrr"]     for r in per_query]
    p95s      = [r["p95_ms"]  for r in per_query]

    # Per-category breakdown
    categories = ["lexical", "mixed", "semantic", "hard"]
    cat_metrics = {}
    for cat in categories:
        cat_qs = [r for r in per_query if r["category"] == cat]
        if cat_qs:
            cat_metrics[cat] = {
                "n":      len(cat_qs),
                "recall": round(sum(r["recall"] for r in cat_qs) / len(cat_qs), 3),
                "mrr":    round(sum(r["mrr"]    for r in cat_qs) / len(cat_qs), 3),
            }

    agg = {
        "config":       name,
        "n_queries":    len(per_query),
        "recall_at_5":  round(sum(recalls) / len(recalls), 3),
        "mrr_at_5":     round(sum(mrrs)    / len(mrrs),    3),
        "p95_ms":       round(sorted(p95s)[math.ceil(0.95 * len(p95s)) - 1], 2),
        "by_category":  cat_metrics,
        "per_query":    per_query,
    }

    print(f"\n  → {name}: R@5={agg['recall_at_5']:.3f}  "
          f"MRR={agg['mrr_at_5']:.3f}  p95={agg['p95_ms']:.0f}ms")
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Ensure results dir exists
    Path("results").mkdir(exist_ok=True)

    # Load data
    print("Loading corpus...")
    chunks, chunk_ids, texts = load_corpus()
    print(f"  {len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} pages")

    queries = load_queries()
    print(f"  {len(queries)} queries loaded")

    # Build indexes
    bm25_cfg    = BM25Config(texts)
    dense_cfg   = DenseConfig(texts)
    hybrid_cfg  = HybridConfig(bm25_cfg, dense_cfg, len(chunks))
    reranker_cfg = RerankerConfig(hybrid_cfg, bm25_cfg, dense_cfg, texts)

    # Evaluate all four
    results = []
    results.append(evaluate_config("BM25",     bm25_cfg.retrieve,     queries, chunk_ids, K))
    results.append(evaluate_config("Dense",    dense_cfg.retrieve,    queries, chunk_ids, K))
    results.append(evaluate_config("Hybrid",   hybrid_cfg.retrieve,   queries, chunk_ids, K))
    results.append(evaluate_config("Reranker", reranker_cfg.retrieve, queries, chunk_ids, K))

    # Summary table
    print("\n" + "="*65)
    print(f"{'Config':<10} {'Recall@5':>10} {'MRR@5':>10} {'p95 (ms)':>12}")
    print("-"*65)
    for r in results:
        print(f"{r['config']:<10} {r['recall_at_5']:>10.3f} "
              f"{r['mrr_at_5']:>10.3f} {r['p95_ms']:>12.1f}")
    print("="*65)

    # Per-category breakdown
    print("\nPer-category Recall@5:")
    categories = ["lexical", "mixed", "semantic", "hard"]
    header = f"{'Category':<12}" + "".join(f"{r['config']:>12}" for r in results)
    print(header)
    print("-" * (12 + 12 * len(results)))
    for cat in categories:
        row = f"{cat:<12}"
        for r in results:
            val = r["by_category"].get(cat, {}).get("recall", 0.0)
            row += f"{val:>12.3f}"
        print(row)

    # Save full results
    output = {
        "corpus_stats": {
            "n_chunks": len(chunks),
            "n_pages":  len(set(c["doc_id"] for c in chunks)),
            "source":   "FastAPI official documentation (GitHub raw, master branch)",
        },
        "config_results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

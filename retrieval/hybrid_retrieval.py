"""
Hybrid retrieval: fuse BM25 (sparse) and dense (fine-tuned bi-encoder) rankings
with Reciprocal Rank Fusion, then rerank the fused candidates with a
cross-encoder for the final top-k.

Why each stage exists:
  - BM25 catches exact keyword/entity matches dense embeddings sometimes miss
    (numbers, rare proper nouns, negations).
  - Dense catches semantic paraphrase matches BM25 misses entirely.
  - RRF fusion is rank-based, not score-based, so it needs no score
    normalization between two very differently-scaled retrievers (BM25 scores
    and cosine similarities aren't comparable) — this is why RRF is the
    standard choice for hybrid search over ad-hoc score averaging.
  - The cross-encoder sees the (claim, candidate) pair jointly through the
    model, rather than comparing two independently-computed embeddings —
    that joint attention is strictly more expressive and typically gives a
    meaningful nDCG bump over bi-encoder + fusion alone, at the cost of being
    too slow to run over the whole corpus (hence: only on the fused top-N).

Usage:
    python retrieval/hybrid_retrieval.py \
        --processed data/processed \
        --dense_model models/verifact-biencoder-v1 \
        --pool_k 100 --fused_top_n 30 --final_k 10
"""
import argparse
import json
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(path):
    doc_ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_ids.append(row["doc_id"])
            texts.append(row["text"])
    return doc_ids, texts


def load_claims_with_qrels(claims_path, qrels_path):
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["claim_id"]] = set(row["relevant_doc_ids"])
    claims = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["claim_id"] in qrels:
                claims.append(row)
    return claims, qrels


def bm25_search_all(claims, doc_ids, texts, k):
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    results = {}
    for c in claims:
        scores = bm25.get_scores(tokenize(c["claim"]))
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:k]
        results[c["claim_id"]] = [d for d, _ in ranked]
    return results


def dense_search_all(claims, doc_ids, texts, model, k):
    corpus_emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=True,
                               batch_size=64, normalize_embeddings=True)
    claim_texts = [c["claim"] for c in claims]
    claim_emb = model.encode(claim_texts, convert_to_tensor=True, show_progress_bar=True,
                              batch_size=64, normalize_embeddings=True)
    hits = util.semantic_search(claim_emb, corpus_emb, top_k=k)
    results = {}
    for c, claim_hits in zip(claims, hits):
        results[c["claim_id"]] = [doc_ids[h["corpus_id"]] for h in claim_hits]
    return results


def reciprocal_rank_fusion(ranked_lists: list, rrf_k: int = 60, top_n: int = 30):
    """
    ranked_lists: list of {claim_id: [doc_id, ...]} dicts, one per retriever,
    each already sorted best-first.

    RRF score for a doc = sum over retrievers of 1 / (rrf_k + rank),
    where rank is 1-indexed and a doc missing from a retriever's list
    simply contributes 0 from that retriever. rrf_k=60 is the standard
    default from the original RRF paper (Cormack et al.) — it discounts
    rank differences at the bottom of the list more than at the top.

    Returns: {claim_id: [doc_id, ...]} fused ranking, truncated to top_n.
    """
    all_claim_ids = set()
    for rl in ranked_lists:
        all_claim_ids.update(rl.keys())

    fused = {}
    for claim_id in all_claim_ids:
        scores = defaultdict(float)
        for rl in ranked_lists:
            for rank, doc_id in enumerate(rl.get(claim_id, []), start=1):
                scores[doc_id] += 1.0 / (rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        fused[claim_id] = [d for d, _ in ranked]
    return fused


def rerank_with_cross_encoder(claims, fused_results, corpus_lookup, model, final_k):
    results = {}
    for c in claims:
        candidate_ids = fused_results.get(c["claim_id"], [])
        if not candidate_ids:
            results[c["claim_id"]] = []
            continue
        pairs = [[c["claim"], corpus_lookup[d]] for d in candidate_ids]
        scores = model.predict(pairs)
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        results[c["claim_id"]] = [d for d, _ in reranked[:final_k]]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--dense_model", default="models/verifact-biencoder-v1")
    parser.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--pool_k", type=int, default=100,
                         help="how many candidates each of BM25/dense contributes before fusion")
    parser.add_argument("--fused_top_n", type=int, default=30,
                         help="how many fused candidates get passed to the (slow) cross-encoder")
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--out", default="retrieval/hybrid_reranked_results.json")
    args = parser.parse_args()

    doc_ids, texts = load_corpus(f"{args.processed}/corpus.jsonl")
    corpus_lookup = dict(zip(doc_ids, texts))
    claims, qrels = load_claims_with_qrels(
        f"{args.processed}/claims.jsonl", f"{args.processed}/qrels.jsonl"
    )
    print(f"Corpus: {len(doc_ids)} | Claims: {len(claims)}")

    print("Running BM25 ...")
    bm25_results = bm25_search_all(claims, doc_ids, texts, k=args.pool_k)

    print(f"Running dense retrieval with {args.dense_model} ...")
    dense_model = SentenceTransformer(args.dense_model)
    dense_results = dense_search_all(claims, doc_ids, texts, dense_model, k=args.pool_k)

    print("Fusing with Reciprocal Rank Fusion ...")
    fused = reciprocal_rank_fusion([bm25_results, dense_results],
                                    rrf_k=args.rrf_k, top_n=args.fused_top_n)

    print(f"Reranking fused candidates with cross-encoder {args.cross_encoder} ...")
    cross_encoder = CrossEncoder(args.cross_encoder)
    final_results = rerank_with_cross_encoder(claims, fused, corpus_lookup,
                                               cross_encoder, final_k=args.final_k)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final_results, f)

    print(f"Saved {len(final_results)} ranked result lists to {args.out}")
    print(f"Run: python eval/run_eval.py --results {args.out} "
          f"--qrels {args.processed}/qrels.jsonl --name hybrid_reranked")


if __name__ == "__main__":
    main()

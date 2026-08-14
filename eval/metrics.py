"""
Retrieval evaluation metrics: Recall@k, MRR, nDCG@k.

Kept dependency-free and generic (BEIR-style) so it works identically for
BM25, dense, hybrid, and reranked results — you just pass in a different
`results` dict each time.

results: {query_id: [ranked doc_id, ...]}   (already sorted, best first)
qrels:   {query_id: set(relevant doc_id)}
"""
import math


def recall_at_k(results: dict, qrels: dict, k: int) -> float:
    scores = []
    for qid, relevant in qrels.items():
        if not relevant or qid not in results:
            continue
        retrieved_k = set(results[qid][:k])
        hit = len(retrieved_k & relevant)
        scores.append(hit / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def mrr(results: dict, qrels: dict, k: int = 10) -> float:
    scores = []
    for qid, relevant in qrels.items():
        if not relevant or qid not in results:
            continue
        rr = 0.0
        for rank, doc_id in enumerate(results[qid][:k], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(results: dict, qrels: dict, k: int = 10) -> float:
    scores = []
    for qid, relevant in qrels.items():
        if not relevant or qid not in results:
            continue
        retrieved_k = results[qid][:k]
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(retrieved_k, start=1)
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(results: dict, qrels: dict, k_values=(1, 5, 10)) -> dict:
    out = {}
    for k in k_values:
        out[f"Recall@{k}"] = round(recall_at_k(results, qrels, k), 4)
        out[f"nDCG@{k}"] = round(ndcg_at_k(results, qrels, k), 4)
    out["MRR@10"] = round(mrr(results, qrels, k=10), 4)
    return out

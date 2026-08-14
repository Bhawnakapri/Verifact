"""
BM25 baseline retriever over the evidence corpus.

This is your Week 1 control group — every later retriever (fine-tuned dense,
hybrid, +reranker) gets compared against these numbers.

Usage:
    python retrieval/bm25_baseline.py --processed data/processed --k 10
"""
import argparse
import json
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, corpus: dict[str, str]):
        """corpus: doc_id -> text"""
        self.doc_ids = list(corpus.keys())
        self.texts = [corpus[d] for d in self.doc_ids]
        tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)[:k]
        return [doc_id for doc_id, _ in ranked]


def load_corpus(path: str) -> dict:
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["doc_id"]] = row["text"]
    return corpus


def load_claims_with_qrels(claims_path: str, qrels_path: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    corpus = load_corpus(f"{args.processed}/corpus.jsonl")
    claims, qrels = load_claims_with_qrels(
        f"{args.processed}/claims.jsonl", f"{args.processed}/qrels.jsonl"
    )

    print(f"Corpus size: {len(corpus)} | Claims with evidence: {len(claims)}")
    retriever = BM25Retriever(corpus)

    # Save results in the generic format eval/metrics.py expects:
    # {claim_id: [ranked doc_id, ...]}
    results = {}
    for c in claims:
        results[c["claim_id"]] = retriever.search(c["claim"], k=args.k)

    with open("retrieval/bm25_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} ranked result lists to retrieval/bm25_results.json")
    print("Run: python eval/run_eval.py --results retrieval/bm25_results.json "
          f"--qrels {args.processed}/qrels.jsonl --name BM25")


if __name__ == "__main__":
    main()

"""
Mine hard negatives for contrastive fine-tuning.

Why hard negatives matter: if you train with random negatives, the model
only learns "claim about coffee" vs "claim about the Eiffel Tower" — trivial
lexical separation. Hard negatives (sentences BM25 thinks are relevant but
aren't) force the model to learn the fine-grained semantic distinction that
actually matters for retrieval quality — e.g. "coffee correlates with lower
Parkinson's risk" vs "a trial found no link between coffee and Parkinson's."

Strategy:
  For each claim, run BM25, take its top-N results, remove any that are
  true positives (from qrels), and keep the remaining top-ranked ones as
  hard negatives — these are exactly the sentences a weak retriever
  confuses with the right answer.

Output: triplets.jsonl — {claim, positive, hard_negative}
One row per (claim, positive) pair that has at least one mined hard negative.
Claims with multiple gold evidence sentences produce multiple triplets.

Usage:
    python retrieval/mine_hard_negatives.py --processed data/processed \
        --bm25_pool 50 --out data/processed/triplets.jsonl
"""
import argparse
import json
import random
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_corpus(path):
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["doc_id"]] = row["text"]
    return corpus


def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["claim_id"]] = set(row["relevant_doc_ids"])
    return qrels


def load_claims(path, qrels):
    claims = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["claim_id"] in qrels:
                claims[row["claim_id"]] = row["claim"]
    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--bm25_pool", type=int, default=50,
                         help="how deep into BM25 results to search for hard negatives")
    parser.add_argument("--negatives_per_positive", type=int, default=1)
    parser.add_argument("--out", default="data/processed/triplets.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    corpus = load_corpus(f"{args.processed}/corpus.jsonl")
    qrels = load_qrels(f"{args.processed}/qrels.jsonl")
    claims = load_claims(f"{args.processed}/claims.jsonl", qrels)

    doc_ids = list(corpus.keys())
    tokenized = [tokenize(corpus[d]) for d in doc_ids]
    bm25 = BM25Okapi(tokenized)

    triplets = []
    no_hard_neg_count = 0

    for claim_id, claim_text in claims.items():
        positives = qrels[claim_id]
        if not positives:
            continue

        scores = bm25.get_scores(tokenize(claim_text))
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        pool = [d for d, _ in ranked[: args.bm25_pool] if d not in positives]

        if not pool:
            no_hard_neg_count += 1
            continue

        for pos_id in positives:
            negs = pool[: args.negatives_per_positive]
            for neg_id in negs:
                triplets.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "positive": corpus[pos_id],
                    "hard_negative": corpus[neg_id],
                })

    with open(args.out, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Claims processed: {len(claims)}")
    print(f"Claims with no hard negative found (BM25 pool too shallow or corpus too small): {no_hard_neg_count}")
    print(f"Triplets written: {len(triplets)} -> {args.out}")


if __name__ == "__main__":
    main()

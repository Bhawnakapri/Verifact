"""
Dense retrieval using a (fine-tuned) sentence-transformer.

Encodes the full corpus once, then for each claim does a brute-force
cosine-similarity search. Fine at FEVER-subset scale (tens of thousands
of sentences); Week 3 swaps this brute-force step for a FAISS index when
the corpus grows and hybrid (BM25 + dense) retrieval is introduced.

Usage:
    # off-the-shelf baseline (compare vs your fine-tuned model):
    python retrieval/dense_retrieval.py --model sentence-transformers/all-MiniLM-L6-v2 \
        --processed data/processed --out retrieval/dense_pretrained_results.json

    # your fine-tuned model:
    python retrieval/dense_retrieval.py --model models/verifact-biencoder-v1 \
        --processed data/processed --out retrieval/dense_finetuned_results.json
"""
import argparse
import json

from sentence_transformers import SentenceTransformer, util


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         help="HF model name or local path to a fine-tuned model")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default="retrieval/dense_results.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    doc_ids, texts = load_corpus(f"{args.processed}/corpus.jsonl")
    claims, qrels = load_claims_with_qrels(
        f"{args.processed}/claims.jsonl", f"{args.processed}/qrels.jsonl"
    )
    print(f"Corpus size: {len(doc_ids)} | Claims with evidence: {len(claims)}")

    print("Encoding corpus ...")
    corpus_emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=True,
                               batch_size=64, normalize_embeddings=True)

    print("Encoding claims and searching ...")
    claim_texts = [c["claim"] for c in claims]
    claim_emb = model.encode(claim_texts, convert_to_tensor=True, show_progress_bar=True,
                              batch_size=64, normalize_embeddings=True)

    hits = util.semantic_search(claim_emb, corpus_emb, top_k=args.k)

    results = {}
    for claim, claim_hits in zip(claims, hits):
        results[claim["claim_id"]] = [doc_ids[h["corpus_id"]] for h in claim_hits]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} ranked result lists to {args.out}")
    print(f"Run: python eval/run_eval.py --results {args.out} "
          f"--qrels {args.processed}/qrels.jsonl --name {args.model.split('/')[-1]}")


if __name__ == "__main__":
    main()

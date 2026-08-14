"""
Turn raw FEVER JSONL (claim, label, evidence sentences) into:

  1. corpus.jsonl        — every unique evidence sentence, given a stable doc_id
  2. qrels.jsonl         — for each claim: which doc_ids are relevant (for eval)
  3. claims.jsonl        — claim text + label, ready for retrieval + NLI stages

This mirrors how BEIR-style benchmarks are structured (corpus / queries / qrels),
so your eval harness in eval/metrics.py can stay generic.

Usage:
    python data/preprocess.py --split train --raw data/raw --out data/processed
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict


def doc_id_for(sentence: str) -> str:
    """Stable content-hash ID so identical evidence sentences across claims
    collapse to a single corpus entry instead of being duplicated."""
    return hashlib.sha1(sentence.strip().encode("utf-8")).hexdigest()[:16]


def flatten_sentences(evidence):
    """The `copenlu/fever_gold_evidence` evidence field is a list of
    [page_title, sentence_id, sentence_text] triplets, e.g.:
        [["Shingles", "31", "The number of new cases per year ranges..."]]
    We only want the sentence_text (index 2) from each triplet — the page
    title and sentence id are metadata, not evidence text, and must be
    discarded rather than treated as separate sentences."""
    out = []
    if not evidence:
        return out
    for item in evidence:
        if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], str):
            s = item[2].strip()
            if s:
                out.append(s)
        elif isinstance(item, str):
            # Fallback in case some rows are already flat strings
            s = item.strip()
            if s:
                out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    in_path = os.path.join(args.raw, f"fever_{args.split}.jsonl")

    corpus = {}          # doc_id -> sentence text
    qrels = defaultdict(list)   # claim_id -> [doc_id, ...]
    claims_out = []
    skipped_nei = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            # NOT ENOUGH INFO claims have no gold evidence — keep them for the
            # NLI stage (Week 3) but they contribute no retrieval qrels.
            if row["label"] == "NOT ENOUGH INFO" or not row["evidence"]:
                skipped_nei += 1
                claims_out.append(row)
                continue

            for sent in flatten_sentences(row["evidence"]):
                did = doc_id_for(sent)
                corpus[did] = sent
                qrels[row["claim_id"]].append(did)

            claims_out.append(row)

    with open(os.path.join(args.out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for did, text in corpus.items():
            f.write(json.dumps({"doc_id": did, "text": text}, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out, "qrels.jsonl"), "w", encoding="utf-8") as f:
        for claim_id, doc_ids in qrels.items():
            f.write(json.dumps({"claim_id": claim_id, "relevant_doc_ids": doc_ids}) + "\n")

    with open(os.path.join(args.out, "claims.jsonl"), "w", encoding="utf-8") as f:
        for row in claims_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Corpus: {len(corpus)} unique evidence sentences")
    print(f"Claims with qrels (SUPPORTS/REFUTES): {len(qrels)}")
    print(f"Claims skipped for retrieval (NOT ENOUGH INFO, kept for NLI): {skipped_nei}")
    print(f"Wrote corpus.jsonl, qrels.jsonl, claims.jsonl to {args.out}")


if __name__ == "__main__":
    main()
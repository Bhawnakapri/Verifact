"""
Build (claim, evidence_sentence, label) pairs for NLI-style stance classification
from FEVER data — the format most NLI transformers expect.

Label mapping (FEVER -> standard NLI convention):
    SUPPORTS         -> entailment     (evidence entails the claim)
    REFUTES          -> contradiction  (evidence contradicts the claim)
    NOT ENOUGH INFO   -> neutral        (no gold evidence exists for this claim)

For SUPPORTS/REFUTES we have real gold evidence sentences straight from FEVER.
For NOT ENOUGH INFO there is no gold evidence by definition — but a stance
classifier still needs *some* sentence to look at, otherwise it can't learn
what "topically related but insufficient" looks like. We pair each NEI claim
with its top BM25 hit as weak-supervision context: it's topically close
(BM25 wouldn't retrieve something unrelated) but, by construction of the NEI
label, doesn't actually confirm or deny the claim. This mirrors how the
original FEVER shared-task baselines handled NEI training data.

Usage:
    python nli/build_nli_pairs.py --processed data/processed --out data/processed/nli_pairs.jsonl
"""
import argparse
import json
import re

from rank_bm25 import BM25Okapi

LABEL_MAP = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral",
}


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def flatten_sentences(evidence):
    """The `copenlu/fever_gold_evidence` evidence field is a list of
    [page_title, sentence_id, sentence_text] triplets. We only want the
    sentence_text (index 2) from each triplet."""
    out = []
    if not evidence:
        return out
    for item in evidence:
        if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], str):
            s = item[2].strip()
            if s:
                out.append(s)
        elif isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def load_corpus(path):
    doc_ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_ids.append(row["doc_id"])
            texts.append(row["text"])
    return doc_ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--out", default="data/processed/nli_pairs.jsonl")
    args = parser.parse_args()

    doc_ids, texts = load_corpus(f"{args.processed}/corpus.jsonl")
    bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    pairs = []
    skipped_nei_no_corpus = 0

    with open(f"{args.processed}/claims.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            label = LABEL_MAP.get(row["label"])
            if label is None:
                continue

            if label in ("entailment", "contradiction"):
                for sent in flatten_sentences(row.get("evidence")):
                    if sent != "dummy":
                        pairs.append({"claim": row["claim"], "evidence": sent, "label": label})
            else:  # neutral (NOT ENOUGH INFO)
                if bm25 is None:
                    skipped_nei_no_corpus += 1
                    continue
                scores = bm25.get_scores(tokenize(row["claim"]))
                top_idx = max(range(len(scores)), key=lambda i: scores[i])
                pairs.append({"claim": row["claim"], "evidence": texts[top_idx], "label": label})

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    counts = Counter(p["label"] for p in pairs)
    print(f"NLI pairs written: {len(pairs)} -> {args.out}")
    print(f"Label distribution: {dict(counts)}")
    if skipped_nei_no_corpus:
        print(f"Skipped {skipped_nei_no_corpus} NEI claims (empty corpus, no BM25 index available)")


if __name__ == "__main__":
    main()
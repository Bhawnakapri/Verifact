"""
Evaluate a retriever's saved results against qrels and print a table.

Run this once per retriever (BM25, dense, hybrid, +reranker) across the
month — append each run's row to results_log.csv so by Week 4 you have
the full comparison table for your README/resume.

Usage:
    python eval/run_eval.py --results retrieval/bm25_results.json \
                             --qrels data/processed/qrels.jsonl \
                             --name BM25
"""
import argparse
import csv
import json
import os

from metrics import evaluate


def load_qrels(path: str) -> dict:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qrels[row["claim_id"]] = set(row["relevant_doc_ids"])
    return qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--name", required=True, help="label for this run, e.g. BM25")
    parser.add_argument("--log", default=None,
                         help="defaults to results_log.csv next to this script")
    args = parser.parse_args()
    if args.log is None:
        args.log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_log.csv")

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    qrels = load_qrels(args.qrels)

    metrics = evaluate(results, qrels)

    print(f"\n=== {args.name} ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    # Append to a running comparison log
    write_header = not os.path.exists(args.log)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", *metrics.keys()])
        if write_header:
            writer.writeheader()
        writer.writerow({"name": args.name, **metrics})

    print(f"\nLogged to {args.log} — this becomes your baseline-vs-fine-tuned comparison table.")


if __name__ == "__main__":
    main()

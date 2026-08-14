"""
Download the FEVER dataset (claims + labels + evidence Wikipedia sentence IDs).

FEVER ships in two parts on Hugging Face:
  - `fever` (claims, labels, evidence pointers into Wikipedia)
  - `fever` wiki-pages config (the actual Wikipedia sentence text)

We use the `copenlu/fever_gold_evidence` mirror which already joins claims
to their gold evidence TEXT (not just page/sentence IDs) — this saves you
from having to separately parse the ~5GB Wikipedia dump for Week 1.
If you want the full corpus later (for realistic large-scale retrieval,
not just gold-evidence classification), switch to the raw `fever/wiki_pages`
config — see the commented block at the bottom.

Usage:
    python data/download_fever.py --split train --limit 20000 --out data/raw
"""
import argparse
import json
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--limit", type=int, default=20000,
                         help="cap rows for a fast first pass; set 0 for full split")
    parser.add_argument("--out", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading FEVER [{args.split}] with gold evidence text ...")
    ds = load_dataset("copenlu/fever_gold_evidence", split=args.split)

    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    out_path = os.path.join(args.out, f"fever_{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            # Normalize to a flat schema we control downstream.
            # label ∈ {SUPPORTS, REFUTES, NOT ENOUGH INFO}
            record = {
                "claim_id": row.get("id"),
                "claim": row["claim"],
                "label": row["label"],
                "evidence": row["evidence"],  # list of evidence sentences (strings)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# For Week 3+, when you want retrieval over the FULL Wikipedia evidence pool
# (not just each claim's gold sentences — this is what makes retrieval hard
# and realistic), load the wiki dump separately:
#
#   wiki = load_dataset("fever", "wiki_pages", split="wikipedia_pages")
#
# and build a single flat corpus of (page_id, sentence_id, sentence_text)
# tuples, then join FEVER's evidence pointers (page + sentence index) against
# it. This is a bigger preprocessing step — tackle it in preprocess.py once
# Week 1's small-scale pipeline is working end to end.
# ---------------------------------------------------------------------------

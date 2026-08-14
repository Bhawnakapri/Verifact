"""
Fine-tune a sentence-transformer bi-encoder on (claim, positive, hard_negative)
triplets using MultipleNegativesRankingLoss.

How the loss works: for each triplet in a batch, the positive is pulled
close to its claim in embedding space, and *every other example's positive
and hard_negative in the batch* act as additional in-batch negatives —
so batch size directly affects how many negatives each example sees.
Including the mined hard_negative as a third column (instead of relying
purely on in-batch negatives) is what makes this meaningfully better than
default MNRL usage — see sentence-transformers docs on "hard negatives".

Usage:
    python retrieval/train_biencoder.py \
        --triplets data/processed/triplets.jsonl \
        --base_model sentence-transformers/all-MiniLM-L6-v2 \
        --epochs 3 --batch_size 32 \
        --out models/verifact-biencoder-v1
"""
import argparse
import json

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def load_triplets(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # texts=[anchor, positive, hard_negative] — MultipleNegativesRankingLoss
            # treats index 0 as query, 1 as positive, 2+ as extra hard negatives.
            examples.append(InputExample(
                texts=[row["claim"], row["positive"], row["hard_negative"]]
            ))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets", default="data/processed/triplets.jsonl")
    parser.add_argument("--base_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                         help="larger batch = more in-batch negatives = usually better; "
                              "reduce if you hit GPU memory limits")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--out", default="models/verifact-biencoder-v1")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)

    train_examples = load_triplets(args.triplets)
    print(f"Training examples: {len(train_examples)}")
    if len(train_examples) < args.batch_size:
        print(f"WARNING: dataset smaller than batch_size ({args.batch_size}); "
              f"reduce --batch_size or mine more triplets (Week 1 corpus was too small).")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.out,
        show_progress_bar=True,
    )

    print(f"Saved fine-tuned model to {args.out}")
    print("Next: python retrieval/dense_retrieval.py --model", args.out)


if __name__ == "__main__":
    main()

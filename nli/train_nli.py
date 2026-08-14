"""
Fine-tune a transformer for 3-way stance classification
(entailment / contradiction / neutral) on the NLI pairs built by
build_nli_pairs.py.

Starts from an MNLI-pretrained checkpoint (already knows the general
entailment/contradiction/neutral task) rather than a vanilla base model —
this transfers much faster than starting from scratch, especially with a
modest FEVER-subset training set.

Reports per-class precision/recall/F1, not just overall accuracy: FEVER's
label distribution is not balanced (NEI is typically the smallest class),
and accuracy alone hides that a lot in this task.

Usage:
    python nli/train_nli.py \
        --pairs data/processed/nli_pairs.jsonl \
        --base_model microsoft/deberta-v3-small \
        --epochs 3 --out models/verifact-nli-v1
"""
import argparse
import json

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_pairs(path):
    claims, evidences, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            claims.append(row["claim"])
            evidences.append(row["evidence"])
            labels.append(LABEL2ID[row["label"]])
    return Dataset.from_dict({"claim": claims, "evidence": evidences, "label": labels})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABELS))), zero_division=0
    )
    acc = accuracy_score(labels, preds)
    metrics = {"accuracy": acc}
    for i, name in enumerate(LABELS):
        metrics[f"f1_{name}"] = f1[i]
        metrics[f"precision_{name}"] = precision[i]
        metrics[f"recall_{name}"] = recall[i]
    metrics["macro_f1"] = f1.mean()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/processed/nli_pairs.jsonl")
    parser.add_argument("--base_model", default="microsoft/deberta-v3-small",
                         help="a smaller MNLI-capable base works fine; swap for "
                              "'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli' for a "
                              "checkpoint already pretrained on FEVER-style NLI")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--out", default="models/verifact-nli-v1")
    args = parser.parse_args()

    print(f"Loading pairs from {args.pairs}")
    dataset = load_pairs(args.pairs)
    dataset = dataset.train_test_split(test_size=args.val_split, seed=42)

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize_fn(batch):
        # NLI convention: premise=evidence, hypothesis=claim — the classifier
        # decides whether the evidence entails/contradicts/is neutral to the claim.
        return tokenizer(batch["evidence"], batch["claim"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nFinal validation metrics (per-class — this is your resume table):")
    final_metrics = trainer.evaluate()
    for k, v in sorted(final_metrics.items()):
        if k.startswith("eval_"):
            print(f"  {k[5:]:>20}: {v:.4f}")

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()

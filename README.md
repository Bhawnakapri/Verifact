# VeriFact — Transparent Fact-Verification Engine

Retrieve evidence for a claim, classify each piece of evidence as
SUPPORTS / REFUTES / NOT ENOUGH INFO, and surface disagreement instead
of a single black-box verdict.

Benchmarked on **FEVER** (Fact Extraction and VERification).

## Architecture

```
claim ─┬─> BM25 retrieval ─┐
       └─> Dense retrieval ┴─> RRF fusion ─> cross-encoder rerank ─> top-k evidence
                                                                          │
                                                              NLI classifier (per sentence)
                                                                          │
                                                          aggregate: verdict + disagreement score
```

## Project layout

```
verifact/
├── data/
│   ├── download_fever.py      # pulls FEVER train/dev via HF datasets
│   └── preprocess.py          # builds evidence corpus + claim-evidence pairs
├── retrieval/
│   ├── bm25_baseline.py         # Week 1 baseline
│   ├── mine_hard_negatives.py   # Week 2 — BM25 near-misses as hard negatives
│   ├── train_biencoder.py       # Week 2 — fine-tune sentence-transformer
│   ├── dense_retrieval.py       # Week 2 — encode + search with fine-tuned model
│   └── hybrid_retrieval.py      # Week 3 — BM25 + dense + RRF + rerank
├── nli/
│   ├── build_nli_pairs.py       # Week 3 — claim+evidence -> entailment/contradiction/neutral pairs
│   └── train_nli.py             # Week 3 — fine-tune stance classifier, per-class F1
├── eval/
│   ├── metrics.py             # Recall@k, MRR, nDCG@k
│   └── run_eval.py            # runs eval for any retriever, prints comparison table
├── aggregation/
│   └── verdict.py              # Week 4 — evidence -> verdict + disagreement score (no models needed)
├── api/
│   ├── pipeline.py             # Week 4 — orchestrates retrieve -> classify -> aggregate
│   ├── main.py                 # Week 4 — FastAPI app, loads models once at startup
│   └── static/index.html       # Week 4 — minimal frontend, no build step
├── Dockerfile
└── requirements.txt
```

## Week-by-week checklist

- [x] **Week 1** — Download FEVER, build evidence corpus, BM25 baseline, eval harness working end-to-end
- [x] **Week 2** — Mine hard negatives, fine-tune sentence-transformer (`MultipleNegativesRankingLoss`), beat BM25 on Recall@10 / MRR
  ```bash
  python retrieval/mine_hard_negatives.py --processed data/processed
  python retrieval/train_biencoder.py --triplets data/processed/triplets.jsonl --out models/verifact-biencoder-v1
  # off-the-shelf comparison point:
  python retrieval/dense_retrieval.py --model sentence-transformers/all-MiniLM-L6-v2 --out retrieval/dense_pretrained_results.json
  python eval/run_eval.py --results retrieval/dense_pretrained_results.json --qrels data/processed/qrels.jsonl --name dense_pretrained
  # your fine-tuned model — this is the number you want to beat BM25 with:
  python retrieval/dense_retrieval.py --model models/verifact-biencoder-v1 --out retrieval/dense_finetuned_results.json
  python eval/run_eval.py --results retrieval/dense_finetuned_results.json --qrels data/processed/qrels.jsonl --name dense_finetuned
  ```
- [x] **Week 3** — Hybrid retrieval (BM25 + dense, RRF) + cross-encoder rerank; fine-tune NLI model on FEVER labels, report per-class F1
  ```bash
  # hybrid retrieval + rerank — this is your best retrieval number of the month:
  python retrieval/hybrid_retrieval.py --processed data/processed --dense_model models/verifact-biencoder-v1
  python eval/run_eval.py --results retrieval/hybrid_reranked_results.json --qrels data/processed/qrels.jsonl --name hybrid_reranked

  # NLI stance classifier — separate model, feeds the aggregation layer in Week 4:
  python nli/build_nli_pairs.py --processed data/processed
  python nli/train_nli.py --pairs data/processed/nli_pairs.jsonl --out models/verifact-nli-v1
  ```
- [ ] **Week 4** — Aggregation logic (verdict + disagreement index), FastAPI + simple frontend, deploy to HF Spaces, record demo GIF
  ```bash
  # local run — needs models/verifact-biencoder-v1 and models/verifact-nli-v1 from Weeks 2-3:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  # then open http://localhost:8000 in a browser

  # quick API test without the frontend:
  curl -X POST http://localhost:8000/verify \
    -H "Content-Type: application/json" \
    -d '{"claim": "Coffee reduces the risk of Parkinson'"'"'s disease", "top_k": 8}'
  ```

  **Deploying to Hugging Face Spaces (Docker SDK):**
  1. Create a new Space, SDK = Docker.
  2. Push this repo to it — the Dockerfile is already set up for `uvicorn api.main:app`.
  3. Either (a) uncomment the `COPY data/processed` and `COPY models` lines in the
     `Dockerfile` and commit the trained artifacts with [Git LFS](https://git-lfs.github.com/)
     (models can be a few hundred MB, use LFS not plain git), or (b) use a Space with
     persistent storage and upload `data/processed/` and `models/` after the container starts.
  4. Free-tier Spaces are CPU-only — NLI + cross-encoder inference will be slower
     than on your GPU dev machine but is fine for a live demo at low request volume.
  5. Record a 20-30s screen capture of a genuinely disputed claim (like the coffee/Parkinson's
     example) resolving to DISPUTED with visible support/refute evidence — this is the
     single most convincing thing you can put in your README and resume link.

## Setup

```bash
pip install -r requirements.txt
python data/download_fever.py --split train --limit 20000   # start small
python data/preprocess.py
python retrieval/bm25_baseline.py
python eval/run_eval.py --retriever bm25
```

## Resume line (fill in numbers once you have them)

> Built VeriFact, a transparent fact-verification engine: fine-tuned a sentence-transformer
> for claim-evidence retrieval and a separate NLI transformer for stance classification
> (support/refute/neutral), benchmarked on FEVER (X% Recall@10, Y% stance F1); combined
> hybrid retrieval with cross-encoder reranking and a confidence-weighted aggregation layer
> that flags contested claims as DISPUTED rather than forcing a single verdict; deployed as
> a live FastAPI + web demo.

## Design decisions worth being ready to explain in an interview

- **Why RRF over score-normalization for fusion?** BM25 scores and cosine similarities
  live on different, non-comparable scales — RRF sidesteps that entirely by fusing on
  rank position instead of raw score.
- **Why hard-negative mining from BM25's own mistakes, not random negatives?** Random
  negatives are trivially separable and teach the model almost nothing; BM25's near-misses
  are exactly the confusable cases that matter for retrieval quality.
- **Why a separate NLI model instead of asking an LLM to classify stance in the prompt?**
  A dedicated, fine-tuned classifier is cheaper to run at scale, gives calibrated
  per-class confidence you can threshold on, and its errors are far more interpretable
  (a confusion matrix) than an LLM's.
- **Why DISPUTED as its own verdict instead of picking a side?** The aggregation logic
  checks disagreement *before* picking a winner — on genuinely contested evidence
  (e.g. one cohort study vs. one RCT on the same question), reporting a confident single
  verdict would misrepresent what the evidence actually shows.

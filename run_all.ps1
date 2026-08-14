# VeriFact -- full end-to-end run, Weeks 1-4, for Windows PowerShell.
# Run from the project root:  .\run_all.ps1
# First pass: keep -limit small (e.g. 5000) to confirm everything connects.
# Real numbers: increase later once this all works.

$ErrorActionPreference = "Stop"   # stop on first error, same as `set -e` in bash

Write-Host "=== Setup ==="
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Week 1: data + BM25 baseline ==="
python data\download_fever.py --split train --limit 5000
python data\preprocess.py --split train
python retrieval\bm25_baseline.py --processed data\processed --k 10
python eval\run_eval.py --results retrieval\bm25_results.json `
    --qrels data\processed\qrels.jsonl --name BM25

Write-Host ""
Write-Host "=== Week 2: hard negatives + fine-tuned bi-encoder ==="
python retrieval\mine_hard_negatives.py --processed data\processed
python retrieval\train_biencoder.py `
    --triplets data\processed\triplets.jsonl `
    --out models\verifact-biencoder-v1 `
    --epochs 3 --batch_size 16

python retrieval\dense_retrieval.py `
    --model sentence-transformers/all-MiniLM-L6-v2 `
    --out retrieval\dense_pretrained_results.json
python eval\run_eval.py --results retrieval\dense_pretrained_results.json `
    --qrels data\processed\qrels.jsonl --name dense_pretrained

python retrieval\dense_retrieval.py `
    --model models\verifact-biencoder-v1 `
    --out retrieval\dense_finetuned_results.json
python eval\run_eval.py --results retrieval\dense_finetuned_results.json `
    --qrels data\processed\qrels.jsonl --name dense_finetuned

Write-Host ""
Write-Host "=== Week 3: hybrid retrieval + rerank, NLI stance classifier ==="
python retrieval\hybrid_retrieval.py `
    --processed data\processed `
    --dense_model models\verifact-biencoder-v1 `
    --out retrieval\hybrid_reranked_results.json
python eval\run_eval.py --results retrieval\hybrid_reranked_results.json `
    --qrels data\processed\qrels.jsonl --name hybrid_reranked

python nli\build_nli_pairs.py --processed data\processed
python nli\train_nli.py `
    --pairs data\processed\nli_pairs.jsonl `
    --out models\verifact-nli-v1 `
    --epochs 3

Write-Host ""
Write-Host "=== Week 4: comparison table + launch API ==="
Write-Host "Retrieval comparison (this is your resume evidence):"
Get-Content eval\results_log.csv
Write-Host ""
Write-Host "All four weeks done. Start the API + demo with:"
Write-Host "  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
Write-Host "then open http://localhost:8000"

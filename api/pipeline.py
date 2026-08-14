"""
Orchestrates the full VeriFact pipeline for a single claim:

  claim -> hybrid retrieval (BM25 + dense, RRF, cross-encoder rerank)
        -> NLI stance classification per evidence sentence
        -> aggregation into a verdict

Kept separate from api/main.py so it can be unit-tested (with mocked models)
without spinning up FastAPI, and so the heavy models are loaded exactly once
at process startup rather than per-request.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aggregation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from verdict import EvidenceItem, aggregate


class VeriFactPipeline:
    """
    Real model loading happens in `load()`, kept separate from `__init__` so
    tests can construct a pipeline instance and monkeypatch `retrieve` /
    `classify_stance` without ever touching PyTorch or Hugging Face.
    """

    def __init__(self):
        self.corpus_lookup = {}
        self.bm25 = None
        self.dense_model = None
        self.cross_encoder = None
        self.nli_tokenizer = None
        self.nli_model = None
        self.nli_id2label = None
        self._loaded = False

    def load(self, processed_dir="data/processed", dense_model_path="models/verifact-biencoder-v1",
              cross_encoder_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
              nli_model_path="models/verifact-nli-v1"):
        import json
        import re
        import torch
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        def tokenize(text):
            return re.findall(r"[a-z0-9]+", text.lower())

        doc_ids, texts = [], []
        with open(f"{processed_dir}/corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                doc_ids.append(row["doc_id"])
                texts.append(row["text"])
        self.corpus_lookup = dict(zip(doc_ids, texts))
        self.doc_ids = doc_ids
        self.texts = texts
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self._tokenize = tokenize

        self.dense_model = SentenceTransformer(dense_model_path)
        self.corpus_emb = self.dense_model.encode(
            texts, convert_to_tensor=True, batch_size=64, normalize_embeddings=True
        )

        self.cross_encoder = CrossEncoder(cross_encoder_path)

        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)
        self.nli_model.eval()
        self.nli_id2label = self.nli_model.config.id2label

        self._loaded = True

    def retrieve(self, claim: str, pool_k: int = 100, fused_top_n: int = 30, final_k: int = 8):
        """BM25 + dense -> RRF fusion -> cross-encoder rerank -> top final_k (doc_id, text)."""
        from sentence_transformers import util
        from collections import defaultdict

        scores = self.bm25.get_scores(self._tokenize(claim))
        bm25_ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)[:pool_k]
        bm25_ids = [d for d, _ in bm25_ranked]

        claim_emb = self.dense_model.encode([claim], convert_to_tensor=True, normalize_embeddings=True)
        hits = util.semantic_search(claim_emb, self.corpus_emb, top_k=pool_k)[0]
        dense_ids = [self.doc_ids[h["corpus_id"]] for h in hits]

        rrf_scores = defaultdict(float)
        for rank, d in enumerate(bm25_ids, start=1):
            rrf_scores[d] += 1.0 / (60 + rank)
        for rank, d in enumerate(dense_ids, start=1):
            rrf_scores[d] += 1.0 / (60 + rank)
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:fused_top_n]
        fused_ids = [d for d, _ in fused]

        if not fused_ids:
            return []
        pairs = [[claim, self.corpus_lookup[d]] for d in fused_ids]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(fused_ids, ce_scores), key=lambda x: x[1], reverse=True)[:final_k]
        return [(d, self.corpus_lookup[d]) for d, _ in reranked]

    def classify_stance(self, claim: str, evidence_text: str):
        """Returns (label, confidence) — premise=evidence, hypothesis=claim, matching training."""
        import torch

        inputs = self.nli_tokenizer(evidence_text, claim, truncation=True,
                                     max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs))
        return self.nli_id2label[top_idx], float(probs[top_idx])

    def verify(self, claim: str, final_k: int = 8):
        """Full pipeline: retrieve -> classify each -> aggregate. Returns a Verdict."""
        retrieved = self.retrieve(claim, final_k=final_k)
        evidence_items = []
        for doc_id, text in retrieved:
            label, confidence = self.classify_stance(claim, text)
            evidence_items.append(EvidenceItem(
                text=text, stance=label, confidence=confidence, source_doc_id=doc_id
            ))
        return aggregate(evidence_items)

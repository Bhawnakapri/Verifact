"""
FastAPI backend for VeriFact.

Loads all models once at startup (retrieval, cross-encoder, NLI), then serves
POST /verify for claim verification. Serves the static frontend at /.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.pipeline import VeriFactPipeline

pipeline = VeriFactPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load once here, not per-request — this is what makes the API
    # fast after the (slow, one-time) startup cost.
    processed_dir = os.environ.get("VERIFACT_PROCESSED_DIR", "data/processed")
    dense_model = os.environ.get("VERIFACT_DENSE_MODEL", "models/verifact-biencoder-v1")
    nli_model = os.environ.get("VERIFACT_NLI_MODEL", "models/verifact-nli-v1")
    pipeline.load(processed_dir=processed_dir, dense_model_path=dense_model, nli_model_path=nli_model)
    yield


app = FastAPI(title="VeriFact", lifespan=lifespan)


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=8, ge=1, le=20)


class EvidenceOut(BaseModel):
    text: str
    stance: str
    confidence: float
    source_doc_id: str


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    disagreement_index: float
    support_mass: float
    refute_mass: float
    neutral_mass: float
    evidence: list[EvidenceOut]


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    if not pipeline._loaded:
        raise HTTPException(status_code=503, detail="Models still loading, try again shortly.")

    result = pipeline.verify(req.claim, final_k=req.top_k)

    return VerifyResponse(
        claim=req.claim,
        verdict=result.label,
        confidence=result.confidence,
        disagreement_index=result.disagreement_index,
        support_mass=result.support_mass,
        refute_mass=result.refute_mass,
        neutral_mass=result.neutral_mass,
        evidence=[
            EvidenceOut(text=e.text, stance=e.stance, confidence=round(e.confidence, 4),
                        source_doc_id=e.source_doc_id)
            for e in result.evidence
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": pipeline._loaded}


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

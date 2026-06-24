"""app.main — FastAPI entrypoint for HW3_B.

Endpoints:
  GET  /              — service root
  GET  /health        — bundle + Qdrant + PG reachability
  GET  /model-info    — bundle metadata + Qdrant vector count
  POST /embed         — text(s) → 384-dim vectors
  POST /predict       — single text → predicted emotion label
  POST /search        — query → Qdrant ANN + PG audit
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status

from . import client_pg, client_qdrant, config
from . import predictor as predictor_mod
from .model_loader import ModelService
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
    SearchRequest,
    SearchResponse,
)
from .search import hybrid_search

log = logging.getLogger("hw3_b")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


model_service = ModelService()


# lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("HW3_B starting. BUNDLE_DIR=%s", config.BUNDLE_DIR)
    model_service.load()
    if model_service.state.loaded:
        log.info("Bundle loaded: %s", model_service.state.bundle_dir)
    else:
        log.error("Bundle load FAILED: %s", model_service.state.error)
    yield
    log.info("HW3_B shutting down.")


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", response_model=RootResponse, tags=["service"])
def root():
    return RootResponse(
        message="QBC12 HW3 Encoder API",
        docs="/docs",
        health="/health",
        version=config.APP_VERSION,
    )

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    bundle_ok = model_service.state.loaded
    qdrant_ok = client_qdrant.ping()
    pg_ok = client_pg.ping()

    status = "ok" if bundle_ok and qdrant_ok and pg_ok else "degraded"

    return HealthResponse(
        status=status,
        bundle_loaded=bundle_ok,
        bundle_dir=str(model_service.state.bundle_dir) if model_service.state.bundle_dir else "",
        qdrant_reachable=qdrant_ok,
        pg_reachable=pg_ok,
        error=model_service.state.error,
    )

# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info():

    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    vector_count = client_qdrant.vector_count(config.QDRANT_COLLECTION)

    meta = model_service.metadata

    return ModelInfoResponse(
        bundle_version=meta.get("framework_version", "unknown"),
        model_id=meta.get("model_name", "unknown"),
        model_revision=meta.get("model_revision", "unknown"),
        device=config.BUNDLE_DEVICE,
        max_seq_len=meta.get("max_seq_len", 256),
        embedding_dim=meta.get("embedding_dim", 384),
        bundle_dir=str(model_service.state.bundle_dir),
        qdrant_collection=config.QDRANT_COLLECTION,
        qdrant_vector_count=vector_count,
    )

# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

@app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
def embed(req: EmbedRequest):

    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    if len(req.texts) > config.EMBED_BATCH_HARD_CAP:
        raise HTTPException(status_code=413, detail="batch too large")

    t0 = time.perf_counter()

    vectors = predictor_mod.embed_texts(
        model_service.require_predictor(),
        req.texts,
    )

    return EmbedResponse(
        count=len(req.texts),
        dim=vectors.shape[1],
        embeddings=vectors.tolist(),
    )

# ---------------------------------------------------------------------------
# /predict — single text → emotion label via nearest neighbor
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["embedding"])
def predict(req: PredictRequest):

    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    t0 = time.perf_counter()

    vec = predictor_mod.embed_texts(
        model_service.require_predictor(),
        [req.text],
    )[0].tolist()

    hits = client_qdrant.search(
        collection=config.QDRANT_COLLECTION,
        vector=vec,
        top_k=1,
        lang=None,
        primary=None,
        exclude_neutral=False,
    )

    if not hits:
        raise HTTPException(status_code=404, detail="no match found in corpus")

    best = hits[0]

    elapsed = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        text=req.text,
        predicted_label=best.payload.get("primary_label", "unknown"),
        confidence=float(best.score),
        matched_text=best.payload.get("text", ""),
        elapsed_ms=elapsed,
    )

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest):

    if not model_service.state.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    query_vec = predictor_mod.embed_texts(
        model_service.require_predictor(),
        [req.query],
    )[0].tolist()

    hits, took_ms = hybrid_search(
        query_vec,
        req.top_k,
        req.lang,
        req.primary,
        req.exclude_neutral,
    )

    return SearchResponse(
        query=req.query,
        count=len(hits),
        top_k=req.top_k,
        took_ms=took_ms,
        hits=hits,
    )

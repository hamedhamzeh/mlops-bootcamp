"""app.schemas — Pydantic request/response models.

extra="forbid" everywhere so Swagger rejects typos and unknown fields loudly
instead of silently dropping them (a real production incident class).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Service — health + model-info + root
# ============================================================================

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    bundle_loaded: bool
    bundle_dir: str
    qdrant_reachable: bool
    pg_reachable: bool
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class ModelInfoResponse(BaseModel):
    bundle_version: str
    model_id: str
    model_revision: str
    device: str
    max_seq_len: int
    embedding_dim: int
    bundle_dir: str
    qdrant_collection: str
    qdrant_vector_count: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class RootResponse(BaseModel):
    message: str
    docs: str
    health: str
    version: str

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# /embed
# ============================================================================

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=256)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"texts": ["I love this so much"]}}
    )


class EmbedResponse(BaseModel):
    count: int
    dim: int
    embeddings: List[List[float]]

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# /search
# ============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(10, ge=1, le=100)
    lang: Optional[Literal["en"]] = None
    primary: Optional[str] = Field(None, max_length=64)
    exclude_neutral: bool = Field(True)

    model_config = ConfigDict(extra="forbid")


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    primary: str
    labels: List[str]
    lang: str
    source: str

    model_config = ConfigDict(extra="forbid")


class SearchResponse(BaseModel):
    query: str
    count: int
    top_k: int
    took_ms: float
    hits: List[SearchHit]

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# /predict — single text → predicted emotion label via nearest neighbor
# ============================================================================

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)

    model_config = ConfigDict(extra="forbid")


class PredictResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    matched_text: str
    elapsed_ms: float

    model_config = ConfigDict(extra="forbid")


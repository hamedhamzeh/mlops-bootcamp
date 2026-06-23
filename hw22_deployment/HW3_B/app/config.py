"""app.config — environment variable contract for HW3_B.

Mirrors the bundle's contract. If you add an env var here, add it to:
  1. .env.example
  2. entrypoint.sh  (if it's a runtime knob, not a build-time secret)
  3. tests/test_env_contract.py  (if it affects determinism)
"""
from __future__ import annotations

import os
from typing import List

# --- App identity ---
APP_TITLE = "QBC12 HW03-B Encoder Embedding & Search API"
APP_VERSION = "0.1.0"

# --- Bundle location (BUNDLE_DIR is the source of truth) ---
# In container: /app/bundle (baked into image)
# In dev:       ../HW3_A/bundle
BUNDLE_DIR = os.getenv("BUNDLE_DIR", "/app/bundle")
BUNDLE_DEVICE = os.getenv("BUNDLE_DEVICE", "cpu")

# --- Qdrant (shared, read-only) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qbc12-qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "qbc12_corpus")

# --- Postgres (shared, read-only) ---
DATABASE_HOST = os.getenv("DATABASE_HOST", "qbc12-postgres")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "qbc12")

DATABASE_API_RO_PASSWORD = os.getenv("DATABASE_API_RO_PASSWORD", "qbc12_ro_password")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://qbc12_hw03_api_ro:{DATABASE_API_RO_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)

# --- MLflow (for the /model-info endpoint) ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

STUDENT_USERNAME = os.getenv(
    "STUDENT_USERNAME",
    MLFLOW_TRACKING_USERNAME or "student_unknown"
)

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "hw3_encoder_search")
MODEL_NAME = os.getenv("MODEL_NAME", "hw3_encoder_bundle")

# --- Search knobs ---
SEARCH_DEFAULT_TOP_K = 10
SEARCH_MAX_TOP_K = 100
SEARCH_MAX_BATCH_TEXTS = 256

# --- Embedding knobs ---
EMBED_MAX_SEQ_LEN = 256
EMBED_BATCH_HARD_CAP = 256
EMBED_DIM = 384
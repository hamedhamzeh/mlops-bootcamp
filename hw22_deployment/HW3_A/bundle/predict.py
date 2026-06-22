#!/usr/bin/env python
"""predict.py — Self-contained embedding inference.

MUST implement exactly 4 functions:
    load_bundle()    → (model, tokenizer)
    embed(texts)     → np.ndarray shape (N, 384)
    similarity(a, b) → float
    info()           → dict

The 7-step pipeline in embed():
    1. Tokenize (padding=True, truncation=True, max_length=256, return_tensors="pt")
    2. Move tensors to device
    3. Forward pass under torch.no_grad()
    4. Mean-pool weighted by attention mask: sum(H * mask) / sum(mask).clamp(min=1e-9)
    5. L2 normalize: F.normalize(pooled, p=2, dim=1)
    6. Move to CPU, convert to numpy float32
    7. Return

DO NOT import sentence_transformers. Use raw transformers only.
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple

import argparse
import json
import sys

from transformers import AutoModel, AutoTokenizer
import torch, torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUNDLE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIM = 384
MAX_SEQ_LEN = 256

_MODEL = None
_TOKENIZER = None

def load_bundle(bundle_dir: str | None = None) -> Tuple:
    """Load model and tokenizer from the bundle directory.

    Args:
        bundle_dir: Path to bundle/model/. Defaults to BUNDLE_DIR env var.

    Returns:
        (model, tokenizer) tuple. model is in eval mode on the correct device.
        tokenizer is loaded from the same directory.
    """
    global _MODEL, _TOKENIZER

    current_bundle_dir = Path(bundle_dir) if bundle_dir is not None else BUNDLE_DIR
    model_dir = current_bundle_dir / "model"
    metadata_path = current_bundle_dir / "metadata.json"

    if not current_bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {current_bundle_dir}")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if bundle_dir is None and _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer

    return _MODEL, _TOKENIZER


def embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into a (N, 384) float32 numpy array.
    Args:
        texts: List of strings to embed. Can be empty.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32.
        For empty input, returns shape (0, 384).
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")

    if len(texts) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if not all(isinstance(t, str) for t in texts):
        raise TypeError("all items in texts must be strings")
    
    model, tokenizer = load_bundle()
    device = next(model.parameters()).device

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state

        attention_mask = encoded["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        pooled = pooled / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        normalized = F.normalize(pooled, p=2, dim=1)

    return normalized.detach().cpu().numpy().astype(np.float32)



def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector (384,).
        b: Second embedding vector (384,).

    Returns:
        float: Cosine similarity in [-1, 1].
    """

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def info() -> dict:
    """Return metadata about the loaded bundle.

    Returns:
        dict with keys: model_name, embedding_dim, max_seq_len, device,
        framework, deterministic, bundle_dir.
    """
    metadata_path = BUNDLE_DIR / "metadata.json"

    if not BUNDLE_DIR.exists():
        raise FileNotFoundError(f"Bundle directory not found: {BUNDLE_DIR}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "model_name": metadata["model_name"],
        "model_revision": metadata["model_revision"],
        "embedding_dim": metadata["embedding_dim"],
        "max_seq_len": metadata["max_seq_len"],
        "framework_version": metadata["framework_version"],
        "transformers_version": metadata["transformers_version"],
        "built_by": metadata["built_by"],
        "build_timestamp_utc": metadata["build_timestamp_utc"],
        "device": device,
    }


# ---------------------------------------------------------------------------
# CLI entry point (used by scripts/gen_manifest.py and for testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bundle embed CLI")
    p.add_argument("--text", action="append", default=[], help="repeatable text input")
    p.add_argument("--texts-file", help="JSON list of strings")
    p.add_argument("--out", help="optional .npy output path")
    p.add_argument("--info", action="store_true", help="print info and exit")
    args = p.parse_args()

    if args.info:
        print(json.dumps(info(), indent=2, default=str))
        raise SystemExit(0)

    texts: list[str] = list(args.text)
    if args.texts_file:
        with open(args.texts_file, encoding="utf-8") as f:
            texts.extend(json.load(f))

    if not texts:
        print("ERROR: provide --text or --texts-file", file=sys.stderr)
        raise SystemExit(2)

    emb = embed(texts)
    if args.out:
        np.save(args.out, emb)
        print(f"Saved {emb.shape} to {args.out}")
    else:
        print(json.dumps([[round(float(x), 6) for x in row] for row in emb]))

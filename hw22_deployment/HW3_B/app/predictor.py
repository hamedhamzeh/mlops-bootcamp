"""app.predictor — thin wrapper over the bundle's BundlePredictor.

This module exists so the FastAPI layer doesn't import from `predict` directly
(safer refactoring — if the bundle's module name ever changes, only this file
needs to change).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def embed_texts(predictor, texts) -> np.ndarray:
    if len(texts) == 0:
        return np.zeros((0, 384), dtype=np.float32)

    arr = predictor.embed(list(texts))

    arr = np.asarray(arr, dtype=np.float32)

    if arr.shape[1] != 384:
        raise ValueError(f"Invalid embedding dim: {arr.shape}")

    return arr


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    return a_norm @ b_norm.T

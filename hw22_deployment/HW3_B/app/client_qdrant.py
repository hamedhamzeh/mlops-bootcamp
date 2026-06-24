"""app.client_qdrant — Qdrant client (read-only API key)."""
from __future__ import annotations

import time
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config


_client: Optional[QdrantClient] = None

def get_client() -> QdrantClient:
    global _client

    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            timeout=10.0,
        )

    return _client


def ping() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def vector_count(collection: str) -> Optional[int]:
    try:
        info = get_client().get_collection(collection_name=collection)

        count = getattr(info, "vectors_count", None)
        if count is not None:
            return count

        count = getattr(info, "points_count", None)
        if count is not None:
            return count

        return None

    except Exception:
        return None
    

def search(
    collection: str,
    vector: List[float],
    top_k: int,
    lang: Optional[str],
    primary: Optional[str],
    exclude_neutral: bool,
) -> List[models.ScoredPoint]:

    client = get_client()

    must_conditions = []

    # language filter
    if lang is not None:
        must_conditions.append(
            models.FieldCondition(
                key="lang",
                match=models.MatchValue(value=lang),
            )
        )

    # primary label filter
    if primary is not None:
        must_conditions.append(
            models.FieldCondition(
                key="primary",
                match=models.MatchValue(value=primary),
            )
        )

    # exclude neutral
    must_not_conditions = []

    if exclude_neutral:
        must_not_conditions.append(
            models.FieldCondition(
                key="primary",
                match=models.MatchValue(value="neutral"),
            )
        )

    query_filter = models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )

    return client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

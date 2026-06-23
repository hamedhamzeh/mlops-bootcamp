"""app.search — hybrid (Qdrant ANN + PG audit) search orchestration."""
from __future__ import annotations

import time
from typing import List, Optional

from . import client_pg, client_qdrant
from . import config
from .schemas import SearchHit


def hybrid_search(
    query_vector,
    top_k,
    lang,
    primary,
    exclude_neutral,
) -> tuple[List[SearchHit], float]:

    t0 = time.perf_counter()

    # 1. Qdrant ANN search
    qdr_hits = client_qdrant.search(
        collection=config.QDRANT_COLLECTION,
        vector=query_vector,
        top_k=top_k,
        lang=lang,
        primary=primary,
        exclude_neutral=exclude_neutral,
    )

    # 2. No results case
    if not qdr_hits:
        return [], (time.perf_counter() - t0) * 1000.0

    # 3. Extract IDs from Qdrant
    ids = [str(h.id) for h in qdr_hits]

    # 4. Fetch ground truth from Postgres
    pg_rows = client_pg.fetch_corpus_hits(ids)

    # 5. Build lookup map (VERY IMPORTANT for alignment)
    pg_map = {row["id"]: row for row in pg_rows}

    # 6. Merge results
    hits: List[SearchHit] = []

    for h in qdr_hits:
        hid = str(h.id)

        if hid not in pg_map:
            continue

        row = pg_map[hid]

        hits.append(
            SearchHit(
                id=hid,
                score=float(h.score),
                text=row["text"],
                primary=row["primary_label"],
                labels=list(row["labels"]),
                lang=row["lang"],
                source=row["source"],
            )
        )

    # 7. return timing
    took_ms = (time.perf_counter() - t0) * 1000.0

    return hits, took_ms

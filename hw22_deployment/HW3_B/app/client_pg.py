"""app.client_pg — Postgres read-only client (audit + source of truth)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

import psycopg
from psycopg.rows import dict_row

from . import config


def ping() -> bool:
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


@contextmanager
def _connect() -> Iterator[psycopg.Connection]:
    conn = None
    try:
        conn = psycopg.connect(
            config.DATABASE_URL,
            connect_timeout=5,
            row_factory=dict_row,
        )
        yield conn
    finally:
        if conn is not None:
            conn.close()


def fetch_corpus_hits(ids: List[str]) -> List[dict]:
    if not ids:
        return []

    query = """
        SELECT
            id::text AS id,
            text,
            primary_label,
            labels,
            lang,
            source
        FROM core.encoder_corpus
        WHERE id = ANY(%s::uuid[])
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (ids,))
            rows = cur.fetchall()

    # build lookup map
    row_map = {row["id"]: row for row in rows}

    # preserve order of input IDs
    return [row_map[i] for i in ids if i in row_map]


def count_corpus() -> Optional[int]:
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM core.encoder_corpus")
                return cur.fetchone()["count"]
    except Exception:
        return None

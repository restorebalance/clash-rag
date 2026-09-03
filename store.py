"""Vector storage.

The only interface is replace_chunks() and search(). Nothing about SQLite or
how vectors are packed leaves this module, so the backing store can be
swapped without touching ingest.py or retrieve.py.
"""

import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path("chunks.db")

FIELDS = ("title", "card", "section", "rarity", "type", "text")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    card      TEXT NOT NULL,
    section   TEXT NOT NULL,
    rarity    TEXT,
    type      TEXT,
    text      TEXT NOT NULL,
    embedding BLOB NOT NULL
)
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def replace_chunks(chunks, embeddings):
    """Replace the stored collection with these chunks and their vectors.

    Wholesale replacement keeps a re-run of the ingest pipeline idempotent;
    there is no partial-update path by design.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    rows = [
        tuple(chunk.get(field) for field in FIELDS)
        + (np.asarray(vector, dtype=np.float32).tobytes(),)
        for chunk, vector in zip(chunks, embeddings)
    ]
    conn = connect()
    with conn:
        conn.execute("DELETE FROM chunks")
        conn.executemany(
            f"INSERT INTO chunks ({', '.join(FIELDS)}, embedding)"
            f" VALUES ({', '.join('?' * (len(FIELDS) + 1))})",
            rows,
        )
    conn.close()
    return len(rows)


def search(embedding, k=5):
    """Return the k stored chunks closest to embedding, best match first."""
    conn = connect()
    rows = conn.execute(
        f"SELECT {', '.join(FIELDS)}, embedding FROM chunks"
    ).fetchall()
    conn.close()
    if not rows:
        return []

    matrix = np.frombuffer(b"".join(row[-1] for row in rows), dtype=np.float32)
    matrix = matrix.reshape(len(rows), -1)
    query = np.asarray(embedding, dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = (matrix @ query) / np.maximum(norms, 1e-9)

    results = []
    for index in np.argsort(-scores)[:k]:
        record = dict(zip(FIELDS, rows[index]))
        record["score"] = float(scores[index])
        results.append(record)
    return results


def count():
    """Number of chunks currently stored."""
    conn = connect()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return total

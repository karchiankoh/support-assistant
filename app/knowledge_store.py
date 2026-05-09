import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = "data/support_knowledge.db"


@dataclass(frozen=True)
class KnowledgeChunk:
    filename: str
    path: str
    kind: str
    chunk_index: int
    text: str
    embedding: list[float]


def get_knowledge_db_path() -> Path:
    return Path(os.getenv("SUPPORT_KB_DB_PATH", DEFAULT_DB_PATH))


def init_db(db_path: Path | None = None) -> None:
    path = db_path or get_knowledge_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(path, chunk_index, embedding_model)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_model
            ON knowledge_chunks (embedding_model)
            """
        )


def upsert_chunks(chunks: list[KnowledgeChunk], embedding_model: str, db_path: Path | None = None) -> int:
    if not chunks:
        return 0

    path = db_path or get_knowledge_db_path()
    init_db(path)

    with sqlite3.connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO knowledge_chunks (
                filename, path, kind, chunk_index, text, embedding_model, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path, chunk_index, embedding_model)
            DO UPDATE SET
                filename = excluded.filename,
                kind = excluded.kind,
                text = excluded.text,
                embedding = excluded.embedding,
                created_at = CURRENT_TIMESTAMP
            """,
            [
                (
                    chunk.filename,
                    chunk.path,
                    chunk.kind,
                    chunk.chunk_index,
                    chunk.text,
                    embedding_model,
                    json.dumps(chunk.embedding),
                )
                for chunk in chunks
            ],
        )
    return len(chunks)


def load_chunks(embedding_model: str, db_path: Path | None = None) -> list[KnowledgeChunk]:
    path = db_path or get_knowledge_db_path()
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT filename, path, kind, chunk_index, text, embedding
            FROM knowledge_chunks
            WHERE embedding_model = ?
            """,
            (embedding_model,),
        ).fetchall()

    return [
        KnowledgeChunk(
            filename=row[0],
            path=row[1],
            kind=row[2],
            chunk_index=row[3],
            text=row[4],
            embedding=json.loads(row[5]),
        )
        for row in rows
    ]

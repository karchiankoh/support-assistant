import logging
import math
import os

from app.config import get_embedding_model, is_dev_environment
from app.knowledge_store import get_knowledge_db_path, load_chunks
from app.openai_client import create_embedding
from app.schemas import RetrievedSource


DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.45
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
MAX_QUERY_CHARS = 12000
logger = logging.getLogger(__name__)


def get_retrieval_top_k() -> int:
    return int(os.getenv("SUPPORT_KB_TOP_K", str(DEFAULT_TOP_K)))


def get_retrieval_min_score() -> float:
    return float(os.getenv("SUPPORT_KB_MIN_SCORE", str(DEFAULT_MIN_SCORE)))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunks.append(clean_text[start:end])
        if end == len(clean_text):
            break
        start = max(0, end - overlap)
    return chunks


async def retrieve_support_context(query: str | None) -> tuple[str | None, list[RetrievedSource]]:
    if is_dev_environment():
        logger.info("Support context retrieval skipped in development environment")
        return None, []

    if not query or not query.strip():
        logger.info("Support context retrieval skipped because query is empty")
        return None, []

    db_path = get_knowledge_db_path()
    if not db_path.exists():
        logger.info(
            "Support context retrieval skipped because knowledge database is missing",
            extra={"knowledge_db_path": str(db_path)},
        )
        return None, []

    logger.info(
        "Support context retrieval started",
        extra={
            "query_characters": len(query),
            "embedded_query_characters": min(len(query), MAX_QUERY_CHARS),
            "knowledge_db_path": str(db_path),
            "embedding_model": get_embedding_model(),
            "top_k": get_retrieval_top_k(),
            "min_score": get_retrieval_min_score(),
        },
    )
    query_embedding = await create_embedding(query[:MAX_QUERY_CHARS])
    rows = load_chunks(get_embedding_model(), db_path)
    ranked = []
    for row in rows:
        score = _cosine_similarity(query_embedding, row.embedding)
        if score >= get_retrieval_min_score():
            ranked.append((score, row))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_matches = ranked[: get_retrieval_top_k()]

    sources = [
        RetrievedSource(
            filename=chunk.filename,
            kind=chunk.kind,
            score=round(score, 4),
            chunk_index=chunk.chunk_index,
            text=chunk.text,
        )
        for score, chunk in top_matches
    ]
    context = _format_retrieved_context(sources)
    logger.info(
        "Support context retrieval completed",
        extra={
            "candidate_chunk_count": len(rows),
            "matching_chunk_count": len(ranked),
            "returned_source_count": len(sources),
            "top_score": round(top_matches[0][0], 4) if top_matches else None,
        },
    )
    return context, sources


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _format_retrieved_context(sources: list[RetrievedSource]) -> str | None:
    if not sources:
        return None
    return "\n\n".join(
        f"--- retrieved {source.kind}: {source.filename} "
        f"(chunk {source.chunk_index}, score {source.score}) ---\n{source.text}"
        for source in sources
    )

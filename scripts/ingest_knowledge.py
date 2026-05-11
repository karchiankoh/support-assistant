import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.files import decode_text
from app.knowledge_store import KnowledgeChunk, get_knowledge_db_path, init_db, upsert_chunks
from app.logging_config import configure_logging
from app.openai_client import embed_texts, get_embedding_model
from app.retrieval import chunk_text


configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_DIRS = {
    "runbook": Path("data/runbooks"),
    "incident": Path("data/incidents"),
}


def discover_files() -> list[tuple[Path, str]]:
    discovered = []
    for kind, directory in DEFAULT_KNOWLEDGE_DIRS.items():
        if not directory.exists():
            logger.info(
                "Knowledge directory skipped because it does not exist",
                extra={
                    "knowledge_kind": kind,
                    "knowledge_directory": str(directory),
                },
            )
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                discovered.append((path, kind))
    logger.info(
        "Knowledge files discovered",
        extra={
            "file_count": len(discovered),
            "knowledge_directories": [str(path) for path in DEFAULT_KNOWLEDGE_DIRS.values()],
        },
    )
    return discovered


async def ingest() -> int:
    db_path = get_knowledge_db_path()
    embedding_model = get_embedding_model()
    logger.info(
        "Knowledge ingestion started",
        extra={
            "knowledge_db_path": str(db_path),
            "embedding_model": embedding_model,
        },
    )
    init_db(db_path)

    discovered_files = discover_files()
    total_chunks = 0
    for path, kind in discovered_files:
        text = decode_text(path.read_bytes())
        chunks = chunk_text(text)
        if not chunks:
            logger.warning(
                "Knowledge file skipped because it produced no chunks",
                extra={
                    "knowledge_path": str(path),
                    "knowledge_kind": kind,
                    "characters": len(text),
                },
            )
            continue

        logger.info(
            "Knowledge file chunked",
            extra={
                "knowledge_path": str(path),
                "knowledge_kind": kind,
                "characters": len(text),
                "chunk_count": len(chunks),
            },
        )
        embeddings = await embed_texts(chunks)
        knowledge_chunks = [
            KnowledgeChunk(
                filename=path.name,
                path=str(path),
                kind=kind,
                chunk_index=index,
                text=chunk,
                embedding=embeddings[index],
            )
            for index, chunk in enumerate(chunks)
        ]
        indexed_chunks = upsert_chunks(knowledge_chunks, embedding_model, db_path)
        total_chunks += indexed_chunks
        logger.info(
            "Knowledge file ingested",
            extra={
                "knowledge_path": str(path),
                "knowledge_kind": kind,
                "indexed_chunk_count": indexed_chunks,
            },
        )

    logger.info(
        "Knowledge ingestion completed",
        extra={
            "file_count": len(discovered_files),
            "total_chunk_count": total_chunks,
            "knowledge_db_path": str(db_path),
        },
    )
    return total_chunks


def main() -> None:
    total_chunks = asyncio.run(ingest())
    logger.info(
        "Knowledge ingestion command completed",
        extra={
            "total_chunk_count": total_chunks,
            "knowledge_db_path": str(get_knowledge_db_path()),
        },
    )


if __name__ == "__main__":
    main()

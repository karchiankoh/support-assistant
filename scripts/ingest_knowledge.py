import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.files import decode_text
from app.knowledge_store import KnowledgeChunk, get_knowledge_db_path, init_db, upsert_chunks
from app.openai_client import embed_texts, get_embedding_model
from app.retrieval import chunk_text


DEFAULT_KNOWLEDGE_DIRS = {
    "runbook": Path("data/runbooks"),
    "incident": Path("data/incidents"),
}


def discover_files() -> list[tuple[Path, str]]:
    discovered = []
    for kind, directory in DEFAULT_KNOWLEDGE_DIRS.items():
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                discovered.append((path, kind))
    return discovered


async def ingest() -> int:
    db_path = get_knowledge_db_path()
    init_db(db_path)

    total_chunks = 0
    for path, kind in discover_files():
        text = decode_text(path.read_bytes())
        chunks = chunk_text(text)
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
        total_chunks += upsert_chunks(knowledge_chunks, get_embedding_model(), db_path)
        print(f"Ingested {len(knowledge_chunks)} chunks from {path}")

    return total_chunks


def main() -> None:
    total_chunks = asyncio.run(ingest())
    print(f"Indexed {total_chunks} chunks in {get_knowledge_db_path()}")


if __name__ == "__main__":
    main()

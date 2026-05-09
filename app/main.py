from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, status

from app.files import UploadedDocument, combine_documents, read_upload
from app.knowledge_store import get_knowledge_db_path
from app.openai_client import analyze_support_context, get_model, is_dev_environment
from app.retrieval import retrieve_support_context
from app.schemas import AnalysisResponse


app = FastAPI(
    title="Support Assistant API",
    version="0.1.0",
    description="Upload support tickets and logs, then summarize issues and debugging steps with the OpenAI Responses API.",
)


async def _read_many(files: list[UploadFile] | None, kind: str) -> list[UploadedDocument]:
    return [await read_upload(file, kind) for file in files or []]


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model": get_model(),
        "dev_environment": is_dev_environment(),
        "knowledge_db_exists": get_knowledge_db_path().exists(),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticket_and_logs(
    ticket: Annotated[UploadFile | None, File(description="Support ticket text file.")] = None,
    logs: Annotated[list[UploadFile] | None, File(description="One or more log files.")] = None,
) -> AnalysisResponse:
    ticket_documents = [await read_upload(ticket, "ticket")] if ticket else []
    log_documents = await _read_many(logs, "log")

    if not ticket_documents and not log_documents:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload at least one ticket or log file.",
        )

    ticket_text = combine_documents(ticket_documents)
    log_text = combine_documents(log_documents)
    try:
        retrieved_context, retrieved_sources = await retrieve_support_context(
            "\n\n".join(part for part in [ticket_text, log_text] if part)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    response_id, model, analysis = await analyze_support_context(
        ticket_text=ticket_text,
        log_text=log_text,
        runbook_text=retrieved_context,
    )

    return AnalysisResponse(
        response_id=response_id,
        model=model,
        sources=[doc.source for doc in ticket_documents + log_documents],
        retrieved_sources=retrieved_sources,
        analysis=analysis,
    )

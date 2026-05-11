import logging
import uuid
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status

from app.files import UploadedDocument, combine_documents, read_upload
from app.knowledge_store import get_knowledge_db_path
from app.logging_config import configure_logging
from app.openai_client import analyze_support_context, get_model, is_dev_environment
from app.retrieval import retrieve_support_context
from app.request_context import request_id_context, request_path_context
from app.schemas import AnalysisResponse


REQUEST_ID_HEADER = "X-Request-ID"

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Support Assistant API",
    version="0.1.0",
    description="Upload support tickets and logs, then summarize issues and debugging steps with the OpenAI Responses API.",
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    request_id_token = request_id_context.set(request_id)
    request_path_token = request_path_context.set(request.url.path)

    try:
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        request_path_context.reset(request_path_token)
        request_id_context.reset(request_id_token)


async def _read_many(files: list[UploadFile] | None, kind: str) -> list[UploadedDocument]:
    return [await read_upload(file, kind) for file in files or []]


@app.get("/health")
async def health() -> dict[str, str | bool]:
    knowledge_db_exists = get_knowledge_db_path().exists()
    logger.info(
        "Health check requested",
        extra={
            "model": get_model(),
            "dev_environment": is_dev_environment(),
            "knowledge_db_exists": knowledge_db_exists,
        },
    )
    return {
        "status": "ok",
        "model": get_model(),
        "dev_environment": is_dev_environment(),
        "knowledge_db_exists": knowledge_db_exists,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticket_and_logs(
    ticket: Annotated[UploadFile | None, File(description="Support ticket text file.")] = None,
    logs: Annotated[list[UploadFile] | None, File(description="One or more log files.")] = None,
) -> AnalysisResponse:
    logger.info(
        "Analyze request received",
        extra={
            "has_ticket": ticket is not None,
            "log_file_count": len(logs or []),
        },
    )

    ticket_documents = [await read_upload(ticket, "ticket")] if ticket else []
    log_documents = await _read_many(logs, "log")

    if not ticket_documents and not log_documents:
        logger.warning("Analyze request rejected because no files were uploaded")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload at least one ticket or log file.",
        )

    ticket_text = combine_documents(ticket_documents)
    log_text = combine_documents(log_documents)
    logger.info(
        "Analyze request files processed",
        extra={
            "ticket_count": len(ticket_documents),
            "log_count": len(log_documents),
            "ticket_characters": len(ticket_text or ""),
            "log_characters": len(log_text or ""),
        },
    )

    try:
        retrieved_context, retrieved_sources = await retrieve_support_context(
            "\n\n".join(part for part in [ticket_text, log_text] if part)
        )
    except RuntimeError as exc:
        logger.exception("Support context retrieval failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    response_id, model, analysis = await analyze_support_context(
        ticket_text=ticket_text,
        log_text=log_text,
        runbook_text=retrieved_context,
    )

    logger.info(
        "Analyze request completed",
        extra={
            "response_id": response_id,
            "model": model,
            "source_count": len(ticket_documents) + len(log_documents),
            "retrieved_source_count": len(retrieved_sources),
            "severity": analysis.severity,
            "affected_component_count": len(analysis.affected_components),
        },
    )

    return AnalysisResponse(
        response_id=response_id,
        model=model,
        sources=[doc.source for doc in ticket_documents + log_documents],
        retrieved_sources=retrieved_sources,
        analysis=analysis,
    )

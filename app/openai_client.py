import json
import os
from typing import Any

from fastapi import HTTPException, status
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.schemas import SupportAnalysis


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEV_ENVIRONMENT = "dev"

SYSTEM_INSTRUCTIONS = """
You are a senior support engineer. Analyze support tickets and application logs.
Return only valid JSON with this shape:
{
  "issue_summary": "string",
  "likely_cause": "string or null",
  "severity": "string or null",
  "affected_components": ["string"],
  "evidence": ["short facts quoted or paraphrased from the supplied files"],
  "customer_facing_explanation": "string or null",
  "debugging_steps": [{"step": "string", "rationale": "string or null"}]
}

Be concrete, avoid inventing facts, and prioritize steps a support engineer can run next.
If evidence is missing, say what should be collected.
""".strip()


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def is_dev_environment() -> bool:
    environment = os.getenv("APP_ENV")
    return environment is not None and environment.strip().lower() == DEV_ENVIRONMENT


def get_client() -> AsyncOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not configured.",
        )
    return AsyncOpenAI()


async def create_embedding(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = get_client()
    try:
        response = await client.embeddings.create(
            model=get_embedding_model(),
            input=texts,
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(f"Could not reach OpenAI embeddings API: {exc}") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"OpenAI embeddings API error {exc.status_code}: {exc.message}") from exc

    return [item.embedding for item in response.data]


def build_prompt(ticket_text: str | None, log_text: str | None, runbook_text: str | None) -> str:
    sections = []
    if ticket_text:
        sections.append(f"<support_ticket>\n{ticket_text}\n</support_ticket>")
    if log_text:
        sections.append(f"<logs>\n{log_text}\n</logs>")
    if runbook_text:
        sections.append(f"<retrieved_support_knowledge>\n{runbook_text}\n</retrieved_support_knowledge>")
    return "\n\n".join(sections)


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _parse_analysis(text: str) -> SupportAnalysis:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return SupportAnalysis(
                issue_summary="The model returned non-JSON output.",
                debugging_steps=[],
                raw_model_output=text,
            )
        payload = json.loads(text[start : end + 1])

    analysis = SupportAnalysis.model_validate(payload)
    analysis.raw_model_output = payload
    return analysis


def _mock_analysis(ticket_text: str | None, log_text: str | None, runbook_text: str | None) -> SupportAnalysis:
    provided_sources = []
    if ticket_text:
        provided_sources.append("support ticket")
    if log_text:
        provided_sources.append("logs")
    if runbook_text:
        provided_sources.append("runbooks")

    sources = ", ".join(provided_sources) if provided_sources else "no source text"
    payload = {
        "issue_summary": f"Mock development analysis generated from {sources}.",
        "likely_cause": "Mock response: no OpenAI API call was made because APP_ENV is set to a development value.",
        "severity": "mock",
        "affected_components": ["development"],
        "evidence": [
            "Development mock mode is enabled.",
            f"Received {len(ticket_text or '')} ticket characters and {len(log_text or '')} log characters.",
        ],
        "debugging_steps": [
            {
                "step": "Set APP_ENV to a non-development value to call the OpenAI API.",
                "rationale": "Development mode intentionally returns a local mock response.",
            },
            {
                "step": "Configure OPENAI_API_KEY before testing production-like behavior.",
                "rationale": "The real OpenAI client requires an API key.",
            },
        ],
    }

    analysis = SupportAnalysis.model_validate(payload)
    analysis.raw_model_output = payload
    return analysis


async def analyze_support_context(
    ticket_text: str | None,
    log_text: str | None,
    runbook_text: str | None,
) -> tuple[str | None, str, SupportAnalysis]:
    model = get_model()
    if is_dev_environment():
        return "mock-response-dev", model, _mock_analysis(ticket_text, log_text, runbook_text)

    client = get_client()

    try:
        response = await client.responses.create(
            model=model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_prompt(ticket_text, log_text, runbook_text),
            temperature=0.2,
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach OpenAI: {exc}",
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error {exc.status_code}: {exc.message}",
        ) from exc

    return getattr(response, "id", None), model, _parse_analysis(_response_text(response))

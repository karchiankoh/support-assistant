import json
import logging
from typing import Any

from app.schemas import SupportAnalysis


logger = logging.getLogger(__name__)

SUPPORT_ANALYSIS_SYSTEM_INSTRUCTIONS = """
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


def build_prompt(ticket_text: str | None, log_text: str | None, runbook_text: str | None) -> str:
    sections = []
    if ticket_text:
        sections.append(f"<support_ticket>\n{ticket_text}\n</support_ticket>")
    if log_text:
        sections.append(f"<logs>\n{log_text}\n</logs>")
    if runbook_text:
        sections.append(f"<retrieved_support_knowledge>\n{runbook_text}\n</retrieved_support_knowledge>")
    return "\n\n".join(sections)


def response_text(response: Any) -> str:
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


def parse_analysis(text: str) -> SupportAnalysis:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Model response was not valid JSON; attempting to extract JSON object")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.error("Model response did not contain a JSON object")
            return SupportAnalysis(
                issue_summary="The model returned non-JSON output.",
                debugging_steps=[],
                raw_model_output=text,
            )
        payload = json.loads(text[start : end + 1])

    analysis = SupportAnalysis.model_validate(payload)
    analysis.raw_model_output = payload
    return analysis


def mock_analysis(ticket_text: str | None, log_text: str | None, runbook_text: str | None) -> SupportAnalysis:
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

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from fastapi import HTTPException, status
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.config import (
    get_embedding_model,
    get_model,
    get_openai_retry_attempts,
    get_openai_retry_backoff_seconds,
    is_dev_environment,
)
from app.schemas import SupportAnalysis
from app.support_analysis import (
    SUPPORT_ANALYSIS_SYSTEM_INSTRUCTIONS,
    build_prompt,
    mock_analysis,
    parse_analysis,
    response_text,
)


RETRYABLE_OPENAI_STATUS_CODES = (408, 409, 429, 500, 502, 503, 504)
T = TypeVar("T")
logger = logging.getLogger(__name__)


def get_client() -> AsyncOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI client requested without OPENAI_API_KEY")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not configured.",
        )
    return AsyncOpenAI()


async def create_embedding(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        logger.info("Embedding request skipped because no texts were provided")
        return []

    logger.info(
        "OpenAI embedding request started",
        extra={
            "embedding_model": get_embedding_model(),
            "text_count": len(texts),
            "total_characters": sum(len(text) for text in texts),
        },
    )
    client = get_client()
    try:
        response = await _call_openai_with_retries(
            lambda: client.embeddings.create(
                model=get_embedding_model(),
                input=texts,
            )
        )
    except (APIConnectionError, APITimeoutError) as exc:
        logger.exception("OpenAI embeddings request failed because the API could not be reached")
        raise RuntimeError(f"Could not reach OpenAI embeddings API: {exc}") from exc
    except APIStatusError as exc:
        logger.exception(
            "OpenAI embeddings request failed with API status error",
            extra={"status_code": exc.status_code},
        )
        raise RuntimeError(f"OpenAI embeddings API error {exc.status_code}: {exc.message}") from exc

    usage = _extract_openai_token_usage(response)
    logger.info(
        "OpenAI embedding request completed",
        extra={
            "embedding_model": get_embedding_model(),
            "embedding_count": len(response.data),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_input_tokens": usage["cached_input_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
        },
    )
    return [item.embedding for item in response.data]


async def analyze_support_context(
    ticket_text: str | None,
    log_text: str | None,
    runbook_text: str | None,
) -> tuple[str | None, str, SupportAnalysis]:
    model = get_model()
    if is_dev_environment():
        logger.info(
            "Analysis request handled with development mock response",
            extra={
                "model": model,
                "ticket_characters": len(ticket_text or ""),
                "log_characters": len(log_text or ""),
                "runbook_characters": len(runbook_text or ""),
            },
        )
        return "mock-response-dev", model, mock_analysis(ticket_text, log_text, runbook_text)

    logger.info(
        "OpenAI responses request started",
        extra={
            "model": model,
            "ticket_characters": len(ticket_text or ""),
            "log_characters": len(log_text or ""),
            "runbook_characters": len(runbook_text or ""),
        },
    )
    client = get_client()

    try:
        response = await _call_openai_with_retries(
            lambda: client.responses.create(
                model=model,
                instructions=SUPPORT_ANALYSIS_SYSTEM_INSTRUCTIONS,
                input=build_prompt(ticket_text, log_text, runbook_text),
                temperature=0.2,
            )
        )
    except (APIConnectionError, APITimeoutError) as exc:
        logger.exception("OpenAI responses request failed because the API could not be reached")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach OpenAI: {exc}",
        ) from exc
    except APIStatusError as exc:
        logger.exception(
            "OpenAI responses request failed with API status error",
            extra={"status_code": exc.status_code},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error {exc.status_code}: {exc.message}",
        ) from exc

    response_id = getattr(response, "id", None)
    usage = _extract_openai_token_usage(response)
    logger.info(
        "OpenAI responses request completed",
        extra={
            "response_id": response_id,
            "model": model,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_input_tokens": usage["cached_input_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
        },
    )
    return response_id, model, parse_analysis(response_text(response))


def _extract_openai_token_usage(response: Any) -> dict[str, int | None]:
    usage = _get_usage_value(response, "usage")
    input_token_details = _get_usage_value(usage, "input_tokens_details")
    output_token_details = _get_usage_value(usage, "output_tokens_details")

    return {
        "input_tokens": _get_first_int_usage_value(usage, ("input_tokens", "prompt_tokens")),
        "output_tokens": _get_int_usage_value(usage, "output_tokens"),
        "total_tokens": _get_int_usage_value(usage, "total_tokens"),
        "cached_input_tokens": _get_int_usage_value(input_token_details, "cached_tokens"),
        "reasoning_tokens": _get_int_usage_value(output_token_details, "reasoning_tokens"),
    }


def _get_first_int_usage_value(source: Any, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _get_int_usage_value(source, key)
        if value is not None:
            return value
    return None


def _get_int_usage_value(source: Any, key: str) -> int | None:
    value = _get_usage_value(source, key)
    return value if isinstance(value, int) else None


def _get_usage_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


async def _call_openai_with_retries(operation: Callable[[], Awaitable[T]]) -> T:
    attempts = get_openai_retry_attempts()
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except (APIConnectionError, APIStatusError, APITimeoutError) as exc:
            if attempt == attempts or not _is_retryable_openai_error(exc):
                logger.exception(
                    "OpenAI request failed",
                    extra={
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "retryable": _is_retryable_openai_error(exc),
                    },
                )
                raise

            delay_seconds = _retry_delay_seconds(attempt)
            logger.warning(
                "OpenAI request failed; retrying",
                extra={
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "delay_seconds": round(delay_seconds, 3),
                    "retryable": True,
                },
            )
            await asyncio.sleep(delay_seconds)

    raise RuntimeError("OpenAI request retry attempts were exhausted.")


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True

    return isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_OPENAI_STATUS_CODES


def _retry_delay_seconds(attempt: int) -> float:
    base_delay = get_openai_retry_backoff_seconds()
    if base_delay == 0:
        return 0.0

    exponential_delay = base_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0, base_delay)
    return exponential_delay + jitter

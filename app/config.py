import os


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_RETRY_ATTEMPTS = 3
DEFAULT_OPENAI_RETRY_BACKOFF_SECONDS = 0.5
DEV_ENVIRONMENT = "dev"


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_openai_retry_attempts() -> int:
    return max(1, int(os.getenv("OPENAI_RETRY_ATTEMPTS", str(DEFAULT_OPENAI_RETRY_ATTEMPTS))))


def get_openai_retry_backoff_seconds() -> float:
    return max(
        0.0,
        float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", str(DEFAULT_OPENAI_RETRY_BACKOFF_SECONDS))),
    )


def is_dev_environment() -> bool:
    environment = os.getenv("APP_ENV")
    return environment is not None and environment.strip().lower() == DEV_ENVIRONMENT

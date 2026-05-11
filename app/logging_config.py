import logging
import logging.config
import os

from pythonjsonlogger import jsonlogger

from app.request_context import request_id_context, request_path_context


DEFAULT_LOG_LEVEL = "INFO"
LOG_RECORD_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(request_path)s"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        record.request_path = request_path_context.get()
        return True


def get_log_level() -> str:
    return os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_id": {
                    "()": RequestIdFilter,
                }
            },
            "formatters": {
                "json": {
                    "()": jsonlogger.JsonFormatter,
                    "fmt": LOG_RECORD_FORMAT,
                    "rename_fields": {
                        "asctime": "timestamp",
                        "levelname": "level",
                    },
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "filters": ["request_id"],
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": get_log_level(),
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console"],
                    "level": get_log_level(),
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console"],
                    "level": get_log_level(),
                    "propagate": False,
                },
            },
        }
    )

from contextvars import ContextVar


request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
request_path_context: ContextVar[str | None] = ContextVar("request_path", default=None)

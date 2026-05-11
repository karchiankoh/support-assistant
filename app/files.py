import logging
import re
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status

from app.schemas import SourceDocument


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "application/octet-stream"
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadedDocument:
    filename: str
    kind: str
    text: str

    @property
    def source(self) -> SourceDocument:
        return SourceDocument(filename=self.filename, kind=self.kind, characters=len(self.text))


async def read_upload(file: UploadFile, kind: str) -> UploadedDocument:
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            "Upload rejected because content type is unsupported",
            extra={
                "upload_filename": file.filename,
                "kind": kind,
                "content_type": content_type,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{file.filename} has unsupported content type {content_type}. Upload text, markdown, or log files.",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        logger.warning(
            "Upload rejected because file is too large",
            extra={
                "upload_filename": file.filename,
                "kind": kind,
                "content_type": content_type,
                "bytes": len(data),
                "max_upload_bytes": MAX_UPLOAD_BYTES,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{file.filename} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    text = decode_text(data)
    if not text.strip():
        logger.warning(
            "Upload rejected because file contained no readable text",
            extra={
                "upload_filename": file.filename,
                "kind": kind,
                "content_type": content_type,
                "bytes": len(data),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{file.filename} did not contain readable text.",
        )

    logger.info(
        "Upload read successfully",
        extra={
            "upload_filename": file.filename,
            "kind": kind,
            "content_type": content_type,
            "bytes": len(data),
            "characters": len(text),
        },
    )
    return UploadedDocument(filename=file.filename or "upload", kind=kind, text=text)


def combine_documents(documents: list[UploadedDocument]) -> str | None:
    if not documents:
        return None
    return "\n\n".join(
        f"--- {document.kind}: {document.filename} ---\n{document.text}" for document in documents
    )


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        decoded = data.decode("utf-8", errors="replace")

    return decoded.strip()

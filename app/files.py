import re
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status

from app.schemas import SourceDocument


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "application/octet-stream",
    "application/rtf",
    "text/rtf",
}


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
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{file.filename} has unsupported content type {content_type}. Upload text, markdown, log, or RTF files.",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{file.filename} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    text = _decode_text(data)
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{file.filename} did not contain readable text.",
        )

    return UploadedDocument(filename=file.filename or "upload", kind=kind, text=text)


def combine_documents(documents: list[UploadedDocument]) -> str | None:
    if not documents:
        return None
    return "\n\n".join(
        f"--- {document.kind}: {document.filename} ---\n{document.text}" for document in documents
    )


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        decoded = data.decode("utf-8", errors="replace")

    if decoded.lstrip().startswith(r"{\rtf"):
        return _strip_rtf(decoded)
    return decoded.strip()


def _strip_rtf(text: str) -> str:
    text = re.sub(r"{\\fonttbl.*?}", "", text, flags=re.DOTALL)
    text = re.sub(r"{\\colortbl.*?}", "", text, flags=re.DOTALL)
    text = re.sub(r"{\\\*\\expandedcolortbl.*?}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}").replace(r"\\", "\\")
    text = text.replace("\\\n", "\n")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

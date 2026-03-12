"""File loaders — read files and convert to Gemini-compatible Parts."""

from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from google.genai import types

logger = logging.getLogger(__name__)


class FileType(StrEnum):
    """Supported file type categories."""

    TEXT = "text"
    CODE = "code"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ETL = "etl"


@dataclass
class LoadedFile:
    """A file loaded and ready for context injection."""

    path: Path
    file_type: FileType
    content: str | None = None  # For text/code files
    part: types.Part | None = None    # For multimodal files (image, audio, pdf)
    size_bytes: int = 0
    mime_type: str = "text/plain"
    token_estimate: int = 0

    @property
    def display_name(self) -> str:
        """Human-readable file name with relative path."""
        return str(self.path)


# ── Extension → Type Mapping ──────────────────────────────

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".sql", ".sh", ".bash", ".zsh", ".yaml", ".yml", ".json", ".xml",
    ".html", ".css", ".scss", ".toml", ".ini", ".cfg", ".tf", ".hcl",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".r", ".m", ".php", ".pl", ".lua", ".vim", ".dockerfile",
}

_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".csv", ".log", ".env", ".gitignore"}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}

_ETL_EXTENSIONS = {".ktr", ".kjb", ".kdb"}  # Pentaho


def classify_file(filepath: Path) -> FileType:
    """Classify a file by its extension."""
    suffix = filepath.suffix.lower()

    if suffix == ".pdf":
        return FileType.PDF
    if suffix in _IMAGE_EXTENSIONS:
        return FileType.IMAGE
    if suffix in _AUDIO_EXTENSIONS:
        return FileType.AUDIO
    if suffix in _VIDEO_EXTENSIONS:
        return FileType.VIDEO
    if suffix in _CODE_EXTENSIONS:
        return FileType.CODE
    if suffix in _ETL_EXTENSIONS:
        return FileType.ETL
    if suffix in _TEXT_EXTENSIONS:
        return FileType.TEXT

    return FileType.TEXT  # Default fallback


def load_file(filepath: Path) -> LoadedFile:
    """Load a single file and prepare it for context injection.

    - Text/Code/ETL files → read as string
    - PDFs → load as binary Part (Gemini handles PDF natively)
    - Images → load as image Part
    - Audio → load as audio Part
    """
    filepath = filepath.resolve()
    file_type = classify_file(filepath)
    size_bytes = filepath.stat().st_size
    mime_type = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
    logger.debug("Loading file: %s (type=%s, size=%d bytes)", filepath.name, file_type.value, size_bytes)

    if file_type in (FileType.TEXT, FileType.CODE, FileType.ETL):
        return _load_text_file(filepath, file_type, size_bytes)

    if file_type == FileType.PDF:
        return _load_binary_part(filepath, file_type, size_bytes, "application/pdf")

    if file_type == FileType.IMAGE:
        return _load_binary_part(filepath, file_type, size_bytes, mime_type)

    if file_type in (FileType.AUDIO, FileType.VIDEO):
        return _load_binary_part(filepath, file_type, size_bytes, mime_type)

    return _load_text_file(filepath, file_type, size_bytes)


def _load_text_file(filepath: Path, file_type: FileType, size_bytes: int) -> LoadedFile:
    """Load a text-based file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with detected encoding
        raw = filepath.read_bytes()
        import chardet
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        content = raw.decode(encoding, errors="replace")

    # Rough token estimate (1 token ≈ 4 chars for English)
    token_estimate = len(content) // 4

    # Wrap code files with file path context
    if file_type in (FileType.CODE, FileType.ETL):
        content = f"```{filepath.suffix.lstrip('.')} ({filepath.name})\n{content}\n```"

    return LoadedFile(
        path=filepath,
        file_type=file_type,
        content=content,
        size_bytes=size_bytes,
        mime_type="text/plain",
        token_estimate=token_estimate,
    )


def _load_binary_part(filepath: Path, file_type: FileType, size_bytes: int, mime_type: str) -> LoadedFile:
    """Load a binary file as a Gemini Part."""
    data = filepath.read_bytes()

    part = types.Part.from_bytes(data=data, mime_type=mime_type)

    return LoadedFile(
        path=filepath,
        file_type=file_type,
        part=part,
        size_bytes=size_bytes,
        mime_type=mime_type,
        token_estimate=size_bytes // 4,  # Rough estimate for binary
    )


def load_pdf_with_text(filepath: Path) -> LoadedFile:
    """Load a PDF extracting text (for cases where you need the raw text).

    Uses pymupdf for text extraction. Falls back to binary Part if extraction fails.
    """
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(filepath))
        text_parts: list[str] = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{text}")
        doc.close()

        content = "\n\n".join(text_parts)
        if content.strip():
            return LoadedFile(
                path=filepath,
                file_type=FileType.PDF,
                content=content,
                size_bytes=filepath.stat().st_size,
                mime_type="application/pdf",
                token_estimate=len(content) // 4,
            )
    except Exception:
        logger.warning("Failed to extract text from PDF: %s — using binary mode", filepath)

    # Fallback to binary part
    return load_file(filepath)

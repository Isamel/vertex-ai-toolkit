"""File filters — .gitignore patterns, binary detection, size limits."""

from __future__ import annotations

import logging
from pathlib import Path

import chardet
import pathspec

from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# Common binary file signatures (magic bytes)
_BINARY_SIGNATURES = [
    b"\x89PNG",       # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF8",          # GIF
    b"PK\x03\x04",   # ZIP/XLSX/DOCX
    b"%PDF",          # PDF
    b"\x7fELF",       # ELF binary
    b"\xca\xfe\xba\xbe",  # Java class
    b"\x00\x00\x01\x00",  # ICO
]


def is_binary_file(filepath: Path, *, sample_size: int = 8192) -> bool:
    """Detect if a file is binary by checking for null bytes and encoding."""
    try:
        with filepath.open("rb") as f:
            chunk = f.read(sample_size)

        if not chunk:
            return False

        # Check magic bytes
        for sig in _BINARY_SIGNATURES:
            if chunk.startswith(sig):
                return True

        # Check for null bytes (strong binary indicator)
        if b"\x00" in chunk:
            return True

        # Try to detect encoding
        result = chardet.detect(chunk)
        if result["encoding"] is None:
            return True

        return False  # noqa: TRY300
    except OSError:
        return True


def build_file_filter(settings: Settings, root_dir: Path) -> pathspec.PathSpec[pathspec.patterns.GitWildMatchPattern]:  # type: ignore[type-arg,name-defined]
    """Build a PathSpec from .gitignore + configured ignore patterns.

    Merges:
    1. The project's .gitignore (if exists)
    2. Global ignore patterns from config
    """
    patterns: list[str] = list(settings.context.ignore_patterns)

    # Try loading .gitignore
    gitignore = root_dir / ".gitignore"
    if gitignore.exists():
        try:
            gitignore_patterns = gitignore.read_text(encoding="utf-8").splitlines()
            patterns.extend(gitignore_patterns)
            logger.debug("Loaded %d patterns from .gitignore", len(gitignore_patterns))
        except OSError:
            logger.warning("Could not read .gitignore at %s", gitignore)

    return pathspec.PathSpec.from_lines("gitignore", patterns)


def should_include_file(
    filepath: Path,
    *,
    settings: Settings,
    spec: pathspec.PathSpec[pathspec.patterns.GitWildMatchPattern],  # type: ignore[type-arg,name-defined]
    root_dir: Path,
) -> bool:
    """Determine if a file should be included in the context.

    Checks:
    1. Not matched by ignore patterns
    2. Not binary (unless it's a supported media type)
    3. Within file size limits
    4. Has a supported extension
    """
    # Check ignore patterns
    try:
        rel_path = filepath.relative_to(root_dir)
    except ValueError:
        rel_path = filepath

    if spec.match_file(str(rel_path)):
        return False

    # Check file size
    try:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > settings.context.max_file_size_mb:
            logger.warning("File too large (%.1f MB): %s", size_mb, filepath)
            return False
    except OSError:
        return False

    # Get all supported extensions
    all_extensions: set[str] = set()
    media_extensions: set[str] = set()
    for category, exts in settings.context.supported_extensions.items():
        all_extensions.update(exts)
        if category == "media":
            media_extensions.update(exts)

    suffix = filepath.suffix.lower()

    # Check if extension is supported
    if suffix not in all_extensions:
        return False

    # Media files are allowed even if binary
    if suffix in media_extensions:
        return True

    # For non-media files, reject binary
    if is_binary_file(filepath):
        logger.debug("Skipping binary file: %s", filepath)
        return False

    return True

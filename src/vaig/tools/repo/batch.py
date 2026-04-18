"""Batch processing models and chunking utilities for repository code analysis.

This module provides:
- :class:`Tier` — priority classification for repository entries.
- :class:`TriagedEntry` — a single file entry with tier assignment.
- :class:`TreeTriageReport` — full triage output for a repository tree.
- :class:`FileChunk` — a language-aware content chunk of a source file.
- :class:`BatchPlan` — an ordered plan of file chunks to process.
- :func:`chunk_file` — split file content into overlapping chunks with
  language-aware boundary detection (Python class/function boundaries).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field


class Tier(StrEnum):
    """Priority tier for repository file triage.

    Attributes:
        TIER_1: Highest priority — core source files, entry points.
        TIER_2: Medium priority — supporting modules, tests.
        TIER_3: Lowest priority — configuration, data, generated files.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


class TriagedEntry(BaseModel):
    """A single file entry from a repository tree with tier assignment.

    Attributes:
        path: Repository-relative file path (e.g. ``src/main.py``).
        tier: Priority tier for processing order.
        reason: Human-readable rationale for the tier assignment.
        size_bytes: File size in bytes (0 when unknown).
        sha: Git blob SHA (empty when unknown).
    """

    path: str
    tier: Tier
    reason: str = ""
    size_bytes: int = 0
    sha: str = ""


class TreeTriageReport(BaseModel):
    """Full triage output for a repository file tree.

    Attributes:
        owner: Repository owner.
        repo: Repository name.
        ref: Branch/tag/SHA used for the triage.
        entries: All triaged file entries, in priority order.
        total_files: Total number of files in the original tree.
        skipped_files: Number of files excluded (binary, oversized, etc.).
    """

    owner: str
    repo: str
    ref: str = "main"
    entries: list[TriagedEntry] = Field(default_factory=list)
    total_files: int = 0
    skipped_files: int = 0

    @property
    def tier_1(self) -> list[TriagedEntry]:
        """Return only TIER_1 entries."""
        return [e for e in self.entries if e.tier == Tier.TIER_1]

    @property
    def tier_2(self) -> list[TriagedEntry]:
        """Return only TIER_2 entries."""
        return [e for e in self.entries if e.tier == Tier.TIER_2]

    @property
    def tier_3(self) -> list[TriagedEntry]:
        """Return only TIER_3 entries."""
        return [e for e in self.entries if e.tier == Tier.TIER_3]


class FileChunk(BaseModel):
    """A single chunk of a source file for batch LLM processing.

    Chunks are created with :func:`chunk_file`.  For Python files, chunk
    boundaries are aligned to ``def`` / ``class`` statement starts to avoid
    splitting a function definition across chunks.

    Attributes:
        path: Repository-relative file path.
        chunk_id: Zero-based index of this chunk within the file.
        line_start: 1-based first line number of the chunk.
        line_end: 1-based last line number of the chunk (inclusive).
        sha: Git blob SHA (empty when unknown).
        content: Raw text content of the chunk.
        overlap_prev: Number of lines shared with the preceding chunk
            (0 for the first chunk).
    """

    path: str
    chunk_id: int
    line_start: int
    line_end: int
    sha: str = ""
    content: str
    overlap_prev: int = 0


class BatchPlan(BaseModel):
    """An ordered plan of file chunks ready for sequential LLM processing.

    Attributes:
        owner: Repository owner.
        repo: Repository name.
        ref: Branch/tag/SHA.
        chunks: Ordered list of :class:`FileChunk` objects.
        total_tokens_estimate: Rough token estimate (chars / 4).
    """

    owner: str
    repo: str
    ref: str = "main"
    chunks: list[FileChunk] = Field(default_factory=list)
    total_tokens_estimate: int = 0


# ── Python boundary detection ─────────────────────────────────

# Regex that matches lines starting a top-level or indented def/class.
_PY_BOUNDARY_RE = re.compile(r"^[ \t]*(def |class )", re.MULTILINE)


def _find_python_boundaries(lines: list[str]) -> list[int]:
    """Return 0-based line indices where a Python def/class begins."""
    boundaries: list[int] = []
    for idx, line in enumerate(lines):
        if _PY_BOUNDARY_RE.match(line):
            boundaries.append(idx)
    return boundaries


def chunk_file(
    path: str,
    content: str,
    *,
    max_lines: int = 300,
    overlap_lines: int = 20,
    sha: str = "",
) -> list[FileChunk]:
    """Split *content* into overlapping :class:`FileChunk` objects.

    For Python files (``path`` ends with ``.py``), chunk boundaries are
    snapped to the nearest ``def``/``class`` statement **before** the
    hard split point to avoid cutting a function in the middle.

    Args:
        path: Repository-relative path (used to detect language).
        content: Full file content as a string.
        max_lines: Target maximum lines per chunk (default 300).
        overlap_lines: Number of lines to repeat at the start of the next
            chunk from the end of the previous one (default 20).
        sha: Optional git blob SHA to embed in each chunk.

    Returns:
        Ordered list of :class:`FileChunk` objects.  A single-chunk list is
        returned for files with fewer than *max_lines* lines.
    """
    all_lines = content.splitlines(keepends=True)
    total = len(all_lines)

    if total == 0:
        return [
            FileChunk(
                path=path,
                chunk_id=0,
                line_start=1,
                line_end=1,
                sha=sha,
                content="",
                overlap_prev=0,
            )
        ]

    # Fast path — file fits in one chunk
    if total <= max_lines:
        return [
            FileChunk(
                path=path,
                chunk_id=0,
                line_start=1,
                line_end=total,
                sha=sha,
                content="".join(all_lines),
                overlap_prev=0,
            )
        ]

    is_python = path.endswith(".py")
    py_boundaries: frozenset[int] = frozenset()
    if is_python:
        py_boundaries = frozenset(_find_python_boundaries(all_lines))

    chunks: list[FileChunk] = []
    start = 0  # 0-based

    while start < total:
        end = min(start + max_lines, total)  # exclusive

        # Snap end to nearest Python boundary before hard cut (Python only)
        if is_python and end < total and py_boundaries:
            # Search backwards from end-1 for a boundary line
            snap: int | None = None
            for candidate in range(end - 1, start, -1):
                if candidate in py_boundaries:
                    snap = candidate
                    break
            if snap is not None and snap > start:
                end = snap  # end is exclusive → chunk ends just before snap

        chunk_lines = all_lines[start:end]
        overlap = min(overlap_lines, start) if start > 0 else 0

        chunks.append(
            FileChunk(
                path=path,
                chunk_id=len(chunks),
                line_start=start + 1,  # 1-based
                line_end=start + len(chunk_lines),  # 1-based inclusive
                sha=sha,
                content="".join(chunk_lines),
                overlap_prev=overlap,
            )
        )

        # Advance, rewinding by overlap_lines for the next chunk
        next_start = end - overlap_lines
        if next_start <= start:
            # Guard against infinite loop when max_lines <= overlap_lines
            next_start = end
        start = next_start

    return chunks


__all__: Sequence[str] = [
    "BatchPlan",
    "FileChunk",
    "Tier",
    "TriagedEntry",
    "TreeTriageReport",
    "chunk_file",
]

"""Manifest-aware chunkers (SPEC-V2-REPO-03).

Per-kind chunker interface and implementations:

- ``Chunk``              – a semantic chunk from a file
- ``ChunkerProtocol``    – protocol for all chunkers
- ``YamlDocChunker``     – splits multi-document YAML on ``---`` separators
- ``TerraformChunker``   – splits on top-level HCL blocks
- ``MarkdownChunker``    – splits on H2 headings, preserves H1 context
- ``FallbackLineChunker``– 400-line windows with 40-line overlap
- ``ChunkingError``      – raised when a chunker cannot parse a file
- ``get_chunker()``      – selects the appropriate chunker for a file kind
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from typing import IO, Protocol

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Data models ───────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A semantic chunk from a file."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    token_estimate: int  # chars / 4 rough estimate
    kind: str  # "yaml_doc" | "tf_resource" | "helm_block" | ...
    outline: str  # Short header, e.g. "Deployment/istio-ingressgateway"


class ChunkingError(Exception):
    """Raised when a chunker cannot parse a file."""


# ── Protocol ──────────────────────────────────────────────────────────────────


class ChunkerProtocol(Protocol):
    """Protocol for manifest-aware chunkers."""

    def chunk(self, content: str, path: str) -> list[Chunk]:
        ...

    def chunk_stream(self, fh: IO[str], path: str) -> Iterator[Chunk]:
        ...


# ── Helpers ───────────────────────────────────────────────────────────────────

_DOC_SEP_RE = re.compile(r"^---\s*$", re.MULTILINE)
_TF_BLOCK_RE = re.compile(
    r'^(resource|module|variable|locals|data|output|provider|terraform)\s*(?:"[^"]*"\s*(?:"[^"]*"\s*)?)?\{',
    re.MULTILINE,
)
_H2_RE = re.compile(r"^## ", re.MULTILINE)
_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


def _token_estimate(text: str) -> int:
    return len(text) // 4


def _make_chunk(
    file_path: str,
    content: str,
    start_line: int,
    end_line: int,
    kind: str,
    outline: str,
) -> Chunk:
    return Chunk(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        token_estimate=_token_estimate(content),
        kind=kind,
        outline=outline,
    )


def _lines_to_range(text: str, base_line: int) -> tuple[int, int]:
    """Return (start_line, end_line) for *text* given *base_line* offset (1-based)."""
    lines = text.splitlines()
    return base_line, base_line + max(len(lines) - 1, 0)


# ── YAML doc chunker ──────────────────────────────────────────────────────────


def _yaml_outline(raw: str) -> str:
    """Extract 'kind/metadata.name' outline from a YAML document string."""
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError:
        return "yaml_doc"
    if not isinstance(doc, dict):
        return "yaml_doc"
    kind = str(doc.get("kind", ""))
    name = ""
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        name = str(metadata.get("name", ""))
    if kind and name:
        return f"{kind}/{name}"
    if kind:
        return kind
    return "yaml_doc"


def _split_yaml_doc_on_keys(
    raw: str,
    path: str,
    base_line: int,
    max_chunk_tokens: int,
) -> list[Chunk]:
    """Split an oversized YAML document on top-level keys.

    Collects each top-level key section, then groups them greedily into
    chunks that stay under *max_chunk_tokens*.
    """
    lines = raw.splitlines(keepends=True)

    # Phase 1: find top-level key boundaries
    key_starts: list[int] = []  # 0-based line indices of top-level keys
    for i, line in enumerate(lines):
        is_top_key = bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*\s*:", line)) and not line.startswith(" ")
        if is_top_key:
            key_starts.append(i)

    if not key_starts:
        # No top-level keys found; return as single chunk
        end_line = base_line + len(lines) - 1
        return [_make_chunk(path, raw, base_line, end_line, "yaml_doc", _yaml_outline(raw))]

    # Build key sections: (start_idx, end_idx, text)
    sections: list[tuple[int, int, str]] = []
    for k, start in enumerate(key_starts):
        end = key_starts[k + 1] - 1 if k + 1 < len(key_starts) else len(lines) - 1
        sections.append((start, end, "".join(lines[start : end + 1])))

    # Phase 2: greedily group sections into chunks ≤ max_chunk_tokens
    chunks: list[Chunk] = []
    group_sections: list[tuple[int, int, str]] = []
    group_tokens = 0

    def flush_group() -> None:
        if not group_sections:
            return
        text = "".join(s[2] for s in group_sections)
        start_line = base_line + group_sections[0][0]
        end_line = base_line + group_sections[-1][1]
        chunks.append(_make_chunk(path, text, start_line, end_line, "yaml_doc", _yaml_outline(text)))

    for sec_start, sec_end, sec_text in sections:
        sec_tokens = _token_estimate(sec_text)
        if group_sections and group_tokens + sec_tokens > max_chunk_tokens:
            flush_group()
            group_sections = []
            group_tokens = 0
        group_sections.append((sec_start, sec_end, sec_text))
        group_tokens += sec_tokens

    flush_group()
    return chunks if chunks else [_make_chunk(path, raw, base_line, base_line + len(lines) - 1, "yaml_doc", _yaml_outline(raw))]


class YamlDocChunker:
    """Splits multi-document YAML files on ``---`` separators."""

    def __init__(self, max_chunk_tokens: int = 2000) -> None:
        self.max_chunk_tokens = max_chunk_tokens

    def chunk(self, content: str, path: str) -> list[Chunk]:
        return list(self.chunk_stream(io.StringIO(content), path))

    def chunk_stream(self, fh: IO[str], path: str) -> Iterator[Chunk]:
        content = fh.read()

        # Split on YAML document separators; keep content between them
        # We split on lines that are exactly '---' (with optional trailing space)
        raw_docs = _DOC_SEP_RE.split(content)

        # Track line numbers across documents
        line_offset = 1
        for i, raw in enumerate(raw_docs):
            # Strip a single leading newline from docs after the first separator;
            # _DOC_SEP_RE.split() leaves a leading \n at the start of subsequent
            # parts which would cause off-by-one errors in start_line/end_line.
            if i > 0 and raw.startswith("\n"):
                raw = raw[1:]
                line_offset += 1

            # Skip empty docs (e.g. leading ---)
            stripped = raw.strip()
            if not stripped:
                # Still advance line counter
                line_offset += raw.count("\n") + 1
                continue

            doc_lines = raw.splitlines()
            start_line = line_offset
            end_line = line_offset + len(doc_lines) - 1

            # Validate YAML
            try:
                yaml.safe_load(stripped)
            except yaml.YAMLError as exc:
                logger.warning("ChunkingError: malformed YAML in %s: %s — falling back", path, exc)
                raise ChunkingError(f"Malformed YAML in {path}") from exc

            if _token_estimate(raw) > self.max_chunk_tokens:
                yield from _split_yaml_doc_on_keys(raw, path, start_line, self.max_chunk_tokens)
            else:
                outline = _yaml_outline(stripped)
                yield _make_chunk(path, raw, start_line, end_line, "yaml_doc", outline)

            line_offset = end_line + 2  # +1 for next line, +1 for separator line


# ── Terraform chunker ─────────────────────────────────────────────────────────


def _tf_outline(header: str) -> str:
    """Build outline from a Terraform block header line."""
    m = re.match(
        r'^(resource|module|variable|locals|data|output|provider|terraform)\s*(?:"([^"]*)"\s*(?:"([^"]*)"\s*)?)?',
        header.strip(),
    )
    if not m:
        return header.strip().split("{")[0].strip()
    block_type = m.group(1)
    arg1 = m.group(2) or ""
    arg2 = m.group(3) or ""
    if arg2:
        return f'{block_type} "{arg1}" "{arg2}"'
    if arg1:
        return f'{block_type} "{arg1}"'
    return block_type


class TerraformChunker:
    """Splits Terraform files on top-level HCL blocks."""

    def __init__(self, max_chunk_tokens: int = 2000) -> None:
        self.max_chunk_tokens = max_chunk_tokens

    def chunk(self, content: str, path: str) -> list[Chunk]:
        return list(self.chunk_stream(io.StringIO(content), path))

    def chunk_stream(self, fh: IO[str], path: str) -> Iterator[Chunk]:
        content = fh.read()
        lines = content.splitlines(keepends=True)

        # Find all top-level block start positions
        block_starts: list[int] = []  # 0-based line indices
        for i, line in enumerate(lines):
            if _TF_BLOCK_RE.match(line):
                block_starts.append(i)

        if not block_starts:
            # No blocks found — yield the whole file as one chunk
            yield _make_chunk(path, content, 1, len(lines), "tf_resource", "terraform_file")
            return

        # Build ranges: each block from its start to the line before the next block
        ranges: list[tuple[int, int]] = []
        for idx, start in enumerate(block_starts):
            end = block_starts[idx + 1] - 1 if idx + 1 < len(block_starts) else len(lines) - 1
            ranges.append((start, end))

        for start, end in ranges:
            chunk_lines = lines[start : end + 1]
            chunk_text = "".join(chunk_lines)
            outline = _tf_outline(lines[start])
            yield _make_chunk(path, chunk_text, start + 1, end + 1, "tf_resource", outline)


# ── Markdown chunker ──────────────────────────────────────────────────────────


class MarkdownChunker:
    """Splits Markdown files on H2 headings, preserving H1 context."""

    def __init__(self, max_chunk_tokens: int = 2000) -> None:
        self.max_chunk_tokens = max_chunk_tokens

    def chunk(self, content: str, path: str) -> list[Chunk]:
        return list(self.chunk_stream(io.StringIO(content), path))

    def chunk_stream(self, fh: IO[str], path: str) -> Iterator[Chunk]:
        content = fh.read()
        lines = content.splitlines(keepends=True)

        # Find H1 context (first H1 in file)
        h1_context = ""
        h1_match = _H1_RE.search(content)
        if h1_match:
            h1_context = h1_match.group(1).strip()

        # Find all H2 positions
        h2_positions: list[int] = []
        for i, line in enumerate(lines):
            if _H2_RE.match(line):
                h2_positions.append(i)

        if not h2_positions:
            # No H2 — yield entire file
            outline = h1_context or path
            yield _make_chunk(path, content, 1, len(lines), "markdown", outline)
            return

        # Preamble before first H2
        if h2_positions[0] > 0:
            preamble = "".join(lines[: h2_positions[0]])
            outline = h1_context or "preamble"
            yield _make_chunk(path, preamble, 1, h2_positions[0], "markdown", outline)

        # Each H2 section
        for idx, start in enumerate(h2_positions):
            end = h2_positions[idx + 1] - 1 if idx + 1 < len(h2_positions) else len(lines) - 1
            section_lines = lines[start : end + 1]
            section_text = "".join(section_lines)
            heading = lines[start].lstrip("#").strip()
            outline = f"{h1_context} / {heading}" if h1_context else heading
            yield _make_chunk(path, section_text, start + 1, end + 1, "markdown", outline)


# ── Fallback line chunker ─────────────────────────────────────────────────────


class FallbackLineChunker:
    """400-line windows with 40-line overlap."""

    def __init__(self, window: int = 400, overlap: int = 40) -> None:
        self.window = window
        self.overlap = overlap

    def chunk(self, content: str, path: str) -> list[Chunk]:
        return list(self.chunk_stream(io.StringIO(content), path))

    def chunk_stream(self, fh: IO[str], path: str) -> Iterator[Chunk]:
        content = fh.read()
        lines = content.splitlines(keepends=True)

        if not lines:
            return

        step = self.window - self.overlap
        pos = 0
        while pos < len(lines):
            end = min(pos + self.window, len(lines))
            chunk_lines = lines[pos:end]
            chunk_text = "".join(chunk_lines)
            start_line = pos + 1
            end_line = end
            outline = f"lines {start_line}-{end_line}"
            yield _make_chunk(path, chunk_text, start_line, end_line, "fallback", outline)
            if end >= len(lines):
                break
            pos += step


# ── Selector ──────────────────────────────────────────────────────────────────


def get_chunker(kind: str) -> ChunkerProtocol:
    """Select the appropriate chunker based on file kind."""
    match kind:
        case "yaml" | "k8s_manifest" | "argocd" | "istio_crd" | "kustomization":
            return YamlDocChunker()
        case "terraform" | "terraform_gke":
            return TerraformChunker()
        case "markdown":
            return MarkdownChunker()
        case _:
            return FallbackLineChunker()


def chunk_file(content: str, path: str, kind: str) -> list[Chunk]:
    """Chunk a file, falling back to FallbackLineChunker on ChunkingError."""
    chunker = get_chunker(kind)
    try:
        return chunker.chunk(content, path)
    except ChunkingError:
        logger.warning("Falling back to FallbackLineChunker for %s", path)
        return FallbackLineChunker().chunk(content, path)

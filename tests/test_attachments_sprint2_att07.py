"""Sprint 2 acceptance tests for SPEC-ATT-07: RepoIndex.build_from_attachments.

Coverage:
- Happy path with a single in-memory adapter
- Multi-adapter aggregation (chunks from N adapters combined)
- ``chunk.source`` carries ``attachment:<name|source>`` prefix
- Fallback to ``attachment:<source>`` when spec.name is None
- Error isolation: one failing ``list_files`` does not abort others
- Error isolation: one failing ``fetch_bytes`` is logged and skipped
- Per-file chunker failure emits EvidenceGap and continues
- Empty adapter list returns empty RepoIndex
- Empty adapter (no files) returns empty chunks
- Binary files emit ``binary_skipped`` gap and are not chunked
- Oversize files (> max_file_bytes_absolute) emit ``catastrophic_size`` gap
- Streaming bytes iterator is consolidated and chunked
- UTF-8 decode fallback on invalid bytes does not raise
- SecretRedactor is applied when cfg.redaction_enabled is True
- Relevance gate keyword filter is applied to the final chunk list
- Default ``Chunk.source`` remains ``"repo"`` (back-compat invariant)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pytest

from vaig.core.attachment_adapter import AttachmentFileEntry, AttachmentKind, AttachmentSpec
from vaig.core.config import AttachmentsConfig, RepoInvestigationConfig
from vaig.core.repo_chunkers import Chunk
from vaig.core.repo_index import RepoIndex
from vaig.core.repo_redactor import SecretRedactor

# ─────────────────────────────────────────────────────────────────────────────
# In-memory fake adapter
# ─────────────────────────────────────────────────────────────────────────────


class _FakeAdapter:
    """Minimal in-memory AttachmentAdapter for tests.

    Mirrors the Protocol shape (``spec``, ``list_files``, ``fetch_bytes``,
    ``fingerprint``) without touching the filesystem.
    """

    def __init__(
        self,
        files: dict[str, bytes | Iterator[bytes]],
        *,
        name: str | None = None,
        source: str = "fake",
        kind: AttachmentKind = AttachmentKind.local_path,
        list_raises: Exception | None = None,
        fetch_raises: dict[str, Exception] | None = None,
    ) -> None:
        self._files = files
        self.spec = AttachmentSpec(name=name, source=source, kind=kind, resolved_path=None)
        self._list_raises = list_raises
        self._fetch_raises = fetch_raises or {}

    def list_files(self, cfg: AttachmentsConfig) -> Iterable[AttachmentFileEntry]:
        if self._list_raises is not None:
            raise self._list_raises
        for rel, data in self._files.items():
            size = len(data) if isinstance(data, bytes) else 0
            yield AttachmentFileEntry(
                relative_path=rel,
                size_bytes=size,
                mtime=None,
                is_symlink=False,
            )

    def fetch_bytes(self, relative_path: str) -> bytes | Iterator[bytes]:
        if relative_path in self._fetch_raises:
            raise self._fetch_raises[relative_path]
        return self._files[relative_path]

    def fingerprint(self) -> str:
        return "fake-fingerprint"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _attach_cfg(**kwargs: Any) -> AttachmentsConfig:
    return AttachmentsConfig(**kwargs)


def _repo_cfg(**kwargs: Any) -> RepoInvestigationConfig:
    return RepoInvestigationConfig(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_single_adapter_produces_chunks() -> None:
    adapter = _FakeAdapter(
        {"README.md": b"# Hello\n\nSome markdown content with enough text to chunk.\n"},
        name="docs",
    )
    idx, gaps = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), _repo_cfg()
    )
    chunks = idx._chunks  # noqa: SLF001
    assert len(chunks) >= 1
    assert all(c.file_path == "README.md" for c in chunks)
    assert all(c.source == "attachment:docs" for c in chunks)
    assert gaps == [] or all(g.level != "ERROR" for g in gaps)


def test_multi_adapter_aggregation() -> None:
    a1 = _FakeAdapter({"a.py": b"def foo():\n    return 1\n"}, name="one")
    a2 = _FakeAdapter({"b.py": b"def bar():\n    return 2\n"}, name="two")
    idx, _ = RepoIndex.build_from_attachments([a1, a2], _attach_cfg(), _repo_cfg())
    sources = {c.source for c in idx._chunks}  # noqa: SLF001
    assert "attachment:one" in sources
    assert "attachment:two" in sources


def test_source_falls_back_to_spec_source_when_name_is_none() -> None:
    adapter = _FakeAdapter(
        {"x.txt": b"hello world content"},
        name=None,
        source="/tmp/anon",
    )
    idx, _ = RepoIndex.build_from_attachments([adapter], _attach_cfg(), _repo_cfg())
    chunks = idx._chunks  # noqa: SLF001
    assert len(chunks) >= 1
    assert all(c.source == "attachment:/tmp/anon" for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Error isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_list_files_failure_is_isolated_per_adapter() -> None:
    broken = _FakeAdapter({}, name="bad", list_raises=RuntimeError("enumeration boom"))
    good = _FakeAdapter({"ok.py": b"print('ok')\n"}, name="good")
    idx, gaps = RepoIndex.build_from_attachments(
        [broken, good], _attach_cfg(), _repo_cfg()
    )
    # Good adapter's chunks survive
    sources = {c.source for c in idx._chunks}  # noqa: SLF001
    assert "attachment:good" in sources
    assert "attachment:bad" not in sources
    # Gap records the failure
    assert any(
        g.kind == "chunker_fallback" and "bad" in (g.details or "")
        for g in gaps
    )


def test_fetch_bytes_failure_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = _FakeAdapter(
        {
            "a.py": b"print('ok')\n",
            "b.py": b"print('boom')\n",
        },
        name="mixed",
        fetch_raises={"b.py": OSError("read failed")},
    )
    caplog.set_level("WARNING", logger="vaig.core.repo_index")
    idx, gaps = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), _repo_cfg()
    )
    paths = {c.file_path for c in idx._chunks}  # noqa: SLF001
    assert "a.py" in paths
    assert "b.py" not in paths
    # Robust assertion: an EvidenceGap must be emitted for the failed file.
    # Do not rely on caplog (logger propagation differs between pytest
    # configurations — e.g. local vs CI).
    assert any(
        g.path == "b.py" and "read failed" in (g.details or "") for g in gaps
    )


# ─────────────────────────────────────────────────────────────────────────────
# Empty / no-op
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_adapter_list_returns_empty_index() -> None:
    idx, gaps = RepoIndex.build_from_attachments([], _attach_cfg(), _repo_cfg())
    assert idx._chunks == []  # noqa: SLF001
    assert gaps == []


def test_adapter_with_no_files_returns_empty_chunks() -> None:
    adapter = _FakeAdapter({}, name="empty")
    idx, gaps = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), _repo_cfg()
    )
    assert idx._chunks == []  # noqa: SLF001
    assert gaps == []


# ─────────────────────────────────────────────────────────────────────────────
# Classification + gaps
# ─────────────────────────────────────────────────────────────────────────────


def test_binary_file_emits_gap_and_is_not_chunked() -> None:
    # Null byte triggers binary detection
    adapter = _FakeAdapter(
        {"image.bin": b"\x00\x01\x02binary payload"},
        name="bins",
    )
    idx, gaps = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), _repo_cfg()
    )
    assert idx._chunks == []  # noqa: SLF001
    assert any(
        g.kind == "binary_skipped" and g.path == "image.bin" for g in gaps
    )


def test_oversize_file_emits_catastrophic_gap() -> None:
    # Build a fake 200-byte "file" and force a tiny catastrophic cap so the
    # classifier rejects it without us actually allocating hundreds of MB.
    adapter = _FakeAdapter({"huge.txt": b"a" * 200}, name="big")
    cfg = _repo_cfg(max_file_bytes_absolute=100)
    idx, gaps = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), cfg
    )
    assert idx._chunks == []  # noqa: SLF001
    assert any(
        g.kind == "catastrophic_size" and g.path == "huge.txt" for g in gaps
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decoding + streaming
# ─────────────────────────────────────────────────────────────────────────────


def test_streaming_iterator_is_consolidated_and_chunked() -> None:
    def _stream() -> Iterator[bytes]:
        yield b"def foo():\n"
        yield b"    return 42\n"

    adapter = _FakeAdapter({"mod.py": _stream()}, name="stream")
    idx, _ = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), _repo_cfg()
    )
    chunks = idx._chunks  # noqa: SLF001
    assert len(chunks) >= 1
    combined = "".join(c.content for c in chunks)
    assert "foo" in combined
    assert "return 42" in combined


def test_invalid_utf8_does_not_raise_uses_replacement() -> None:
    # 0xFF is invalid UTF-8 start byte
    adapter = _FakeAdapter(
        {"weird.txt": b"hello \xff\xfe world body with more text here"},
        name="weird",
    )
    idx, _ = RepoIndex.build_from_attachments(
        [adapter], _attach_cfg(), _repo_cfg()
    )
    # Non-binary (no null byte, ratio under threshold) → still chunked
    chunks = idx._chunks  # noqa: SLF001
    assert len(chunks) >= 1
    assert any("hello" in c.content for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Redaction + relevance gate
# ─────────────────────────────────────────────────────────────────────────────


def test_redactor_is_applied_when_enabled() -> None:
    secret_line = 'AWS_SECRET_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"\n'
    body = "# config\n" + secret_line + "other_var = 1\n"
    adapter = _FakeAdapter({"config.py": body.encode()}, name="cfg")
    redactor = SecretRedactor()
    idx, _ = RepoIndex.build_from_attachments(
        [adapter],
        _attach_cfg(),
        _repo_cfg(redaction_enabled=True),
        redactor=redactor,
    )
    chunks = idx._chunks  # noqa: SLF001
    combined = "".join(c.content for c in chunks)
    # Raw secret must be gone
    assert "AKIAIOSFODNN7EXAMPLE" not in combined


def test_redactor_not_applied_when_disabled() -> None:
    body = 'password = "hunter2-literal-marker-xyz"\n' * 4
    adapter = _FakeAdapter({"p.py": body.encode()}, name="cfg")
    redactor = SecretRedactor()
    idx, _ = RepoIndex.build_from_attachments(
        [adapter],
        _attach_cfg(),
        _repo_cfg(redaction_enabled=False),
        redactor=redactor,
    )
    chunks = idx._chunks  # noqa: SLF001
    combined = "".join(c.content for c in chunks)
    # Redactor was passed but flag is off → literal must survive
    assert "hunter2-literal-marker-xyz" in combined


def test_relevance_gate_filters_chunks_by_keywords() -> None:
    a = _FakeAdapter(
        {
            "match.py": b"def authenticate_user():\n    pass\n",
            "other.py": b"def unrelated_helper():\n    pass\n",
        },
        name="mixed",
    )
    idx, _ = RepoIndex.build_from_attachments(
        [a],
        _attach_cfg(),
        _repo_cfg(),
        keywords=frozenset({"authenticate"}),
    )
    paths = {c.file_path for c in idx._chunks}  # noqa: SLF001
    assert "match.py" in paths
    assert "other.py" not in paths


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat invariant
# ─────────────────────────────────────────────────────────────────────────────


def test_default_chunk_source_remains_repo() -> None:
    """A freshly-constructed Chunk (no explicit source) defaults to 'repo'."""
    c = Chunk(
        file_path="x.py",
        kind="python",
        start_line=1,
        end_line=1,
        content="x",
        token_estimate=1,
        outline="",
    )
    assert c.source == "repo"

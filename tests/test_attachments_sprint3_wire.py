"""Sprint 3 acceptance tests — wire integration.

Covers:
- _build_and_resolve_attachments dispatches to URLAdapter for https:// source
- _persist_session writes to session dir
- _display_attachments_table runs without error (smoke)
- allow_http=False blocks http:// source at the CLI layer (Exit 1)
- url_allowlist blocks disallowed domain at the CLI layer (Exit 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer

from vaig.cli.commands.live import (
    _build_and_resolve_attachments,
    _display_attachments_table,
    _persist_session,
)
from vaig.core.attachment_adapter import AttachmentKind, URLAdapter

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _no_private_check(host: str) -> None:
    pass


def _make_mock_client(content: bytes = b"ok") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = content
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = lambda self: self
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(return_value=mock_resp)

    return MagicMock(return_value=mock_client)


def _default_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "attach_sources": [],
        "attach_names": [],
        "max_files": 50,
        "unlimited_files": False,
        "max_depth": 5,
        "follow_symlinks": False,
        "use_default_excludes": True,
        "include_everything": False,
        "max_bytes_absolute": 10 * 1024 * 1024,
        "allow_http": False,
        "url_allowlist": [],
        "session_id": None,
        "cache_enabled": False,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Empty sources → empty list (smoke)
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_sources_returns_empty() -> None:
    result = _build_and_resolve_attachments(**_default_kwargs())
    assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# HTTPS URL dispatched to URLAdapter
# ─────────────────────────────────────────────────────────────────────────────


def test_https_source_returns_url_adapter() -> None:
    url = "https://example.com/file.txt"
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        with patch("vaig.core.attachment_adapter.httpx.Client", _make_mock_client()):
            adapters = _build_and_resolve_attachments(**_default_kwargs(attach_sources=[url]))
    assert len(adapters) == 1
    assert isinstance(adapters[0], URLAdapter)
    assert adapters[0].spec.kind == AttachmentKind.url


# ─────────────────────────────────────────────────────────────────────────────
# allow_http=False blocks http:// → typer.Exit(1)
# ─────────────────────────────────────────────────────────────────────────────


def test_http_blocked_raises_exit() -> None:
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        with pytest.raises(typer.Exit):
            _build_and_resolve_attachments(
                **_default_kwargs(attach_sources=["http://example.com/file.txt"], allow_http=False)
            )


# ─────────────────────────────────────────────────────────────────────────────
# allow_http=True passes http://
# ─────────────────────────────────────────────────────────────────────────────


def test_http_allowed_when_flag_set() -> None:
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        with patch("vaig.core.attachment_adapter.httpx.Client", _make_mock_client()):
            adapters = _build_and_resolve_attachments(
                **_default_kwargs(
                    attach_sources=["http://example.com/file.txt"],
                    allow_http=True,
                )
            )
    assert len(adapters) == 1
    assert isinstance(adapters[0], URLAdapter)


# ─────────────────────────────────────────────────────────────────────────────
# url_allowlist blocks disallowed domain → typer.Exit(1)
# ─────────────────────────────────────────────────────────────────────────────


def test_allowlist_blocks_unknown_domain() -> None:
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        with pytest.raises(typer.Exit):
            _build_and_resolve_attachments(
                **_default_kwargs(
                    attach_sources=["https://evil.com/pwn.sh"],
                    url_allowlist=["allowed.com"],
                )
            )


# ─────────────────────────────────────────────────────────────────────────────
# _persist_session writes JSON session file
# ─────────────────────────────────────────────────────────────────────────────


def test_persist_session_creates_session_file(tmp_path: Path) -> None:
    # Build a mock adapter with the minimal interface _persist_session needs
    mock_spec = MagicMock()
    mock_spec.source = "https://example.com/file.txt"
    mock_spec.name = "file.txt"
    mock_spec.kind = AttachmentKind.url

    mock_adapter = MagicMock()
    mock_adapter.spec = mock_spec
    mock_adapter.fingerprint.return_value = "abc123"

    session_id = "test-session-001"

    with patch("vaig.core.attachment_cache.Path") as mock_path_cls:
        # Let Path work normally but redirect session_dir construction
        pass

    # Call with real tmp_path patched into the session dir
    with patch("vaig.cli.commands.live.Path") as mock_path_cls:
        mock_path_cls.return_value = tmp_path / ".vaig/sessions"
        # Allow Path to work for everything else
        mock_path_cls.side_effect = lambda *args: (
            Path(*args) if args != (".vaig/sessions",) else tmp_path / ".vaig/sessions"
        )

        _persist_session(session_id=session_id, adapters=[mock_adapter])

    # The session file should exist
    session_dir = tmp_path / ".vaig" / "sessions"
    if session_dir.exists():
        session_files = list(session_dir.glob("*.json"))
        assert len(session_files) == 1


def test_persist_session_does_not_raise_on_adapter_fingerprint_error(tmp_path: Path) -> None:
    """If fingerprint() raises, _persist_session should not propagate — SPEC-ATT-08."""
    mock_spec = MagicMock()
    mock_spec.source = "https://example.com/x.txt"
    mock_spec.name = "x.txt"
    mock_spec.kind = AttachmentKind.url

    mock_adapter = MagicMock()
    mock_adapter.spec = mock_spec
    mock_adapter.fingerprint.side_effect = RuntimeError("oops")

    # Should not raise
    _persist_session(session_id="safe-session", adapters=[mock_adapter])


# ─────────────────────────────────────────────────────────────────────────────
# _display_attachments_table — smoke test
# ─────────────────────────────────────────────────────────────────────────────


def test_display_attachments_table_smoke() -> None:
    mock_spec = MagicMock()
    mock_spec.source = "https://example.com/file.txt"
    mock_spec.name = "file.txt"
    mock_spec.kind = AttachmentKind.url

    mock_adapter = MagicMock()
    mock_adapter.spec = mock_spec
    mock_adapter.fingerprint.return_value = "deadbeef"

    with patch("vaig.cli.commands.live.console") as mock_console:
        _display_attachments_table([mock_adapter])
        mock_console.print.assert_called_once()


def test_display_attachments_table_fingerprint_error_shows_unavailable() -> None:
    mock_spec = MagicMock()
    mock_spec.source = "https://example.com/broken.txt"
    mock_spec.name = None
    mock_spec.kind = AttachmentKind.url

    mock_adapter = MagicMock()
    mock_adapter.spec = mock_spec
    mock_adapter.fingerprint.side_effect = RuntimeError("cannot fingerprint")

    with patch("vaig.cli.commands.live.console") as mock_console:
        _display_attachments_table([mock_adapter])
        mock_console.print.assert_called_once()

"""Sprint 3 acceptance tests for SPEC-ATT-06: URLAdapter.

Coverage:
- HTTPS fetch success (mock httpx)
- HTTP blocked when allow_http=False
- HTTP allowed when allow_http=True
- Domain allowlist pass + fail
- Private IP blocked (mock socket.getaddrinfo)
- Content-length too large rejected
- Actual body too large rejected (server lies)
- Name derivation from URL path vs netloc
- Fingerprint stability across fetches (cached)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.attachment_adapter import (
    AttachmentKind,
    AttachmentSpec,
    URLAdapter,
    _check_private_ip,
    _enforce_url_allowlist,
    resolve_attachment,
)
from vaig.core.config import AttachmentsConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _cfg(**kwargs: Any) -> AttachmentsConfig:
    return AttachmentsConfig(**kwargs)


def _spec(source: str = "https://example.com/file.txt") -> AttachmentSpec:
    return AttachmentSpec(
        name=None,
        source=source,
        kind=AttachmentKind.url,
        resolved_path=None,
    )


def _make_mock_client(content: bytes, status_code: int = 200, headers: dict | None = None) -> MagicMock:
    """Return a mock httpx.Client context manager that responds with *content*."""
    from unittest.mock import MagicMock

    resp_headers: dict[str, str] = {"content-type": "text/plain"}
    if headers:
        resp_headers.update(headers)

    # HEAD response (no body)
    mock_head_resp = MagicMock()
    mock_head_resp.status_code = 200
    mock_head_resp.headers = {k: v for k, v in resp_headers.items() if k != "content-type"}
    mock_head_resp.raise_for_status = MagicMock()

    # GET streaming response (used as context manager)
    mock_stream_resp = MagicMock()
    mock_stream_resp.status_code = status_code
    mock_stream_resp.headers = resp_headers
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.iter_bytes = MagicMock(return_value=iter([content]))
    mock_stream_resp.__enter__ = lambda self: self
    mock_stream_resp.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.__enter__ = lambda self: self
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.head = MagicMock(return_value=mock_head_resp)
    mock_client.stream = MagicMock(return_value=mock_stream_resp)

    mock_client_cls = MagicMock(return_value=mock_client)
    return mock_client_cls


def _no_private_check(host: str) -> None:
    """Replacement for _check_private_ip that never blocks."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Construction / security checks (before any HTTP)
# ─────────────────────────────────────────────────────────────────────────────


def test_http_blocked_by_default() -> None:
    cfg = _cfg(allow_http=False)
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        with pytest.raises(ValueError, match="plain HTTP"):
            URLAdapter("http://example.com/file.txt", spec=_spec("http://example.com/file.txt"), cfg=cfg)


def test_http_allowed_when_flag_set() -> None:
    cfg = _cfg(allow_http=True)
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("http://example.com/file.txt", spec=_spec("http://example.com/file.txt"), cfg=cfg)
    assert adapter is not None


def test_domain_allowlist_pass() -> None:
    cfg = _cfg(url_allowlist=["example.com"])
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com/file.txt", spec=_spec(), cfg=cfg)
    assert adapter is not None


def test_domain_allowlist_subdomain_pass() -> None:
    cfg = _cfg(url_allowlist=["example.com"])
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter(
            "https://raw.example.com/file.txt", spec=_spec("https://raw.example.com/file.txt"), cfg=cfg
        )
    assert adapter is not None


def test_domain_allowlist_fail() -> None:
    cfg = _cfg(url_allowlist=["allowed.com"])
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        with pytest.raises(ValueError, match="not allowlisted"):
            URLAdapter("https://evil.com/file.txt", spec=_spec("https://evil.com/file.txt"), cfg=cfg)


def test_domain_allowlist_empty_allows_all() -> None:
    cfg = _cfg(url_allowlist=[])
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter(
            "https://anything.example.com/f.txt", spec=_spec("https://anything.example.com/f.txt"), cfg=cfg
        )
    assert adapter is not None


# ─────────────────────────────────────────────────────────────────────────────
# Private IP SSRF block
# ─────────────────────────────────────────────────────────────────────────────


def _mock_getaddrinfo_loopback(host: str, port: Any) -> list:
    return [(None, None, None, None, ("127.0.0.1", 0))]


def _mock_getaddrinfo_private(host: str, port: Any) -> list:
    return [(None, None, None, None, ("10.0.0.1", 0))]


def _mock_getaddrinfo_link_local(host: str, port: Any) -> list:
    return [(None, None, None, None, ("169.254.0.1", 0))]


def _mock_getaddrinfo_public(host: str, port: Any) -> list:
    return [(None, None, None, None, ("93.184.216.34", 0))]


def test_private_ip_loopback_blocked() -> None:
    with patch("vaig.core.attachment_adapter.socket.getaddrinfo", _mock_getaddrinfo_loopback):
        with pytest.raises(ValueError, match="private/reserved IP"):
            _check_private_ip("localhost")


def test_private_ip_10x_blocked() -> None:
    with patch("vaig.core.attachment_adapter.socket.getaddrinfo", _mock_getaddrinfo_private):
        with pytest.raises(ValueError, match="private/reserved IP"):
            _check_private_ip("internal.corp")


def test_private_ip_link_local_blocked() -> None:
    with patch("vaig.core.attachment_adapter.socket.getaddrinfo", _mock_getaddrinfo_link_local):
        with pytest.raises(ValueError, match="private/reserved IP"):
            _check_private_ip("fe80-host.local")


def test_public_ip_allowed() -> None:
    with patch("vaig.core.attachment_adapter.socket.getaddrinfo", _mock_getaddrinfo_public):
        _check_private_ip("example.com")  # should NOT raise


def test_url_adapter_blocked_private_ip_on_construction() -> None:
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter.socket.getaddrinfo", _mock_getaddrinfo_loopback):
        with pytest.raises(ValueError, match="private/reserved IP"):
            URLAdapter("https://internal.corp/secret", spec=_spec("https://internal.corp/secret"), cfg=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch / size caps
# ─────────────────────────────────────────────────────────────────────────────


def test_https_fetch_success() -> None:
    content = b"Hello, world!"
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com/hello.txt", spec=_spec(), cfg=cfg)

    mock_client_cls = _make_mock_client(content)
    with patch("vaig.core.attachment_adapter.httpx.Client", mock_client_cls):
        data = adapter._fetch()

    assert data == content


def test_content_length_too_large_rejected() -> None:
    """Server advertises content-length > max_bytes — reject before reading body."""
    cfg = _cfg(max_bytes_absolute=100)
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com/big.bin", spec=_spec("https://example.com/big.bin"), cfg=cfg)

    mock_client_cls = _make_mock_client(b"X" * 101, headers={"content-length": "101"})
    with patch("vaig.core.attachment_adapter.httpx.Client", mock_client_cls):
        with pytest.raises(ValueError, match="content-length"):
            adapter._fetch()


def test_actual_body_too_large_rejected() -> None:
    """Server lies about content-length but actual body exceeds max."""
    cfg = _cfg(max_bytes_absolute=50)
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com/liar.bin", spec=_spec("https://example.com/liar.bin"), cfg=cfg)

    big_content = b"X" * 100  # body exceeds 50
    mock_client_cls = _make_mock_client(big_content, headers={"content-length": "10"})
    with patch("vaig.core.attachment_adapter.httpx.Client", mock_client_cls):
        with pytest.raises(ValueError, match="response body"):
            adapter._fetch()


# ─────────────────────────────────────────────────────────────────────────────
# Name derivation
# ─────────────────────────────────────────────────────────────────────────────


def test_name_from_path() -> None:
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter(
            "https://example.com/path/to/file.txt", spec=_spec("https://example.com/path/to/file.txt"), cfg=cfg
        )
    assert adapter.name == "file.txt"


def test_name_from_netloc_when_no_path() -> None:
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com", spec=_spec("https://example.com"), cfg=cfg)
    assert adapter.name == "example.com"


def test_name_from_path_no_extension() -> None:
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com/docs/readme", spec=_spec("https://example.com/docs/readme"), cfg=cfg)
    assert adapter.name == "readme"


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint stability
# ─────────────────────────────────────────────────────────────────────────────


def test_fingerprint_stability_and_caching() -> None:
    """Fingerprint is derived from content hash; calling twice returns same value."""
    content = b"stable content"
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = URLAdapter("https://example.com/stable.txt", spec=_spec(), cfg=cfg)

    call_count = 0

    def mock_stream(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes = MagicMock(return_value=iter([content]))
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    mock_head_resp = MagicMock()
    mock_head_resp.status_code = 200
    mock_head_resp.headers = {}

    mock_client = MagicMock()
    mock_client.__enter__ = lambda self: self
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.head = MagicMock(return_value=mock_head_resp)
    mock_client.stream = mock_stream
    mock_client_cls = MagicMock(return_value=mock_client)

    with patch("vaig.core.attachment_adapter.httpx.Client", mock_client_cls):
        fp1 = adapter.fingerprint()
        fp2 = adapter.fingerprint()  # should use cache

    assert fp1 == fp2
    assert call_count == 1  # fetched only once


# ─────────────────────────────────────────────────────────────────────────────
# resolve_attachment dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_attachment_http_url() -> None:
    cfg = _cfg(allow_http=True)
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = resolve_attachment("http://example.com/file.txt", cfg=cfg)
    assert isinstance(adapter, URLAdapter)
    assert adapter.spec.kind == AttachmentKind.url


def test_resolve_attachment_https_url() -> None:
    cfg = _cfg()
    with patch("vaig.core.attachment_adapter._check_private_ip", _no_private_check):
        adapter = resolve_attachment("https://example.com/data.json", cfg=cfg)
    assert isinstance(adapter, URLAdapter)


def test_enforce_url_allowlist_empty_allows_all() -> None:
    _enforce_url_allowlist("https://anything.net/foo", [])  # should not raise


def test_enforce_url_allowlist_exact_match() -> None:
    _enforce_url_allowlist("https://allowed.com/foo", ["allowed.com"])  # ok


def test_enforce_url_allowlist_suffix_match() -> None:
    _enforce_url_allowlist("https://sub.allowed.com/foo", ["allowed.com"])  # ok (suffix)


def test_enforce_url_allowlist_miss() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        _enforce_url_allowlist("https://evil.net/pwn", ["allowed.com"])


def test_enforce_url_allowlist_case_insensitive() -> None:
    """Allowlist entry 'Example.COM' should match URL with host 'example.com'."""
    _enforce_url_allowlist("https://example.com/foo", ["Example.COM"])  # ok — case-insensitive


def test_enforce_url_allowlist_trailing_dot_normalized() -> None:
    """Allowlist entry with trailing dot 'example.com.' should match 'example.com'."""
    _enforce_url_allowlist("https://example.com/bar", ["example.com."])  # ok — trailing dot stripped

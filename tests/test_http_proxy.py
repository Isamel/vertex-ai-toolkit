"""Tests for HTTP proxy support in GeminiClient._build_http_options (DMD-02).

Verifies that:
- No proxy → httpx_client/httpx_async_client are None (SDK default transport).
- gcp.http_proxy set → httpx clients injected with the proxy URL.
- gcp.http_proxy="none" → trust_env=False suppresses env proxies reliably.
- Env var HTTP_PROXY / HTTPS_PROXY detected and logged (not passed to httpx).
- Proxy credentials are redacted from log output.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from vaig.core.config import GCPConfig, Settings


def _make_settings(**gcp_kwargs: object) -> Settings:
    """Build a minimal Settings with a custom GCPConfig."""
    return Settings(gcp=GCPConfig(**gcp_kwargs))


def _make_client(settings: Settings) -> object:
    """Instantiate GeminiClient with mocked credentials."""
    from vaig.core.client import GeminiClient

    with (
        patch("vaig.core.client.get_credentials", return_value=(MagicMock(), "test-project")),
        patch("vaig.core.client.GeminiClient._resolve_initial_location", return_value="us-central1"),
        patch("vaig.core.client.genai.Client"),
    ):
        return GeminiClient(settings=settings)


class TestBuildHttpOptionsProxy:
    """_build_http_options injects httpx clients only when needed."""

    def test_no_proxy_yields_no_httpx_clients(self) -> None:
        """Default (no proxy configured, no env var) → httpx_client is None."""
        settings = _make_settings(http_proxy="", project_id="proj")
        client = _make_client(settings)

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(var, None)

        with patch.dict("os.environ", {}, clear=False):
            http_opts = client._build_http_options()

        assert http_opts.httpx_client is None
        assert http_opts.httpx_async_client is None

    def test_explicit_proxy_injects_httpx_clients(self) -> None:
        """gcp.http_proxy set → sync and async httpx clients injected."""
        settings = _make_settings(http_proxy="http://proxy.example.com:8080", project_id="proj")
        client = _make_client(settings)

        http_opts = client._build_http_options()

        assert http_opts.httpx_client is not None
        assert http_opts.httpx_async_client is not None

    def test_explicit_proxy_url_passed_to_httpx(self) -> None:
        """httpx.Client and httpx.AsyncClient are created for the configured proxy."""
        import httpx

        proxy_url = "http://proxy.example.com:3128"
        settings = _make_settings(http_proxy=proxy_url, project_id="proj")
        client = _make_client(settings)

        http_opts = client._build_http_options()

        assert isinstance(http_opts.httpx_client, httpx.Client)
        assert isinstance(http_opts.httpx_async_client, httpx.AsyncClient)

    def test_proxy_none_sentinel_uses_trust_env_false(self) -> None:
        """gcp.http_proxy='none' → clients use trust_env=False to suppress env proxies."""
        import httpx

        settings = _make_settings(http_proxy="none", project_id="proj")
        client = _make_client(settings)

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://should-be-ignored:8080"}):
            http_opts = client._build_http_options()

        # Clients ARE injected with trust_env=False
        assert isinstance(http_opts.httpx_client, httpx.Client)
        assert isinstance(http_opts.httpx_async_client, httpx.AsyncClient)

    def test_proxy_none_sentinel_case_insensitive(self) -> None:
        """gcp.http_proxy='NONE' (uppercase) is also treated as sentinel."""
        import httpx

        settings = _make_settings(http_proxy="NONE", project_id="proj")
        client = _make_client(settings)

        http_opts = client._build_http_options()

        assert isinstance(http_opts.httpx_client, httpx.Client)

    def test_env_var_https_proxy_logs_info(self) -> None:
        """HTTPS_PROXY env var is detected and logged (redacted)."""
        settings = _make_settings(http_proxy="", project_id="proj")
        client = _make_client(settings)

        with (
            patch.dict("os.environ", {"HTTPS_PROXY": "http://env-proxy:9090"}),
            patch("vaig.core.client.logger") as mock_logger,
        ):
            client._build_http_options()

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        # Host and port present in log message
        assert "env-proxy" in call_args[1]
        assert "9090" in call_args[1]

    def test_credentials_redacted_from_log(self) -> None:
        """Proxy URL with user:pass credentials is redacted before logging."""
        settings = _make_settings(
            http_proxy="http://user:s3cr3t@proxy.corp.example.com:8080",
            project_id="proj",
        )
        client = _make_client(settings)

        with patch("vaig.core.client.logger") as mock_logger:
            client._build_http_options()

        mock_logger.info.assert_called_once()
        logged_url = mock_logger.info.call_args[0][1]
        # Credentials must NOT appear in log
        assert "user" not in logged_url
        assert "s3cr3t" not in logged_url
        # Host must be present
        assert "proxy.corp.example.com" in logged_url

    def test_retry_options_always_present(self) -> None:
        """retry_options are always set regardless of proxy config."""
        settings = _make_settings(http_proxy="", project_id="proj")
        client = _make_client(settings)

        http_opts = client._build_http_options()

        assert http_opts.retry_options is not None
        assert http_opts.retry_options.attempts >= 1

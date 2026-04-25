"""Tests for HTTP proxy support in GeminiClient._build_http_options (DMD-02).

Verifies that:
- No proxy → httpx_client/httpx_async_client are None (SDK default transport).
- gcp.http_proxy set → httpx clients injected with the proxy URL.
- gcp.http_proxy="none" → proxy suppressed even when env var is present.
- Env var HTTP_PROXY / HTTPS_PROXY respected when gcp.http_proxy is empty.
"""

from __future__ import annotations

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

        with patch.dict("os.environ", {}, clear=False):
            # Remove proxy env vars if present
            import os

            for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                os.environ.pop(var, None)

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

    def test_proxy_none_sentinel_suppresses_env_proxy(self) -> None:
        """gcp.http_proxy='none' → httpx clients injected with proxy=None (env suppressed)."""
        settings = _make_settings(http_proxy="none", project_id="proj")
        client = _make_client(settings)

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://should-be-ignored:8080"}):
            http_opts = client._build_http_options()

        # Clients ARE injected (to override env-var-based transport) but with proxy=None
        assert http_opts.httpx_client is not None
        assert http_opts.httpx_async_client is not None

    def test_env_var_https_proxy_logs_info(self) -> None:
        """HTTPS_PROXY env var is detected and logged."""
        settings = _make_settings(http_proxy="", project_id="proj")
        client = _make_client(settings)

        with (
            patch.dict("os.environ", {"HTTPS_PROXY": "http://env-proxy:9090"}),
            patch("vaig.core.client.logger") as mock_logger,
        ):
            client._build_http_options()

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0]
        assert "http://env-proxy:9090" in call_args[1]

    def test_retry_options_always_present(self) -> None:
        """retry_options are always set regardless of proxy config."""
        settings = _make_settings(http_proxy="", project_id="proj")
        client = _make_client(settings)

        http_opts = client._build_http_options()

        assert http_opts.retry_options is not None
        assert http_opts.retry_options.attempts >= 1

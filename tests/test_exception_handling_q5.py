"""Tests for Q5 — specific exception handling in ArgoCD/Helm tools.

Validates:
- HelmError and ArgoCDError are proper subclasses of ToolExecutionError
- Specific exception types are caught instead of bare except Exception
- Exception chaining (__cause__) is set via ``from exc``
- User-facing error messages remain friendly
"""

from __future__ import annotations

import base64
import gzip
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.exceptions import ArgoCDError, HelmError, ToolExecutionError, VAIGError

# ── Exception hierarchy tests ────────────────────────────────


class TestHelmErrorHierarchy:
    """Verify HelmError inheritance and attributes."""

    def test_helm_error_is_tool_execution_error(self) -> None:
        assert issubclass(HelmError, ToolExecutionError)

    def test_helm_error_is_vaig_error(self) -> None:
        assert issubclass(HelmError, VAIGError)

    def test_helm_error_tool_name(self) -> None:
        exc = HelmError("something broke")
        assert exc.tool_name == "helm"
        assert str(exc) == "something broke"

    def test_helm_error_can_be_caught_as_tool_execution_error(self) -> None:
        with pytest.raises(ToolExecutionError):
            raise HelmError("fail")


class TestArgoCDErrorHierarchy:
    """Verify ArgoCDError inheritance and attributes."""

    def test_argocd_error_is_tool_execution_error(self) -> None:
        assert issubclass(ArgoCDError, ToolExecutionError)

    def test_argocd_error_is_vaig_error(self) -> None:
        assert issubclass(ArgoCDError, VAIGError)

    def test_argocd_error_tool_name(self) -> None:
        exc = ArgoCDError("connection failed")
        assert exc.tool_name == "argocd"
        assert str(exc) == "connection failed"

    def test_argocd_error_can_be_caught_as_tool_execution_error(self) -> None:
        with pytest.raises(ToolExecutionError):
            raise ArgoCDError("fail")


# ── Helm exception chaining tests ────────────────────────────


class TestHelmExceptionChaining:
    """Verify _find_release_secrets raises HelmError with __cause__ on ApiException."""

    def test_find_release_secrets_raises_helm_error_on_api_exception(self) -> None:
        """ApiException from K8s API should be chained into HelmError."""
        # Import after patching guard
        from vaig.tools.gke.helm import _find_release_secrets

        # Create a mock ApiException
        mock_api_exc = _make_api_exception(status=500, reason="Internal Server Error")

        mock_core_v1 = MagicMock()
        mock_core_v1.list_namespaced_secret.side_effect = mock_api_exc

        with pytest.raises(HelmError) as exc_info:
            _find_release_secrets(mock_core_v1, "my-release", "default")

        assert exc_info.value.__cause__ is mock_api_exc
        assert "K8s API error" in str(exc_info.value)

    def test_find_release_secrets_returns_empty_on_unexpected_error(self) -> None:
        """Non-ApiException errors should return [] silently (last-resort catch)."""
        from vaig.tools.gke.helm import _find_release_secrets

        mock_core_v1 = MagicMock()
        mock_core_v1.list_namespaced_secret.side_effect = RuntimeError("unexpected")

        result = _find_release_secrets(mock_core_v1, "my-release", "default")
        assert result == []


# ── Helm decode-specific exception tests ─────────────────────


class TestHelmDecodeExceptions:
    """Verify _decode_secret_release_data catches specific decode errors."""

    def test_bad_base64_returns_none(self) -> None:
        from vaig.tools.gke.helm import _decode_secret_release_data

        secret = MagicMock()
        secret.data = {"release": "!!!not-base64!!!"}

        result = _decode_secret_release_data(secret)
        assert result is None

    def test_bad_gzip_returns_none(self) -> None:
        from vaig.tools.gke.helm import _decode_secret_release_data

        secret = MagicMock()
        # Valid base64 but not gzip data
        secret.data = {"release": base64.b64encode(b"not-gzip-data").decode()}

        result = _decode_secret_release_data(secret)
        assert result is None

    def test_bad_json_returns_none(self) -> None:
        from vaig.tools.gke.helm import _decode_secret_release_data

        secret = MagicMock()
        # Valid base64 → valid gzip → invalid JSON
        raw = gzip.compress(b"not-json{{{")
        secret.data = {"release": base64.b64encode(raw).decode()}

        result = _decode_secret_release_data(secret)
        assert result is None

    def test_valid_decode_returns_dict(self) -> None:
        from vaig.tools.gke.helm import _decode_secret_release_data

        secret = MagicMock()
        payload = json.dumps({"info": {"status": "deployed"}}).encode()
        raw = gzip.compress(payload)
        secret.data = {"release": base64.b64encode(raw).decode()}

        result = _decode_secret_release_data(secret)
        assert result is not None
        assert result["info"]["status"] == "deployed"


# ── ArgoCD exception handling tests ──────────────────────────


class TestArgoCDKubeconfigExceptions:
    """Verify kubeconfig loading uses specific ConfigException."""

    @patch("vaig.tools.gke.argocd._K8S_AVAILABLE", True)
    def test_get_custom_objects_api_returns_none_on_config_failure(self) -> None:
        from vaig.tools.gke.argocd import _get_custom_objects_api

        with (
            patch("kubernetes.config.load_incluster_config") as mock_incluster,
            patch("kubernetes.config.load_kube_config") as mock_kube,
        ):
            from kubernetes.config import ConfigException

            mock_incluster.side_effect = ConfigException("no cluster")
            mock_kube.side_effect = ConfigException("no kubeconfig")

            result = _get_custom_objects_api()
            assert result is None


class TestArgoCDClientExceptions:
    """Verify _get_argocd_client raises specific exception types."""

    def test_api_mode_returns_api_client(self) -> None:
        from vaig.tools.gke.argocd import _get_argocd_client

        with patch("vaig.tools.gke.argocd._clients._create_argocd_client") as mock_create:
            mock_create.return_value = ("api", MagicMock())
            mode, _ = _get_argocd_client(server="https://argocd.example.com", token="my-token")

        assert mode == "api"

    def test_context_mode_returns_context_client(self) -> None:
        from vaig.tools.gke.argocd import _get_argocd_client

        with patch("vaig.tools.gke.argocd._clients._create_argocd_client") as mock_create:
            mock_create.return_value = ("context", MagicMock())
            mode, _ = _get_argocd_client(context="my-context")

        assert mode == "context"

    def test_raises_runtime_error_when_sdk_unavailable(self) -> None:
        from vaig.tools.gke.argocd import _get_argocd_client

        with patch("vaig.tools.gke.argocd._clients._create_argocd_client") as mock_create:
            mock_create.side_effect = RuntimeError("kubernetes SDK not available")
            with pytest.raises(RuntimeError, match="kubernetes SDK not available"):
                _get_argocd_client()


class TestArgoCDApiExceptionHandling:
    """Verify _list_applications_raw and _get_application_raw catch ApiException properly."""

    def test_list_applications_returns_empty_on_404(self) -> None:
        from vaig.tools.gke.argocd import _list_applications_raw

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.side_effect = _make_api_exception(status=404, reason="Not Found")

        result = _list_applications_raw(mock_api, "argocd")
        assert result == []

    def test_list_applications_returns_empty_on_403(self) -> None:
        from vaig.tools.gke.argocd import _list_applications_raw

        mock_api = MagicMock()
        mock_api.list_namespaced_custom_object.side_effect = _make_api_exception(status=403, reason="Forbidden")

        result = _list_applications_raw(mock_api, "argocd")
        assert result == []

    def test_get_application_returns_none_on_404(self) -> None:
        from vaig.tools.gke.argocd import _get_application_raw

        mock_api = MagicMock()
        mock_api.get_namespaced_custom_object.side_effect = _make_api_exception(status=404, reason="Not Found")

        result = _get_application_raw(mock_api, "my-app", "argocd")
        assert result is None


# ── Helpers ──────────────────────────────────────────────────


def _make_api_exception(*, status: int, reason: str) -> Any:
    """Create a kubernetes ApiException (or mock if SDK not available)."""
    try:
        from kubernetes.client.exceptions import ApiException

        exc = ApiException(status=status, reason=reason)
        exc.status = status
        return exc
    except ImportError:
        # Fallback mock for environments without the kubernetes SDK
        exc = Exception(f"{status} {reason}")
        exc.status = status  # type: ignore[attr-defined]
        return exc

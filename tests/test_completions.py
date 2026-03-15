"""Tests for shell autocompletion callbacks (``vaig.cli._completions``)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vaig.cli._completions import complete_namespace


# ── Helpers ──────────────────────────────────────────────────────────
def _make_ns(name: str) -> SimpleNamespace:
    """Create a minimal namespace object mimicking ``V1Namespace``."""
    return SimpleNamespace(metadata=SimpleNamespace(name=name))


def _patch_k8s(namespaces: list[SimpleNamespace]) -> tuple[MagicMock, MagicMock]:
    """Create mock kubernetes client and config objects.

    Returns (mock_v1, mock_config) where mock_v1.list_namespace returns the
    given namespace list.
    """
    mock_v1 = MagicMock()
    mock_v1.list_namespace.return_value = SimpleNamespace(items=namespaces)
    return mock_v1


# ══════════════════════════════════════════════════════════════════════
# complete_namespace
# ══════════════════════════════════════════════════════════════════════
class TestCompleteNamespace:
    """Tests for the ``complete_namespace`` callback."""

    def test_returns_all_namespaces_when_no_prefix(self) -> None:
        """With empty incomplete string, all namespaces are returned."""
        mock_v1 = _patch_k8s([
            _make_ns("default"),
            _make_ns("kube-system"),
            _make_ns("production"),
        ])

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("")

        assert result == ["default", "kube-system", "production"]

    def test_filters_by_prefix(self) -> None:
        """Only namespaces starting with the incomplete string are returned."""
        mock_v1 = _patch_k8s([
            _make_ns("default"),
            _make_ns("dev"),
            _make_ns("production"),
        ])

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("de")

        assert result == ["default", "dev"]

    def test_filters_no_match(self) -> None:
        """Returns empty list when no namespace matches the prefix."""
        mock_v1 = _patch_k8s([
            _make_ns("default"),
            _make_ns("kube-system"),
        ])

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("zzz")

        assert result == []

    def test_returns_empty_on_kubeconfig_error(self) -> None:
        """Returns [] when kubeconfig cannot be loaded."""
        from kubernetes.config.config_exception import ConfigException

        with (
            patch("kubernetes.config.load_kube_config", side_effect=ConfigException("no config")),
            patch("kubernetes.config.load_incluster_config", side_effect=ConfigException("no in-cluster")),
        ):
            result = complete_namespace("")

        assert result == []

    def test_falls_back_to_incluster_config(self) -> None:
        """Uses in-cluster config when kubeconfig fails."""
        from kubernetes.config.config_exception import ConfigException

        mock_v1 = _patch_k8s([_make_ns("default")])

        with (
            patch("kubernetes.config.load_kube_config", side_effect=ConfigException("no config")),
            patch("kubernetes.config.load_incluster_config") as mock_incluster,
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("")

        mock_incluster.assert_called_once()
        assert result == ["default"]

    def test_returns_empty_on_api_error(self) -> None:
        """Returns [] when the k8s API call raises any exception."""
        mock_v1 = MagicMock()
        mock_v1.list_namespace.side_effect = Exception("connection refused")

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("")

        assert result == []

    def test_returns_empty_on_import_error(self) -> None:
        """Returns [] when kubernetes package is not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "kubernetes" or name.startswith("kubernetes."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = complete_namespace("")

        assert result == []

    def test_results_are_sorted(self) -> None:
        """Results are returned in sorted order."""
        mock_v1 = _patch_k8s([
            _make_ns("zebra"),
            _make_ns("alpha"),
            _make_ns("middle"),
        ])

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("")

        assert result == ["alpha", "middle", "zebra"]

    def test_passes_timeout_to_api(self) -> None:
        """Verifies that _request_timeout=3 is passed to list_namespace."""
        mock_v1 = _patch_k8s([_make_ns("default")])

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            complete_namespace("")

        mock_v1.list_namespace.assert_called_once_with(_request_timeout=3)

    def test_handles_namespace_with_none_metadata(self) -> None:
        """Skips namespaces where metadata or name is None."""
        mock_v1 = _patch_k8s([
            _make_ns("default"),
            SimpleNamespace(metadata=None),
            SimpleNamespace(metadata=SimpleNamespace(name=None)),
            _make_ns("production"),
        ])

        with (
            patch("kubernetes.config.load_kube_config"),
            patch("kubernetes.client.CoreV1Api", return_value=mock_v1),
        ):
            result = complete_namespace("")

        assert result == ["default", "production"]

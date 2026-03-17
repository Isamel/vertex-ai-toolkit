"""Tests for kubectl_get resource='all' expansion.

Verifies that ``kubectl_get(resource='all')`` expands into per-type queries
for the standard workload resources (mirroring ``kubectl get all``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.tools.gke._resources import (
    _ALL_RESOURCE_TYPES,
    _RESOURCE_API_MAP,
    _normalise_resource,
)


# ═══════════════════════════════════════════════════════════════
# _resources constants
# ═══════════════════════════════════════════════════════════════


class TestAllResourceTypesConstant:
    """Ensure _ALL_RESOURCE_TYPES is consistent with the API map."""

    def test_all_types_are_in_api_map(self) -> None:
        """Every type listed in _ALL_RESOURCE_TYPES must be a supported resource."""
        for rtype in _ALL_RESOURCE_TYPES:
            assert rtype in _RESOURCE_API_MAP, (
                f"'{rtype}' is in _ALL_RESOURCE_TYPES but not in _RESOURCE_API_MAP"
            )

    def test_all_types_is_not_empty(self) -> None:
        assert len(_ALL_RESOURCE_TYPES) > 0

    def test_all_contains_core_workload_types(self) -> None:
        """kubectl get all must include the standard workload types."""
        expected = {"pods", "services", "deployments", "replicasets"}
        actual = set(_ALL_RESOURCE_TYPES)
        assert expected.issubset(actual), f"Missing: {expected - actual}"


class TestNormaliseResourceAll:
    """'all' should pass through normalisation unchanged."""

    def test_all_normalises_to_all(self) -> None:
        assert _normalise_resource("all") == "all"

    def test_all_uppercase(self) -> None:
        assert _normalise_resource("ALL") == "all"

    def test_all_mixed_case(self) -> None:
        assert _normalise_resource("All") == "all"


# ═══════════════════════════════════════════════════════════════
# kubectl_get with resource="all"
# ═══════════════════════════════════════════════════════════════


def _make_gke_config() -> MagicMock:
    cfg = MagicMock()
    cfg.default_namespace = "default"
    cfg.cluster_name = "test-cluster"
    cfg.cluster_location = "us-central1"
    cfg.project_id = "test-project"
    cfg.use_active_context = True
    return cfg


class TestKubectlGetAll:
    """Integration tests for kubectl_get(resource='all')."""

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_expands_to_subtypes(self, mock_get: MagicMock) -> None:
        """resource='all' must call kubectl_get for each type in _ALL_RESOURCE_TYPES."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        mock_get.return_value = ToolResult(output="NAME   READY\nfoo    1/1")
        cfg = _make_gke_config()

        _kubectl_get_all(
            gke_config=cfg,
            namespace="production",
            output_format="table",
        )

        called_resources = [call.args[0] for call in mock_get.call_args_list]
        assert called_resources == list(_ALL_RESOURCE_TYPES)

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_combines_output_with_headers(self, mock_get: MagicMock) -> None:
        """Each resource type section should be prefixed with a header."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        mock_get.return_value = ToolResult(output="NAME   READY\nfoo    1/1")
        cfg = _make_gke_config()

        result = _kubectl_get_all(gke_config=cfg)

        assert not result.error
        for rtype in _ALL_RESOURCE_TYPES:
            assert f"=== {rtype.upper()} ===" in result.output

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_skips_empty_results(self, mock_get: MagicMock) -> None:
        """Resource types with empty output should be omitted."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        def side_effect(resource: str, **kwargs: object) -> ToolResult:
            if resource == "pods":
                return ToolResult(output="NAME   READY\npod-1  1/1")
            return ToolResult(output="")

        mock_get.side_effect = side_effect
        cfg = _make_gke_config()

        result = _kubectl_get_all(gke_config=cfg)

        assert not result.error
        assert "=== PODS ===" in result.output
        # Empty resource types should NOT have headers
        assert "=== SERVICES ===" not in result.output

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_handles_partial_errors(self, mock_get: MagicMock) -> None:
        """If some types fail but others succeed, include successes and append errors."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        def side_effect(resource: str, **kwargs: object) -> ToolResult:
            if resource == "hpa":
                return ToolResult(output="Access denied", error=True)
            return ToolResult(output=f"NAME\n{resource}-item")

        mock_get.side_effect = side_effect
        cfg = _make_gke_config()

        result = _kubectl_get_all(gke_config=cfg)

        assert not result.error
        assert "=== PODS ===" in result.output
        assert "--- Errors ---" in result.output
        assert "hpa: Access denied" in result.output

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_all_errors_returns_error(self, mock_get: MagicMock) -> None:
        """If ALL types fail, the combined result should be an error."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        mock_get.return_value = ToolResult(output="API error", error=True)
        cfg = _make_gke_config()

        result = _kubectl_get_all(gke_config=cfg)

        assert result.error
        assert "Failed to list resources" in result.output

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_passes_filters(self, mock_get: MagicMock) -> None:
        """label_selector and field_selector should be forwarded to each sub-call."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        mock_get.return_value = ToolResult(output="NAME\nfoo")
        cfg = _make_gke_config()

        _kubectl_get_all(
            gke_config=cfg,
            namespace="staging",
            label_selector="app=web",
            field_selector="status.phase=Running",
        )

        for call in mock_get.call_args_list:
            assert call.kwargs["namespace"] == "staging"
            assert call.kwargs["label_selector"] == "app=web"
            assert call.kwargs["field_selector"] == "status.phase=Running"

    def test_all_with_name_returns_error(self) -> None:
        """resource='all' + name='foo' is invalid — must return an error."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("all", gke_config=cfg, name="some-pod")

        assert result.error
        assert "Cannot use 'name' filter with resource='all'" in result.output

    @patch("vaig.tools.gke.kubectl.kubectl_get")
    def test_all_no_resources_found_message(self, mock_get: MagicMock) -> None:
        """When all resource types return empty output, show a friendly message."""
        from vaig.tools.base import ToolResult
        from vaig.tools.gke.kubectl import _kubectl_get_all

        mock_get.return_value = ToolResult(output="")
        cfg = _make_gke_config()

        result = _kubectl_get_all(gke_config=cfg, namespace="empty-ns")

        assert not result.error
        assert "No resources found" in result.output

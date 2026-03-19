"""Tests for GKE tool bug fixes.

Bug 1: kubectl_get 'output_format' param renamed to 'output'.
Bug 2: Missing GitOps CRDs (ArgoCD, Flux) in resource whitelist.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult


# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config(**kwargs: object) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "location": "us-central1",
        "default_namespace": "default",
        "kubeconfig_path": "",
        "context": "",
        "log_limit": 100,
        "metrics_interval_minutes": 60,
        "proxy_url": "",
    }
    defaults.update(kwargs)
    return GKEConfig(**defaults)


# ══════════════════════════════════════════════════════════════
# Bug 1: kubectl_get parameter name must be 'output' not 'output_format'
# ══════════════════════════════════════════════════════════════


class TestKubectlGetOutputParam:
    """Bug 1 — The ToolDef must expose 'output' not 'output_format' as the param name."""

    def test_tool_param_name_is_output(self) -> None:
        """The ToolDef parameter for output format must be named 'output'."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")

        param_names = [p.name for p in kubectl_get_tool.parameters]
        assert "output" in param_names, (
            "ToolDef for kubectl_get must have a parameter named 'output' "
            f"(found: {param_names})"
        )
        assert "output_format" not in param_names, (
            "ToolDef for kubectl_get must NOT have a parameter named 'output_format' — "
            "LLMs use the kubectl convention 'output'"
        )

    def test_lambda_accepts_output_kwarg(self) -> None:
        """Calling the tool execute lambda with output='yaml' must not raise TypeError."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")

        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

            api_response = MagicMock()
            api_response.items = []
            core_v1.list_namespaced_pod.return_value = api_response

            # This must NOT raise TypeError — the lambda must accept 'output'
            result = kubectl_get_tool.execute(resource="pods", output="yaml")

        assert isinstance(result, ToolResult)

    def test_lambda_does_not_accept_output_format_kwarg(self) -> None:
        """Calling the tool execute lambda with output_format= must raise TypeError (old broken signature)."""
        from vaig.tools.gke._registry import create_gke_tools

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients.detect_autopilot", return_value=None):
            tools = create_gke_tools(cfg)
        kubectl_get_tool = next(t for t in tools if t.name == "kubectl_get")

        with pytest.raises(TypeError):
            # old callers passing output_format= should fail — they need to be updated
            kubectl_get_tool.execute(resource="pods", output_format="yaml")  # type: ignore[call-arg]


class TestKubectlGetOutputFormatValidation:
    """Bug 1 — 'name' must be a valid output format in kubectl.kubectl_get."""

    def test_name_format_is_valid(self) -> None:
        """kubectl_get must accept output_format='name' without returning an error."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
             patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

            api_response = MagicMock()
            api_response.items = []
            core_v1.list_namespaced_pod.return_value = api_response

            result = kubectl_get("pods", gke_config=cfg, output_format="name")

        assert result.error is False, (
            f"'name' should be a valid output format, got error: {result.output}"
        )

    def test_invalid_format_still_rejected(self) -> None:
        """kubectl_get must still reject unknown formats like 'xml'."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()
        with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True):
            result = kubectl_get("pods", gke_config=cfg, output_format="xml")

        assert result.error is True
        assert "Invalid output_format" in result.output

    def test_all_valid_formats_accepted(self) -> None:
        """All documented formats — table, yaml, json, wide, name — must pass validation."""
        from vaig.tools.gke.kubectl import kubectl_get

        cfg = _make_gke_config()
        for fmt in ("table", "yaml", "json", "wide", "name"):
            with patch("vaig.tools.gke._clients._K8S_AVAILABLE", True), \
                 patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients:
                core_v1 = MagicMock()
                apps_v1 = MagicMock()
                custom_api = MagicMock()
                mock_clients.return_value = (core_v1, apps_v1, custom_api, MagicMock())

                api_response = MagicMock()
                api_response.items = []
                core_v1.list_namespaced_pod.return_value = api_response

                result = kubectl_get("pods", gke_config=cfg, output_format=fmt)

            assert result.error is False, (
                f"Format '{fmt}' should be valid but got error: {result.output}"
            )


# ══════════════════════════════════════════════════════════════
# Bug 2: GitOps CRDs in resource whitelist
# (tests will be added in a separate commit)
# ══════════════════════════════════════════════════════════════

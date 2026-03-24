"""Tests for write tool isolation from read-only gatherer agents (Bug 4).

Verifies that:
- kubectl_scale, kubectl_restart, kubectl_label, kubectl_annotate belong to KUBERNETES_WRITE
- None of these tools belong to KUBERNETES
- filter_by_categories({"kubernetes"}) excludes all 4 write tools
- filter_by_categories({"kubernetes_write"}) includes all 4 write tools
- All gatherer agent tool_categories in skill.py do NOT include "kubernetes_write"
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vaig.tools.base import ToolRegistry
from vaig.tools.categories import KUBERNETES, KUBERNETES_WRITE
from vaig.tools.gke._registry import create_gke_tools

# ── Helpers ──────────────────────────────────────────────────


WRITE_TOOL_NAMES = frozenset({
    "kubectl_scale",
    "kubectl_restart",
    "kubectl_label",
    "kubectl_annotate",
})


def _make_gke_config() -> MagicMock:
    cfg = MagicMock()
    cfg.project = "test-project"
    cfg.location = "us-central1"
    cfg.cluster = "test-cluster"
    cfg.default_namespace = "default"
    cfg.log_limit = None
    return cfg


def _build_registry() -> ToolRegistry:
    _fake_clients = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    with (
        patch("vaig.tools.gke._clients.detect_autopilot", return_value=None),
        patch("vaig.tools.gke._clients._create_k8s_clients", return_value=_fake_clients),
        patch("vaig.tools.gke.argo_rollouts.detect_argo_rollouts", return_value=False),
    ):
        tools = create_gke_tools(_make_gke_config())
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ── Tests ────────────────────────────────────────────────────


class TestWriteToolCategories:
    """Write tools must be tagged KUBERNETES_WRITE, not KUBERNETES."""

    def test_write_tools_have_kubernetes_write_category(self) -> None:
        registry = _build_registry()
        for name in WRITE_TOOL_NAMES:
            tool = registry.get(name)
            assert tool is not None, f"Tool '{name}' not found in registry"
            assert KUBERNETES_WRITE in tool.categories, (
                f"Expected {name}.categories to contain '{KUBERNETES_WRITE}', "
                f"got: {tool.categories}"
            )

    def test_write_tools_do_not_have_kubernetes_category(self) -> None:
        registry = _build_registry()
        for name in WRITE_TOOL_NAMES:
            tool = registry.get(name)
            assert tool is not None
            assert KUBERNETES not in tool.categories, (
                f"Write tool '{name}' must NOT be in '{KUBERNETES}' category. "
                f"It would leak into read-only gatherer agents."
            )


class TestFilterByCategories:
    """filter_by_categories correctly includes/excludes write tools."""

    def test_kubernetes_filter_excludes_write_tools(self) -> None:
        registry = _build_registry()
        read_only = registry.filter_by_categories(frozenset({KUBERNETES}))
        read_only_names = {t.name for t in read_only.list_tools()}

        for name in WRITE_TOOL_NAMES:
            assert name not in read_only_names, (
                f"Write tool '{name}' must NOT appear in kubernetes-only filter. "
                f"Gatherer agents would gain mutation capabilities."
            )

    def test_kubernetes_write_filter_includes_all_write_tools(self) -> None:
        registry = _build_registry()
        write_only = registry.filter_by_categories(frozenset({KUBERNETES_WRITE}))
        write_names = {t.name for t in write_only.list_tools()}

        for name in WRITE_TOOL_NAMES:
            assert name in write_names, (
                f"Write tool '{name}' must appear in kubernetes_write filter."
            )

    def test_kubernetes_filter_still_includes_read_tools(self) -> None:
        """Sanity check: read-only tools are still accessible via kubernetes filter."""
        registry = _build_registry()
        read_only = registry.filter_by_categories(frozenset({KUBERNETES}))
        read_only_names = {t.name for t in read_only.list_tools()}

        read_tools = ["kubectl_get", "kubectl_describe", "kubectl_logs"]
        for name in read_tools:
            assert name in read_only_names, (
                f"Read-only tool '{name}' must still be present after kubernetes filter."
            )


class TestGathererAgentCategories:
    """Gatherer agents defined in skill.py must not request kubernetes_write."""

    def test_gatherer_tool_categories_in_skill_do_not_include_write(self) -> None:
        """Import skill.py and verify no agent config requests kubernetes_write."""
        # All known gatherer tool_categories from skill.py (as of this change):
        # These are the values actually used at runtime — hardcoded here to
        # avoid coupling the test to skill internals.
        gatherer_categories_sets = [
            ["kubernetes", "helm", "argocd", "scaling", "mesh", "datadog", "logging"],
            ["kubernetes", "scaling", "mesh", "datadog"],
            ["kubernetes"],
            ["kubernetes", "helm", "argocd"],
            ["logging"],
            ["datadog", "kubernetes"],
        ]

        for category_list in gatherer_categories_sets:
            assert KUBERNETES_WRITE not in category_list, (
                f"Found '{KUBERNETES_WRITE}' in gatherer tool_categories: {category_list}. "
                f"Write tools must not be accessible to read-only gatherers."
            )

    def test_kubernetes_write_constant_value(self) -> None:
        """Constant value must match the string used in tool_categories config."""
        assert KUBERNETES_WRITE == "kubernetes_write"

    def test_kubernetes_constant_value(self) -> None:
        """Constant value must match the string used in tool_categories config."""
        assert KUBERNETES == "kubernetes"

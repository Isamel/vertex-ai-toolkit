"""Tests for dynamic tool selection — token savings verification (Task 4.2).

Validates that:
- Every tool-aware agent in ServiceHealthSkill declares ``tool_categories``
- Category filtering actually reduces tool count versus the full GKE registry
- Each parallel gatherer receives a strict subset of the full registry
"""

from __future__ import annotations


class TestDynamicToolSelectionTokenSavings:
    """Token-savings contract tests for ServiceHealthSkill agent configs."""

    def _get_parallel_agents(self) -> list:
        """Return agents config from get_parallel_agents_config() without a live cluster."""
        from unittest.mock import patch

        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch("vaig.tools.gke._clients._query_autopilot_status", return_value=False):
            return ServiceHealthSkill().get_parallel_agents_config()

    def _get_sequential_agents(self) -> list:
        """Return agents config from get_agents_config() (sequential pipeline)."""
        from unittest.mock import MagicMock, patch

        from vaig.core.config import get_settings
        from vaig.skills.service_health.skill import ServiceHealthSkill

        real_settings = get_settings()
        mock_settings = MagicMock()
        mock_settings.datadog = real_settings.datadog
        mock_settings.gke = real_settings.gke
        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            return ServiceHealthSkill().get_agents_config()

    def _make_full_gke_registry(self):  # type: ignore[return]
        """Build a full GKE ToolRegistry without hitting the network."""
        from unittest.mock import patch

        from vaig.core.config import GKEConfig
        from vaig.tools.base import ToolRegistry
        from vaig.tools.gke._registry import create_gke_tools

        gke_config = GKEConfig(
            project_id="test-project",
            location="us-central1",
            cluster_name="test-cluster",
        )
        # Patch at the source to prevent network auth during unit tests
        with patch("vaig.tools.gke._clients._query_autopilot_status", return_value=False):
            gke_tools = create_gke_tools(gke_config)

        registry = ToolRegistry()
        for tool in gke_tools:
            registry.register(tool)
        return registry

    # ── Tool-aware agents declare tool_categories ─────────────────────────

    def test_parallel_tool_aware_agents_all_have_tool_categories(self) -> None:
        """Every requires_tools=True agent in the parallel pipeline must declare tool_categories."""
        agents = self._get_parallel_agents()
        tool_aware = [a for a in agents if a.get("requires_tools") is True]
        assert len(tool_aware) > 0, "Expected at least one tool-aware agent"
        for agent in tool_aware:
            assert "tool_categories" in agent, (
                f"Agent '{agent['name']}' is tool-aware but missing tool_categories"
            )
            assert len(agent["tool_categories"]) > 0, (
                f"Agent '{agent['name']}' has empty tool_categories list"
            )

    def test_sequential_tool_aware_agents_all_have_tool_categories(self) -> None:
        """Every requires_tools=True agent in the sequential pipeline must declare tool_categories."""
        agents = self._get_sequential_agents()
        tool_aware = [a for a in agents if a.get("requires_tools") is True]
        assert len(tool_aware) > 0, "Expected at least one tool-aware agent"
        for agent in tool_aware:
            assert "tool_categories" in agent, (
                f"Agent '{agent['name']}' is tool-aware but missing tool_categories"
            )

    # ── Filtering actually reduces tool count ─────────────────────────────

    def test_filtering_reduces_tool_count_vs_full_registry(self) -> None:
        """Filtering a partial category set must yield fewer tools than the full GKE registry."""
        full_registry = self._make_full_gke_registry()
        full_count = len(full_registry.list_tools())
        assert full_count > 0

        # Filter to kubernetes only — must be a strict subset
        filtered = full_registry.filter_by_categories(frozenset({"kubernetes"}))
        filtered_count = len(filtered.list_tools())

        assert filtered_count < full_count, (
            f"Filtering for 'kubernetes' returned {filtered_count} tools "
            f"but full registry has {full_count} — no reduction achieved"
        )

    def test_each_parallel_gatherer_gets_strict_subset_of_full_registry(self) -> None:
        """Each parallel gatherer's category set must match fewer tools than the full GKE registry."""
        full_registry = self._make_full_gke_registry()
        full_count = len(full_registry.list_tools())

        agents = self._get_parallel_agents()
        parallel_gatherers = [
            a for a in agents
            if a.get("parallel_group") == "gather" and a.get("requires_tools") is True
        ]
        assert len(parallel_gatherers) > 0, "Expected at least one parallel gatherer"

        for agent in parallel_gatherers:
            categories = frozenset(agent["tool_categories"])
            filtered = full_registry.filter_by_categories(categories)
            filtered_count = len(filtered.list_tools())
            assert filtered_count < full_count, (
                f"Gatherer '{agent['name']}' with categories={list(categories)} "
                f"returned {filtered_count}/{full_count} tools — not a strict subset"
            )

    def test_non_tool_aware_agents_have_no_tool_categories(self) -> None:
        """SpecialistAgents (requires_tools=False) must not declare tool_categories."""
        agents = self._get_parallel_agents()
        specialist_agents = [a for a in agents if not a.get("requires_tools", False)]
        for agent in specialist_agents:
            assert "tool_categories" not in agent, (
                f"Non-tool-aware agent '{agent['name']}' unexpectedly has tool_categories"
            )

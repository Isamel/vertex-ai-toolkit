"""Tests for BaseSkill.route_agents hook and service_health capabilities coverage.

Tests cover:
- BaseSkill.route_agents delegates to vaig.core.router.route_agents by default
- A custom skill can override route_agents with its own logic
- All service_health parallel gatherer configs have a non-empty capabilities field
- Representative queries route to the expected gatherers
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.router import route_agents as core_route_agents
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase

# ── Minimal concrete subclass ─────────────────────────────────────────────────


class _MinimalSkill(BaseSkill):
    """Concrete skill that satisfies all abstract requirements (no overrides)."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="minimal",
            display_name="Minimal Skill",
            description="A minimal test skill",
        )

    def get_system_instruction(self) -> str:
        return "You are minimal."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"{phase.value}: {user_input}"


class _CustomRoutingSkill(_MinimalSkill):
    """Skill that overrides route_agents with custom logic."""

    def route_agents(self, query: str, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Custom logic: always return only the first config
        if configs:
            return [configs[0]]
        return configs


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_gatherer(name: str, capabilities: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "parallel_group": "gather",
        "capabilities": capabilities,
        "system_instruction": f"You are {name}.",
    }


def _make_sequential(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "system_instruction": f"You are {name}.",
    }


@pytest.fixture
def minimal_skill() -> _MinimalSkill:
    return _MinimalSkill()


@pytest.fixture
def custom_routing_skill() -> _CustomRoutingSkill:
    return _CustomRoutingSkill()


@pytest.fixture
def sample_configs() -> list[dict[str, Any]]:
    return [
        _make_gatherer("node_gatherer", ["node", "nodes", "cpu", "memory"]),
        _make_gatherer("workload_gatherer", ["pod", "pods", "deployment", "restart"]),
        _make_gatherer("logging_gatherer", ["log", "logs", "logging", "error"]),
        _make_sequential("health_analyzer"),
        _make_sequential("health_reporter"),
    ]


# ── TestBaseSkillRouteAgents ──────────────────────────────────────────────────


class TestBaseSkillRouteAgents:
    """Tests for the default BaseSkill.route_agents implementation."""

    def test_delegates_to_core_router(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        """Default route_agents must call vaig.core.router.route_agents."""
        with patch("vaig.skills.base._core_route_agents", wraps=core_route_agents) as mock_router:
            result = minimal_skill.route_agents("check pod crashes", sample_configs)
            mock_router.assert_called_once_with("check pod crashes", sample_configs)
        assert isinstance(result, list)

    def test_returns_list_of_dicts(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        result = minimal_skill.route_agents("any query", sample_configs)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_empty_configs_returns_empty(self, minimal_skill: _MinimalSkill) -> None:
        result = minimal_skill.route_agents("pods crashing", [])
        assert result == []

    def test_sequential_only_returns_all(self, minimal_skill: _MinimalSkill) -> None:
        """When all configs are sequential (no parallel_group), all pass through."""
        configs = [_make_sequential("agent_a"), _make_sequential("agent_b")]
        result = minimal_skill.route_agents("pods crashing", configs)
        assert result == configs

    def test_matching_gatherer_is_included(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        """A focused query should include only the matching gatherer + sequential agents."""
        result = minimal_skill.route_agents("show me the pod status", sample_configs)
        names = [c["name"] for c in result]
        assert "workload_gatherer" in names
        # Sequential agents must always be included
        assert "health_analyzer" in names
        assert "health_reporter" in names

    def test_non_matching_query_returns_all(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        """When no gatherer matches (e.g. short/irrelevant query), safe-all fallback applies."""
        result = minimal_skill.route_agents("xyzzy", sample_configs)
        assert len(result) == len(sample_configs)

    def test_empty_query_returns_all(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        """Empty query → safe-all fallback."""
        result = minimal_skill.route_agents("", sample_configs)
        assert len(result) == len(sample_configs)

    def test_node_query_routes_to_node_gatherer(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        result = minimal_skill.route_agents("cpu pressure on nodes", sample_configs)
        names = [c["name"] for c in result]
        assert "node_gatherer" in names
        assert "workload_gatherer" not in names
        assert "logging_gatherer" not in names

    def test_log_query_routes_to_logging_gatherer(
        self,
        minimal_skill: _MinimalSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        result = minimal_skill.route_agents("show error logs", sample_configs)
        names = [c["name"] for c in result]
        assert "logging_gatherer" in names
        assert "health_analyzer" in names  # sequential always passes through


# ── TestCustomRouteAgentsOverride ────────────────────────────────────────────


class TestCustomRouteAgentsOverride:
    """Tests that subclasses can override route_agents with custom logic."""

    def test_custom_override_is_called(
        self,
        custom_routing_skill: _CustomRoutingSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        """Custom implementation should replace the default, not call core router."""
        with patch("vaig.skills.base._core_route_agents") as mock_router:
            result = custom_routing_skill.route_agents("any query", sample_configs)
            # Core router must NOT be called — custom logic is in effect
            mock_router.assert_not_called()
        # Custom logic returns only the first config
        assert len(result) == 1
        assert result[0]["name"] == sample_configs[0]["name"]

    def test_custom_override_can_return_all(
        self,
        custom_routing_skill: _CustomRoutingSkill,
    ) -> None:
        """Custom routing can return any subset it wants."""
        configs = [_make_sequential("agent_a"), _make_sequential("agent_b")]
        result = custom_routing_skill.route_agents("irrelevant", configs)
        # Our custom logic: first config only
        assert result == [configs[0]]

    def test_custom_override_handles_empty(
        self,
        custom_routing_skill: _CustomRoutingSkill,
    ) -> None:
        result = custom_routing_skill.route_agents("some query", [])
        assert result == []

    def test_base_skill_default_and_override_coexist(
        self,
        minimal_skill: _MinimalSkill,
        custom_routing_skill: _CustomRoutingSkill,
        sample_configs: list[dict[str, Any]],
    ) -> None:
        """Default and overridden skills produce different results for the same query."""
        default_result = minimal_skill.route_agents("node cpu pressure", sample_configs)
        custom_result = custom_routing_skill.route_agents("node cpu pressure", sample_configs)
        # Default uses core router — may return a subset; custom always returns first config only
        assert custom_result == [sample_configs[0]]
        assert default_result != custom_result or len(default_result) != 1


# ── TestServiceHealthCapabilities ────────────────────────────────────────────


class TestServiceHealthCapabilities:
    """Tests that service_health gatherer configs have the capabilities field."""

    @pytest.fixture
    def service_health_configs(self) -> list[dict[str, Any]]:
        """Return the parallel agents config from ServiceHealthSkill (datadog disabled).

        Uses the real settings + patches datadog.enabled=False, exactly as
        the existing test_service_health.py pattern (patch vaig.core.config.get_settings).
        """
        from vaig.core.config import get_settings
        from vaig.skills.service_health.skill import ServiceHealthSkill

        real_settings = get_settings()
        mock_settings = MagicMock()
        mock_settings.gke = real_settings.gke
        mock_settings.datadog.enabled = False
        mock_settings.investigation.enabled = False  # prevent autonomous branch

        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            return ServiceHealthSkill().get_parallel_agents_config()

    @pytest.fixture
    def service_health_configs_with_datadog(self) -> list[dict[str, Any]]:
        """Return configs with Datadog enabled."""
        from vaig.core.config import get_settings
        from vaig.skills.service_health.skill import ServiceHealthSkill

        real_settings = get_settings()
        mock_settings = MagicMock()
        mock_settings.gke = real_settings.gke
        mock_settings.datadog.enabled = True
        mock_settings.investigation.enabled = False  # prevent autonomous branch

        with patch("vaig.core.config.get_settings", return_value=mock_settings):
            return ServiceHealthSkill().get_parallel_agents_config()

    def _get_parallel_gatherers(self, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only configs that are parallel gatherers (have parallel_group)."""
        return [c for c in configs if c.get("parallel_group") == "gather"]

    def _get_sequential(self, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only sequential agents (no parallel_group)."""
        return [c for c in configs if "parallel_group" not in c]

    def test_all_gatherers_have_capabilities(
        self,
        service_health_configs: list[dict[str, Any]],
    ) -> None:
        """Every parallel gatherer must have a non-empty capabilities list."""
        gatherers = self._get_parallel_gatherers(service_health_configs)
        assert len(gatherers) >= 4, "Expected at least 4 parallel gatherers"
        for g in gatherers:
            assert "capabilities" in g, f"Gatherer '{g['name']}' missing capabilities"
            assert isinstance(g["capabilities"], list), f"capabilities must be a list for '{g['name']}'"
            assert len(g["capabilities"]) > 0, f"capabilities must be non-empty for '{g['name']}'"

    def test_expected_gatherer_names_present(
        self,
        service_health_configs: list[dict[str, Any]],
    ) -> None:
        """Core gatherer names must be present in the parallel group."""
        gatherer_names = {c["name"] for c in self._get_parallel_gatherers(service_health_configs)}
        assert "node_gatherer" in gatherer_names
        assert "workload_gatherer" in gatherer_names
        assert "event_gatherer" in gatherer_names
        assert "logging_gatherer" in gatherer_names

    def test_datadog_gatherer_has_capabilities_when_enabled(
        self,
        service_health_configs_with_datadog: list[dict[str, Any]],
    ) -> None:
        """When Datadog is enabled, datadog_gatherer must also have capabilities."""
        gatherers = self._get_parallel_gatherers(service_health_configs_with_datadog)
        datadog = next((g for g in gatherers if g["name"] == "datadog_gatherer"), None)
        assert datadog is not None, "datadog_gatherer missing when datadog.enabled=True"
        assert "capabilities" in datadog
        assert len(datadog["capabilities"]) > 0

    def test_sequential_agents_have_no_capabilities(
        self,
        service_health_configs: list[dict[str, Any]],
    ) -> None:
        """Sequential tail agents (analyzer, verifier, reporter) must NOT have capabilities."""
        sequential = self._get_sequential(service_health_configs)
        for s in sequential:
            assert "capabilities" not in s, (
                f"Sequential agent '{s['name']}' should not have capabilities "
                "(capabilities are only for parallel gatherers)"
            )

    @pytest.mark.parametrize("query,expected_gatherers,excluded_gatherers", [
        # Highly specific queries that uniquely match a single domain
        (
            "pods are crashing with OOMKilled",
            ["workload_gatherer"],
            ["node_gatherer", "logging_gatherer", "event_gatherer"],
        ),
        (
            "crashloop restart count exceeded in deployment",
            ["workload_gatherer"],
            [],  # Not asserting exclusions — "in" matches "ingress" etc.
        ),
        (
            "networking issues with service DNS",
            ["event_gatherer"],
            ["node_gatherer", "workload_gatherer", "logging_gatherer"],
        ),
        (
            "check cloud logs for exceptions",
            ["logging_gatherer"],
            ["node_gatherer", "workload_gatherer", "event_gatherer"],
        ),
        (
            "dns connectivity and ingress",
            ["event_gatherer"],
            ["node_gatherer", "workload_gatherer", "logging_gatherer"],
        ),
        # CPU/node queries: 'cpu' and 'nodes' match node_gatherer precisely
        (
            "cpu nodes pressure",
            ["node_gatherer"],
            [],  # Not asserting exclusions — short tokens like 'on' can spread
        ),
    ])
    def test_query_routes_to_expected_gatherer(
        self,
        service_health_configs: list[dict[str, Any]],
        query: str,
        expected_gatherers: list[str],
        excluded_gatherers: list[str],
    ) -> None:
        """Representative queries should route to the correct gatherer."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        result = skill.route_agents(query, service_health_configs)
        result_names = [c["name"] for c in result]

        for expected in expected_gatherers:
            assert expected in result_names, (
                f"Expected '{expected}' in result for query: {query!r}\n"
                f"Got: {result_names}"
            )
        for excluded in excluded_gatherers:
            assert excluded not in result_names, (
                f"Expected '{excluded}' to be filtered out for query: {query!r}\n"
                f"Got: {result_names}"
            )

    def test_ambiguous_query_uses_safe_all_fallback(
        self,
        service_health_configs: list[dict[str, Any]],
    ) -> None:
        """A query that matches no capabilities triggers safe-all fallback."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        result = skill.route_agents("xyzzy frobnicate quux", service_health_configs)
        # Safe-all: all original configs must be returned
        assert len(result) == len(service_health_configs)

    def test_datadog_query_routes_to_datadog_gatherer(
        self,
        service_health_configs_with_datadog: list[dict[str, Any]],
    ) -> None:
        """A Datadog/APM-focused query routes to datadog_gatherer."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        result = skill.route_agents("check APM traces and latency in Datadog", service_health_configs_with_datadog)
        result_names = [c["name"] for c in result]
        assert "datadog_gatherer" in result_names

    def test_sequential_agents_always_in_result(
        self,
        service_health_configs: list[dict[str, Any]],
    ) -> None:
        """Sequential agents (analyzer, verifier, reporter) always pass through."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        result = skill.route_agents("pods crashing", service_health_configs)
        result_names = [c["name"] for c in result]
        sequential = self._get_sequential(service_health_configs)
        for s in sequential:
            assert s["name"] in result_names, (
                f"Sequential agent '{s['name']}' must always be in result"
            )

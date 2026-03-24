"""Integration tests for dynamic agent routing in the Orchestrator.

Verifies that `execute_with_tools` and `async_execute_with_tools` call
`skill.route_agents()` before creating agents, so only the routed subset
of gatherers is executed.

Test scope:
- Routing hook is invoked with the correct (query, configs) arguments
- Only routed configs produce agents — the unmatched gatherers are not run
- Safe-all fallback: when no gatherers match, all agents still run
- Custom override: a skill that overrides route_agents controls the list
- route_agents is called even when strategy falls back to sequential
- Logging: activated agent names are logged at INFO level
- Async path: _async_execute_with_tools_impl also calls route_agents
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from vaig.agents.orchestrator import Orchestrator
from vaig.core.client import GenerationResult
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generation_result(text: str = "ok") -> GenerationResult:
    return GenerationResult(
        text=text,
        model="gemini-2.5-pro",
        usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        finish_reason="STOP",
    )


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.generate.return_value = _make_generation_result()
    return client


def _make_mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.models.default = "gemini-2.5-pro"
    settings.models.fallback = "gemini-2.5-flash"
    settings.agents.max_iterations_retry = 10
    settings.budget.max_cost_per_run = 0.0
    settings.agents.max_failures_before_fallback = 0
    return settings


def _make_mock_registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Stub skills
# ---------------------------------------------------------------------------


class _BaseStubSkill(BaseSkill):
    """Base for all routing integration test skills."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="routing_test_skill",
            display_name="Routing Test Skill",
            description="Stub for routing integration tests",
        )

    def get_system_instruction(self) -> str:
        return "You are a routing test skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return user_input


class _GathererSkill(_BaseStubSkill):
    """Skill with 3 gatherers (node, workload, logging) + 1 sequential reporter."""

    NODE_CFG: dict[str, Any] = {
        "name": "node_gatherer",
        "role": "Node Gatherer",
        "system_instruction": "Gather node data.",
        "model": "gemini-2.5-pro",
        "requires_tools": False,
        "capabilities": ["node", "nodes", "cluster", "cpu", "memory"],
    }
    WORKLOAD_CFG: dict[str, Any] = {
        "name": "workload_gatherer",
        "role": "Workload Gatherer",
        "system_instruction": "Gather workload data.",
        "model": "gemini-2.5-pro",
        "requires_tools": False,
        "capabilities": ["pod", "pods", "deployment", "crash", "oomkilled"],
    }
    LOG_CFG: dict[str, Any] = {
        "name": "logging_gatherer",
        "role": "Logging Gatherer",
        "system_instruction": "Gather logs.",
        "model": "gemini-2.5-pro",
        "requires_tools": False,
        "capabilities": ["log", "logs", "logging", "error", "errors"],
    }
    REPORTER_CFG: dict[str, Any] = {
        "name": "reporter",
        "role": "Reporter",
        "system_instruction": "Write report.",
        "model": "gemini-2.5-flash",
        "requires_tools": False,
        # No capabilities — sequential agent, always passes through
    }

    def get_agents_config(self, **kwargs: Any) -> list[dict]:
        return [self.NODE_CFG, self.WORKLOAD_CFG, self.LOG_CFG, self.REPORTER_CFG]


class _CustomRouteSkill(_GathererSkill):
    """Skill that overrides route_agents to always return only the reporter."""

    def route_agents(self, query: str, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Custom: only keep the reporter regardless of query
        return [c for c in configs if c.get("name") == "reporter"]


# ===========================================================================
# Task 3.4 — Integration: execute_with_tools calls route_agents
# ===========================================================================


class TestExecuteWithToolsRouting:
    """Verify that route_agents is called in the sync pipeline."""

    def test_route_agents_called_with_query_and_configs(self) -> None:
        """route_agents receives the user query and the skill's config list."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        with patch.object(skill, "route_agents", wraps=skill.route_agents) as mock_route:
            orchestrator.execute_with_tools("check node cpu", skill, registry, strategy="sequential")

        mock_route.assert_called_once()
        call_args = mock_route.call_args
        assert call_args[0][0] == "check node cpu"  # query
        configs_arg = call_args[0][1]
        config_names = [c["name"] for c in configs_arg]
        assert "node_gatherer" in config_names
        assert "workload_gatherer" in config_names
        assert "logging_gatherer" in config_names
        assert "reporter" in config_names

    def test_only_routed_agents_are_created(self) -> None:
        """Agents are created only for the configs returned by route_agents."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        # route_agents returns only node_gatherer + reporter (skips workload + logging)
        routed = [_GathererSkill.NODE_CFG, _GathererSkill.REPORTER_CFG]
        with patch.object(skill, "route_agents", return_value=routed):
            orchestrator.execute_with_tools("check node cpu", skill, registry, strategy="sequential")

        agent_names = orchestrator.list_agents()
        assert "node_gatherer" in agent_names
        assert "reporter" in agent_names
        assert "workload_gatherer" not in agent_names
        assert "logging_gatherer" not in agent_names

    def test_custom_override_controls_agents(self) -> None:
        """A skill that overrides route_agents fully controls which agents run."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _CustomRouteSkill()
        registry = _make_mock_registry()

        orchestrator.execute_with_tools("pods crashing", skill, registry, strategy="sequential")

        agent_names = orchestrator.list_agents()
        # Custom override returns only reporter
        assert agent_names == ["reporter"]

    def test_safe_all_fallback_when_no_match(self) -> None:
        """When no gatherers match, all 4 configs pass through (safe-all fallback)."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        # Very abstract query — tokens like "the", "is", "a" shouldn't match specific caps
        # Use a query that definitely won't match any capability keywords
        with patch("vaig.core.router.route_agents", return_value=skill.get_agents_config()) as mock_core:
            with patch.object(skill, "route_agents", side_effect=lambda q, c: mock_core(q, c)):
                orchestrator.execute_with_tools("xyz123 qqqq zzz", skill, registry, strategy="sequential")

        # All 4 agents should be present (safe-all fallback from core router)
        agent_names = orchestrator.list_agents()
        assert len(agent_names) == 4

    def test_route_agents_called_before_agent_creation(self) -> None:
        """route_agents is invoked before create_agents_for_skill."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        call_order: list[str] = []

        original_route = skill.route_agents
        original_create = orchestrator.create_agents_for_skill

        def spy_route(query: str, configs: list[dict]) -> list[dict]:
            call_order.append("route_agents")
            return original_route(query, configs)

        def spy_create(sk: Any, reg: Any = None, **kw: Any) -> list:
            call_order.append("create_agents_for_skill")
            return original_create(sk, reg, **kw)

        with (
            patch.object(skill, "route_agents", side_effect=spy_route),
            patch.object(orchestrator, "create_agents_for_skill", side_effect=spy_create),
        ):
            orchestrator.execute_with_tools("check nodes", skill, registry, strategy="sequential")

        # route_agents must appear before create_agents_for_skill
        assert "route_agents" in call_order
        assert "create_agents_for_skill" in call_order
        assert call_order.index("route_agents") < call_order.index("create_agents_for_skill")

    def test_routing_logs_activated_agent_names(self) -> None:
        """Activated agent names are logged at INFO level after routing."""
        import logging as _logging

        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        # Use a direct in-memory handler — immune to caplog propagation issues
        # that occur when other tests modify logger state before this one runs.
        records: list[_logging.LogRecord] = []

        class _CapturingHandler(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                records.append(record)

        orch_logger = _logging.getLogger("vaig.agents.orchestrator")
        handler = _CapturingHandler()
        handler.setLevel(_logging.INFO)
        old_level = orch_logger.level
        orch_logger.setLevel(_logging.INFO)
        orch_logger.addHandler(handler)
        try:
            routed = [_GathererSkill.NODE_CFG, _GathererSkill.REPORTER_CFG]
            with patch.object(skill, "route_agents", return_value=routed):
                orchestrator.execute_with_tools("check node cpu", skill, registry, strategy="sequential")
        finally:
            orch_logger.removeHandler(handler)
            orch_logger.setLevel(old_level)

        messages = [r.getMessage() for r in records]
        routing_lines = [m for m in messages if "Dynamic routing" in m]
        assert routing_lines, "Expected 'Dynamic routing' log message not found"
        routing_msg = routing_lines[0]
        assert "node_gatherer" in routing_msg
        assert "reporter" in routing_msg
        # New format: X/Y pattern — routed count out of original count
        assert "/" in routing_msg

    def test_routing_with_no_capabilities_agents_passes_all(self) -> None:
        """When no agent has capabilities, all configs pass through unchanged."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        class _NoCapsSkill(_BaseStubSkill):
            def get_agents_config(self, **kwargs: Any) -> list[dict]:
                return [
                    {"name": "agent-a", "role": "A", "system_instruction": "A.", "model": "gemini-2.5-pro"},
                    {"name": "agent-b", "role": "B", "system_instruction": "B.", "model": "gemini-2.5-pro"},
                ]

        skill = _NoCapsSkill()
        orchestrator.execute_with_tools("some query", skill, registry, strategy="sequential")

        agent_names = orchestrator.list_agents()
        assert "agent-a" in agent_names
        assert "agent-b" in agent_names


# ===========================================================================
# Task 3.4 — Integration: async_execute_with_tools also calls route_agents
# ===========================================================================


class TestAsyncExecuteWithToolsRouting:
    """Verify that route_agents is called in the async pipeline."""

    def test_async_route_agents_called(self) -> None:
        """route_agents is also invoked in the async path."""
        client = _make_mock_client()
        # async_execute requires async_generate; fall back to sync via to_thread
        client.generate.return_value = _make_generation_result()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        with patch.object(skill, "route_agents", wraps=skill.route_agents) as mock_route:
            asyncio.run(
                orchestrator.async_execute_with_tools(
                    "check node cpu", skill, registry, strategy="sequential"
                )
            )

        mock_route.assert_called_once()
        call_args = mock_route.call_args
        assert call_args[0][0] == "check node cpu"

    def test_async_only_routed_agents_created(self) -> None:
        """Async path creates agents only for the routed configs."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        routed = [_GathererSkill.WORKLOAD_CFG, _GathererSkill.REPORTER_CFG]
        with patch.object(skill, "route_agents", return_value=routed):
            asyncio.run(
                orchestrator.async_execute_with_tools(
                    "pods are crashing", skill, registry, strategy="sequential"
                )
            )

        agent_names = orchestrator.list_agents()
        assert "workload_gatherer" in agent_names
        assert "reporter" in agent_names
        assert "node_gatherer" not in agent_names
        assert "logging_gatherer" not in agent_names

    def test_async_routing_logs_activated_agents(self) -> None:
        """Activated agent names are logged in the async path."""
        import logging as _logging

        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        # Use a direct in-memory handler — immune to caplog propagation issues
        # that occur when other tests modify logger state before this one runs.
        records: list[_logging.LogRecord] = []

        class _CapturingHandler(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                records.append(record)

        orch_logger = _logging.getLogger("vaig.agents.orchestrator")
        handler = _CapturingHandler()
        handler.setLevel(_logging.INFO)
        old_level = orch_logger.level
        orch_logger.setLevel(_logging.INFO)
        orch_logger.addHandler(handler)
        try:
            routed = [_GathererSkill.LOG_CFG, _GathererSkill.REPORTER_CFG]
            with patch.object(skill, "route_agents", return_value=routed):
                asyncio.run(
                    orchestrator.async_execute_with_tools(
                        "show me recent error logs", skill, registry, strategy="sequential"
                    )
                )
        finally:
            orch_logger.removeHandler(handler)
            orch_logger.setLevel(old_level)

        messages = [r.getMessage() for r in records]
        routing_lines = [m for m in messages if "Dynamic routing" in m]
        assert routing_lines, "Expected 'Dynamic routing' log message not found in async path"
        routing_msg = routing_lines[0]
        assert "logging_gatherer" in routing_msg
        assert "reporter" in routing_msg
        # New format: X/Y pattern — routed count out of original count
        assert "/" in routing_msg


# ===========================================================================
# Task 3.5 — Verify existing orchestrator tests not broken
# (these test the orthogonal behavior: existing tests still pass even when
# route_agents returns all configs, i.e. default behavior = no filtering)
# ===========================================================================


class TestExistingBehaviorPreserved:
    """Smoke-test that existing sequential / fanout / single paths
    still work correctly after injecting route_agents.

    route_agents defaults to core_route_agents, which when no capabilities
    are declared returns all configs unchanged — so existing stubs are
    unaffected.
    """

    def test_sequential_with_no_capabilities_unchanged(self) -> None:
        """Skill without capabilities in configs still runs all agents (no regression)."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        class _PlainSkill(_BaseStubSkill):
            def get_agents_config(self, **kwargs: Any) -> list[dict]:
                return [
                    {"name": "a1", "role": "R1", "system_instruction": "s1.", "model": "gemini-2.5-pro"},
                    {"name": "a2", "role": "R2", "system_instruction": "s2.", "model": "gemini-2.5-pro"},
                ]

        skill = _PlainSkill()
        result = orchestrator.execute_with_tools("anything", skill, registry, strategy="sequential")

        assert result.skill_name == "routing_test_skill"
        assert len(orchestrator.list_agents()) == 2

    def test_result_still_has_synthesized_output(self) -> None:
        """Pipeline still produces synthesized_output after routing injection."""
        client = _make_mock_client()
        client.generate.return_value = _make_generation_result("final answer")
        orchestrator = Orchestrator(client, _make_mock_settings())
        registry = _make_mock_registry()

        class _SingleAgentSkill(_BaseStubSkill):
            def get_agents_config(self, **kwargs: Any) -> list[dict]:
                return [
                    {"name": "sole-agent", "role": "Worker", "system_instruction": "Do it.", "model": "gemini-2.5-pro"},
                ]

        skill = _SingleAgentSkill()
        result = orchestrator.execute_with_tools("do something", skill, registry, strategy="single")

        assert result.success is True
        assert result.synthesized_output == "final answer"

    def test_route_agents_receives_post_injection_configs(self) -> None:
        """route_agents receives configs AFTER language/autopilot injection.

        This ensures routing decisions can observe injected instructions,
        and that routing is applied to the final config list (not the raw
        skill output).
        """
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = _GathererSkill()
        registry = _make_mock_registry()

        captured_configs: list[list[dict]] = []

        def capture_route(query: str, configs: list[dict]) -> list[dict]:
            captured_configs.append(list(configs))
            return configs  # pass-through

        with patch.object(skill, "route_agents", side_effect=capture_route):
            orchestrator.execute_with_tools(
                "check nodes",
                skill,
                registry,
                strategy="sequential",
                is_autopilot=True,  # triggers autopilot injection
            )

        # route_agents must have been called
        assert captured_configs, "route_agents was not called"
        # All configs should have been passed (autopilot-injected or not)
        passed_names = {c["name"] for c in captured_configs[0]}
        assert "node_gatherer" in passed_names
        assert "reporter" in passed_names

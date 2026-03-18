"""Tests for Phase 2: pre_execute_parallel hook and K8s client pre-warming."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import Orchestrator
from vaig.agents.specialist import SpecialistAgent
from vaig.core.client import GeminiClient, GenerationResult
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> MagicMock:
    client = MagicMock(spec=GeminiClient)
    client.generate.return_value = GenerationResult(
        text="Agent response",
        model="gemini-2.5-pro",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="STOP",
    )
    return client


def _make_mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.models.default = "gemini-2.5-pro"
    settings.budget.max_cost_per_run = 0.0
    settings.agents.max_failures_before_fallback = 0
    return settings


def _make_agent_result(
    name: str, *, success: bool = True, content: str | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        content=content or f"Result from {name}",
        success=success,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


class SkillWithHook(BaseSkill):
    """Skill that implements pre_execute_parallel — records calls."""

    def __init__(self) -> None:
        self.hook_calls: list[str] = []

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="hook_skill",
            display_name="Hook Skill",
            description="Skill with pre_execute_parallel hook for testing.",
        )

    def get_system_instruction(self) -> str:
        return "Hook skill instruction."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "data_gatherer",
                "role": "Data Gatherer",
                "system_instruction": "gather data",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            },
            {
                "name": "analyzer",
                "role": "Analyzer",
                "system_instruction": "analyze",
                "model": "gemini-2.5-pro",
                "requires_tools": False,
            },
        ]

    def pre_execute_parallel(self, query: str) -> None:
        self.hook_calls.append(query)


class SkillWithoutHook(BaseSkill):
    """Skill that does NOT override pre_execute_parallel — tests backward compat."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="no_hook_skill",
            display_name="No Hook Skill",
            description="Skill without pre_execute_parallel for backward-compat testing.",
        )

    def get_system_instruction(self) -> str:
        return "No hook skill instruction."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "data_gatherer",
                "role": "Data Gatherer",
                "system_instruction": "gather data",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            },
        ]


# ---------------------------------------------------------------------------
# Tests: BaseSkill.pre_execute_parallel default
# ---------------------------------------------------------------------------


class TestBaseSkillHookDefault:
    """Verify BaseSkill provides a no-op default for pre_execute_parallel."""

    def test_base_skill_hook_exists_and_is_callable(self) -> None:
        """BaseSkill must have a pre_execute_parallel method."""
        skill = SkillWithoutHook()
        # Should not raise — default is a no-op
        result = skill.pre_execute_parallel("test query")
        assert result is None  # default returns None

    def test_base_skill_hook_accepts_query_string(self) -> None:
        """pre_execute_parallel must accept a str query argument."""
        skill = SkillWithoutHook()
        # No exception means the signature is correct
        skill.pre_execute_parallel("check all namespaces are healthy")


# ---------------------------------------------------------------------------
# Tests: Orchestrator calls hook before parallel execution
# ---------------------------------------------------------------------------


class TestOrchestratorCallsHook:
    """Verify the orchestrator calls pre_execute_parallel before threads launch."""

    def test_hook_is_called_before_parallel_execution(self) -> None:
        """pre_execute_parallel must be called before any gatherer agent executes."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SkillWithHook()
        tool_registry = MagicMock(spec=ToolRegistry)

        call_order: list[str] = []

        def hook_side_effect(q: str) -> None:
            call_order.append("hook")

        def gatherer_execute(query: str, **kwargs: Any) -> AgentResult:
            call_order.append("gatherer")
            return _make_agent_result("data_gatherer")

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = MagicMock(spec=SpecialistAgent)
            gatherer.name = "data_gatherer"
            gatherer.role = "Data Gatherer"
            gatherer.execute.side_effect = gatherer_execute

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "Analyzer"
            analyzer.execute.return_value = _make_agent_result("analyzer")

            mock_create.return_value = [gatherer, analyzer]

            with patch.object(skill, "pre_execute_parallel", side_effect=hook_side_effect):
                orchestrator.execute_with_tools(
                    "check health",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )

        # hook must appear before gatherer in call order
        assert "hook" in call_order
        assert "gatherer" in call_order
        assert call_order.index("hook") < call_order.index("gatherer")

    def test_hook_receives_query_string(self) -> None:
        """pre_execute_parallel must receive the original query string."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SkillWithHook()
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = MagicMock(spec=SpecialistAgent)
            gatherer.name = "data_gatherer"
            gatherer.role = "Data Gatherer"
            gatherer.execute.return_value = _make_agent_result("data_gatherer")

            mock_create.return_value = [gatherer]

            orchestrator.execute_with_tools(
                "specific query text",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert skill.hook_calls == ["specific query text"]

    def test_hook_called_once_not_per_gatherer(self) -> None:
        """pre_execute_parallel must be called once regardless of gatherer count."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SkillWithHook()
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "Node Gatherer"
            g0.execute.return_value = _make_agent_result("node_gatherer")

            g1 = MagicMock(spec=SpecialistAgent)
            g1.name = "workload_gatherer"
            g1.role = "Workload Gatherer"
            g1.execute.return_value = _make_agent_result("workload_gatherer")

            mock_create.return_value = [g0, g1]

            # Override skill to track calls using the SkillWithHook
            skill2 = SkillWithHook()
            skill2.get_agents_config = lambda: [  # type: ignore[method-assign]
                {"name": "node_gatherer", "role": "Node Gatherer",
                 "system_instruction": "gather", "model": "gemini-2.5-pro", "requires_tools": True},
                {"name": "workload_gatherer", "role": "Workload Gatherer",
                 "system_instruction": "gather", "model": "gemini-2.5-pro", "requires_tools": True},
            ]
            orchestrator.execute_with_tools(
                "multi gatherer query",
                skill2,
                tool_registry,
                strategy="parallel_sequential",
            )

        # hook called exactly once, not twice for two gatherers
        assert len(skill2.hook_calls) == 1

    def test_skill_without_hook_still_works(self) -> None:
        """Skills that don't override pre_execute_parallel must work unchanged."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SkillWithoutHook()
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = MagicMock(spec=SpecialistAgent)
            gatherer.name = "data_gatherer"
            gatherer.role = "Data Gatherer"
            gatherer.execute.return_value = _make_agent_result("data_gatherer")

            mock_create.return_value = [gatherer]

            # Should not raise even though skill has no custom hook
            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.success


# ---------------------------------------------------------------------------
# Tests: Async orchestrator calls hook
# ---------------------------------------------------------------------------


class TestAsyncOrchestratorCallsHook:
    """Verify the async path also calls pre_execute_parallel before parallel launch."""

    def test_async_hook_is_called_before_parallel(self) -> None:
        """Async path: hook must be called before gatherer agents start."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SkillWithHook()
        tool_registry = MagicMock(spec=ToolRegistry)

        call_order: list[str] = []

        def hook_side_effect(q: str) -> None:
            call_order.append("hook")

        def gatherer_execute(query: str, **kwargs: Any) -> AgentResult:
            call_order.append("gatherer")
            return _make_agent_result("data_gatherer")

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = MagicMock(spec=SpecialistAgent)
            gatherer.name = "data_gatherer"
            gatherer.role = "Data Gatherer"
            gatherer.execute.side_effect = gatherer_execute

            mock_create.return_value = [gatherer]

            with patch.object(skill, "pre_execute_parallel", side_effect=hook_side_effect):
                asyncio.run(
                    orchestrator.async_execute_with_tools(
                        "async check health",
                        skill,
                        tool_registry,
                        strategy="parallel_sequential",
                    )
                )

        assert "hook" in call_order
        assert "gatherer" in call_order
        assert call_order.index("hook") < call_order.index("gatherer")

    def test_async_skill_without_hook_still_works(self) -> None:
        """Async path: skills without hook must run without errors."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SkillWithoutHook()
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = MagicMock(spec=SpecialistAgent)
            gatherer.name = "data_gatherer"
            gatherer.role = "Data Gatherer"
            gatherer.execute.return_value = _make_agent_result("data_gatherer")

            mock_create.return_value = [gatherer]

            result = asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )
            )

        assert result.success


# ---------------------------------------------------------------------------
# Tests: ensure_client_initialized
# ---------------------------------------------------------------------------


class TestEnsureClientInitialized:
    """Verify ensure_client_initialized populates _CLIENT_CACHE idempotently."""

    def test_ensure_client_initialized_function_exists(self) -> None:
        """ensure_client_initialized must be importable from _clients."""
        from vaig.tools.gke._clients import ensure_client_initialized  # noqa: PLC0415
        assert callable(ensure_client_initialized)

    def test_ensure_client_initialized_populates_cache(self) -> None:
        """Calling ensure_client_initialized should trigger _create_k8s_clients."""
        from vaig.tools.gke._clients import (  # noqa: PLC0415
            _CLIENT_CACHE,
            _cache_key,
            clear_k8s_client_cache,
            ensure_client_initialized,
        )

        clear_k8s_client_cache()

        gke_config = MagicMock()
        gke_config.kubeconfig_path = "/fake/kubeconfig"
        gke_config.context = "test-context"
        gke_config.proxy_url = ""

        mock_clients = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        key = _cache_key(gke_config)

        # Write directly to cache (simulating what _create_k8s_clients would do)
        def fake_create(cfg: Any) -> Any:
            _CLIENT_CACHE[key] = mock_clients
            return mock_clients

        with patch("vaig.tools.gke._clients._create_k8s_clients", side_effect=fake_create):
            ensure_client_initialized(gke_config)

        assert key in _CLIENT_CACHE
        assert _CLIENT_CACHE[key] == mock_clients

        # Cleanup
        clear_k8s_client_cache()

    def test_ensure_client_initialized_is_idempotent(self) -> None:
        """Calling ensure_client_initialized twice must not create clients twice."""
        from vaig.tools.gke._clients import (  # noqa: PLC0415
            _CLIENT_CACHE,
            _cache_key,
            clear_k8s_client_cache,
            ensure_client_initialized,
        )

        clear_k8s_client_cache()

        gke_config = MagicMock()
        gke_config.kubeconfig_path = ""
        gke_config.context = ""
        gke_config.proxy_url = ""

        mock_clients = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        key = _cache_key(gke_config)
        call_count = 0

        def counting_create(cfg: Any) -> Any:
            nonlocal call_count
            call_count += 1
            # Simulate what _create_k8s_clients does — write to cache
            _CLIENT_CACHE[key] = mock_clients
            return mock_clients

        with patch("vaig.tools.gke._clients._create_k8s_clients", side_effect=counting_create):
            ensure_client_initialized(gke_config)
            ensure_client_initialized(gke_config)  # second call — should be no-op

        assert call_count == 1  # _create_k8s_clients called only once

        # Cleanup
        clear_k8s_client_cache()

    def test_ensure_client_initialized_handles_tool_result_error_gracefully(self) -> None:
        """If _create_k8s_clients returns a ToolResult(error=True), no exception is raised."""
        from vaig.tools.base import ToolResult  # noqa: PLC0415
        from vaig.tools.gke._clients import (  # noqa: PLC0415
            clear_k8s_client_cache,
            ensure_client_initialized,
        )

        clear_k8s_client_cache()

        gke_config = MagicMock()
        gke_config.kubeconfig_path = ""
        gke_config.context = ""
        gke_config.proxy_url = ""

        error_result = ToolResult(output="K8s not available", error=True)

        with patch("vaig.tools.gke._clients._create_k8s_clients", return_value=error_result):
            # Must not raise — errors should be swallowed with a log warning
            ensure_client_initialized(gke_config)

        # Cache should remain empty since init failed
        clear_k8s_client_cache()


# ---------------------------------------------------------------------------
# Tests: ServiceHealthSkill.pre_execute_parallel
# ---------------------------------------------------------------------------


class TestServiceHealthPreExecuteHook:
    """Verify ServiceHealthSkill.pre_execute_parallel pre-warms the K8s client."""

    def test_service_health_has_pre_execute_parallel(self) -> None:
        """ServiceHealthSkill must implement pre_execute_parallel."""
        from vaig.skills.service_health.skill import ServiceHealthSkill  # noqa: PLC0415

        skill = ServiceHealthSkill()
        assert hasattr(skill, "pre_execute_parallel")
        assert callable(skill.pre_execute_parallel)

    def test_service_health_hook_calls_ensure_client_initialized(self) -> None:
        """ServiceHealthSkill.pre_execute_parallel must call ensure_client_initialized."""
        from vaig.skills.service_health.skill import ServiceHealthSkill  # noqa: PLC0415

        skill = ServiceHealthSkill()

        with patch(
            "vaig.skills.service_health.skill.ensure_client_initialized",
        ) as mock_ensure:
            skill.pre_execute_parallel("check health of production cluster")

        mock_ensure.assert_called_once()

    def test_service_health_hook_does_not_raise_on_error(self) -> None:
        """ServiceHealthSkill.pre_execute_parallel must be silent on errors."""
        from vaig.skills.service_health.skill import ServiceHealthSkill  # noqa: PLC0415

        skill = ServiceHealthSkill()

        with patch(
            "vaig.skills.service_health.skill.ensure_client_initialized",
            side_effect=Exception("K8s not reachable"),
        ):
            # Must not propagate the exception
            skill.pre_execute_parallel("health check")

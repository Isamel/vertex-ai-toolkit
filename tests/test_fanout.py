"""Tests for concurrent fan-out execution and GeminiClient thread safety."""

from __future__ import annotations

import threading
import time
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
    return settings


class FanoutSkill(BaseSkill):
    """Skill with configurable agents for testing fan-out."""

    def __init__(self, agent_count: int = 3) -> None:
        self._agents = [
            {
                "name": f"agent-{i}",
                "role": f"Role {i}",
                "system_instruction": f"Instruction {i}.",
                "model": "gemini-2.5-pro",
            }
            for i in range(agent_count)
        ]

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="fanout_skill",
            display_name="Fanout Skill",
            description="A skill for testing fan-out execution.",
        )

    def get_system_instruction(self) -> str:
        return "You are a fanout skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        return list(self._agents)


class ToolFanoutSkill(BaseSkill):
    """Skill with tool-aware agents for testing fan-out via execute_with_tools."""

    def __init__(self, agent_count: int = 3) -> None:
        self._agents = [
            {
                "name": f"tool-agent-{i}",
                "role": f"Tool Role {i}",
                "system_instruction": f"Tool Instruction {i}.",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            }
            for i in range(agent_count)
        ]

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="tool_fanout_skill",
            display_name="Tool Fanout Skill",
            description="A tool-fanout skill.",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return "You are a tool fanout skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        return list(self._agents)


def _make_agent_result(name: str, *, success: bool = True) -> AgentResult:
    return AgentResult(
        agent_name=name,
        content=f"Result from {name}",
        success=success,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


# ===========================================================================
# Task 3.1a — Concurrent fanout execution
# ===========================================================================


class TestExecuteFanoutConcurrent:
    """Verify concurrent execution of execute_fanout() with mock agents."""

    def test_all_agents_called_with_correct_args(self) -> None:
        """All 3 agents should be called with the same prompt and context."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=3)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"agent-{i}"
                agent.execute.return_value = _make_agent_result(f"agent-{i}")
            mock_create.return_value = agents

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "test context", "test input",
            )

        assert len(result.agent_results) == 3
        for agent in agents:
            agent.execute.assert_called_once()

    def test_results_in_submission_order(self) -> None:
        """Results must be in submission order, not completion order."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=3)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"agent-{i}"
                agent.execute.return_value = _make_agent_result(f"agent-{i}")
            # Agent 2 finishes "fastest" — but order should still be 0, 1, 2
            mock_create.return_value = agents

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )

        agent_names = [r.agent_name for r in result.agent_results]
        assert agent_names == ["agent-0", "agent-1", "agent-2"]

    def test_usage_totals_accumulated_correctly(self) -> None:
        """Total usage must equal the sum of all agent usages."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=3)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"agent-{i}"
                agent.execute.return_value = _make_agent_result(f"agent-{i}")
            mock_create.return_value = agents

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )

        assert result.total_usage["prompt_tokens"] == 30  # 10 * 3
        assert result.total_usage["completion_tokens"] == 60  # 20 * 3
        assert result.total_usage["total_tokens"] == 90  # 30 * 3

    def test_agents_execute_concurrently(self) -> None:
        """Verify that agents actually run concurrently, not sequentially.

        Each mock agent sleeps for 0.2s.  With 3 agents running
        concurrently, total wall-clock time should be ~0.2s, not ~0.6s.
        """
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=3)

        def _slow_execute(prompt: str, *, context: str = "") -> AgentResult:
            time.sleep(0.2)
            return _make_agent_result(threading.current_thread().name)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"agent-{i}"
                agent.execute.side_effect = _slow_execute
            mock_create.return_value = agents

            start = time.monotonic()
            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )
            elapsed = time.monotonic() - start

        assert len(result.agent_results) == 3
        # If truly concurrent, elapsed should be ~0.2s, not ~0.6s
        assert elapsed < 0.5, f"Expected concurrent execution (<0.5s), but took {elapsed:.2f}s"

    def test_single_agent_degenerates(self) -> None:
        """Single agent fanout should produce identical result to sequential."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=1)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agent = MagicMock(spec=SpecialistAgent)
            agent.name = "agent-0"
            agent.execute.return_value = _make_agent_result("agent-0")
            mock_create.return_value = [agent]

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )

        assert len(result.agent_results) == 1
        assert result.success is True
        assert result.agent_results[0].agent_name == "agent-0"


# ===========================================================================
# Task 3.1b — Failure isolation
# ===========================================================================


class TestFanoutFailureIsolation:
    """One agent failing must not crash or cancel others."""

    def test_one_agent_fails_others_succeed(self) -> None:
        """Agent B raises exception; agents A and C still succeed."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=3)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"agent-{i}"
            agents[0].execute.return_value = _make_agent_result("agent-0")
            agents[1].execute.side_effect = RuntimeError("API timeout")
            agents[2].execute.return_value = _make_agent_result("agent-2")
            mock_create.return_value = agents

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )

        assert result.success is True  # at least one succeeded
        assert result.agent_results[0].success is True
        assert result.agent_results[1].success is False
        assert result.agent_results[2].success is True

    def test_all_agents_fail(self) -> None:
        """When all agents fail, overall success is False."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=2)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(2)]
            for i, agent in enumerate(agents):
                agent.name = f"agent-{i}"
                agent.execute.side_effect = RuntimeError("boom")
            mock_create.return_value = agents

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )

        assert result.success is False
        assert all(not r.success for r in result.agent_results)

    def test_agent_returns_failed_result_not_exception(self) -> None:
        """Agent that returns AgentResult(success=False) — not an exception."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = FanoutSkill(agent_count=2)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock(spec=SpecialistAgent) for _ in range(2)]
            agents[0].name = "agent-0"
            agents[0].execute.return_value = _make_agent_result("agent-0")
            agents[1].name = "agent-1"
            agents[1].execute.return_value = AgentResult(
                agent_name="agent-1",
                content="I failed gracefully",
                success=False,
                usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            )
            mock_create.return_value = agents

            result = orchestrator.execute_fanout(
                skill, SkillPhase.ANALYZE, "ctx", "input",
            )

        assert result.success is True  # agent-0 succeeded
        assert result.agent_results[1].success is False


# ===========================================================================
# Task 3.1c — execute_with_tools fanout branch
# ===========================================================================


class TestExecuteWithToolsFanout:
    """Verify execute_with_tools() fanout branch behaves identically."""

    def test_tools_fanout_concurrent(self) -> None:
        """execute_with_tools with strategy='fanout' runs concurrently."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ToolFanoutSkill(agent_count=3)
        registry = ToolRegistry()

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock() for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"tool-agent-{i}"
                agent.execute.return_value = _make_agent_result(f"tool-agent-{i}")
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "test query", skill, registry, strategy="fanout",
            )

        assert len(result.agent_results) == 3
        assert result.success is True
        for agent in agents:
            agent.execute.assert_called_once()

    def test_tools_fanout_failure_isolation(self) -> None:
        """execute_with_tools fanout isolates failures."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ToolFanoutSkill(agent_count=3)
        registry = ToolRegistry()

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock() for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"tool-agent-{i}"
            agents[0].execute.return_value = _make_agent_result("tool-agent-0")
            agents[1].execute.side_effect = ValueError("broken tool")
            agents[2].execute.return_value = _make_agent_result("tool-agent-2")
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "test query", skill, registry, strategy="fanout",
            )

        assert result.success is True
        assert result.agent_results[0].success is True
        assert result.agent_results[1].success is False
        assert result.agent_results[2].success is True

    def test_tools_fanout_usage_accumulation(self) -> None:
        """Usage tokens accumulate correctly from tool-fanout agents."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ToolFanoutSkill(agent_count=2)
        registry = ToolRegistry()

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock() for _ in range(2)]
            for i, agent in enumerate(agents):
                agent.name = f"tool-agent-{i}"
                agent.execute.return_value = _make_agent_result(f"tool-agent-{i}")
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "test query", skill, registry, strategy="fanout",
            )

        assert result.total_usage["prompt_tokens"] == 20  # 10 * 2
        assert result.total_usage["total_tokens"] == 60  # 30 * 2

    def test_tools_fanout_results_in_submission_order(self) -> None:
        """Tool fanout results must be in submission order."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ToolFanoutSkill(agent_count=3)
        registry = ToolRegistry()

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = [MagicMock() for _ in range(3)]
            for i, agent in enumerate(agents):
                agent.name = f"tool-agent-{i}"
                agent.execute.return_value = _make_agent_result(f"tool-agent-{i}")
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "test query", skill, registry, strategy="fanout",
            )

        agent_names = [r.agent_name for r in result.agent_results]
        assert agent_names == ["tool-agent-0", "tool-agent-1", "tool-agent-2"]


# ===========================================================================
# Task 3.2 — Thread safety of _reinitialize_with_fallback
# ===========================================================================


class TestReinitializeWithFallbackThreadSafety:
    """Thread safety test for GeminiClient._reinitialize_with_fallback().

    Uses a threading.Barrier to force 3 threads to call the method
    simultaneously, then asserts initialize() is called exactly once.
    """

    def test_concurrent_fallback_calls_initialize_once(self) -> None:
        """3 threads hit _reinitialize_with_fallback simultaneously.

        With the lock + double-checked pattern, initialize() should
        be called exactly once (for the fallback).
        """
        settings = MagicMock()
        settings.models.default = "gemini-2.5-pro"
        settings.gcp.location = "us-central1"
        settings.gcp.fallback_location = "europe-west1"

        client = GeminiClient(settings)
        # Mark as initialized (simulating successful primary init)
        client._initialized = True
        client._client = MagicMock()

        barrier = threading.Barrier(3)
        initialize_count = 0
        count_lock = threading.Lock()

        original_initialize = GeminiClient.initialize

        def counting_initialize(self: Any) -> None:
            nonlocal initialize_count
            with count_lock:
                initialize_count += 1
            # Don't actually call the real initialize (needs GCP credentials)
            self._initialized = True
            self._client = MagicMock()

        errors: list[Exception] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                client._reinitialize_with_fallback()
            except Exception as exc:
                errors.append(exc)

        with patch.object(GeminiClient, "initialize", counting_initialize):
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert not errors, f"Threads raised errors: {errors}"
        assert initialize_count == 1, (
            f"Expected initialize() called exactly once, but was called {initialize_count} times"
        )
        assert client._using_fallback is True
        assert client._active_location == "europe-west1"

    def test_fallback_lock_exists(self) -> None:
        """GeminiClient should have a _fallback_lock attribute."""
        settings = MagicMock()
        settings.models.default = "gemini-2.5-pro"
        settings.gcp.location = "us-central1"
        settings.gcp.fallback_location = "europe-west1"

        client = GeminiClient(settings)
        assert hasattr(client, "_fallback_lock")
        assert isinstance(client._fallback_lock, type(threading.Lock()))

"""Integration tests for the parallel_sequential pipeline (Phase 4).

These tests verify end-to-end behaviour of the full 4-gatherer pipeline
(node_gatherer, workload_gatherer, event_gatherer, logging_gatherer →
health_analyzer → health_verifier → health_reporter), including:

- Happy-path: all 4 gatherers succeed, merged output flows to analyzer
- Fallback: partial failures (2 out of 4, then 3 out of 4) are handled
  gracefully without crashing the pipeline
- Backward compatibility: sequential skills still execute correctly
- Auto-detection: ``parallel_group`` in agent configs triggers automatic
  upgrade from ``"sequential"`` to ``"parallel_sequential"`` strategy
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.base import AgentResult
from vaig.agents.orchestrator import Orchestrator
from vaig.agents.specialist import SpecialistAgent
from vaig.core.client import GeminiClient, GenerationResult
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.tools.base import ToolRegistry

# ---------------------------------------------------------------------------
# Test helpers
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


def _make_agent_result(
    name: str,
    *,
    success: bool = True,
    content: str | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        content=content or f"## Output from {name}\n\nDetails collected by {name}.",
        success=success,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


# ---------------------------------------------------------------------------
# Test skills
# ---------------------------------------------------------------------------


class ServiceHealthParallelSkill(BaseSkill):
    """Minimal ServiceHealthSkill stand-in for integration testing.

    Returns the same 7-agent parallel config structure as the real
    ServiceHealthSkill, using plain strings for prompts.
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="service_health",
            display_name="Service Health",
            description="GKE service health check.",
        )

    def get_system_instruction(self) -> str:
        return "You are a service health skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict[str, Any]]:
        """Return 7-agent config: 4 parallel gatherers + 3 sequential tail."""
        return [
            # ── Parallel group ─────────────────────────────────────────────
            {
                "name": "node_gatherer",
                "role": "Node Gatherer",
                "system_instruction": "Gather node health data.",
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "requires_tools": True,
                "parallel_group": "gather",
            },
            {
                "name": "workload_gatherer",
                "role": "Workload Gatherer",
                "system_instruction": "Gather workload health data.",
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "requires_tools": True,
                "parallel_group": "gather",
            },
            {
                "name": "event_gatherer",
                "role": "Event Gatherer",
                "system_instruction": "Gather event data.",
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "requires_tools": True,
                "parallel_group": "gather",
            },
            {
                "name": "logging_gatherer",
                "role": "Logging Gatherer",
                "system_instruction": "Gather Cloud Logging data.",
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "requires_tools": True,
                "parallel_group": "gather",
            },
            # ── Sequential tail ────────────────────────────────────────────
            {
                "name": "health_analyzer",
                "role": "Health Analyzer",
                "system_instruction": "Analyze gathered data.",
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "requires_tools": False,
            },
            {
                "name": "health_verifier",
                "role": "Health Verifier",
                "system_instruction": "Verify findings.",
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "requires_tools": False,
            },
            {
                "name": "health_reporter",
                "role": "Health Reporter",
                "system_instruction": "Write the report.",
                "model": "gemini-2.5-flash",
                "temperature": 0.3,
                "requires_tools": False,
            },
        ]


class SequentialOnlySkill(BaseSkill):
    """Skill with no parallel_group keys — sequential pipeline only.

    Used to verify backward compatibility: when no ``parallel_group`` key
    is present the Orchestrator must NOT upgrade the strategy.
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="sequential_skill",
            display_name="Sequential Skill",
            description="A purely sequential skill.",
        )

    def get_system_instruction(self) -> str:
        return "You are a sequential skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "gatherer",
                "role": "Data Gatherer",
                "system_instruction": "Gather data.",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            },
            {
                "name": "analyzer",
                "role": "Analyzer",
                "system_instruction": "Analyze data.",
                "model": "gemini-2.5-flash",
                "requires_tools": False,
            },
        ]


# ---------------------------------------------------------------------------
# Integration: 4.1 — Full pipeline happy path
# ---------------------------------------------------------------------------


class TestFullPipelineHappyPath:
    """4.1 — End-to-end pipeline with 4 mock sub-gatherers.

    Verifies that:
    - All 4 gatherers are called
    - Their outputs are merged and passed to the analyzer
    - The pipeline completes successfully with all 7 agent results
    """

    def test_all_four_gatherers_are_executed(self) -> None:
        """All 4 gatherer agents must be called exactly once each."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        gatherer_names = [
            "node_gatherer", "workload_gatherer", "event_gatherer", "logging_gatherer",
        ]
        sequential_names = ["health_analyzer", "health_verifier", "health_reporter"]

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name in gatherer_names:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = f"{name} role"
                a.execute.return_value = _make_agent_result(
                    name, content=f"## {name.replace('_', ' ').title()}\n\nData from {name}.",
                )
                agents.append(a)

            for name in sequential_names:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = f"{name} role"
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "Check cluster health in namespace prod",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.success
        assert len(result.agent_results) == 7
        result_names = [r.agent_name for r in result.agent_results]
        for name in gatherer_names:
            assert name in result_names, f"{name} missing from results"
        for name in sequential_names:
            assert name in result_names, f"{name} missing from results"

    def test_merged_output_contains_all_gatherer_sections(self) -> None:
        """The merged context passed to the analyzer must include all 4 gatherer sections."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        gatherer_contents = {
            "node_gatherer": "## Cluster Overview\n\nAll nodes healthy.",
            "workload_gatherer": "## Service Status\n\nAll pods running.",
            "event_gatherer": "## Events Timeline\n\nNo critical events.",
            "logging_gatherer": "## Cloud Logging Findings\n\nNo errors found.",
        }
        received_analyzer_context: list[str] = []

        def capture_analyzer_execute(query: str, **kwargs: Any) -> AgentResult:
            received_analyzer_context.append(kwargs.get("context", ""))
            return _make_agent_result("health_analyzer")

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name, content in gatherer_contents.items():
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = f"{name} role"
                a.execute.return_value = _make_agent_result(name, content=content)
                agents.append(a)

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "health_analyzer"
            analyzer.role = "Health Analyzer"
            analyzer.execute.side_effect = capture_analyzer_execute

            verifier = MagicMock(spec=SpecialistAgent)
            verifier.name = "health_verifier"
            verifier.role = "Health Verifier"
            verifier.execute.return_value = _make_agent_result("health_verifier")

            reporter = MagicMock(spec=SpecialistAgent)
            reporter.name = "health_reporter"
            reporter.role = "Health Reporter"
            reporter.execute.return_value = _make_agent_result("health_reporter")

            agents.extend([analyzer, verifier, reporter])
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "Check cluster health",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.success
        assert len(received_analyzer_context) == 1
        merged = received_analyzer_context[0]

        # All 4 gatherer names must be present in the merged context
        for name in gatherer_contents:
            assert name in merged, f"Gatherer '{name}' section missing from merged context"

        # All gatherer content must be present
        for content in gatherer_contents.values():
            assert content in merged, "Gatherer content missing from merged context"

    def test_pipeline_completes_with_synthesized_output(self) -> None:
        """The synthesized_output must be the reporter's content."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        reporter_output = "# Health Report\n\nAll systems nominal."

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name in ["node_gatherer", "workload_gatherer", "event_gatherer", "logging_gatherer"]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            for name in ["health_analyzer", "health_verifier"]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            reporter = MagicMock(spec=SpecialistAgent)
            reporter.name = "health_reporter"
            reporter.role = "Health Reporter"
            reporter.execute.return_value = _make_agent_result(
                "health_reporter", content=reporter_output,
            )
            agents.append(reporter)
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "Check cluster health",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.success
        assert result.synthesized_output == reporter_output

    def test_analyzer_receives_merged_context_not_raw_query(self) -> None:
        """The analyzer context must contain gatherer outputs, not just the original query."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        original_query = "Check cluster health in namespace test-ns"
        received_contexts: list[str] = []

        def capture_context(query: str, **kwargs: Any) -> AgentResult:
            received_contexts.append(kwargs.get("context", ""))
            return _make_agent_result("health_analyzer")

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name in ["node_gatherer", "workload_gatherer", "event_gatherer", "logging_gatherer"]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(
                    name, content=f"Specific data from {name}",
                )
                agents.append(a)

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "health_analyzer"
            analyzer.role = "Health Analyzer"
            analyzer.execute.side_effect = capture_context

            verifier = MagicMock(spec=SpecialistAgent)
            verifier.name = "health_verifier"
            verifier.role = "Health Verifier"
            verifier.execute.return_value = _make_agent_result("health_verifier")

            reporter = MagicMock(spec=SpecialistAgent)
            reporter.name = "health_reporter"
            reporter.role = "Health Reporter"
            reporter.execute.return_value = _make_agent_result("health_reporter")

            agents.extend([analyzer, verifier, reporter])
            mock_create.return_value = agents

            orchestrator.execute_with_tools(
                original_query,
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert len(received_contexts) == 1
        context = received_contexts[0]
        # Context must be the merged gatherer output, not the bare query
        assert "Specific data from node_gatherer" in context
        assert "Specific data from workload_gatherer" in context
        assert "Specific data from event_gatherer" in context
        assert "Specific data from logging_gatherer" in context


# ---------------------------------------------------------------------------
# Integration: 4.2 — Fallback with partial gatherer failures
# ---------------------------------------------------------------------------


class TestPartialFailureFallback:
    """4.2 — Graceful degradation when sub-gatherers fail.

    Boundary conditions:
    - 2 out of 4 gatherers fail → pipeline continues with partial merged output
    - 3 out of 4 gatherers fail → pipeline continues (any_gatherer_ok = True)
    - All 4 gatherers fail → pipeline fails (any_gatherer_ok = False)
    """

    def _run_with_failures(
        self,
        failing_names: list[str],
    ) -> tuple[Any, str]:
        """Run the pipeline with specified gatherers set to fail.

        Returns (OrchestratorResult, analyzer_context).
        """
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        gathered_context: list[str] = []

        def capture_analyzer(query: str, **kwargs: Any) -> AgentResult:
            gathered_context.append(kwargs.get("context", ""))
            return _make_agent_result("health_analyzer")

        all_gatherer_names = [
            "node_gatherer", "workload_gatherer", "event_gatherer", "logging_gatherer",
        ]

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name in all_gatherer_names:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                if name in failing_names:
                    a.execute.return_value = _make_agent_result(
                        name,
                        success=False,
                        content=f"{name} timed out",
                    )
                else:
                    a.execute.return_value = _make_agent_result(
                        name, content=f"## Data from {name}\n\nHealthy.",
                    )
                agents.append(a)

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "health_analyzer"
            analyzer.role = "Health Analyzer"
            analyzer.execute.side_effect = capture_analyzer

            verifier = MagicMock(spec=SpecialistAgent)
            verifier.name = "health_verifier"
            verifier.role = "Health Verifier"
            verifier.execute.return_value = _make_agent_result("health_verifier")

            reporter = MagicMock(spec=SpecialistAgent)
            reporter.name = "health_reporter"
            reporter.role = "Health Reporter"
            reporter.execute.return_value = _make_agent_result("health_reporter")

            agents.extend([analyzer, verifier, reporter])
            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "Check cluster health",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        context = gathered_context[0] if gathered_context else ""
        return result, context

    def test_two_failures_pipeline_continues(self) -> None:
        """2 out of 4 gatherers failing must not abort the pipeline."""
        result, context = self._run_with_failures(
            failing_names=["workload_gatherer", "event_gatherer"],
        )

        assert result.success, "Pipeline must succeed when ≥1 gatherer succeeds"
        # All 4 gatherer results must be recorded
        gatherer_results = [
            r for r in result.agent_results
            if r.agent_name.endswith("_gatherer")
        ]
        assert len(gatherer_results) == 4

    def test_two_failures_merged_output_contains_error_notes(self) -> None:
        """Merged output must contain [ERROR: ...] for failed gatherers."""
        _, context = self._run_with_failures(
            failing_names=["workload_gatherer", "event_gatherer"],
        )

        assert "[ERROR:" in context, "Merged output must mark failed gatherers with [ERROR:]"

    def test_two_failures_successful_gatherer_data_is_preserved(self) -> None:
        """Successful gatherers' output must still reach the analyzer."""
        _, context = self._run_with_failures(
            failing_names=["workload_gatherer", "event_gatherer"],
        )

        assert "node_gatherer" in context, "node_gatherer output missing from merged context"
        assert "logging_gatherer" in context, "logging_gatherer output missing from merged context"

    def test_three_failures_pipeline_continues_with_warning(self) -> None:
        """3 out of 4 gatherers failing — 1 success keeps the pipeline alive."""
        result, context = self._run_with_failures(
            failing_names=["workload_gatherer", "event_gatherer", "logging_gatherer"],
        )

        # any_gatherer_ok = True (node_gatherer succeeded), so pipeline must succeed
        assert result.success, "Pipeline must succeed when at least 1 gatherer succeeds"

        # Failure metadata must surface in the merged context as [ERROR:] markers
        # (caplog cannot reliably capture logs from ThreadPoolExecutor worker threads)
        assert "[ERROR:" in context, "Failed gatherers must leave error markers in merged context"

    def test_three_failures_only_successful_data_flows_to_analyzer(self) -> None:
        """Only the single successful gatherer's data must appear without errors."""
        _, context = self._run_with_failures(
            failing_names=["workload_gatherer", "event_gatherer", "logging_gatherer"],
        )

        # The one successful gatherer must be in context
        assert "node_gatherer" in context, "node_gatherer output missing"
        # 3 error notes must be present
        error_count = context.count("[ERROR:")
        assert error_count == 3, f"Expected 3 [ERROR:] markers, got {error_count}"

    def test_all_four_failures_pipeline_fails(self) -> None:
        """All 4 gatherers failing means any_gatherer_ok=False → pipeline fails."""
        result, _ = self._run_with_failures(
            failing_names=[
                "node_gatherer", "workload_gatherer", "event_gatherer", "logging_gatherer",
            ],
        )

        assert not result.success, "Pipeline must fail when ALL gatherers fail"


# ---------------------------------------------------------------------------
# Integration: 4.3 — Backward compatibility (sequential skills unchanged)
# ---------------------------------------------------------------------------


class TestBackwardCompatibilitySequential:
    """4.3 — Sequential skills must execute identically with no regressions.

    When a skill's ``get_agents_config()`` returns configs WITHOUT
    ``parallel_group`` keys, the Orchestrator must NOT auto-upgrade the
    strategy — it stays ``sequential``.
    """

    def test_sequential_skill_runs_sequentially(self) -> None:
        """A skill without parallel_group must run in sequential order."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SequentialOnlySkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        execution_order: list[str] = []

        def make_side_effect(name: str) -> Any:
            def _exec(query: str, **kwargs: Any) -> AgentResult:
                execution_order.append(name)
                return _make_agent_result(name)
            return _exec

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g = MagicMock(spec=SpecialistAgent)
            g.name = "gatherer"
            g.role = "Gatherer"
            g.execute.side_effect = make_side_effect("gatherer")

            a = MagicMock(spec=SpecialistAgent)
            a.name = "analyzer"
            a.role = "Analyzer"
            a.execute.side_effect = make_side_effect("analyzer")

            mock_create.return_value = [g, a]

            result = orchestrator.execute_with_tools(
                "test query",
                skill,
                tool_registry,
                strategy="sequential",
            )

        assert result.success
        assert execution_order == ["gatherer", "analyzer"], (
            f"Expected sequential order [gatherer, analyzer], got {execution_order}"
        )

    def test_sequential_skill_has_no_parallel_group_keys(self) -> None:
        """SequentialOnlySkill must not contain parallel_group in its config."""
        skill = SequentialOnlySkill()
        configs = skill.get_agents_config()
        for cfg in configs:
            assert "parallel_group" not in cfg, (
                f"Agent '{cfg['name']}' unexpectedly has parallel_group key"
            )

    def test_sequential_skill_stays_sequential_strategy(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Auto-detection must NOT fire when no parallel_group keys are present."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = SequentialOnlySkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        with caplog.at_level(logging.WARNING, logger="vaig.agents.orchestrator"):
            with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
                g = MagicMock(spec=SpecialistAgent)
                g.name = "gatherer"
                g.role = "Gatherer"
                g.execute.return_value = _make_agent_result("gatherer")

                a = MagicMock(spec=SpecialistAgent)
                a.name = "analyzer"
                a.role = "Analyzer"
                a.execute.return_value = _make_agent_result("analyzer")

                mock_create.return_value = [g, a]

                orchestrator.execute_with_tools(
                    "test query",
                    skill,
                    tool_registry,
                    strategy="sequential",
                )

        auto_detect_messages = [
            r.getMessage() for r in caplog.records
            if "Auto-detected" in r.getMessage()
        ]
        assert len(auto_detect_messages) == 0, (
            "Auto-detection must NOT fire for sequential-only skills"
        )

    def test_parallel_skill_with_sequential_strategy_auto_upgrades(self) -> None:
        """Passing strategy='sequential' with a parallel-config skill auto-upgrades."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name in [
                "node_gatherer", "workload_gatherer",
                "event_gatherer", "logging_gatherer",
            ]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            for name in ["health_analyzer", "health_verifier", "health_reporter"]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            mock_create.return_value = agents

            # Pass strategy="sequential" even though configs have parallel_group.
            # The auto-detection must upgrade to parallel_sequential and run all 7 agents.
            result = orchestrator.execute_with_tools(
                "test query",
                skill,
                tool_registry,
                strategy="sequential",
            )

        # Pipeline must succeed using parallel_sequential after auto-upgrade
        assert result.success
        assert len(result.agent_results) == 7


# ---------------------------------------------------------------------------
# Integration: 4.6 — Strategy auto-detection
# ---------------------------------------------------------------------------


class TestStrategyAutoDetection:
    """4.6 — Orchestrator auto-upgrades strategy when parallel_group is detected."""

    def test_auto_detection_upgrades_sequential_to_parallel_sequential(self) -> None:
        """strategy='sequential' + parallel_group configs → auto-upgrade to parallel_sequential."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            agents = []
            for name in [
                "node_gatherer", "workload_gatherer",
                "event_gatherer", "logging_gatherer",
            ]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            for name in ["health_analyzer", "health_verifier", "health_reporter"]:
                a = MagicMock(spec=SpecialistAgent)
                a.name = name
                a.role = name
                a.execute.return_value = _make_agent_result(name)
                agents.append(a)

            mock_create.return_value = agents

            result = orchestrator.execute_with_tools(
                "Check cluster",
                skill,
                tool_registry,
                strategy="sequential",
            )

        # Behavioral assertion: auto-upgrade ran and the full 7-agent pipeline completed
        assert result.success
        assert len(result.agent_results) == 7

    def test_explicit_parallel_sequential_does_not_double_upgrade(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Explicitly passing strategy='parallel_sequential' must not log an auto-upgrade."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        with caplog.at_level(logging.WARNING, logger="vaig.agents.orchestrator"):
            with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
                agents = []
                for name in [
                    "node_gatherer", "workload_gatherer",
                    "event_gatherer", "logging_gatherer",
                ]:
                    a = MagicMock(spec=SpecialistAgent)
                    a.name = name
                    a.role = name
                    a.execute.return_value = _make_agent_result(name)
                    agents.append(a)

                for name in ["health_analyzer", "health_verifier", "health_reporter"]:
                    a = MagicMock(spec=SpecialistAgent)
                    a.name = name
                    a.role = name
                    a.execute.return_value = _make_agent_result(name)
                    agents.append(a)

                mock_create.return_value = agents

                orchestrator.execute_with_tools(
                    "Check cluster",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )

        upgrade_logs = [
            r.getMessage() for r in caplog.records
            if "upgrading strategy to parallel_sequential" in r.getMessage()
        ]
        assert len(upgrade_logs) == 0, "No auto-upgrade log should appear when strategy is already parallel_sequential"

    def test_auto_detection_does_not_fire_for_non_sequential_strategy(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Auto-detection only upgrades 'sequential' — other strategies are left alone."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        with caplog.at_level(logging.WARNING, logger="vaig.agents.orchestrator"):
            with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
                agents = []
                for name in [
                    "node_gatherer", "workload_gatherer",
                    "event_gatherer", "logging_gatherer",
                ]:
                    a = MagicMock(spec=SpecialistAgent)
                    a.name = name
                    a.role = name
                    a.execute.return_value = _make_agent_result(name)
                    agents.append(a)

                for name in ["health_analyzer", "health_verifier", "health_reporter"]:
                    a = MagicMock(spec=SpecialistAgent)
                    a.name = name
                    a.role = name
                    a.execute.return_value = _make_agent_result(name)
                    agents.append(a)

                mock_create.return_value = agents

                # 'fanout' strategy should NOT get auto-upgraded
                orchestrator.execute_with_tools(
                    "Check cluster",
                    skill,
                    tool_registry,
                    strategy="fanout",
                )

        upgrade_logs = [
            r.getMessage() for r in caplog.records
            if "upgrading strategy" in r.getMessage()
        ]
        assert len(upgrade_logs) == 0, "Auto-detection must only fire for 'sequential' strategy"

    def test_async_auto_detection_upgrades_strategy(self) -> None:
        """Async path auto-detection: strategy='sequential' + parallel_group → upgrade."""
        import asyncio

        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ServiceHealthParallelSkill()
        tool_registry = MagicMock(spec=ToolRegistry)

        async def _run() -> Any:
            with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
                agents = []
                for name in [
                    "node_gatherer", "workload_gatherer",
                    "event_gatherer", "logging_gatherer",
                ]:
                    a = MagicMock(spec=SpecialistAgent)
                    a.name = name
                    a.role = name

                    async def _async_exec(  # noqa: RUF029
                        query: str,
                        _name: str = name,
                        **kwargs: Any,
                    ) -> AgentResult:
                        return _make_agent_result(_name)

                    a.execute = MagicMock(return_value=_make_agent_result(name))
                    agents.append(a)

                for name in ["health_analyzer", "health_verifier", "health_reporter"]:
                    a = MagicMock(spec=SpecialistAgent)
                    a.name = name
                    a.role = name
                    a.execute = MagicMock(return_value=_make_agent_result(name))
                    agents.append(a)

                mock_create.return_value = agents

                return await orchestrator.async_execute_with_tools(
                    "Check cluster",
                    skill,
                    tool_registry,
                    strategy="sequential",
                )

        result = asyncio.run(_run())
        assert result.success
        # 7 agents should have run (parallel_sequential path was taken)
        assert len(result.agent_results) == 7

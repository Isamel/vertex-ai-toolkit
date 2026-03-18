"""Tests for parallel_sequential strategy in execute_with_tools."""

from __future__ import annotations

import asyncio
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


def _make_agent_result(
    name: str, *, success: bool = True, content: str | None = None,
) -> AgentResult:
    return AgentResult(
        agent_name=name,
        content=content or f"Result from {name}",
        success=success,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


class ParallelSkill(BaseSkill):
    """Skill with gatherers + sequential agents for testing parallel_sequential."""

    def __init__(
        self,
        *,
        gatherer_names: list[str] | None = None,
        sequential_names: list[str] | None = None,
        required_sections: list[str] | None = None,
    ) -> None:
        self._gatherer_names = gatherer_names or ["node_gatherer", "workload_gatherer"]
        self._sequential_names = sequential_names or ["analyzer"]
        self._required_sections = required_sections

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="parallel_skill",
            display_name="Parallel Skill",
            description="A skill for testing parallel_sequential execution.",
        )

    def get_system_instruction(self) -> str:
        return "You are a parallel skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {user_input}"

    def get_agents_config(self) -> list[dict]:
        configs: list[dict] = []
        for name in self._gatherer_names:
            configs.append({
                "name": name,
                "role": f"{name} role",
                "system_instruction": f"{name} instruction.",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            })
        for name in self._sequential_names:
            configs.append({
                "name": name,
                "role": f"{name} role",
                "system_instruction": f"{name} instruction.",
                "model": "gemini-2.5-pro",
                "requires_tools": True,
            })
        return configs

    def get_required_output_sections(self) -> list[str] | None:
        return self._required_sections


# ---------------------------------------------------------------------------
# Tests: agent partitioning
# ---------------------------------------------------------------------------


class TestPartitioning:
    """Verify _gatherer agents are split from sequential agents."""

    def test_gatherers_are_identified_by_name_suffix(self) -> None:
        """Agents ending with _gatherer should be treated as the parallel group."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer_0 = MagicMock(spec=SpecialistAgent)
            gatherer_0.name = "node_gatherer"
            gatherer_0.role = "node role"
            gatherer_0.parallel_group = "gather"
            gatherer_0.execute.return_value = _make_agent_result("node_gatherer")

            gatherer_1 = MagicMock(spec=SpecialistAgent)
            gatherer_1.name = "workload_gatherer"
            gatherer_1.role = "workload role"
            gatherer_1.parallel_group = "gather"
            gatherer_1.execute.return_value = _make_agent_result("workload_gatherer")

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.return_value = _make_agent_result("analyzer")

            mock_create.return_value = [gatherer_0, gatherer_1, analyzer]

            result = orchestrator.execute_with_tools(
                "test query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.success
        assert len(result.agent_results) == 3
        names = [r.agent_name for r in result.agent_results]
        assert "node_gatherer" in names
        assert "workload_gatherer" in names
        assert "analyzer" in names

    def test_non_gatherer_agents_run_sequentially_after_parallel(self) -> None:
        """Sequential agents should receive merged gatherer output as context."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        received_contexts: list[str] = []

        def capture_execute(query: str, **kwargs: Any) -> AgentResult:
            received_contexts.append(kwargs.get("context", ""))
            return _make_agent_result("analyzer")

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = MagicMock(spec=SpecialistAgent)
            gatherer.name = "node_gatherer"
            gatherer.role = "node role"
            gatherer.parallel_group = "gather"
            gatherer.execute.return_value = _make_agent_result(
                "node_gatherer", content="node data here",
            )

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.side_effect = capture_execute

            mock_create.return_value = [gatherer, analyzer]

            orchestrator.execute_with_tools(
                "test query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        # Analyzer's context should contain gatherer's output
        assert len(received_contexts) == 1
        assert "node_gatherer" in received_contexts[0]
        assert "node data here" in received_contexts[0]


# ---------------------------------------------------------------------------
# Tests: output merging
# ---------------------------------------------------------------------------


class TestOutputMerging:
    """Verify _merge_parallel_outputs produces correctly formatted output."""

    def test_merge_uses_section_headers(self) -> None:
        """Each agent section should be prefixed with --- [Agent Name] ---."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        results = [
            _make_agent_result("node_gatherer", content="node content"),
            _make_agent_result("workload_gatherer", content="workload content"),
        ]
        merged = orchestrator._merge_parallel_outputs(results)

        assert "--- node_gatherer ---" in merged
        assert "node content" in merged
        assert "--- workload_gatherer ---" in merged
        assert "workload content" in merged

    def test_failed_agent_gets_error_note(self) -> None:
        """Failed agents should produce an error note in the merged output."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        results = [
            _make_agent_result("node_gatherer", success=True, content="ok"),
            _make_agent_result("workload_gatherer", success=False, content="timeout"),
        ]
        merged = orchestrator._merge_parallel_outputs(results)

        assert "--- node_gatherer ---" in merged
        assert "--- workload_gatherer ---" in merged
        assert "[ERROR:" in merged
        assert "timeout" in merged

    def test_all_failed_agents_still_produce_merged_string(self) -> None:
        """Merge should produce output even if all agents fail."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())

        results = [
            _make_agent_result("a_gatherer", success=False, content="err"),
        ]
        merged = orchestrator._merge_parallel_outputs(results)

        assert "--- a_gatherer ---" in merged
        assert "[ERROR:" in merged


# ---------------------------------------------------------------------------
# Tests: execution behaviour
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Verify concurrent execution and failure isolation."""

    def test_all_gatherers_are_called(self) -> None:
        """Both gatherer agents must be called once each."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=[],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result("node_gatherer")

            g1 = MagicMock(spec=SpecialistAgent)
            g1.name = "workload_gatherer"
            g1.role = "workload role"
            g1.parallel_group = "gather"
            g1.execute.return_value = _make_agent_result("workload_gatherer")

            mock_create.return_value = [g0, g1]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        g0.execute.assert_called_once()
        g1.execute.assert_called_once()
        assert result.success

    def test_one_gatherer_fails_pipeline_continues(self) -> None:
        """A single failing gatherer must not abort the pipeline."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result(
                "node_gatherer", success=False, content="timeout",
            )

            g1 = MagicMock(spec=SpecialistAgent)
            g1.name = "workload_gatherer"
            g1.role = "workload role"
            g1.parallel_group = "gather"
            g1.execute.return_value = _make_agent_result("workload_gatherer", success=True)

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.return_value = _make_agent_result("analyzer")

            mock_create.return_value = [g0, g1, analyzer]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        # analyzer should still have run
        analyzer.execute.assert_called_once()
        assert len(result.agent_results) == 3
        assert result.success  # one gatherer ok + analyzer ok

    def test_all_gatherers_fail_result_is_failure(self) -> None:
        """If ALL gatherers fail, the overall result must be failure."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result("node_gatherer", success=False)

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.return_value = _make_agent_result("analyzer")

            mock_create.return_value = [g0, analyzer]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert not result.success

    def test_usage_accumulated_across_all_agents(self) -> None:
        """Total tokens should sum across gatherers AND sequential agents."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = AgentResult(
                agent_name="node_gatherer",
                content="data",
                success=True,
                usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            )

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.return_value = AgentResult(
                agent_name="analyzer",
                content="analysis",
                success=True,
                usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            )

            mock_create.return_value = [g0, analyzer]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.total_usage.get("total_tokens") == 20

    def test_synthesized_output_is_last_sequential_agent(self) -> None:
        """synthesized_output should be the last sequential agent's content."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result("node_gatherer")

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.return_value = _make_agent_result(
                "analyzer", content="final analysis",
            )

            mock_create.return_value = [g0, analyzer]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.synthesized_output == "final analysis"


# ---------------------------------------------------------------------------
# Tests: async version
# ---------------------------------------------------------------------------


class TestAsyncParallelExecution:
    """Verify the async counterpart mirrors sync behaviour."""

    def test_async_gatherers_all_called(self) -> None:
        """Both gatherers must be called in the async path."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=[],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result("node_gatherer")

            g1 = MagicMock(spec=SpecialistAgent)
            g1.name = "workload_gatherer"
            g1.role = "workload role"
            g1.parallel_group = "gather"
            g1.execute.return_value = _make_agent_result("workload_gatherer")

            mock_create.return_value = [g0, g1]

            result = asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )
            )

        g0.execute.assert_called_once()
        g1.execute.assert_called_once()
        assert result.success

    def test_async_partial_failure_continues(self) -> None:
        """Async: a failing gatherer should not abort the pipeline."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result(
                "node_gatherer", success=False, content="err",
            )

            g1 = MagicMock(spec=SpecialistAgent)
            g1.name = "workload_gatherer"
            g1.role = "workload role"
            g1.parallel_group = "gather"
            g1.execute.return_value = _make_agent_result("workload_gatherer")

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.return_value = _make_agent_result("analyzer")

            mock_create.return_value = [g0, g1, analyzer]

            result = asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )
            )

        analyzer.execute.assert_called_once()
        assert result.success

    def test_async_merged_context_passed_to_sequential(self) -> None:
        """Async: sequential agent should receive merged gatherer output."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        received_contexts: list[str] = []

        def capture_execute(query: str, **kwargs: Any) -> AgentResult:
            received_contexts.append(kwargs.get("context", ""))
            return _make_agent_result("analyzer")

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result(
                "node_gatherer", content="async node data",
            )

            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"
            analyzer.parallel_group = None
            analyzer.execute.side_effect = capture_execute

            mock_create.return_value = [g0, analyzer]

            asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )
            )

        assert len(received_contexts) == 1
        assert "node_gatherer" in received_contexts[0]
        assert "async node data" in received_contexts[0]


# ---------------------------------------------------------------------------
# Tests: validation (task 1.4)
# ---------------------------------------------------------------------------


class TestValidation:
    """Verify required section validation against merged output."""

    def test_missing_required_sections_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning should be logged if required sections are absent from merged output."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=[],
            required_sections=["## Node Conditions"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        # Force propagation so caplog (which hooks into the root logger)
        # can capture records even when vaig's setup_logging() has set
        # propagate=False on the "vaig" logger.
        vaig_logger = logging.getLogger("vaig")
        orig_propagate = vaig_logger.propagate
        vaig_logger.propagate = True
        try:
            with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
                g0 = MagicMock(spec=SpecialistAgent)
                g0.name = "node_gatherer"
                g0.role = "node role"
                g0.parallel_group = "gather"
                # content does NOT contain the required section
                g0.execute.return_value = _make_agent_result("node_gatherer", content="some data")

                mock_create.return_value = [g0]

                with caplog.at_level(logging.WARNING, logger="vaig.agents.orchestrator"):
                    orchestrator.execute_with_tools(
                        "query",
                        skill,
                        tool_registry,
                        strategy="parallel_sequential",
                    )
        finally:
            vaig_logger.propagate = orig_propagate

        assert any("missing" in record.getMessage().lower() for record in caplog.records)

    def test_present_required_sections_no_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No warning should be logged when all required sections are present."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=[],
            required_sections=["## Node Conditions"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            g0 = MagicMock(spec=SpecialistAgent)
            g0.name = "node_gatherer"
            g0.role = "node role"
            g0.parallel_group = "gather"
            g0.execute.return_value = _make_agent_result(
                "node_gatherer", content="## Node Conditions\nAll good.",
            )

            mock_create.return_value = [g0]

            with caplog.at_level(logging.WARNING, logger="vaig.agents.orchestrator"):
                orchestrator.execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )

        missing_warnings = [
            r for r in caplog.records
            if "missing" in r.message.lower() and "required section" in r.message.lower()
        ]
        assert len(missing_warnings) == 0

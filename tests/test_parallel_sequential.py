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
    settings.budget.max_cost_per_run = 0.0
    settings.agents.max_failures_before_fallback = 0
    return settings


def _make_agent_result(
    name: str,
    *,
    success: bool = True,
    content: str | None = None,
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

    def get_agents_config(self, **kwargs: Any) -> list[dict]:
        configs: list[dict] = []
        for name in self._gatherer_names:
            configs.append(
                {
                    "name": name,
                    "role": f"{name} role",
                    "system_instruction": f"{name} instruction.",
                    "model": "gemini-2.5-pro",
                    "requires_tools": True,
                }
            )
        for name in self._sequential_names:
            configs.append(
                {
                    "name": name,
                    "role": f"{name} role",
                    "system_instruction": f"{name} instruction.",
                    "model": "gemini-2.5-pro",
                    "requires_tools": True,
                }
            )
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
                "node_gatherer",
                content="node data here",
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
                "node_gatherer",
                success=False,
                content="timeout",
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
                "analyzer",
                content="final analysis",
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
                "node_gatherer",
                success=False,
                content="err",
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
                "node_gatherer",
                content="async node data",
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
        self,
        caplog: pytest.LogCaptureFixture,
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
        self,
        caplog: pytest.LogCaptureFixture,
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
                "node_gatherer",
                content="## Node Conditions\nAll good.",
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
            r for r in caplog.records if "missing" in r.message.lower() and "required section" in r.message.lower()
        ]
        assert len(missing_warnings) == 0


# ---------------------------------------------------------------------------
# Tests: structured_report population (bug fix — parallel_sequential path)
# ---------------------------------------------------------------------------


class _FakeSchema:
    """Minimal Pydantic-like schema stub for testing model_validate_json."""

    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def model_validate_json(cls, content: str) -> _FakeSchema:
        import json

        return cls(json.loads(content))


class TestStructuredReportPopulation:
    """Verify structured_report is populated when the last sequential agent is a reporter.

    Regression test for the bug where _execute_parallel_then_sequential() never
    called skill.post_process_report() or set result.structured_report, causing
    --format html to always fall back to raw markdown output.
    """

    def _make_reporter_agent(
        self,
        *,
        name: str = "health_reporter",
        content: str,
    ) -> MagicMock:
        agent = MagicMock(spec=SpecialistAgent)
        agent.name = name
        agent.role = "Health Report Generator"  # contains "report"
        agent.parallel_group = None
        agent.model = "gemini-2.5-pro"
        # Simulate config.response_schema = _FakeSchema
        cfg = MagicMock()
        cfg.response_schema = _FakeSchema
        agent.config = cfg
        agent.execute.return_value = _make_agent_result(name, content=content)
        return agent

    def _make_gatherer_agent(self, *, name: str) -> MagicMock:
        agent = MagicMock(spec=SpecialistAgent)
        agent.name = name
        agent.role = f"{name} role"
        agent.parallel_group = "gather"
        agent.model = "gemini-2.5-pro"
        agent.config = MagicMock()
        agent.config.response_schema = None
        agent.execute.return_value = _make_agent_result(name, content=f"{name} data")
        return agent

    def test_structured_report_populated_from_reporter_json(self) -> None:
        """structured_report must be set when reporter agent has response_schema."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["health_reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        reporter_json = '{"status": "healthy", "services": []}'

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = self._make_gatherer_agent(name="node_gatherer")
            reporter = self._make_reporter_agent(content=reporter_json)
            mock_create.return_value = [gatherer, reporter]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.structured_report is not None, (
            "structured_report must be populated when reporter agent has response_schema"
        )
        assert isinstance(result.structured_report, _FakeSchema)
        assert result.structured_report.data == {"status": "healthy", "services": []}

    def test_structured_report_none_when_reporter_json_invalid(self) -> None:
        """structured_report stays None (best-effort) when reporter output is invalid JSON."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["health_reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = self._make_gatherer_agent(name="node_gatherer")
            reporter = self._make_reporter_agent(content="not valid json at all")
            mock_create.return_value = [gatherer, reporter]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        # Must not crash — best-effort means structured_report stays None
        assert result.structured_report is None

    def test_post_process_report_called_on_reporter_content(self) -> None:
        """skill.post_process_report must be called on reporter's output."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["health_reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        reporter_json = '{"status": "ok"}'

        post_process_calls: list[str] = []

        def _fake_post_process(content: str, **kwargs: object) -> str:
            post_process_calls.append(content)
            return "## Processed Markdown"

        skill.post_process_report = _fake_post_process  # type: ignore[method-assign]

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = self._make_gatherer_agent(name="node_gatherer")
            reporter = self._make_reporter_agent(content=reporter_json)
            mock_create.return_value = [gatherer, reporter]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert post_process_calls == [reporter_json], (
            "post_process_report must be called with the raw reporter JSON output"
        )
        assert result.synthesized_output == "## Processed Markdown", (
            "synthesized_output must reflect the post-processed content"
        )

    def test_structured_report_not_populated_for_non_reporter(self) -> None:
        """structured_report stays None when last agent is NOT a reporter."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["analyzer"],  # role is "analyzer role" — no "report"
        )
        tool_registry = MagicMock(spec=ToolRegistry)

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = self._make_gatherer_agent(name="node_gatherer")
            analyzer = MagicMock(spec=SpecialistAgent)
            analyzer.name = "analyzer"
            analyzer.role = "analyzer role"  # does NOT contain "report"
            analyzer.parallel_group = None
            analyzer.model = "gemini-2.5-pro"
            cfg = MagicMock()
            cfg.response_schema = _FakeSchema
            analyzer.config = cfg
            analyzer.execute.return_value = _make_agent_result("analyzer", content='{"x": 1}')
            mock_create.return_value = [gatherer, analyzer]

            result = orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
            )

        assert result.structured_report is None, (
            "structured_report must NOT be set for agents without 'report' in their role"
        )

    def test_async_structured_report_populated_from_reporter_json(self) -> None:
        """Async path: structured_report must be set when reporter has response_schema."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["health_reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        reporter_json = '{"status": "healthy", "services": []}'

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = self._make_gatherer_agent(name="node_gatherer")
            reporter = self._make_reporter_agent(content=reporter_json)
            mock_create.return_value = [gatherer, reporter]

            result = asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )
            )

        assert result.structured_report is not None, (
            "Async path: structured_report must be populated when reporter has response_schema"
        )
        assert isinstance(result.structured_report, _FakeSchema)

    def test_async_post_process_report_called_on_reporter_content(self) -> None:
        """Async path: skill.post_process_report must be called on reporter output."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer"],
            sequential_names=["health_reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        reporter_json = '{"status": "ok"}'

        post_process_calls: list[str] = []

        def _fake_post_process(content: str, **kwargs: object) -> str:
            post_process_calls.append(content)
            return "## Async Processed Markdown"

        skill.post_process_report = _fake_post_process  # type: ignore[method-assign]

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            gatherer = self._make_gatherer_agent(name="node_gatherer")
            reporter = self._make_reporter_agent(content=reporter_json)
            mock_create.return_value = [gatherer, reporter]

            result = asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                )
            )

        assert post_process_calls == [reporter_json]
        assert result.synthesized_output == "## Async Processed Markdown"


class TestProgressCounterConsistency:
    """Validate that on_agent_progress callbacks use the combined total (parallel +
    sequential) as the denominator in BOTH the parallel and sequential phases.

    Before the fix, the parallel phase used `total_parallel` (e.g. 4) while the
    sequential phase used `total_agents` (e.g. 7), making the counter jump from
    [4/4] back to [5/7].  After the fix, both phases must use `total_agents`.
    """

    def _make_gatherer(self, name: str) -> MagicMock:
        agent = MagicMock(spec=SpecialistAgent)
        agent.name = name
        agent.role = f"{name} role"
        agent.parallel_group = "gather"
        agent.execute.return_value = _make_agent_result(name)
        return agent

    def _make_sequential(self, name: str) -> MagicMock:
        agent = MagicMock(spec=SpecialistAgent)
        agent.name = name
        agent.role = f"{name} role"
        agent.parallel_group = None
        agent.execute.return_value = _make_agent_result(name)
        return agent

    def test_sync_parallel_phase_uses_combined_total(self) -> None:
        """Parallel phase on_agent_progress callbacks must report the combined
        (parallel + sequential) total, not just the parallel count."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        # 2 parallel gatherers + 1 sequential analyzer = 3 total
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        progress_calls: list[tuple[str, int, int, str]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,  # noqa: ARG001
        ) -> None:
            progress_calls.append((agent_name, agent_index, total_agents, event))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_sequential("analyzer"),
            ]
            orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
                on_agent_progress=_capture,
            )

        # All callbacks must use total=3 (2 parallel + 1 sequential), never total=2
        totals_seen = {total for (_name, _idx, total, _event) in progress_calls}
        assert totals_seen == {3}, (
            f"Expected all progress callbacks to use total=3 (combined count), "
            f"but saw totals: {totals_seen}. "
            "Parallel phase must not use total_parallel=2 as the denominator."
        )

    def test_sync_no_denominator_reset_between_phases(self) -> None:
        """The total denominator must never change between the parallel and
        sequential phases — there must be no [N/parallel_count] then [M/total] jump."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        # 4 parallel gatherers + 3 sequential = 7 total (realistic pipeline shape)
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer", "event_gatherer", "log_gatherer"],
            sequential_names=["analyzer", "verifier", "reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        progress_calls: list[tuple[str, int, int, str]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,  # noqa: ARG001
        ) -> None:
            progress_calls.append((agent_name, agent_index, total_agents, event))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_gatherer("event_gatherer"),
                self._make_gatherer("log_gatherer"),
                self._make_sequential("analyzer"),
                self._make_sequential("verifier"),
                self._make_sequential("reporter"),
            ]
            orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
                on_agent_progress=_capture,
            )

        totals_seen = {total for (_name, _idx, total, _event) in progress_calls}
        assert len(totals_seen) == 1, (
            f"Progress denominator changed between phases! Saw totals: {totals_seen}. "
            "All callbacks must use the same combined total=7."
        )
        assert 7 in totals_seen, (
            f"Expected combined total=7 in progress callbacks, got: {totals_seen}"
        )

    def test_async_parallel_phase_uses_combined_total(self) -> None:
        """Async parallel phase on_agent_progress callbacks must also use the
        combined total as the denominator."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        # 2 parallel + 1 sequential = 3 total
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        progress_calls: list[tuple[str, int, int, str]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,  # noqa: ARG001
        ) -> None:
            progress_calls.append((agent_name, agent_index, total_agents, event))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_sequential("analyzer"),
            ]
            asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                    on_agent_progress=_capture,
                )
            )

        totals_seen = {total for (_name, _idx, total, _event) in progress_calls}
        assert totals_seen == {3}, (
            f"Async path: expected all progress callbacks to use total=3 (combined), "
            f"but saw totals: {totals_seen}."
        )

    def test_sync_parallel_phase_fires_single_collective_start(self) -> None:
        """Parallel phase must fire exactly ONE 'start' event for the whole group,
        not one per gatherer agent. Individual per-agent start events would all fire
        synchronously before any thread executes, causing the spinner to jump to
        [N/total] and get stuck.  The collective event uses agent_name='parallel
        gatherers' at idx=0."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer", "event_gatherer", "log_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        start_events: list[tuple[str, int, int]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,  # noqa: ARG001
        ) -> None:
            if event == "start":
                start_events.append((agent_name, agent_index, total_agents))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_gatherer("event_gatherer"),
                self._make_gatherer("log_gatherer"),
                self._make_sequential("analyzer"),
            ]
            orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
                on_agent_progress=_capture,
            )

        # Parallel phase: exactly one collective start event
        parallel_start_events = [e for e in start_events if e[0] == "parallel gatherers"]
        assert len(parallel_start_events) == 1, (
            f"Expected exactly 1 collective 'parallel gatherers' start event, "
            f"got {len(parallel_start_events)}: {parallel_start_events}"
        )
        # The collective event must be at idx=0 with the combined total
        name, idx, total = parallel_start_events[0]
        assert idx == 0, f"Collective start event must be at idx=0, got idx={idx}"
        assert total == 5, f"Collective start total must be 5 (4+1), got {total}"

        # No individual per-gatherer start events
        individual_gatherer_starts = [
            e for e in start_events
            if e[0] in {"node_gatherer", "workload_gatherer", "event_gatherer", "log_gatherer"}
        ]
        assert individual_gatherer_starts == [], (
            f"Expected no individual per-gatherer start events, got: {individual_gatherer_starts}"
        )

        # Sequential phase still fires its own start event
        sequential_starts = [e for e in start_events if e[0] == "analyzer"]
        assert len(sequential_starts) == 1, (
            f"Sequential agent must still fire its own start event, got: {sequential_starts}"
        )

    def test_async_parallel_phase_fires_single_collective_start(self) -> None:
        """Async path: parallel phase must fire exactly ONE collective start event."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer", "event_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        start_events: list[tuple[str, int, int]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,  # noqa: ARG001
        ) -> None:
            if event == "start":
                start_events.append((agent_name, agent_index, total_agents))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_gatherer("event_gatherer"),
                self._make_sequential("analyzer"),
            ]
            asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                    on_agent_progress=_capture,
                )
            )

        # Parallel phase: exactly one collective start event
        parallel_start_events = [e for e in start_events if e[0] == "parallel gatherers"]
        assert len(parallel_start_events) == 1, (
            f"Async: expected exactly 1 collective 'parallel gatherers' start event, "
            f"got {len(parallel_start_events)}: {parallel_start_events}"
        )

        # No individual per-gatherer start events
        individual_gatherer_starts = [
            e for e in start_events
            if e[0] in {"node_gatherer", "workload_gatherer", "event_gatherer"}
        ]
        assert individual_gatherer_starts == [], (
            f"Async: expected no individual per-gatherer start events, got: {individual_gatherer_starts}"
        )

    def test_sync_parallel_collective_event_includes_end_agent_index(self) -> None:
        """Parallel collective start/end events must include end_agent_index=total_parallel-1
        so the CLI can display '[1-4/7]' range notation instead of '[1/7]'."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        # 4 parallel gatherers + 3 sequential = 7 total
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer", "event_gatherer", "log_gatherer"],
            sequential_names=["analyzer", "verifier", "reporter"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        parallel_events: list[tuple[str, int, int, str, int | None]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,
        ) -> None:
            if agent_name == "parallel gatherers":
                parallel_events.append((agent_name, agent_index, total_agents, event, end_agent_index))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_gatherer("event_gatherer"),
                self._make_gatherer("log_gatherer"),
                self._make_sequential("analyzer"),
                self._make_sequential("verifier"),
                self._make_sequential("reporter"),
            ]
            orchestrator.execute_with_tools(
                "query",
                skill,
                tool_registry,
                strategy="parallel_sequential",
                on_agent_progress=_capture,
            )

        assert len(parallel_events) == 2, (  # start + end
            f"Expected 2 parallel collective events (start + end), got: {parallel_events}"
        )

        for _name, idx, total, _event, end_idx in parallel_events:
            assert idx == 0, f"Collective parallel event agent_index must be 0, got {idx}"
            assert total == 7, f"Collective parallel event total must be 7, got {total}"
            assert end_idx == 3, (
                f"Collective parallel event end_agent_index must be 3 (total_parallel-1=4-1), "
                f"got {end_idx}"
            )

    def test_async_parallel_collective_event_includes_end_agent_index(self) -> None:
        """Async path: parallel collective events must also include end_agent_index."""
        client = _make_mock_client()
        orchestrator = Orchestrator(client, _make_mock_settings())
        # 3 parallel gatherers + 1 sequential = 4 total
        skill = ParallelSkill(
            gatherer_names=["node_gatherer", "workload_gatherer", "event_gatherer"],
            sequential_names=["analyzer"],
        )
        tool_registry = MagicMock(spec=ToolRegistry)
        parallel_events: list[tuple[str, int, int, str, int | None]] = []

        def _capture(
            agent_name: str,
            agent_index: int,
            total_agents: int,
            event: str,
            end_agent_index: int | None = None,
        ) -> None:
            if agent_name == "parallel gatherers":
                parallel_events.append((agent_name, agent_index, total_agents, event, end_agent_index))

        with patch.object(orchestrator, "create_agents_for_skill") as mock_create:
            mock_create.return_value = [
                self._make_gatherer("node_gatherer"),
                self._make_gatherer("workload_gatherer"),
                self._make_gatherer("event_gatherer"),
                self._make_sequential("analyzer"),
            ]
            asyncio.run(
                orchestrator.async_execute_with_tools(
                    "query",
                    skill,
                    tool_registry,
                    strategy="parallel_sequential",
                    on_agent_progress=_capture,
                )
            )

        assert len(parallel_events) == 2, (  # start + end
            f"Async: expected 2 parallel collective events (start + end), got: {parallel_events}"
        )

        for _name, idx, total, _event, end_idx in parallel_events:
            assert idx == 0, f"Async: collective parallel event agent_index must be 0, got {idx}"
            assert total == 4, f"Async: collective parallel event total must be 4, got {total}"
            assert end_idx == 2, (
                f"Async: collective parallel event end_agent_index must be 2 (total_parallel-1=3-1), "
                f"got {end_idx}"
            )


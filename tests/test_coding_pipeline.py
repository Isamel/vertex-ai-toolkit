"""Tests for CodingSkillOrchestrator and CodingPipelineResult."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.agents.base import AgentResult
from vaig.agents.coding_pipeline import (
    _IMPLEMENTER_SYSTEM_PROMPT,
    _PLANNER_SYSTEM_PROMPT,
    _VERIFIER_SYSTEM_PROMPT,
    CodingPipelineResult,
    CodingSkillOrchestrator,
)
from vaig.core.config import CodingConfig

# ── Helpers ───────────────────────────────────────────────────


def _make_mock_client(current_model: str = "gemini-2.5-pro") -> MagicMock:
    """Create a MagicMock that behaves like GeminiClientProtocol."""
    client = MagicMock()
    client.current_model = current_model
    return client


def _make_coding_config(
    workspace_root: str = "/tmp/test-workspace",
    max_tool_iterations: int = 5,
    pipeline_mode: bool = False,
) -> CodingConfig:
    return CodingConfig(
        workspace_root=workspace_root,
        max_tool_iterations=max_tool_iterations,
        pipeline_mode=pipeline_mode,
    )


def _make_agent_result(
    content: str = "Agent output",
    success: bool = True,
    usage: dict[str, int] | None = None,
) -> AgentResult:
    """Create a minimal AgentResult for mock returns."""
    return AgentResult(
        success=success,
        content=content,
        agent_name="mock-agent",
        usage=usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        metadata={"iterations": 1, "tools_executed": []},
    )


# ===========================================================================
# TestCodingPipelineResult
# ===========================================================================


class TestCodingPipelineResult:
    """Tests for CodingPipelineResult dataclass."""

    def test_required_fields(self) -> None:
        """CodingPipelineResult can be constructed with required fields."""
        result = CodingPipelineResult(
            task="Add retry logic",
            plan="Step 1: ...",
            implementation_summary="Files written: x.py",
            verification_report="PASS ✅",
            success=True,
        )
        assert result.task == "Add retry logic"
        assert result.plan == "Step 1: ..."
        assert result.implementation_summary == "Files written: x.py"
        assert result.verification_report == "PASS ✅"
        assert result.success is True

    def test_default_usage_is_empty_dict(self) -> None:
        result = CodingPipelineResult(
            task="task",
            plan="plan",
            implementation_summary="impl",
            verification_report="report",
            success=True,
        )
        assert result.usage == {}

    def test_default_metadata_is_empty_dict(self) -> None:
        result = CodingPipelineResult(
            task="task",
            plan="plan",
            implementation_summary="impl",
            verification_report="report",
            success=True,
        )
        assert result.metadata == {}

    def test_custom_usage(self) -> None:
        usage = {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        result = CodingPipelineResult(
            task="t",
            plan="p",
            implementation_summary="i",
            verification_report="v",
            success=False,
            usage=usage,
        )
        assert result.usage == usage

    def test_success_false(self) -> None:
        result = CodingPipelineResult(
            task="t",
            plan="p",
            implementation_summary="i",
            verification_report="FAIL ❌",
            success=False,
        )
        assert result.success is False

    def test_is_dataclass(self) -> None:
        """CodingPipelineResult is a proper dataclass."""
        field_names = {f.name for f in fields(CodingPipelineResult)}
        assert "task" in field_names
        assert "plan" in field_names
        assert "implementation_summary" in field_names
        assert "verification_report" in field_names
        assert "success" in field_names
        assert "usage" in field_names
        assert "metadata" in field_names


# ===========================================================================
# TestCodingSkillOrchestratorInit
# ===========================================================================


class TestCodingSkillOrchestratorInit:
    """Tests for CodingSkillOrchestrator.__init__()."""

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    def test_basic_init(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        client = _make_mock_client()
        config = _make_coding_config()

        orchestrator = CodingSkillOrchestrator(client, config)

        assert orchestrator._client is client
        assert orchestrator._workspace == Path("/tmp/test-workspace").resolve()
        assert orchestrator._max_iterations == 5

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    def test_default_model_from_client(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Default models derive from client.current_model."""
        client = _make_mock_client(current_model="gemini-2.5-flash")
        config = _make_coding_config()

        orchestrator = CodingSkillOrchestrator(client, config)

        assert orchestrator._planner_model == "gemini-2.5-flash"
        assert orchestrator._implementer_model == "gemini-2.5-flash"

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    def test_model_overrides(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Explicit model overrides are respected."""
        client = _make_mock_client()
        config = _make_coding_config()

        orchestrator = CodingSkillOrchestrator(
            client, config,
            planner_model="gemini-2.5-pro",
            implementer_model="gemini-2.5-flash",
            verifier_model="gemini-2.5-flash",
        )

        assert orchestrator._planner_model == "gemini-2.5-pro"
        assert orchestrator._implementer_model == "gemini-2.5-flash"
        assert orchestrator._verifier_model == "gemini-2.5-flash"

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    def test_verifier_model_falls_back_to_client_model_when_no_settings(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Without settings, verifier defaults to client.current_model."""
        client = _make_mock_client(current_model="gemini-2.5-pro")
        config = _make_coding_config()

        orchestrator = CodingSkillOrchestrator(client, config, settings=None)

        assert orchestrator._verifier_model == "gemini-2.5-pro"

    @patch("vaig.agents.coding_pipeline.create_shell_tools")
    @patch("vaig.agents.coding_pipeline.create_file_tools")
    def test_file_tools_created_with_workspace(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """create_file_tools is called with the resolved workspace path."""
        mock_file_tools.return_value = []
        mock_shell_tools.return_value = []
        client = _make_mock_client()
        config = _make_coding_config(workspace_root="/tmp/ws")

        CodingSkillOrchestrator(client, config)

        mock_file_tools.assert_called_once_with(Path("/tmp/ws").resolve())

    @patch("vaig.agents.coding_pipeline.create_shell_tools")
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    def test_shell_tools_created_with_denied_commands(
        self,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """create_shell_tools is called with the denied_commands list."""
        mock_shell_tools.return_value = []
        client = _make_mock_client()
        config = _make_coding_config()

        CodingSkillOrchestrator(client, config)

        call_kwargs = mock_shell_tools.call_args[1]
        assert call_kwargs["denied_commands"] is not None
        assert len(call_kwargs["denied_commands"]) > 0


# ===========================================================================
# TestCodingSkillOrchestratorRun
# ===========================================================================


class TestCodingSkillOrchestratorRun:
    """Tests for CodingSkillOrchestrator.run()."""

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.ToolAwareAgent")
    def test_run_returns_pipeline_result(
        self,
        mock_agent_cls: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """run() returns a CodingPipelineResult with all fields populated."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        mock_agent.execute.side_effect = [
            _make_agent_result("PLAN.md written"),
            _make_agent_result("Files implemented"),
            _make_agent_result("PASS ✅"),
        ]

        client = _make_mock_client()
        config = _make_coding_config()
        orchestrator = CodingSkillOrchestrator(client, config)

        result = orchestrator.run("Add retry logic")

        assert isinstance(result, CodingPipelineResult)
        assert result.task == "Add retry logic"
        assert result.plan == "PLAN.md written"
        assert result.implementation_summary == "Files implemented"
        assert result.verification_report == "PASS ✅"
        assert result.success is True

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.ToolAwareAgent")
    def test_run_three_agents_created(
        self,
        mock_agent_cls: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """run() creates exactly 3 ToolAwareAgent instances."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.execute.return_value = _make_agent_result("ok")

        client = _make_mock_client()
        config = _make_coding_config()
        orchestrator = CodingSkillOrchestrator(client, config)
        orchestrator.run("Task")

        assert mock_agent_cls.call_count == 3

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.ToolAwareAgent")
    def test_run_fail_when_verifier_reports_fail(
        self,
        mock_agent_cls: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """success=False when verification report contains FAIL."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.execute.side_effect = [
            _make_agent_result("plan content"),
            _make_agent_result("impl content"),
            _make_agent_result("FAIL ❌ — missing tests"),
        ]

        client = _make_mock_client()
        config = _make_coding_config()
        orchestrator = CodingSkillOrchestrator(client, config)

        result = orchestrator.run("Fix bug")

        assert result.success is False

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.ToolAwareAgent")
    def test_run_aggregates_usage(
        self,
        mock_agent_cls: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """Token usage is summed across all 3 agents."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.execute.side_effect = [
            _make_agent_result("plan", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}),
            _make_agent_result("impl", usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}),
            _make_agent_result("verify", usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}),
        ]

        client = _make_mock_client()
        config = _make_coding_config()
        orchestrator = CodingSkillOrchestrator(client, config)

        result = orchestrator.run("Task")

        assert result.usage["prompt_tokens"] == 30
        assert result.usage["completion_tokens"] == 55
        assert result.usage["total_tokens"] == 85

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.ToolAwareAgent")
    def test_run_metadata_has_three_agent_entries(
        self,
        mock_agent_cls: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """result.metadata contains planner, implementer, and verifier entries."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.execute.return_value = _make_agent_result("ok")

        orchestrator = CodingSkillOrchestrator(_make_mock_client(), _make_coding_config())
        result = orchestrator.run("Task")

        assert "planner" in result.metadata
        assert "implementer" in result.metadata
        assert "verifier" in result.metadata

    @patch("vaig.agents.coding_pipeline.create_shell_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.create_file_tools", return_value=[])
    @patch("vaig.agents.coding_pipeline.ToolAwareAgent")
    def test_run_with_context_is_included(
        self,
        mock_agent_cls: MagicMock,
        mock_file_tools: MagicMock,
        mock_shell_tools: MagicMock,
    ) -> None:
        """When context is provided, the planner prompt includes it."""
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.execute.return_value = _make_agent_result("ok")

        orchestrator = CodingSkillOrchestrator(_make_mock_client(), _make_coding_config())
        orchestrator.run("Task", context="Some existing code here")

        first_call = mock_agent.execute.call_args_list[0]
        prompt_arg = first_call[0][0]
        assert "Some existing code here" in prompt_arg


# ===========================================================================
# TestParseSuccess
# ===========================================================================


class TestParseSuccess:
    """Tests for CodingSkillOrchestrator._parse_success()."""

    def test_pass_emoji_is_success(self) -> None:
        assert CodingSkillOrchestrator._parse_success("All files verified. PASS ✅") is True

    def test_fail_emoji_is_failure(self) -> None:
        assert CodingSkillOrchestrator._parse_success("Missing tests. FAIL ❌") is False

    def test_pass_word_is_success(self) -> None:
        assert CodingSkillOrchestrator._parse_success("Overall result: pass") is True

    def test_fail_word_is_failure(self) -> None:
        assert CodingSkillOrchestrator._parse_success("Overall result: fail") is False

    def test_fail_overrides_pass(self) -> None:
        """When report contains both FAIL and PASS, FAIL wins."""
        assert CodingSkillOrchestrator._parse_success("Some passed, some failed") is False

    def test_empty_report_defaults_to_success(self) -> None:
        """No verdict → optimistic default (True)."""
        assert CodingSkillOrchestrator._parse_success("") is True

    def test_ambiguous_report_defaults_to_success(self) -> None:
        assert CodingSkillOrchestrator._parse_success("All files written.") is True


# ===========================================================================
# TestAggregateUsage
# ===========================================================================


class TestAggregateUsage:
    """Tests for CodingSkillOrchestrator._aggregate_usage()."""

    def test_sums_three_dicts(self) -> None:
        a = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        b = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        c = {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}

        result = CodingSkillOrchestrator._aggregate_usage(a, b, c)

        assert result["prompt_tokens"] == 60
        assert result["completion_tokens"] == 30
        assert result["total_tokens"] == 90

    def test_missing_keys_default_to_zero(self) -> None:
        a: dict[str, int] = {}
        b: dict[str, int] = {}

        result = CodingSkillOrchestrator._aggregate_usage(a, b)

        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_single_dict(self) -> None:
        usage = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        result = CodingSkillOrchestrator._aggregate_usage(usage)
        assert result == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}


# ===========================================================================
# TestWrapAgentOutput
# ===========================================================================


class TestWrapAgentOutput:
    """Tests for CodingSkillOrchestrator._wrap_agent_output()."""

    def test_wraps_in_xml_tags(self) -> None:
        result = CodingSkillOrchestrator._wrap_agent_output(label="PLANNER_OUTPUT", content="plan here")
        assert result.startswith("<planner_output>")
        assert result.endswith("</planner_output>")
        assert "plan here" in result

    def test_escapes_ampersands(self) -> None:
        result = CodingSkillOrchestrator._wrap_agent_output(label="X", content="a & b")
        assert "&amp;" in result
        assert "a & b" not in result

    def test_escapes_less_than(self) -> None:
        result = CodingSkillOrchestrator._wrap_agent_output(label="X", content="a < b")
        assert "&lt;" in result

    def test_label_lowercased(self) -> None:
        result = CodingSkillOrchestrator._wrap_agent_output(label="SOME_LABEL", content="c")
        assert "<some_label>" in result


# ===========================================================================
# TestSystemPromptContents
# ===========================================================================


class TestSystemPromptContents:
    """Tests for pipeline system prompt constants."""

    def test_planner_mentions_plan_md(self) -> None:
        assert "PLAN.md" in _PLANNER_SYSTEM_PROMPT

    def test_implementer_mentions_zero_placeholders(self) -> None:
        assert "placeholder" in _IMPLEMENTER_SYSTEM_PROMPT.lower()

    def test_verifier_mentions_verification_report(self) -> None:
        assert "Verification Report" in _VERIFIER_SYSTEM_PROMPT

    def test_all_prompts_have_cot_instruction(self) -> None:
        """All three prompts include COT instruction from prompt_defense."""
        for prompt in (_PLANNER_SYSTEM_PROMPT, _IMPLEMENTER_SYSTEM_PROMPT, _VERIFIER_SYSTEM_PROMPT):
            assert "Chain-of-Thought" in prompt or "chain" in prompt.lower()

    def test_verifier_mentions_verify_completeness(self) -> None:
        assert "verify_completeness" in _VERIFIER_SYSTEM_PROMPT

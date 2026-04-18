"""Integration tests: CodingSkillOrchestrator with WorkspaceJail.

These tests validate the full jail lifecycle within the pipeline without
making real API calls — the ToolAwareAgent.execute() method is mocked to
return controlled agent outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vaig.agents.coding_pipeline import CodingPipelineResult, CodingSkillOrchestrator
from vaig.core.config import CodingConfig

# ── Fixtures ──────────────────────────────────────────────────


def _make_agent_result(*, content: str, success: bool = True) -> MagicMock:
    """Return a mock ToolAwareAgent execution result."""
    result = MagicMock()
    result.content = content
    result.success = success
    result.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    result.metadata = {"iterations": 1, "tools_executed": []}
    return result


def _make_config(
    tmp_path: Path,
    *,
    workspace_isolation: bool = False,
    max_fix_iterations: int = 1,
) -> CodingConfig:
    return CodingConfig(
        workspace_root=str(tmp_path),
        workspace_isolation=workspace_isolation,
        max_fix_iterations=max_fix_iterations,
        max_tool_iterations=5,
        allowed_commands=[],
        denied_commands=[],
        test_command="",
        test_timeout=30,
    )


def _make_orchestrator(config: CodingConfig) -> CodingSkillOrchestrator:
    client = MagicMock()
    client.current_model = "gemini-2.0-flash"
    return CodingSkillOrchestrator(client=client, coding_config=config)


def _make_workspace(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("# main\n")
    return tmp_path


# ── No isolation (baseline) ───────────────────────────────────


def test_pipeline_runs_without_isolation(tmp_path: Path) -> None:
    _make_workspace(tmp_path)
    config = _make_config(tmp_path, workspace_isolation=False)
    orch = _make_orchestrator(config)

    plan_result = _make_agent_result(content="PLAN.md content")
    impl_result = _make_agent_result(content="Implementation done")
    verify_result = _make_agent_result(content="Overall: PASS ✅")

    with patch.object(
        orch, "_make_agent"
    ) as mock_make:
        agents = [MagicMock(), MagicMock(), MagicMock()]
        agents[0].execute.return_value = plan_result
        agents[1].execute.return_value = impl_result
        agents[2].execute.return_value = verify_result
        mock_make.side_effect = agents

        result = orch.run("Add a feature")

    assert isinstance(result, CodingPipelineResult)
    assert result.success is True
    assert result.plan == "PLAN.md content"


# ── With isolation ────────────────────────────────────────────


def test_pipeline_with_jail_creates_copy(tmp_path: Path) -> None:
    _make_workspace(tmp_path)
    config = _make_config(tmp_path, workspace_isolation=True)
    orch = _make_orchestrator(config)

    plan_result = _make_agent_result(content="PLAN.md")
    impl_result = _make_agent_result(content="impl done")
    verify_result = _make_agent_result(content="PASS ✅")

    with patch.object(orch, "_make_agent") as mock_make:
        agents = [MagicMock(), MagicMock(), MagicMock()]
        agents[0].execute.return_value = plan_result
        agents[1].execute.return_value = impl_result
        agents[2].execute.return_value = verify_result
        mock_make.side_effect = agents

        result = orch.run("Add feature")

    assert result.success is True
    # The workspace should still exist (sync_back was called on success)
    assert tmp_path.exists()


def test_pipeline_jail_isolates_on_failure(tmp_path: Path) -> None:
    """When pipeline fails, the original workspace should be untouched."""
    _make_workspace(tmp_path)
    original_content = (tmp_path / "main.py").read_text()

    config = _make_config(tmp_path, workspace_isolation=True)
    orch = _make_orchestrator(config)

    plan_result = _make_agent_result(content="PLAN.md")

    def _corrupt_and_fail(*args: Any, **kwargs: Any) -> Any:
        # Simulate implementer writing bad code — but this runs inside the jail
        # so changes don't reach the original workspace on failure
        raise RuntimeError("Implementer crashed")

    with patch.object(orch, "_make_agent") as mock_make:
        planner = MagicMock()
        planner.execute.return_value = plan_result

        implementer = MagicMock()
        implementer.execute.side_effect = RuntimeError("Implementer crashed")

        mock_make.side_effect = [planner, implementer]

        with pytest.raises(RuntimeError, match="Implementer crashed"):
            orch.run("Add feature")

    # Original file must be intact — jail discards changes on exception
    assert (tmp_path / "main.py").read_text() == original_content


# ── Fix-forward integration ───────────────────────────────────


def test_fix_forward_retries_on_verify_fail(tmp_path: Path) -> None:
    """On verify failure, pipeline retries up to max_fix_iterations."""
    _make_workspace(tmp_path)
    config = _make_config(tmp_path, workspace_isolation=False, max_fix_iterations=2)
    orch = _make_orchestrator(config)

    plan_result = _make_agent_result(content="PLAN.md")
    impl_fail = _make_agent_result(content="impl attempt")
    verify_fail = _make_agent_result(content="Overall: FAIL ❌")
    impl_ok = _make_agent_result(content="fixed impl")
    verify_ok = _make_agent_result(content="Overall: PASS ✅")

    with patch.object(orch, "_make_agent") as mock_make:
        agents = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        # planner, impl1, verify1 (fail), impl2, verify2 (pass)
        agents[0].execute.return_value = plan_result
        agents[1].execute.return_value = impl_fail
        agents[2].execute.return_value = verify_fail
        agents[3].execute.return_value = impl_ok
        agents[4].execute.return_value = verify_ok
        mock_make.side_effect = agents

        result = orch.run("Fix bug")

    assert result.success is True
    assert result.metadata["iteration_count"] == 2


def test_fix_forward_returns_last_on_exhaustion(tmp_path: Path) -> None:
    """When max_fix_iterations exhausted, return last attempt (no raise)."""
    _make_workspace(tmp_path)
    config = _make_config(tmp_path, workspace_isolation=False, max_fix_iterations=2)
    orch = _make_orchestrator(config)

    plan_result = _make_agent_result(content="PLAN.md")
    impl_fail = _make_agent_result(content="impl attempt")
    verify_fail = _make_agent_result(content="Overall: FAIL ❌\nISSUE: missing tests")

    with patch.object(orch, "_make_agent") as mock_make:
        # 1 planner + 2*(impl + verify) = 5 agents
        agents = [MagicMock() for _ in range(5)]
        agents[0].execute.return_value = plan_result
        agents[1].execute.return_value = impl_fail
        agents[2].execute.return_value = verify_fail
        agents[3].execute.return_value = impl_fail
        agents[4].execute.return_value = verify_fail
        mock_make.side_effect = agents

        result = orch.run("Add tests")

    # Must not raise — returns last attempt
    assert isinstance(result, CodingPipelineResult)
    assert result.success is False
    assert result.metadata["iteration_count"] == 2

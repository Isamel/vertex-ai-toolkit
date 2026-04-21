"""Sprint 6 tests: budget manager, HTML reporter, lint gate."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.core.migration.budget import BudgetEvent, BudgetEventKind, MigrationBudgetManager
from vaig.core.migration.domain import Chunk, DomainNode
from vaig.core.migration.reporter import MigrationReporter
from vaig.core.migration.state import FileRecord, FileStatus, MigrationState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk() -> Chunk:
    node = DomainNode(
        step_name="test_step",
        step_type="TRANSFORM",
        semantic_kind="TRANSFORM",
    )
    return Chunk(node=node, text="test chunk")


def _make_state(
    source_kind: str = "spark",
    target_kind: str = "glue",
) -> MigrationState:
    state = MigrationState.new(source_kind, target_kind)
    return state


# ---------------------------------------------------------------------------
# CM-17: MigrationBudgetManager
# ---------------------------------------------------------------------------


def test_budget_manager_record_and_totals() -> None:
    mgr = MigrationBudgetManager(max_tokens=1000, max_cost_usd=5.0)
    mgr.record(BudgetEvent(kind=BudgetEventKind.LLM_CALL, tokens_used=100, cost_usd=0.50))
    mgr.record(BudgetEvent(kind=BudgetEventKind.GATE_CHECK, tokens_used=50, cost_usd=0.25))

    assert mgr.total_tokens() == 150
    assert abs(mgr.total_cost() - 0.75) < 1e-9


def test_budget_manager_is_over_budget_tokens() -> None:
    mgr = MigrationBudgetManager(max_tokens=100, max_cost_usd=10.0)
    mgr.record(BudgetEvent(kind=BudgetEventKind.LLM_CALL, tokens_used=101, cost_usd=0.0))
    assert mgr.is_over_budget() is True


def test_budget_manager_is_over_budget_cost() -> None:
    mgr = MigrationBudgetManager(max_tokens=100_000, max_cost_usd=1.0)
    mgr.record(BudgetEvent(kind=BudgetEventKind.LLM_CALL, tokens_used=1, cost_usd=1.01))
    assert mgr.is_over_budget() is True


def test_budget_manager_not_over_budget() -> None:
    mgr = MigrationBudgetManager(max_tokens=1000, max_cost_usd=5.0)
    mgr.record(BudgetEvent(kind=BudgetEventKind.LLM_CALL, tokens_used=300, cost_usd=1.0))
    assert mgr.is_over_budget() is False


def test_budget_manager_remaining() -> None:
    mgr = MigrationBudgetManager(max_tokens=1000, max_cost_usd=5.0)
    mgr.record(BudgetEvent(kind=BudgetEventKind.LLM_CALL, tokens_used=300, cost_usd=1.0))
    assert mgr.remaining_tokens() == 700
    assert abs(mgr.remaining_cost() - 4.0) < 1e-9


def test_budget_manager_compact_history() -> None:
    mgr = MigrationBudgetManager(max_tokens=100_000, max_cost_usd=10.0)
    for i in range(15):
        mgr.record(BudgetEvent(kind=BudgetEventKind.LLM_CALL, tokens_used=10, cost_usd=0.01))

    original_total_tokens = mgr.total_tokens()
    original_total_cost = mgr.total_cost()

    compacted = mgr.compact_history(keep_last=5)

    # At most keep_last + 1 summary event
    assert len(compacted.events) <= 6
    # Totals must be preserved
    assert compacted.total_tokens() == original_total_tokens
    assert abs(compacted.total_cost() - original_total_cost) < 1e-9


def test_budget_manager_summary_keys() -> None:
    mgr = MigrationBudgetManager(max_tokens=1000, max_cost_usd=5.0)
    s = mgr.summary()
    assert "total_tokens" in s
    assert "total_cost" in s
    assert "event_count" in s
    assert "is_over_budget" in s
    assert "remaining_tokens" in s
    assert "remaining_cost" in s


# ---------------------------------------------------------------------------
# CM-17: Orchestrator accepts budget param
# ---------------------------------------------------------------------------


def test_orchestrator_accepts_budget(tmp_path: Path) -> None:
    from vaig.core.migration.budget import MigrationBudgetManager
    from vaig.core.migration.config import MigrationConfig
    from vaig.core.migration.orchestrator import MigrationOrchestrator

    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    config = MigrationConfig(
        from_dirs=[src],
        to_dir=out,
        source_kind="spark",
        target_kind="glue",
    )
    budget = MigrationBudgetManager(max_tokens=500, max_cost_usd=2.0)
    orch = MigrationOrchestrator(migration_config=config, budget=budget)
    assert orch.budget is budget


# ---------------------------------------------------------------------------
# CM-19: MigrationReporter
# ---------------------------------------------------------------------------


def _state_with_files() -> MigrationState:
    state = _make_state()
    state.files["a.py"] = FileRecord(
        source_path="a.py",
        target_path="out/a.py",
        status=FileStatus.COMPLETED,
    )
    state.files["b.py"] = FileRecord(
        source_path="b.py",
        status=FileStatus.FAILED,
        error="some error",
    )
    state.files["c.py"] = FileRecord(
        source_path="c.py",
        status=FileStatus.PENDING,
    )
    return state


def test_migration_report_build() -> None:
    state = _state_with_files()
    reporter = MigrationReporter(state=state)
    report = reporter.build_report()

    assert report.files_total == 3
    assert report.files_completed == 1
    assert report.files_failed == 1
    assert report.files_skipped == 0
    assert report.change_id == state.change_id
    assert "b.py: some error" in report.errors


def test_reporter_to_html_contains_summary() -> None:
    state = _state_with_files()
    reporter = MigrationReporter(state=state)
    report = reporter.build_report()
    html = reporter.to_html(report)

    assert "completed" in html.lower() or "Completed" in html
    assert "failed" in html.lower() or "Failed" in html
    assert report.change_id in html


def test_reporter_save_html(tmp_path: Path) -> None:
    state = _state_with_files()
    reporter = MigrationReporter(state=state)
    report = reporter.build_report()
    out = tmp_path / "report.html"
    reporter.save_html(report, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<html" in content


# ---------------------------------------------------------------------------
# CM-13: LintGate
# ---------------------------------------------------------------------------


def test_lint_gate_ruff_not_available() -> None:
    from vaig.core.migration.gates.lint_gate import LintGate

    gate = LintGate()
    chunk = _make_chunk()

    with patch("subprocess.run", side_effect=FileNotFoundError("ruff not found")):
        result = gate.check(chunk, "x = 1")

    assert result.passed is True
    assert "ruff not available" in result.notes


def test_lint_gate_no_violations_passes() -> None:
    from vaig.core.migration.gates.lint_gate import LintGate

    gate = LintGate()
    chunk = _make_chunk()

    mock_result = MagicMock()
    mock_result.stdout = "[]"
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        result = gate.check(chunk, "x = 1\n")

    assert result.passed is True
    assert result.violations == []


def test_lint_gate_violations_strict_false() -> None:
    from vaig.core.migration.gates.lint_gate import LintGate

    gate = LintGate()
    chunk = _make_chunk()

    violation = [{"code": "E501", "message": "line too long", "location": {"row": 1}}]
    mock_result = MagicMock()
    mock_result.stdout = json.dumps(violation)
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        result = gate.check(chunk, "x = 1\n", strict=False)

    assert result.passed is True
    assert len(result.violations) == 1
    assert "E501" in result.notes


def test_lint_gate_violations_strict_true() -> None:
    from vaig.core.migration.gates.lint_gate import LintGate

    gate = LintGate()
    chunk = _make_chunk()

    violation = [{"code": "E501", "message": "line too long", "location": {"row": 1}}]
    mock_result = MagicMock()
    mock_result.stdout = json.dumps(violation)
    mock_result.returncode = 1

    with patch("subprocess.run", return_value=mock_result):
        result = gate.check(chunk, "x = 1\n", strict=True)

    assert result.passed is False
    assert len(result.violations) == 1

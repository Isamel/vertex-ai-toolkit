"""Tests for SPEC-RATE-05: reporter Run Quality section rendering.

Tests the _render_run_quality_section static method of ServiceHealthSkill,
which renders a ## Run Quality table when any issues are present.
"""

from __future__ import annotations

from vaig.core.quality import QualityIssue, QualityIssueKind
from vaig.skills.service_health.skill import ServiceHealthSkill

# ── _render_run_quality_section ───────────────────────────────


class TestRenderRunQualitySection:
    """Unit tests for ServiceHealthSkill._render_run_quality_section."""

    def test_empty_list_returns_empty_string(self) -> None:
        """No issues → empty string (no section rendered)."""
        result = ServiceHealthSkill._render_run_quality_section([])
        assert result == ""

    def test_none_returns_empty_string(self) -> None:
        """None run_quality → empty string."""
        result = ServiceHealthSkill._render_run_quality_section(None)
        assert result == ""

    def test_any_issue_type_renders_section(self) -> None:
        """Any issue kind → section rendered (not just MODEL_DEGRADED)."""
        issues = [
            QualityIssue(kind=QualityIssueKind.agent_failed, where="agent-x"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result != ""
        assert "## Run Quality" in result
        assert "agent-x" in result

    def test_heading_format(self) -> None:
        """Section heading is '## Run Quality ⚠ (N issue/issues)'."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-alpha"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "## Run Quality ⚠ (1 issue)" in result

    def test_heading_plural(self) -> None:
        """Plural label when more than one issue."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-a"),
            QualityIssue(kind=QualityIssueKind.agent_failed, where="agent-b"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "## Run Quality ⚠ (2 issues)" in result

    def test_table_header_present(self) -> None:
        """Markdown table header row is present."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="ag"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "| Issue | Where | Consequence |" in result
        assert "|---|---|---|" in result

    def test_issue_row_contains_kind_and_where(self) -> None:
        """Each issue appears as a table row with kind and where."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-alpha"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "model_degraded" in result
        assert "agent-alpha" in result

    def test_all_issues_appear_in_table(self) -> None:
        """All issues (regardless of kind) appear in the table."""
        issues = [
            QualityIssue(kind=QualityIssueKind.agent_failed, where="failed-agent"),
            QualityIssue(kind=QualityIssueKind.model_degraded, where="degraded-agent"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "failed-agent" in result
        assert "degraded-agent" in result

    def test_suggested_action_line_present(self) -> None:
        """'Suggested action' line is included in the section."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-z"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "Suggested action" in result
        assert "lower-quota window" in result

    def test_section_ends_with_double_newline(self) -> None:
        """Section ends with \\n\\n for proper Markdown separation."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="agent-z"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result.endswith("\n\n")

    def test_section_starts_with_heading(self) -> None:
        """Section starts with ## heading (not a blockquote)."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="ag"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result.startswith("## Run Quality")

    def test_section_prepended_to_report(self) -> None:
        """When prepended to a report string, quality section appears at TOP."""
        issues = [
            QualityIssue(kind=QualityIssueKind.model_degraded, where="ag"),
        ]
        section = ServiceHealthSkill._render_run_quality_section(issues)
        report_body = "## Summary\nSome content."
        full_report = section + report_body
        assert full_report.startswith("## Run Quality")
        assert "## Summary" in full_report

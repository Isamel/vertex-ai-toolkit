"""Tests for SPEC-RATE-05: reporter Run Quality section rendering.

Tests the _render_run_quality_section static method of ServiceHealthSkill,
which prepends a degradation notice when MODEL_DEGRADED issues are present.
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

    def test_no_model_degraded_returns_empty_string(self) -> None:
        """Issues without MODEL_DEGRADED → empty string."""
        issues = [
            QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="agent-x"),
            QualityIssue(kind=QualityIssueKind.TOOL_ERROR, where="tool-y"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result == ""

    def test_model_degraded_renders_notice(self) -> None:
        """One MODEL_DEGRADED issue → blockquote notice returned."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-alpha"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result != ""
        assert "⚠️" in result
        assert "agent-alpha" in result

    def test_model_degraded_notice_contains_agent_names(self) -> None:
        """Multiple MODEL_DEGRADED issues → agent names appear in notice."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-a"),
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-b"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "agent-a" in result
        assert "agent-b" in result

    def test_mixed_issues_only_renders_for_model_degraded(self) -> None:
        """Only MODEL_DEGRADED issues contribute to the notice."""
        issues = [
            QualityIssue(kind=QualityIssueKind.AGENT_FAILED, where="failed-agent"),
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="degraded-agent"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert "degraded-agent" in result
        # failed-agent should NOT appear in the run quality notice
        assert "failed-agent" not in result

    def test_section_ends_with_double_newline(self) -> None:
        """Section ends with \\n\\n for proper Markdown separation."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-z"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result.endswith("\n\n")

    def test_notice_is_blockquote_format(self) -> None:
        """Notice is rendered as a Markdown blockquote (starts with '>') ."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="ag"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        assert result.startswith(">")

    def test_deduplicates_agent_names_in_notice(self) -> None:
        """Duplicate agent names (same where) appear only once in notice."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-a"),
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="agent-a"),  # same
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        # agent-a should appear only once
        assert result.count("agent-a") == 1

    def test_agents_sorted_alphabetically(self) -> None:
        """Agent names in the notice are sorted alphabetically."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="zebra"),
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="alpha"),
        ]
        result = ServiceHealthSkill._render_run_quality_section(issues)
        alpha_pos = result.find("alpha")
        zebra_pos = result.find("zebra")
        assert alpha_pos < zebra_pos

    def test_section_prepended_to_report(self) -> None:
        """When prepended to a report string, quality section appears at TOP."""
        issues = [
            QualityIssue(kind=QualityIssueKind.MODEL_DEGRADED, where="ag"),
        ]
        section = ServiceHealthSkill._render_run_quality_section(issues)
        report_body = "## Summary\nSome content."
        full_report = section + report_body
        assert full_report.startswith(">")
        assert "## Summary" in full_report

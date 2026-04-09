"""Tests for the GitHub Actions health-check entrypoint.

Covers:
- SC-01 through SC-09 scenarios from the spec
- Unit tests for parse_inputs, extract_severity, format_comment, set_outputs
- Integration test for main() with mocked execute_skill_headless
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# The entrypoint is a standalone script in .github/actions/health-check/.
# We import it by manipulating sys.path so pytest can find it regardless
# of whether ruff includes .github/.
ENTRYPOINT_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    ".github",
    "actions",
    "health-check",
)
sys.path.insert(0, os.path.abspath(ENTRYPOINT_DIR))

import entrypoint  # noqa: E402, I001


# ── Helpers ──────────────────────────────────────────────────────


class _FakeFinding:
    """Minimal mock for a Finding object."""

    def __init__(self, severity: str) -> None:
        self.severity = severity


class _FakeReport:
    """Minimal mock for a HealthReport."""

    def __init__(self, findings: list[_FakeFinding] | None = None) -> None:
        self.findings = findings if findings is not None else []

    def to_markdown(self) -> str:
        if not self.findings:
            return "# Service Health Report\n\nNo findings."
        lines = ["# Service Health Report", ""]
        for f in self.findings:
            lines.append(f"- [{f.severity}] Finding")
        return "\n".join(lines)


@dataclass
class _FakeOrchestratorResult:
    """Minimal stand-in for OrchestratorResult."""

    skill_name: str = "discovery"
    phase: str = "execute"
    synthesized_output: str = "test output"
    success: bool = True
    run_cost_usd: float = 0.0042
    structured_report: Any = None
    agent_results: list[Any] = field(default_factory=list)


# ── Task 3.1: parse_inputs tests ────────────────────────────────


class TestParseInputs:
    """Unit tests for parse_inputs()."""

    def test_valid_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INPUT_CLUSTER", "my-cluster")
        monkeypatch.setenv("INPUT_PROJECT-ID", "my-project")
        monkeypatch.setenv("INPUT_LOCATION", "us-central1")
        monkeypatch.setenv("INPUT_NAMESPACE", "production")
        monkeypatch.setenv("INPUT_FAIL-ON", "HIGH")
        monkeypatch.setenv("INPUT_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("INPUT_COMMENT", "false")
        monkeypatch.setenv("INPUT_TIMEOUT", "120")

        result = entrypoint.parse_inputs()

        assert result.cluster == "my-cluster"
        assert result.project_id == "my-project"
        assert result.location == "us-central1"
        assert result.namespace == "production"
        assert result.fail_on == "HIGH"
        assert result.model == "gemini-2.5-pro"
        assert result.comment is False
        assert result.timeout == 120

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INPUT_CLUSTER", "c")
        monkeypatch.setenv("INPUT_PROJECT-ID", "p")
        monkeypatch.setenv("INPUT_LOCATION", "loc")
        # Clear optional inputs
        monkeypatch.delenv("INPUT_NAMESPACE", raising=False)
        monkeypatch.delenv("INPUT_FAIL-ON", raising=False)
        monkeypatch.delenv("INPUT_MODEL", raising=False)
        monkeypatch.delenv("INPUT_COMMENT", raising=False)
        monkeypatch.delenv("INPUT_TIMEOUT", raising=False)

        result = entrypoint.parse_inputs()

        assert result.namespace == "default"
        assert result.fail_on == "CRITICAL"
        assert result.model == "gemini-2.5-flash"
        assert result.comment is True
        assert result.timeout == 300

    def test_missing_required_cluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SC-08: Missing required input."""
        monkeypatch.delenv("INPUT_CLUSTER", raising=False)
        monkeypatch.setenv("INPUT_PROJECT-ID", "p")
        monkeypatch.setenv("INPUT_LOCATION", "loc")

        with pytest.raises(SystemExit) as exc_info:
            entrypoint.parse_inputs()
        assert exc_info.value.code == 1

    def test_missing_required_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INPUT_CLUSTER", "c")
        monkeypatch.delenv("INPUT_PROJECT-ID", raising=False)
        monkeypatch.setenv("INPUT_LOCATION", "loc")

        with pytest.raises(SystemExit) as exc_info:
            entrypoint.parse_inputs()
        assert exc_info.value.code == 1

    def test_missing_required_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INPUT_CLUSTER", "c")
        monkeypatch.setenv("INPUT_PROJECT-ID", "p")
        monkeypatch.delenv("INPUT_LOCATION", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            entrypoint.parse_inputs()
        assert exc_info.value.code == 1

    def test_invalid_fail_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SC-08 variant: Invalid fail-on value."""
        monkeypatch.setenv("INPUT_CLUSTER", "c")
        monkeypatch.setenv("INPUT_PROJECT-ID", "p")
        monkeypatch.setenv("INPUT_LOCATION", "loc")
        monkeypatch.setenv("INPUT_FAIL-ON", "INVALID")

        with pytest.raises(SystemExit) as exc_info:
            entrypoint.parse_inputs()
        assert exc_info.value.code == 1

    def test_fail_on_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INPUT_CLUSTER", "c")
        monkeypatch.setenv("INPUT_PROJECT-ID", "p")
        monkeypatch.setenv("INPUT_LOCATION", "loc")
        monkeypatch.setenv("INPUT_FAIL-ON", "high")

        result = entrypoint.parse_inputs()
        assert result.fail_on == "HIGH"


# ── Task 3.2: extract_severity tests ────────────────────────────


class TestExtractSeverity:
    """Unit tests for extract_severity()."""

    @pytest.mark.parametrize(
        ("severities", "expected_max", "expected_level", "expected_count"),
        [
            # SC-01: INFO only
            (["INFO"], "INFO", 0, 1),
            # SC-02: CRITICAL present
            (["INFO", "CRITICAL", "LOW"], "CRITICAL", 4, 3),
            # SC-03: MEDIUM (WARNING equivalent)
            (["LOW", "MEDIUM"], "MEDIUM", 2, 2),
            # Multiple HIGHs
            (["HIGH", "HIGH", "INFO"], "HIGH", 3, 3),
            # Single CRITICAL
            (["CRITICAL"], "CRITICAL", 4, 1),
        ],
    )
    def test_severity_extraction(
        self,
        severities: list[str],
        expected_max: str,
        expected_level: int,
        expected_count: int,
    ) -> None:
        findings = [_FakeFinding(s) for s in severities]
        report = _FakeReport(findings)
        result = _FakeOrchestratorResult(structured_report=report)

        max_sev, level, count = entrypoint.extract_severity(result)

        assert max_sev == expected_max
        assert level == expected_level
        assert count == expected_count

    def test_empty_findings(self) -> None:
        """SC-04: Clean cluster, zero findings."""
        report = _FakeReport(findings=[])
        result = _FakeOrchestratorResult(structured_report=report)

        max_sev, level, count = entrypoint.extract_severity(result)

        assert max_sev == "NONE"
        assert level == -1
        assert count == 0

    def test_no_report(self) -> None:
        """structured_report is None (parse failed)."""
        result = _FakeOrchestratorResult(structured_report=None)

        max_sev, level, count = entrypoint.extract_severity(result)

        assert max_sev == "NONE"
        assert level == -1
        assert count == 0


# ── Task 3.3: format_comment tests ──────────────────────────────


class TestFormatComment:
    """Unit tests for format_comment()."""

    def test_marker_present(self) -> None:
        body = entrypoint.format_comment("report", "pass", 0, 0.0)
        assert entrypoint.COMMENT_MARKER in body

    def test_details_wrapping(self) -> None:
        body = entrypoint.format_comment("# Report\n\nSome findings", "fail", 3, 0.01)
        assert "<details>" in body
        assert "</details>" in body
        assert "<summary>Full Health Report</summary>" in body

    def test_summary_line(self) -> None:
        body = entrypoint.format_comment("report", "pass", 5, 0.0042)
        assert "Status: `pass`" in body
        assert "Findings: 5" in body
        assert "$0.0042" in body

    def test_fail_status_in_summary(self) -> None:
        body = entrypoint.format_comment("report", "fail", 1, 0.01)
        assert "Status: `fail`" in body


# ── Task 3.4: set_outputs tests ─────────────────────────────────


class TestSetOutputs:
    """Unit tests for set_outputs()."""

    def test_outputs_written(self, tmp_path: Any) -> None:
        output_file = tmp_path / "github_output"
        output_file.write_text("")

        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            entrypoint.set_outputs("pass", 3, "HIGH", "# Report\nDetails here")

        content = output_file.read_text()
        assert "status=pass" in content
        assert "findings-count=3" in content
        assert "max-severity=HIGH" in content
        assert "report<<VAIG_EOF" in content
        assert "# Report\nDetails here" in content
        assert "VAIG_EOF" in content

    def test_multiline_report(self, tmp_path: Any) -> None:
        output_file = tmp_path / "github_output"
        output_file.write_text("")

        multiline = "Line 1\nLine 2\nLine 3"
        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(output_file)}):
            entrypoint.set_outputs("fail", 0, "NONE", multiline)

        content = output_file.read_text()
        assert "Line 1\nLine 2\nLine 3" in content

    def test_no_output_file(self) -> None:
        """GITHUB_OUTPUT not set — should not crash."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_OUTPUT", None)
            # Should not raise
            entrypoint.set_outputs("pass", 0, "NONE", "report")


# ── Task 3.5: severity threshold gating ─────────────────────────


class TestSeverityGating:
    """Unit tests for severity threshold logic."""

    @pytest.mark.parametrize(
        ("max_severity", "fail_on", "expected_exit"),
        [
            # CRITICAL finding, threshold CRITICAL → fail
            ("CRITICAL", "CRITICAL", 1),
            # INFO finding, threshold CRITICAL → pass
            ("INFO", "CRITICAL", 0),
            # MEDIUM finding, threshold MEDIUM → fail (SC-03)
            ("MEDIUM", "MEDIUM", 1),
            # HIGH finding, threshold CRITICAL → pass
            ("HIGH", "CRITICAL", 0),
            # LOW finding, threshold LOW → fail
            ("LOW", "LOW", 1),
            # NONE (no findings), threshold INFO → pass
            ("NONE", "INFO", 0),
            # HIGH finding, threshold HIGH → fail
            ("HIGH", "HIGH", 1),
            # LOW finding, threshold HIGH → pass
            ("LOW", "HIGH", 0),
        ],
    )
    def test_threshold_gating(
        self, max_severity: str, fail_on: str, expected_exit: int
    ) -> None:
        severity_level = entrypoint.SEVERITY_LEVELS.get(max_severity, -1)
        threshold_level = entrypoint.SEVERITY_LEVELS.get(fail_on, 4)

        if severity_level >= threshold_level:
            exit_code = 1
        else:
            exit_code = 0

        assert exit_code == expected_exit


# ── Task 3.6: main() integration tests ──────────────────────────


class TestMain:
    """Integration tests for the full main() flow."""

    def _setup_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
        """Set up common environment for main() tests."""
        monkeypatch.setenv("INPUT_CLUSTER", "test-cluster")
        monkeypatch.setenv("INPUT_PROJECT-ID", "test-project")
        monkeypatch.setenv("INPUT_LOCATION", "us-central1")
        monkeypatch.setenv("INPUT_FAIL-ON", "CRITICAL")
        monkeypatch.setenv("INPUT_COMMENT", "true")
        monkeypatch.setenv("INPUT_TIMEOUT", "300")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

        # Event file
        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"pull_request": {"number": 42}}))
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))

        # Output file
        output_file = tmp_path / "github_output"
        output_file.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

        return output_file

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_happy_path_pass(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-01: Happy path — INFO findings, threshold CRITICAL → pass."""
        output_file = self._setup_env(monkeypatch, tmp_path)

        report = _FakeReport([_FakeFinding("INFO")])
        mock_headless.return_value = _FakeOrchestratorResult(
            structured_report=report,
            run_cost_usd=0.005,
        )

        exit_code = entrypoint.main()

        assert exit_code == 0
        mock_post.assert_called_once()
        content = output_file.read_text()
        assert "status=pass" in content
        assert "max-severity=INFO" in content
        assert "findings-count=1" in content

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_critical_finding_fails(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-02: CRITICAL finding with threshold CRITICAL → fail."""
        output_file = self._setup_env(monkeypatch, tmp_path)

        report = _FakeReport([_FakeFinding("INFO"), _FakeFinding("CRITICAL")])
        mock_headless.return_value = _FakeOrchestratorResult(
            structured_report=report,
            run_cost_usd=0.01,
        )

        exit_code = entrypoint.main()

        assert exit_code == 1
        content = output_file.read_text()
        assert "status=fail" in content
        assert "max-severity=CRITICAL" in content

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_medium_threshold_medium(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-03: MEDIUM finding with threshold MEDIUM → fail."""
        output_file = self._setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("INPUT_FAIL-ON", "MEDIUM")

        report = _FakeReport([_FakeFinding("MEDIUM")])
        mock_headless.return_value = _FakeOrchestratorResult(
            structured_report=report,
        )

        exit_code = entrypoint.main()

        assert exit_code == 1
        content = output_file.read_text()
        assert "status=fail" in content

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_clean_cluster(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-04: Clean cluster — zero findings."""
        output_file = self._setup_env(monkeypatch, tmp_path)

        report = _FakeReport(findings=[])
        mock_headless.return_value = _FakeOrchestratorResult(
            structured_report=report,
        )

        exit_code = entrypoint.main()

        assert exit_code == 0
        content = output_file.read_text()
        assert "status=pass" in content
        assert "findings-count=0" in content
        assert "max-severity=NONE" in content

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_auth_failure(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-05: Auth failure — no ADC credentials."""
        output_file = self._setup_env(monkeypatch, tmp_path)

        mock_headless.side_effect = RuntimeError(
            "Default credentials not found"
        )

        exit_code = entrypoint.main()

        assert exit_code == 1
        mock_post.assert_called_once()  # Error comment posted
        content = output_file.read_text()
        assert "status=fail" in content

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_timeout(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-06: Pipeline timeout."""
        output_file = self._setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("INPUT_TIMEOUT", "60")

        mock_headless.side_effect = TimeoutError("Health check timed out")

        exit_code = entrypoint.main()

        assert exit_code == 1
        content = output_file.read_text()
        assert "status=fail" in content

    @patch("requests.get")
    @patch("requests.patch")
    @patch("requests.post")
    def test_comment_update_existing(
        self,
        mock_post: MagicMock,
        mock_patch: MagicMock,
        mock_get: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """SC-07: Update existing comment (idempotent)."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_API_URL", "https://api.github.com")

        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"pull_request": {"number": 10}}))
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))

        # Existing comment with marker
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"id": 999, "body": f"old report {entrypoint.COMMENT_MARKER}"},
            ],
        )
        mock_patch.return_value = MagicMock(status_code=200)

        entrypoint.post_comment("new report body")

        # Should PATCH, not POST
        mock_patch.assert_called_once()
        mock_post.assert_not_called()
        assert "/issues/comments/999" in mock_patch.call_args[0][0]

    @patch("requests.get")
    @patch("requests.post")
    def test_comment_permission_denied(
        self,
        mock_post: MagicMock,
        mock_get: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SC-09: GITHUB_TOKEN lacks write permission — fallback to stdout."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_readonly")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_API_URL", "https://api.github.com")

        event_file = tmp_path / "event.json"
        event_file.write_text(json.dumps({"pull_request": {"number": 5}}))
        monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))

        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        mock_post.return_value = MagicMock(status_code=403, text="Forbidden")

        entrypoint.post_comment("report body")

        captured = capsys.readouterr()
        assert "report body" in captured.out

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_no_report_fallback(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """structured_report is None — fall back to synthesized_output."""
        output_file = self._setup_env(monkeypatch, tmp_path)

        mock_headless.return_value = _FakeOrchestratorResult(
            structured_report=None,
            synthesized_output="Fallback text report",
        )

        exit_code = entrypoint.main()

        assert exit_code == 0  # No findings → pass
        content = output_file.read_text()
        assert "Fallback text report" in content

    @patch("entrypoint.post_comment")
    @patch("vaig.core.headless.execute_skill_headless")
    def test_comment_disabled(
        self,
        mock_headless: MagicMock,
        mock_post: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """comment=false skips PR commenting."""
        self._setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("INPUT_COMMENT", "false")

        report = _FakeReport([_FakeFinding("INFO")])
        mock_headless.return_value = _FakeOrchestratorResult(
            structured_report=report,
        )

        exit_code = entrypoint.main()

        assert exit_code == 0
        mock_post.assert_not_called()

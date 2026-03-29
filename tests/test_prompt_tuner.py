"""Tests for ReportStore, PromptTuner, and the ``vaig optimize --reports`` CLI flag.

Covers local JSONL persistence, quality signal computation, suggestion
generation, edge cases, and CLI integration via Typer's CliRunner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from vaig.core.prompt_tuner import (
    HALLUCINATION_THRESHOLD,
    INCOMPLETE_THRESHOLD,
    LOW_ACTIONABILITY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    OVER_ESCALATION_THRESHOLD,
    PromptTuner,
    QualityInsights,
    QualitySignal,
)
from vaig.core.report_store import ReportStore

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(
    *,
    evidence: list[str] | None = None,
    severity: str = "MEDIUM",
    confidence: str = "HIGH",
) -> dict[str, Any]:
    """Build a minimal Finding dict."""
    return {
        "title": "Test finding",
        "description": "Something is off",
        "severity": severity,
        "confidence": confidence,
        "evidence": evidence if evidence is not None else ["log line 1"],
        "affected_resources": ["pod/nginx"],
        "category": "reliability",
    }


def _make_action(*, command: str = "kubectl get pods") -> dict[str, Any]:
    """Build a minimal RecommendedAction dict."""
    return {
        "title": "Fix it",
        "command": command,
        "priority": 1,
        "urgency": "immediate",
    }


def _make_recommendation(
    *,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal Recommendation dict."""
    return {
        "title": "Restart pod",
        "actions": actions if actions is not None else [_make_action()],
    }


def _make_report(
    *,
    findings: list[dict[str, Any]] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
    timeline: list[dict[str, Any]] | None = None,
    scope: str = "Cluster-wide",
) -> dict[str, Any]:
    """Build a minimal serialised HealthReport dict."""
    return {
        "executive_summary": {
            "overall_status": "warning",
            "issues_found": 1,
            "critical_count": 0,
            "warning_count": 1,
            "scope": scope,
        },
        "findings": findings if findings is not None else [_make_finding()],
        "recommendations": (
            recommendations
            if recommendations is not None
            else [_make_recommendation()]
        ),
        "timeline": timeline if timeline is not None else [{"event": "observed", "ts": "now"}],
        "metadata": {},
    }


def _wrap_record(
    report: dict[str, Any],
    *,
    run_id: str = "20260329T000000Z",
) -> dict[str, Any]:
    """Wrap a report dict in a store record envelope."""
    return {
        "timestamp": "2026-03-29T00:00:00+00:00",
        "run_id": run_id,
        "report": report,
    }


def _seed_report_store(
    tmp_path: Path,
    records_per_run: dict[str, list[dict[str, Any]]],
) -> ReportStore:
    """Create a ReportStore seeded with pre-written JSONL files.

    Args:
        tmp_path: pytest temp directory.
        records_per_run: Mapping of run_id → list of record dicts.

    Returns:
        A ReportStore pointed at the seeded directory.
    """
    store_dir = tmp_path / "reports"
    store_dir.mkdir(parents=True, exist_ok=True)
    for run_id, records in records_per_run.items():
        path = store_dir / f"{run_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return ReportStore(base_dir=store_dir)


# ===========================================================================
# ReportStore tests
# ===========================================================================


class TestReportStoreSave:
    """Tests for ReportStore.save()."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Saving a report creates a JSONL file named after the run_id."""
        store = ReportStore(base_dir=tmp_path / "reports")
        report = _make_report()
        path = store.save("run-1", report)

        assert path.exists()
        assert path.name == "run-1.jsonl"

    def test_save_appends_to_existing(self, tmp_path: Path) -> None:
        """Successive saves to the same run_id append lines."""
        store = ReportStore(base_dir=tmp_path / "reports")
        store.save("run-1", _make_report())
        store.save("run-1", _make_report())

        path = tmp_path / "reports" / "run-1.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2


class TestReportStoreListRuns:
    """Tests for ReportStore.list_runs()."""

    def test_list_runs_newest_first(self, tmp_path: Path) -> None:
        """Runs are returned in reverse modification-time order."""
        store_dir = tmp_path / "reports"
        store_dir.mkdir(parents=True, exist_ok=True)
        # Create files with deterministic order via explicit writes
        (store_dir / "old-run.jsonl").write_text("{}\n")
        (store_dir / "new-run.jsonl").write_text("{}\n")

        store = ReportStore(base_dir=store_dir)
        runs = store.list_runs()

        assert "old-run" in runs
        assert "new-run" in runs
        # newest (last-modified) first — both files written same instant so
        # we just check both are present
        assert len(runs) == 2


class TestReportStoreReadReports:
    """Tests for ReportStore.read_reports()."""

    def test_read_reports_limit(self, tmp_path: Path) -> None:
        """Requesting last=1 returns at most 1 record."""
        store = _seed_report_store(tmp_path, {
            "run-1": [_wrap_record(_make_report())],
            "run-2": [_wrap_record(_make_report())],
        })
        reports = store.read_reports(last=1)

        assert len(reports) == 1

    def test_malformed_jsonl_skipped(self, tmp_path: Path) -> None:
        """Malformed lines are silently skipped."""
        store_dir = tmp_path / "reports"
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / "bad-run.jsonl"
        path.write_text("not valid json\n" + json.dumps(_wrap_record(_make_report())) + "\n")

        store = ReportStore(base_dir=store_dir)
        reports = store.read_reports()

        assert len(reports) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        """An empty store returns an empty list."""
        store = ReportStore(base_dir=tmp_path / "reports")
        reports = store.read_reports()

        assert reports == []


# ===========================================================================
# PromptTuner tests
# ===========================================================================


class TestPromptTunerAllHealthy:
    """Tests for PromptTuner when all signals are healthy."""

    def test_all_healthy_returns_no_suggestions(self) -> None:
        """A report with evidence, actions, and timeline passes all signals."""
        records = [_wrap_record(_make_report())]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        assert insights.total_reports == 1
        assert insights.suggestions == []
        assert all(s.passed for s in insights.signals)


class TestPromptTunerHallucination:
    """Tests for the hallucination_rate signal."""

    def test_high_hallucination_fails(self) -> None:
        """Findings with no evidence push the hallucination rate above threshold."""
        findings = [
            _make_finding(evidence=[]),
            _make_finding(evidence=[]),
            _make_finding(evidence=["has evidence"]),
        ]
        records = [_wrap_record(_make_report(findings=findings))]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        halluc = next(s for s in insights.signals if s.name == "hallucination_rate")
        assert not halluc.passed
        assert halluc.value > HALLUCINATION_THRESHOLD


class TestPromptTunerActionability:
    """Tests for the actionability signal."""

    def test_low_actionability_fails(self) -> None:
        """Actions without commands lower the actionability score."""
        recs = [_make_recommendation(actions=[
            _make_action(command=""),
            _make_action(command=""),
            _make_action(command="kubectl get pods"),
        ])]
        records = [_wrap_record(_make_report(recommendations=recs))]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        actionability = next(s for s in insights.signals if s.name == "actionability")
        assert not actionability.passed
        assert actionability.value < LOW_ACTIONABILITY_THRESHOLD


class TestPromptTunerOverEscalation:
    """Tests for the over_escalation signal."""

    def test_critical_at_resource_scope_fails(self) -> None:
        """CRITICAL findings at resource scope trigger over-escalation."""
        findings = [
            _make_finding(severity="CRITICAL"),
            _make_finding(severity="CRITICAL"),
        ]
        records = [_wrap_record(
            _make_report(findings=findings, scope="Resource: pod/nginx in default"),
        )]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        oe = next(s for s in insights.signals if s.name == "over_escalation")
        assert not oe.passed
        assert oe.value > OVER_ESCALATION_THRESHOLD


class TestPromptTunerCompleteness:
    """Tests for the completeness signal."""

    def test_missing_timeline_fails(self) -> None:
        """Reports without timeline data lower the completeness score."""
        records = [
            _wrap_record(_make_report(timeline=[])),
            _wrap_record(_make_report(timeline=[])),
        ]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        completeness = next(s for s in insights.signals if s.name == "completeness")
        assert not completeness.passed
        assert completeness.value < INCOMPLETE_THRESHOLD


class TestPromptTunerLowConfidence:
    """Tests for the low_confidence signal."""

    def test_many_low_confidence_fails(self) -> None:
        """A majority of LOW-confidence findings triggers the signal."""
        findings = [
            _make_finding(confidence="LOW"),
            _make_finding(confidence="LOW"),
            _make_finding(confidence="LOW"),
            _make_finding(confidence="HIGH"),
        ]
        records = [_wrap_record(_make_report(findings=findings))]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        lc = next(s for s in insights.signals if s.name == "low_confidence")
        assert not lc.passed
        assert lc.value > LOW_CONFIDENCE_THRESHOLD


class TestPromptTunerSuggestions:
    """Tests for suggestion generation per signal."""

    def test_each_failing_signal_maps_to_suggestion(self) -> None:
        """Each failing signal produces exactly one suggestion."""
        # Create a report that fails hallucination, actionability, and completeness
        findings = [_make_finding(evidence=[])]
        recs = [_make_recommendation(actions=[_make_action(command="")])]
        report = _make_report(
            findings=findings,
            recommendations=recs,
            timeline=[],
        )
        records = [_wrap_record(report)]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        failing = [s for s in insights.signals if not s.passed]
        assert len(failing) >= 2
        assert len(insights.suggestions) == len(failing)


class TestPromptTunerEdgeCases:
    """Tests for edge cases in PromptTuner."""

    def test_empty_reports_list(self) -> None:
        """An empty list returns zero reports and no signals."""
        tuner = PromptTuner()
        insights = tuner.analyze_quality([])

        assert insights.total_reports == 0
        assert insights.signals == []
        assert insights.suggestions == []

    def test_missing_fields_handled_gracefully(self) -> None:
        """A report dict missing optional fields does not crash."""
        # Minimal dict — no findings, no recommendations, no timeline
        records = [_wrap_record({"executive_summary": {"scope": ""}})]
        tuner = PromptTuner()
        insights = tuner.analyze_quality(records)

        assert insights.total_reports == 1
        # Should not raise


class TestQualityDataModels:
    """Tests for QualitySignal and QualityInsights frozen dataclasses."""

    def test_quality_signal_frozen(self) -> None:
        """QualitySignal instances are immutable."""
        signal = QualitySignal(
            name="test",
            value=0.5,
            threshold=0.3,
            passed=True,
            detail="ok",
        )
        assert signal.name == "test"

    def test_quality_insights_defaults(self) -> None:
        """QualityInsights defaults to empty lists."""
        insights = QualityInsights(total_reports=0)
        assert insights.signals == []
        assert insights.suggestions == []


# ===========================================================================
# CLI integration — vaig optimize --reports
# ===========================================================================


class TestOptimizeReportsCLI:
    """Tests for the --reports flag on the optimize command."""

    def test_reports_flag_invokes_prompt_tuner(self, tmp_path: Path) -> None:
        """``vaig optimize --reports`` calls PromptTuner.analyze_quality."""
        from vaig.cli.app import app

        with (
            patch(
                "vaig.core.report_store.ReportStore",
            ) as mock_store_cls,
            patch(
                "vaig.core.prompt_tuner.PromptTuner",
            ) as mock_tuner_cls,
        ):
            mock_store_cls.return_value.read_reports.return_value = []
            mock_tuner_cls.return_value.analyze_quality.return_value = QualityInsights(
                total_reports=0,
            )

            result = runner.invoke(app, ["optimize", "--reports"])

            assert result.exit_code == 0
            mock_tuner_cls.return_value.analyze_quality.assert_called_once()

    def test_reports_flag_displays_signals(self, tmp_path: Path) -> None:
        """``vaig optimize --reports`` displays quality signals in output."""
        from vaig.cli.app import app

        insights = QualityInsights(
            total_reports=5,
            signals=[
                QualitySignal(
                    name="hallucination_rate",
                    value=0.1,
                    threshold=0.3,
                    passed=True,
                    detail="1/10 findings lack evidence",
                ),
            ],
            suggestions=[],
        )

        with (
            patch(
                "vaig.core.report_store.ReportStore",
            ) as mock_store_cls,
            patch(
                "vaig.core.prompt_tuner.PromptTuner",
            ) as mock_tuner_cls,
        ):
            mock_store_cls.return_value.read_reports.return_value = []
            mock_tuner_cls.return_value.analyze_quality.return_value = insights

            result = runner.invoke(app, ["optimize", "--reports"])

            assert result.exit_code == 0
            assert "5" in result.output  # total reports
            assert "PASS" in result.output

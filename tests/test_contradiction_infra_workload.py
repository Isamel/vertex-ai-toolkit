"""Tests for SPEC-V2-AUDIT-10 — Infra → workload causal contradiction rule."""
from __future__ import annotations

import pytest

from vaig.skills.service_health.contradiction_validator import (
    _check_infra_degrades_workload,
    apply_contradiction_rules,
)
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    Severity,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_finding(
    fid: str,
    title: str,
    description: str = "",
    severity: Severity = Severity.HIGH,
    caused_by: list[str] | None = None,
) -> Finding:
    return Finding(
        id=fid,
        title=title,
        description=description,
        severity=severity,
        caused_by=caused_by or [],
    )


def _make_report(findings: list[Finding]) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED,
            scope="test",
            summary_text="test",
        ),
        findings=findings,
    )


# ── tests ─────────────────────────────────────────────────────────────────────


class TestCheckInfraDegrades:
    def test_infra_and_workload_sets_caused_by(self) -> None:
        """When both infra (CNI) and workload (istiod) findings exist, workload gets caused_by."""
        infra = _make_finding("cni-fail", "istio-cni-node DaemonSet not ready")
        workload = _make_finding("istiod-conn", "connection to istiod failed", severity=Severity.HIGH)
        report = _make_report([infra, workload])

        result = _check_infra_degrades_workload(report)

        assert result is not None
        # Workload finding in report should now have caused_by set
        updated = next(f for f in report.findings if f.id == "istiod-conn")
        assert "cni-fail" in updated.caused_by

    def test_emits_medium_contradiction_finding(self) -> None:
        """The emitted finding should be MEDIUM severity with id contradiction-infra-degrades-workload."""
        infra = _make_finding("cni-fail", "cni plugin NetworkNotReady error")
        workload = _make_finding("envoy-err", "envoy proxy upstream request timeout")
        report = _make_report([infra, workload])

        result = _check_infra_degrades_workload(report)

        assert result is not None
        assert result.severity == Severity.MEDIUM
        assert result.id == "contradiction-infra-degrades-workload"

    def test_no_infra_finding_returns_none(self) -> None:
        """When no infra keyword is present, rule returns None."""
        workload = _make_finding("istiod-conn", "connection to istiod failed")
        report = _make_report([workload])

        result = _check_infra_degrades_workload(report)

        assert result is None

    def test_no_workload_finding_returns_none(self) -> None:
        """When no workload keyword is present, rule returns None."""
        infra = _make_finding("cni-fail", "istio-cni-node DaemonSet crash")
        report = _make_report([infra])

        result = _check_infra_degrades_workload(report)

        assert result is None

    def test_workload_already_has_caused_by_skipped(self) -> None:
        """When workload finding already has caused_by, rule does not override it."""
        infra = _make_finding("cni-fail", "anetd CNI plugin crash")
        workload = _make_finding(
            "istiod-conn", "istiod connection lost", caused_by=["other-finding"]
        )
        report = _make_report([infra, workload])

        result = _check_infra_degrades_workload(report)

        assert result is None

    def test_same_finding_is_not_linked_to_itself(self) -> None:
        """A finding matching both infra and workload keywords is not linked to itself."""
        both = _make_finding("combined", "istio-cni-node and istiod error combined")
        report = _make_report([both])

        result = _check_infra_degrades_workload(report)

        assert result is None

    def test_only_low_severity_ignored(self) -> None:
        """LOW severity findings are not considered for this rule."""
        infra = _make_finding("cni-fail", "istio-cni-node warning", severity=Severity.LOW)
        workload = _make_finding("istiod-conn", "istiod warning", severity=Severity.LOW)
        report = _make_report([infra, workload])

        result = _check_infra_degrades_workload(report)

        assert result is None


class TestApplyContradictionRules:
    def test_no_regression_without_infra_keywords(self) -> None:
        """Report without infra/workload keywords is unchanged and returns empty list."""
        finding = _make_finding("generic", "Some unrelated error")
        report = _make_report([finding])

        new_findings = apply_contradiction_rules(report)

        assert new_findings == []
        assert report.findings[0].caused_by == []

    def test_returns_list_of_findings(self) -> None:
        """apply_contradiction_rules returns a list of Finding objects."""
        infra = _make_finding("cni-fail", "calico CNI node error")
        workload = _make_finding("envoy-down", "envoy proxy failing")
        report = _make_report([infra, workload])

        new_findings = apply_contradiction_rules(report)

        assert isinstance(new_findings, list)
        assert all(isinstance(f, Finding) for f in new_findings)
        assert len(new_findings) == 1

"""Tests for SPEC-SH-13 — Contradiction detector."""
from __future__ import annotations

from vaig.skills.service_health.contradiction_validator import (
    _check_apm_catalog_conflict,
    _check_argocd_managed_no_status,
    _check_helm_labels_no_release,
    _check_pods_ready_apm_error,
    detect_contradictions,
)
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    Finding,
    HealthReport,
    OverallStatus,
    ServiceStatus,
    Severity,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_finding(title: str = "Test", description: str = "", category: str = "general", fid: str = "test") -> Finding:
    return Finding(id=fid, title=title, severity=Severity.MEDIUM, category=category, description=description)


def _make_report(
    findings: list[Finding] | None = None,
    service_statuses: list[ServiceStatus] | None = None,
) -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.DEGRADED, scope="test", summary_text="test summary"
        ),
        findings=findings or [],
        service_statuses=service_statuses or [],
    )


def _make_svc(pods_ready: str = "3/3", service: str = "my-svc") -> ServiceStatus:
    return ServiceStatus(service=service, pods_ready=pods_ready)


# ── detect_contradictions (integration) ──────────────────────────────────────


class TestDetectContradictions:
    def test_empty_report_returns_empty(self) -> None:
        report = _make_report()
        assert detect_contradictions(report) == []

    def test_no_contradiction_returns_empty(self) -> None:
        report = _make_report(findings=[_make_finding(title="pod crash")])
        assert detect_contradictions(report) == []

    def test_returns_list_of_findings(self) -> None:
        apm_finding = _make_finding(title="APM traces detected", fid="apm")
        catalog_finding = _make_finding(title="service not registered in catalog", fid="cat")
        report = _make_report(findings=[apm_finding, catalog_finding])
        result = detect_contradictions(report)
        assert isinstance(result, list)
        assert all(isinstance(f, Finding) for f in result)

    def test_multiple_contradictions_all_returned(self) -> None:
        apm_finding = _make_finding(title="APM traces detected", fid="apm")
        catalog_finding = _make_finding(title="no catalog entry", fid="cat")
        helm_label_finding = _make_finding(title="helm label present meta.helm.sh", fid="hl")
        no_release_finding = _make_finding(title="no helm release found", fid="nr")
        report = _make_report(
            findings=[apm_finding, catalog_finding, helm_label_finding, no_release_finding],
            service_statuses=[_make_svc("3/3")],
        )
        result = detect_contradictions(report)
        ids = [f.id for f in result]
        assert "contradiction-apm-catalog-gap" in ids
        assert "contradiction-helm-labels-no-release" in ids


# ── _check_apm_catalog_conflict ───────────────────────────────────────────────


class TestApmCatalogConflict:
    def test_no_apm_returns_none(self) -> None:
        report = _make_report(findings=[_make_finding(title="pod crash")])
        assert _check_apm_catalog_conflict(report) is None

    def test_apm_without_catalog_issue_returns_none(self) -> None:
        report = _make_report(findings=[_make_finding(title="APM traces detected")])
        assert _check_apm_catalog_conflict(report) is None

    def test_apm_with_catalog_missing_returns_finding(self) -> None:
        apm = _make_finding(title="APM traces detected", fid="apm")
        cat = _make_finding(title="service not registered in catalog", fid="cat")
        report = _make_report(findings=[apm, cat])
        result = _check_apm_catalog_conflict(report)
        assert result is not None
        assert result.id == "contradiction-apm-catalog-gap"
        assert result.severity == Severity.MEDIUM
        assert result.category == "contradiction"

    def test_tracing_keyword_triggers_check(self) -> None:
        apm = _make_finding(description="opentelemetry tracing configured", fid="apm")
        cat = _make_finding(title="missing catalog entry", fid="cat")
        report = _make_report(findings=[apm, cat])
        result = _check_apm_catalog_conflict(report)
        assert result is not None


# ── _check_pods_ready_apm_error ───────────────────────────────────────────────


class TestPodsReadyApmError:
    def test_no_error_rate_returns_none(self) -> None:
        report = _make_report(service_statuses=[_make_svc("3/3")])
        assert _check_pods_ready_apm_error(report) is None

    def test_high_error_without_ready_pods_returns_none(self) -> None:
        err = _make_finding(title="high error rate spike", fid="err")
        report = _make_report(
            findings=[err],
            service_statuses=[_make_svc("0/3")],
        )
        assert _check_pods_ready_apm_error(report) is None

    def test_high_error_with_all_ready_pods_returns_finding(self) -> None:
        err = _make_finding(title="high error rate spike", fid="err")
        report = _make_report(
            findings=[err],
            service_statuses=[_make_svc("3/3")],
        )
        result = _check_pods_ready_apm_error(report)
        assert result is not None
        assert result.id == "contradiction-pods-ready-apm-error"
        assert result.severity == Severity.HIGH

    def test_0_of_0_pods_does_not_trigger(self) -> None:
        """0/0 is not 'all ready' — no pods exist."""
        err = _make_finding(title="high error rate", fid="err")
        report = _make_report(
            findings=[err],
            service_statuses=[_make_svc("0/0")],
        )
        assert _check_pods_ready_apm_error(report) is None

    def test_na_pods_ready_does_not_trigger(self) -> None:
        err = _make_finding(title="error rate elevated", fid="err")
        report = _make_report(
            findings=[err],
            service_statuses=[_make_svc("N/A")],
        )
        assert _check_pods_ready_apm_error(report) is None


# ── _check_helm_labels_no_release ─────────────────────────────────────────────


class TestHelmLabelsNoRelease:
    def test_no_helm_signal_returns_none(self) -> None:
        report = _make_report(findings=[_make_finding(title="pod crash")])
        assert _check_helm_labels_no_release(report) is None

    def test_helm_label_only_returns_none(self) -> None:
        f = _make_finding(title="meta.helm.sh label found", fid="hl")
        report = _make_report(findings=[f])
        assert _check_helm_labels_no_release(report) is None

    def test_helm_label_with_no_release_returns_finding(self) -> None:
        hl = _make_finding(title="meta.helm.sh annotation present", fid="hl")
        nr = _make_finding(title="no helm release found", fid="nr")
        report = _make_report(findings=[hl, nr])
        result = _check_helm_labels_no_release(report)
        assert result is not None
        assert result.id == "contradiction-helm-labels-no-release"
        assert result.severity == Severity.LOW

    def test_chart_keyword_triggers_helm_check(self) -> None:
        chart = _make_finding(description="helm.sh/chart annotation on deployment", fid="ch")
        no_rel = _make_finding(title="helm release absent", fid="nr")
        report = _make_report(findings=[chart, no_rel])
        result = _check_helm_labels_no_release(report)
        assert result is not None


# ── _check_argocd_managed_no_status ──────────────────────────────────────────


class TestArgoCdManagedNoStatus:
    def test_no_argocd_signal_returns_none(self) -> None:
        report = _make_report(findings=[_make_finding(title="pod crash")])
        assert _check_argocd_managed_no_status(report) is None

    def test_argocd_without_missing_status_returns_none(self) -> None:
        f = _make_finding(title="argocd app synced", fid="a")
        report = _make_report(findings=[f])
        assert _check_argocd_managed_no_status(report) is None

    def test_argocd_with_missing_status_returns_finding(self) -> None:
        argo = _make_finding(title="argocd application found", fid="argo")
        ns = _make_finding(title="sync status missing for app", fid="ns")
        report = _make_report(findings=[argo, ns])
        result = _check_argocd_managed_no_status(report)
        assert result is not None
        assert result.id == "contradiction-argocd-managed-no-status"
        assert result.severity == Severity.MEDIUM

    def test_gitops_keyword_triggers_check(self) -> None:
        gitops = _make_finding(description="gitops pipeline detected", fid="gp")
        ns = _make_finding(title="no sync status found", fid="ns")
        report = _make_report(findings=[gitops, ns])
        result = _check_argocd_managed_no_status(report)
        assert result is not None

    def test_result_has_remediation(self) -> None:
        argo = _make_finding(title="argocd application found", fid="argo")
        ns = _make_finding(title="argocd status absent", fid="ns")
        report = _make_report(findings=[argo, ns])
        result = _check_argocd_managed_no_status(report)
        assert result is not None
        assert result.remediation is not None and len(result.remediation) > 0

"""Contradiction detector for service health reports (SPEC-SH-13).

Detects logical contradictions between findings and service statuses in a
:class:`~vaig.skills.service_health.schema.HealthReport` and emits additional
:class:`~vaig.skills.service_health.schema.Finding` entries to surface the gaps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vaig.skills.service_health.schema import Confidence, Finding, Severity

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import HealthReport

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def _finding_has_pattern(finding: Finding, *patterns: str) -> bool:
    """Return True if any of *patterns* appears (case-insensitive) in the
    finding's title, description, category, or root_cause."""
    haystack = " ".join(
        [finding.title, finding.description, finding.category, finding.root_cause]
    ).lower()
    return any(p.lower() in haystack for p in patterns)


def _any_finding(findings: list[Finding], *patterns: str) -> bool:
    """Return True when at least one finding in *findings* matches any pattern."""
    return any(_finding_has_pattern(f, *patterns) for f in findings)


# ── private checkers ──────────────────────────────────────────────────────────


def _check_apm_catalog_conflict(report: HealthReport) -> Finding | None:
    """Detect APM-present but no service-catalog registration.

    Signal: findings mention APM / tracing **and** findings mention
    "no catalog", "not registered", "missing catalog", or "service catalog".
    """
    findings = report.findings
    has_apm = _any_finding(findings, "apm", "trace", "tracing", "datadog apm", "opentelemetry")
    has_no_catalog = _any_finding(
        findings, "no catalog", "not registered", "missing catalog", "service catalog"
    )
    if not (has_apm and has_no_catalog):
        return None

    logger.debug("Contradiction detected: APM present but service not registered in catalog")
    return Finding(
        id="contradiction-apm-catalog-gap",
        title="APM instrumented but service not registered in catalog",
        severity=Severity.MEDIUM,
        category="contradiction",
        description=(
            "APM/tracing signals were detected for this service, but the service is missing "
            "from the service catalog.  This is a registration gap — traces will not be "
            "correlated with SLOs or alerts defined in the catalog."
        ),
        root_cause="Service was instrumented for APM before being registered in the service catalog.",
        evidence=[
            "Finding mentions APM/tracing signals.",
            "Finding mentions absence of service catalog entry.",
        ],
        confidence=Confidence.MEDIUM,
        remediation="Register the service in the catalog and link the APM instrumentation key.",
    )


def _check_pods_ready_apm_error(report: HealthReport) -> Finding | None:
    """Detect pods-ready contradiction with high error rate in APM.

    Signal: service_statuses show all pods ready (e.g. "3/3") **and**
    findings mention high error rate / error rate spike.
    """
    findings = report.findings
    has_high_error = _any_finding(
        findings, "high error rate", "error rate spike", "error rate elevated", "error spike"
    )
    if not has_high_error:
        return None

    # Check if at least one service shows all pods ready.
    pods_all_ready = False
    for svc in report.service_statuses:
        pr = svc.pods_ready
        if pr and pr != "N/A" and "/" in pr:
            parts = pr.split("/")
            if len(parts) == 2 and parts[0].strip() == parts[1].strip() and parts[0].strip() != "0":  # noqa: SIM102
                pods_all_ready = True
                break

    if not pods_all_ready:
        return None

    logger.debug("Contradiction detected: pods ready but APM shows high error rate")
    return Finding(
        id="contradiction-pods-ready-apm-error",
        title="All pods ready but APM reports high error rate",
        severity=Severity.HIGH,
        category="contradiction",
        description=(
            "Service pods are all in Ready state, yet APM data shows an elevated error rate. "
            "This contradiction suggests an application-level fault that does not crash pods — "
            "e.g. a broken dependency, a degraded downstream service, or a configuration error."
        ),
        root_cause=(
            "Application is running (pods healthy) but returning errors, pointing to a "
            "logic/config/dependency issue rather than infrastructure failure."
        ),
        evidence=[
            "Service status table shows all pods ready.",
            "APM finding reports high error rate.",
        ],
        confidence=Confidence.HIGH,
        remediation=(
            "Inspect application logs and APM traces for error patterns.  "
            "Check downstream dependencies and recent configuration changes."
        ),
    )


def _check_helm_labels_no_release(report: HealthReport) -> Finding | None:
    """Detect Helm labels present but no Helm release found.

    Signal: findings mention helm labels / helm annotations **and** findings
    mention "no release", "release not found", "helm release missing".
    """
    findings = report.findings
    has_helm_labels = _any_finding(
        findings, "helm label", "helm annotation", "meta.helm.sh", "helm.sh/chart"
    )
    has_no_release = _any_finding(
        findings,
        "no release",
        "release not found",
        "helm release missing",
        "no helm release",
        "release absent",
    )
    if not (has_helm_labels and has_no_release):
        return None

    logger.debug("Contradiction detected: Helm labels present but no Helm release")
    return Finding(
        id="contradiction-helm-labels-no-release",
        title="Helm labels present but no Helm release found",
        severity=Severity.LOW,
        category="contradiction",
        description=(
            "Workload resources carry Helm management labels/annotations, but no corresponding "
            "Helm release was found.  This typically means the chart was applied manually after "
            "a failed release, or the Helm secret was deleted."
        ),
        root_cause=(
            "Helm release secret is absent or was deleted, leaving orphaned resources with "
            "stale Helm labels."
        ),
        evidence=[
            "Workload carries meta.helm.sh/release-name or helm.sh/chart annotation.",
            "helm_list_releases / helm_release_status returned no matching release.",
        ],
        confidence=Confidence.LOW,
        remediation=(
            "Re-install or upgrade the Helm chart to re-create the release secret, "
            "or remove stale Helm labels if the chart is no longer managed by Helm."
        ),
    )


def _check_argocd_managed_no_status(report: HealthReport) -> Finding | None:
    """Detect ArgoCD-managed resource with no sync status.

    Signal: findings mention argocd / gitops **and** findings mention
    "no status", "status not found", "sync status missing", "argocd status absent".
    """
    findings = report.findings
    has_argocd = _any_finding(findings, "argocd", "gitops", "argo cd", "application.argoproj.io")
    has_no_status = _any_finding(
        findings,
        "no status",
        "status not found",
        "sync status missing",
        "argocd status absent",
        "no sync status",
        "missing status",
    )
    if not (has_argocd and has_no_status):
        return None

    logger.debug("Contradiction detected: ArgoCD managed but no sync status")
    return Finding(
        id="contradiction-argocd-managed-no-status",
        title="ArgoCD-managed workload but sync status unavailable",
        severity=Severity.MEDIUM,
        category="contradiction",
        description=(
            "The workload appears to be managed by ArgoCD, but no sync status was found.  "
            "This may indicate the ArgoCD application object is missing, the resource is "
            "orphaned, or the ArgoCD API server is unreachable.  "
            "A re-gather with ArgoCD tools enabled is recommended."
        ),
        root_cause=(
            "ArgoCD application object is absent or the ArgoCD API server was not queried "
            "during data collection."
        ),
        evidence=[
            "Workload or findings reference ArgoCD / GitOps management.",
            "No ArgoCD sync status was returned by tool calls.",
        ],
        confidence=Confidence.MEDIUM,
        remediation=(
            "Verify the ArgoCD Application object exists in the argocd namespace.  "
            "Re-run the investigation with ArgoCD tools explicitly enabled."
        ),
    )


# ── public API ────────────────────────────────────────────────────────────────


def detect_contradictions(report: HealthReport) -> list[Finding]:
    """Detect logical contradictions in *report* and return new findings.

    Runs all registered contradiction checkers against the report.
    Each checker returns either a :class:`~vaig.skills.service_health.schema.Finding`
    or ``None``.  Only non-None results are included in the output.

    Args:
        report: A fully-populated :class:`~vaig.skills.service_health.schema.HealthReport`
            as produced by the health-reporter agent.

    Returns:
        A (possibly empty) list of new :class:`~vaig.skills.service_health.schema.Finding`
        objects describing detected contradictions.  These should be *appended* to
        ``report.findings`` by the caller.
    """
    checkers = [
        _check_apm_catalog_conflict,
        _check_pods_ready_apm_error,
        _check_helm_labels_no_release,
        _check_argocd_managed_no_status,
    ]
    contradictions: list[Finding] = []
    for checker in checkers:
        try:
            result = checker(report)
        except Exception:  # noqa: BLE001
            logger.warning("Contradiction checker %s failed (non-fatal)", checker.__name__, exc_info=True)
            continue
        if result is not None:
            contradictions.append(result)

    if contradictions:
        logger.info(
            "Contradiction detector found %d contradiction(s): %s",
            len(contradictions),
            [f.id for f in contradictions],
        )
    return contradictions

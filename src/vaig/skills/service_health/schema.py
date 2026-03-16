"""Pydantic v2 models for structured service health reports.

These models define the schema for Gemini's structured output mode
(``response_schema`` parameter).  The root ``HealthReport`` model
can be:

* Passed as a class to ``types.GenerateContentConfig(response_schema=HealthReport)``
* Validated from JSON via ``HealthReport.model_validate_json(raw)``
* Rendered to Markdown via ``report.to_markdown()``
* Serialised to a dict via ``report.to_dict()``

All enum fields use ``StrEnum`` so they serialise as plain strings,
which is required for Gemini's ``response_schema`` compatibility.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────


class OverallStatus(StrEnum):
    """Top-level cluster / scope health status."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class Severity(StrEnum):
    """Finding severity — matches the reporter prompt severity scale."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Confidence(StrEnum):
    """Confidence level for findings and root-cause hypotheses."""

    CONFIRMED = "CONFIRMED"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Effort(StrEnum):
    """Estimated effort for a recommended action."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ActionUrgency(StrEnum):
    """Time horizon for recommended actions."""

    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM = "LONG_TERM"


class ServiceHealthStatus(StrEnum):
    """Health status for individual services in the status table."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


# ── Severity → emoji mapping (matches reporter prompt) ───────


_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "🟢",
}

_SEVERITY_LABEL: dict[Severity, str] = {
    Severity.CRITICAL: "Critical",
    Severity.HIGH: "High",
    Severity.MEDIUM: "Medium",
    Severity.LOW: "Low",
    Severity.INFO: "Informational",
}

_STATUS_EMOJI: dict[ServiceHealthStatus, str] = {
    ServiceHealthStatus.HEALTHY: "🟢",
    ServiceHealthStatus.DEGRADED: "🟡",
    ServiceHealthStatus.FAILED: "🔴",
    ServiceHealthStatus.UNKNOWN: "⚪",
}


# ── Sub-models ───────────────────────────────────────────────


class ExecutiveSummary(BaseModel):
    """Executive summary section of the health report."""

    overall_status: OverallStatus
    scope: str = Field(
        description=(
            "Blast radius: 'Cluster-wide', 'Namespace: <name>', "
            "or 'Resource: <type>/<name> in <namespace>'"
        ),
    )
    summary_text: str = Field(description="1-2 sentence overview of the situation")
    services_checked: int = Field(default=0, ge=0)
    issues_found: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)


class ClusterMetric(BaseModel):
    """A single row in the Cluster Overview table."""

    metric: str
    value: str


class ServiceStatus(BaseModel):
    """A single row in the Service Status table."""

    service: str
    namespace: str = ""
    status: ServiceHealthStatus = ServiceHealthStatus.UNKNOWN
    pods_ready: str = Field(default="N/A", description="e.g. '3/3'")
    restarts_1h: str = Field(default="N/A", description="Restart count in last hour")
    cpu_usage: str = Field(default="N/A")
    memory_usage: str = Field(default="N/A")
    issues: str = Field(default="")


class Finding(BaseModel):
    """An individual health finding (issue or observation)."""

    id: str = Field(description="Slug identifier, e.g. 'crashloop-payment-svc'")
    title: str
    severity: Severity
    category: str = Field(default="", description="e.g. 'pod-health', 'scaling', 'networking'")
    service: str = Field(default="")
    description: str = Field(default="", description="What is happening")
    root_cause: str = Field(default="", description="The causal mechanism")
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    impact: str = Field(default="")
    affected_resources: list[str] = Field(default_factory=list)
    remediation: str | None = None


class DowngradedFinding(BaseModel):
    """A finding that was downgraded during the verification pass."""

    title: str
    original_confidence: Confidence = Confidence.MEDIUM
    final_confidence: Confidence = Confidence.LOW
    reason: str = ""


class RootCauseHypothesis(BaseModel):
    """A root-cause hypothesis for a critical/high/medium finding."""

    finding_title: str
    mechanism: str = Field(description="Chain of events that produced the issue")
    confidence: Confidence = Confidence.MEDIUM
    supporting_evidence: list[str] = Field(default_factory=list)
    what_would_confirm: str = Field(default="N/A")


class EvidenceDetail(BaseModel):
    """Structured evidence detail with optional YAML/code blocks."""

    title: str
    description: str = ""
    evidence_text: str = Field(default="", description="Raw evidence (code/YAML block content)")
    corrected_text: str = Field(default="", description="Corrected version if applicable")


class RecommendedAction(BaseModel):
    """A recommended remediation action."""

    priority: int = Field(ge=1, description="1 = highest priority")
    title: str
    description: str = ""
    urgency: ActionUrgency = ActionUrgency.SHORT_TERM
    effort: Effort = Effort.MEDIUM
    command: str = Field(default="", description="Exact kubectl / gcloud command")
    why: str = Field(default="")
    risk: str = Field(default="")
    related_findings: list[str] = Field(
        default_factory=list,
        description="References to Finding.id values",
    )


class ManualInvestigation(BaseModel):
    """A finding that could not be automatically verified."""

    finding_title: str
    reason: str = Field(default="", description="What tool call failed")
    investigation_steps: str = Field(default="", description="Manual steps to verify")


class TimelineEvent(BaseModel):
    """A single event in the chronological timeline."""

    time: str = Field(description="Timestamp — relative ('7m ago') or absolute ISO 8601")
    event: str
    severity: Severity = Severity.INFO


class ReportMetadata(BaseModel):
    """Metadata about how and when the report was generated."""

    generated_at: str = Field(default="", description="ISO 8601 timestamp")
    cluster_name: str = Field(default="")
    project_id: str = Field(default="")
    model_used: str = Field(default="")
    skill_version: str = Field(default="")


# ── Root model ───────────────────────────────────────────────


class HealthReport(BaseModel):
    """Root model for a structured service health report.

    Mirrors the mandatory report structure defined in the reporter
    prompt (``HEALTH_REPORTER_PROMPT``).  Every section in the prompt
    maps to a field here.
    """

    executive_summary: ExecutiveSummary
    cluster_overview: list[ClusterMetric] = Field(default_factory=list)
    service_statuses: list[ServiceStatus] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    downgraded_findings: list[DowngradedFinding] = Field(default_factory=list)
    root_cause_hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)
    evidence_details: list[EvidenceDetail] = Field(default_factory=list)
    recommendations: list[RecommendedAction] = Field(default_factory=list)
    manual_investigations: list[ManualInvestigation] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)

    # ── Serialisation helpers ────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-friendly dict serialisation (delegates to Pydantic)."""
        return self.model_dump()

    # ── Markdown rendering ───────────────────────────────────

    def to_markdown(self) -> str:
        """Render the report as Markdown matching the reporter prompt format.

        The output follows the MANDATORY report structure defined in
        ``HEALTH_REPORTER_PROMPT``: Executive Summary → Cluster Overview →
        Service Status → Findings (by severity) → Downgraded Findings →
        Root Cause Hypotheses → Evidence Details → Recommended Actions →
        Manual Investigation Required → Timeline.
        """
        parts: list[str] = []
        parts.append("# Service Health Report")
        parts.append("")
        self._render_executive_summary(parts)
        self._render_cluster_overview(parts)
        self._render_service_status(parts)
        self._render_findings(parts)
        self._render_downgraded_findings(parts)
        self._render_root_cause_hypotheses(parts)
        self._render_evidence_details(parts)
        self._render_recommendations(parts)
        self._render_manual_investigations(parts)
        self._render_timeline(parts)
        return "\n".join(parts)

    # ── Private rendering helpers ────────────────────────────

    def _render_executive_summary(self, parts: list[str]) -> None:
        es = self.executive_summary
        parts.append("## Executive Summary")
        parts.append(f"- **Status**: {es.overall_status.value}")
        parts.append(f"- **Scope**: {es.scope}")
        parts.append(f"- **Summary**: {es.summary_text}")
        parts.append("")

    def _render_cluster_overview(self, parts: list[str]) -> None:
        parts.append("## Cluster Overview")
        if not self.cluster_overview:
            parts.append(
                "Cluster overview data was not collected by the diagnostic pipeline. "
                "Run `kubectl get nodes` and `kubectl top nodes` for manual assessment."
            )
            parts.append("")
            return
        parts.append("| Metric | Value |")
        parts.append("|--------|-------|")
        for row in self.cluster_overview:
            parts.append(f"| {row.metric} | {row.value} |")
        parts.append("")

    def _render_service_status(self, parts: list[str]) -> None:
        parts.append("## Service Status")
        parts.append("")
        if not self.service_statuses:
            parts.append("No service status data available.")
            parts.append("")
            return
        parts.append(
            "| Service | Namespace | Status | Pods Ready "
            "| Restarts (1h) | CPU Usage | Memory Usage | Issues |"
        )
        parts.append(
            "|---------|-----------|--------|------------"
            "|---------------|-----------|--------------|--------|"
        )
        for svc in self.service_statuses:
            emoji = _STATUS_EMOJI.get(svc.status, "⚪")
            parts.append(
                f"| {svc.service} | {svc.namespace} | {emoji} | {svc.pods_ready} "
                f"| {svc.restarts_1h} | {svc.cpu_usage} | {svc.memory_usage} | {svc.issues} |"
            )
        parts.append("")

    def _render_findings(self, parts: list[str]) -> None:
        parts.append("## Findings")
        parts.append("")

        if not self.findings:
            parts.append("No findings to report.")
            parts.append("")
            return

        # Group findings by severity in the mandated order
        severity_order = [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]

        grouped: dict[Severity, list[Finding]] = {s: [] for s in severity_order}
        for f in self.findings:
            grouped[f.severity].append(f)

        for severity in severity_order:
            findings_at_level = grouped[severity]
            if not findings_at_level:
                continue

            emoji = _SEVERITY_EMOJI[severity]
            label = _SEVERITY_LABEL[severity]
            parts.append(f"### {emoji} {label}")
            parts.append("")

            if severity in (Severity.LOW, Severity.INFO):
                # Low and Info use bullet-point format per the prompt
                for f in findings_at_level:
                    line = f"- **{f.title}**"
                    if f.description:
                        line += f": {f.description}"
                    if f.evidence:
                        line += f" — Evidence: {'; '.join(f.evidence)}"
                    parts.append(line)
                parts.append("")
            else:
                # Critical, High, Medium use full structured format
                for f in findings_at_level:
                    parts.append(f"#### {f.title}")
                    if f.description:
                        parts.append(f"- **What**: {f.description}")
                    if f.root_cause:
                        parts.append(f"- **Root Cause**: {f.root_cause}")
                    if f.evidence:
                        parts.append(f"- **Evidence**: {'; '.join(f.evidence)}")
                    parts.append(f"- **Confidence**: {f.confidence.value}")
                    if f.impact:
                        parts.append(f"- **Impact**: {f.impact}")
                    if f.affected_resources:
                        parts.append(f"- **Affected Resources**: {', '.join(f.affected_resources)}")
                    parts.append("")

    def _render_downgraded_findings(self, parts: list[str]) -> None:
        parts.append("## Downgraded Findings")
        if not self.downgraded_findings:
            parts.append(
                "No findings were downgraded during verification "
                "— all findings maintained or increased confidence."
            )
            parts.append("")
            return

        parts.append("| Finding | Original Confidence | Final Confidence | Reason for Downgrade |")
        parts.append("|---------|---------------------|------------------|----------------------|")
        for df in self.downgraded_findings:
            parts.append(
                f"| {df.title} | {df.original_confidence} "
                f"| {df.final_confidence} | {df.reason} |"
            )
        parts.append("")

    def _render_root_cause_hypotheses(self, parts: list[str]) -> None:
        parts.append("## Root Cause Hypotheses")
        parts.append("")

        if not self.root_cause_hypotheses:
            parts.append("No root cause hypotheses to report.")
            parts.append("")
            return

        for hyp in self.root_cause_hypotheses:
            parts.append(f"#### {hyp.finding_title}")
            parts.append(f"- **Mechanism**: {hyp.mechanism}")
            parts.append(f"- **Confidence**: {hyp.confidence.value}")
            if hyp.supporting_evidence:
                parts.append(f"- **Supporting Evidence**: {'; '.join(hyp.supporting_evidence)}")
            parts.append(f"- **What Would Confirm This**: {hyp.what_would_confirm}")
            parts.append("")

    def _render_evidence_details(self, parts: list[str]) -> None:
        if not self.evidence_details:
            return

        parts.append("## Evidence Details")
        parts.append("")

        for ev in self.evidence_details:
            parts.append(f"### {ev.title}")
            if ev.description:
                parts.append(ev.description)
            if ev.evidence_text:
                parts.append("```")
                parts.append(ev.evidence_text)
                parts.append("```")
            if ev.corrected_text:
                parts.append("")
                parts.append("**Corrected**:")
                parts.append("```")
                parts.append(ev.corrected_text)
                parts.append("```")
            parts.append("")

    def _render_recommendations(self, parts: list[str]) -> None:
        parts.append("## Recommended Actions")
        parts.append("")

        if not self.recommendations:
            parts.append("No recommended actions at this time.")
            parts.append("")
            return

        urgency_order = [
            ActionUrgency.IMMEDIATE,
            ActionUrgency.SHORT_TERM,
            ActionUrgency.LONG_TERM,
        ]
        urgency_labels: dict[ActionUrgency, str] = {
            ActionUrgency.IMMEDIATE: "Immediate (next 5 minutes)",
            ActionUrgency.SHORT_TERM: "Short-term (next 1 hour)",
            ActionUrgency.LONG_TERM: "Long-term (next sprint)",
        }

        grouped: dict[ActionUrgency, list[RecommendedAction]] = {u: [] for u in urgency_order}
        for action in self.recommendations:
            grouped[action.urgency].append(action)

        for urgency in urgency_order:
            actions = sorted(grouped[urgency], key=lambda a: a.priority)
            if not actions:
                continue

            parts.append(f"### {urgency_labels[urgency]}")
            for action in actions:
                parts.append(f"{action.priority}. {action.title}")
                if action.command:
                    parts.append("   ```")
                    parts.append(f"   {action.command}")
                    parts.append("   ```")
                if action.why:
                    parts.append(f"   - Why: {action.why}")
                if action.risk:
                    parts.append(f"   - Risk: {action.risk}")
            parts.append("")

    def _render_manual_investigations(self, parts: list[str]) -> None:
        if not self.manual_investigations:
            return

        parts.append("### Manual Investigation Required")
        for mi in self.manual_investigations:
            line = f"- **{mi.finding_title}**"
            if mi.reason:
                line += f": {mi.reason}"
            if mi.investigation_steps:
                line += f" — {mi.investigation_steps}"
            parts.append(line)
        parts.append("")

    def _render_timeline(self, parts: list[str]) -> None:
        parts.append("## Timeline")
        if not self.timeline:
            parts.append("No timeline events available.")
            parts.append("")
            return

        parts.append("| Time | Event | Severity |")
        parts.append("|------|-------|----------|")
        for ev in self.timeline:
            parts.append(f"| {ev.time} | {ev.event} | {ev.severity.value} |")
        parts.append("")

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

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# ── Text-normalisation regexes (for smart timeline collapse) ──

_POD_HASH_RE = re.compile(r'-[a-z0-9]{5,10}-[a-z0-9]{4,7}\b')   # strip -59967f9ccc-4zdx6
_COUNTER_RE = re.compile(r'\s*\(\d+(?:st|nd|rd|th)\s+time\)')    # strip "(3rd time)"
_TIMESTAMP_RE = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z')  # strip ISO timestamps
_IP_RE = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?\b')  # strip IPs / IP:port
_UUID_RE = re.compile(                                             # strip UUIDs
    r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
    re.IGNORECASE,
)

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

_OVERALL_STATUS_EMOJI: dict[OverallStatus, str] = {
    OverallStatus.HEALTHY: "🟢",
    OverallStatus.DEGRADED: "🟡",
    OverallStatus.CRITICAL: "🔴",
    OverallStatus.UNKNOWN: "⚪",
}


class ContentType(StrEnum):
    """Content type hint for evidence code fences in Markdown rendering."""

    YAML = "yaml"
    JSON = "json"
    LOG = "log"
    TEXT = "text"
    COMMAND = "command"
    UNKNOWN = "unknown"


CONTENT_TYPE_FENCE_MAP: dict[ContentType, str] = {
    ContentType.YAML: "yaml",
    ContentType.JSON: "json",
    ContentType.LOG: "log",
    ContentType.TEXT: "text",
    ContentType.COMMAND: "bash",
    ContentType.UNKNOWN: "text",
}


# ── Enum coercion factory ─────────────────────────────────────


def _make_enum_coercer(enum_class: type, default_value: object) -> Callable[[object], object]:
    """Return a Pydantic v2 field-validator-compatible coercion function.

    Produces a classmethod that:
    * Returns the value unchanged when it is already an instance of
      *enum_class*.
    * Tries ``enum_class(str(v).lower())`` for string values, falling
      back to *default_value* on ``ValueError``.

    Usage inside a Pydantic model::

        @field_validator("my_field", mode="before")
        @classmethod
        def coerce_my_field(cls, v):
            return _make_enum_coercer(MyEnum, MyEnum.DEFAULT)(v)

    Or via direct assignment (bypasses the decorator boilerplate entirely)::

        _coerce_severity = field_validator("severity", mode="before")(
            classmethod(_make_enum_coercer(Severity, Severity.INFO))
        )

    Args:
        enum_class: The ``StrEnum`` (or any ``Enum``) subclass.
        default_value: Value returned when *v* cannot be coerced.

    Returns:
        A coercion callable ``(v: object) -> object``.
    """
    def _coerce(v: object) -> object:
        if isinstance(v, enum_class):
            return v
        if isinstance(v, str):
            # Try the value as-is first, then uppercase and lowercase variants
            for candidate in (v, v.upper(), v.lower()):
                try:
                    return enum_class(candidate)
                except ValueError:
                    pass
            logger.debug(
                "Unknown %s value %r — coercing to %r",
                enum_class.__name__, v, default_value,
            )
            return default_value
        return v

    return _coerce


# ── Smart timeline collapse helpers ──────────────────────────


def _normalize_event_text(text: str) -> str:
    """Return a normalised form of *text* used only for grouping comparisons.

    Strips volatile tokens (pod hashes, embedded timestamps, occurrence
    counters, IPv4 addresses, and UUIDs) so that semantically identical
    events compare equal even when their raw text differs slightly.
    """
    normalised = _POD_HASH_RE.sub('', text)
    normalised = _COUNTER_RE.sub('', normalised)
    normalised = _TIMESTAMP_RE.sub('', normalised)
    normalised = _IP_RE.sub('<IP>', normalised)
    normalised = _UUID_RE.sub('<UUID>', normalised)
    return normalised.strip()


@dataclass
class _CollapsedEvent:
    """Internal view of a (potentially collapsed) group of timeline events.

    This is an INTERNAL dataclass — never passed to Gemini or exposed via
    ``response_schema``.  The Pydantic ``TimelineEvent`` model is kept
    unchanged for full API compatibility.
    """

    time_first: str
    time_last: str
    event: str          # original (non-normalised) text from the first occurrence
    normalized_event: str  # normalised text, used for display when count > 1
    severity: Severity
    service: str
    count: int

    @property
    def display_event(self) -> str:
        """Return the event text for rendering, with ×N notation when collapsed.

        When count == 1, shows the raw original text.
        When count > 1, uses the normalized text (volatile tokens stripped)
        so the collapsed entry doesn't misleadingly show data from the first
        occurrence only (e.g. a specific pod hash or timestamp).
        """
        if self.count == 1:
            return self.event
        return f"{self.normalized_event} (×{self.count}, {self.time_first} → {self.time_last})"

    @property
    def display_time(self) -> str:
        """Return the time value for rendering (always the first-seen time)."""
        return self.time_first


def _collapse_repeated_events(events: list[TimelineEvent]) -> list[_CollapsedEvent]:
    """Collapse semantically identical *consecutive* timeline events.

    Grouping key: (normalised event text, severity, service).  Only
    back-to-back runs of matching events are merged — non-consecutive
    repetitions are kept as separate entries to preserve chronological
    order.  For example::

        [A, B, A]       → [A, B, A]   (3 entries — not consecutive)
        [A, A, A, B, B] → [A×3, B×2] (2 entries — each is a consecutive run)

    The ``normalized_event`` field of every resulting ``_CollapsedEvent``
    holds the de-volatilised text; ``event`` holds the raw first-occurrence
    text.  ``display_event`` picks which one to show (see property docs).
    """
    result: list[_CollapsedEvent] = []
    for ev in events:
        norm = _normalize_event_text(ev.event)
        key = (norm, ev.severity, ev.service)
        if result:
            prev = result[-1]
            prev_key = (_normalize_event_text(prev.event), prev.severity, prev.service)
            if key == prev_key:
                prev.count += 1
                prev.time_last = ev.time
                continue
        result.append(_CollapsedEvent(
            time_first=ev.time,
            time_last=ev.time,
            event=ev.event,
            normalized_event=norm,
            severity=ev.severity,
            service=ev.service,
            count=1,
        ))
    return result


# ── Sub-models ───────────────────────────────────────────────


class ExecutiveSummary(BaseModel):
    """Executive summary section of the health report."""

    model_config = ConfigDict(extra="ignore")

    overall_status: OverallStatus

    @field_validator("overall_status", mode="before")
    @classmethod
    def coerce_overall_status(cls, v: object) -> object:
        if isinstance(v, str):
            upper = v.upper()
            try:
                return OverallStatus(upper)
            except ValueError:
                logger.debug("Unknown overall_status %r — coercing to UNKNOWN", v)
                return OverallStatus.UNKNOWN
        return v

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

    model_config = ConfigDict(extra="ignore")

    metric: str
    value: str


class ServiceStatus(BaseModel):
    """A single row in the Service Status table."""

    model_config = ConfigDict(extra="ignore")

    service: str
    namespace: str = ""
    status: ServiceHealthStatus = ServiceHealthStatus.UNKNOWN

    @field_validator("status", mode="before")
    @classmethod
    def coerce_service_health_status(cls, v: object) -> object:
        if isinstance(v, str):
            upper = v.upper()
            try:
                return ServiceHealthStatus(upper)
            except ValueError:
                logger.debug("Unknown service health status %r — coercing to UNKNOWN", v)
                return ServiceHealthStatus.UNKNOWN
        return v

    pods_ready: str = Field(default="N/A", description="e.g. '3/3'")
    restarts_1h: str = Field(default="N/A", description="Restart count in last hour")
    cpu_usage: str = Field(default="N/A")
    memory_usage: str = Field(default="N/A")
    issues: str = Field(default="")

    # Argo Rollouts enrichment fields (optional — only present when Argo Rollouts manages
    # the workload; absent for standard Deployment-based services)
    rollout_strategy: str | None = Field(
        default=None,
        description=(
            "Argo Rollout strategy in canonical hyphenated form: 'canary' | 'blue-green' | None. "
            "Note: tool output may show 'blueGreen' (camelCase); always normalize to 'blue-green'."
        ),
    )
    rollout_status: str | None = Field(
        default=None,
        description="Argo Rollout phase mapped to: 'Healthy' | 'Progressing' | 'Paused' | 'Degraded' | None",
    )
    hpa_conditions: list[str] = Field(
        default_factory=list,
        description="HPA status.conditions messages when HPA scaleTargetRef points to a Rollout",
    )


class Finding(BaseModel):
    """An individual health finding (issue or observation)."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Slug identifier, e.g. 'crashloop-payment-svc'")
    title: str
    severity: Severity

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: object) -> object:
        return _make_enum_coercer(Severity, Severity.INFO)(v)

    category: str = Field(default="", description="e.g. 'pod-health', 'scaling', 'networking'")
    service: str = Field(default="")
    description: str = Field(default="", description="What is happening")
    root_cause: str = Field(default="", description="The causal mechanism")
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: object) -> object:
        return _make_enum_coercer(Confidence, Confidence.MEDIUM)(v)

    impact: str = Field(default="")
    affected_resources: list[str] = Field(default_factory=list)
    remediation: str | None = None


class DowngradedFinding(BaseModel):
    """A finding that was downgraded during the verification pass."""

    model_config = ConfigDict(extra="ignore")

    title: str
    original_confidence: Confidence = Confidence.MEDIUM
    final_confidence: Confidence = Confidence.LOW

    @field_validator("original_confidence", "final_confidence", mode="before")
    @classmethod
    def coerce_downgraded_confidence(cls, v: object) -> object:
        return _make_enum_coercer(Confidence, Confidence.MEDIUM)(v)

    reason: str = ""


class RootCauseHypothesis(BaseModel):
    """A root-cause hypothesis for a critical/high/medium finding."""

    model_config = ConfigDict(extra="ignore")

    finding_title: str
    mechanism: str = Field(description="Chain of events that produced the issue")
    confidence: Confidence = Confidence.MEDIUM

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: object) -> object:
        return _make_enum_coercer(Confidence, Confidence.MEDIUM)(v)

    supporting_evidence: list[str] = Field(default_factory=list)
    what_would_confirm: str = Field(default="N/A")


class EvidenceDetail(BaseModel):
    """Structured evidence detail with optional YAML/code blocks."""

    model_config = ConfigDict(extra="ignore")

    title: str
    description: str = ""
    evidence_text: str = Field(default="", description="Raw evidence (code/YAML block content)")
    corrected_text: str = Field(default="", description="Corrected version if applicable")
    content_type: ContentType = Field(
        default=ContentType.TEXT,
        description="Content type for syntax-highlighted code fences",
    )

    @field_validator("content_type", mode="before")
    @classmethod
    def coerce_content_type(cls, v: object) -> object:
        return _make_enum_coercer(ContentType, ContentType.TEXT)(v)


class RecommendedAction(BaseModel):
    """A recommended remediation action."""

    model_config = ConfigDict(extra="ignore")

    priority: int = Field(ge=1, description="1 = highest priority")
    title: str
    description: str = ""
    urgency: ActionUrgency = ActionUrgency.SHORT_TERM

    @field_validator("urgency", mode="before")
    @classmethod
    def coerce_urgency(cls, v: object) -> object:
        return _make_enum_coercer(ActionUrgency, ActionUrgency.SHORT_TERM)(v)

    effort: Effort = Effort.MEDIUM

    @field_validator("effort", mode="before")
    @classmethod
    def coerce_effort(cls, v: object) -> object:
        return _make_enum_coercer(Effort, Effort.MEDIUM)(v)

    command: str = Field(default="", description="Exact kubectl / gcloud command")
    expected_output: str = Field(
        default="",
        description="What the command output looks like when healthy",
    )
    interpretation: str = Field(
        default="",
        description="How to read the output and decide next steps",
    )
    why: str = Field(default="")
    risk: str = Field(default="")
    related_findings: list[str] = Field(
        default_factory=list,
        description="References to Finding.id values",
    )


class ManualInvestigation(BaseModel):
    """A finding that could not be automatically verified."""

    model_config = ConfigDict(extra="ignore")

    finding_title: str
    reason: str = Field(default="", description="What tool call failed")
    investigation_steps: str = Field(default="", description="Manual steps to verify")


class TimelineEvent(BaseModel):
    """A single event in the chronological timeline."""

    model_config = ConfigDict(extra="ignore")

    time: str = Field(description="Timestamp — relative ('7m ago') or absolute ISO 8601")
    event: str
    severity: Severity = Severity.INFO

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: object) -> object:
        return _make_enum_coercer(Severity, Severity.INFO)(v)

    service: str = Field(
        default="",
        description="Service or component name, e.g. 'payment-svc', 'node/gke-pool-1'",
    )


class GKEResourceCost(BaseModel):
    """Cost breakdown for a single resource dimension (CPU, RAM, or Ephemeral)."""

    model_config = ConfigDict(extra="ignore")

    resource_type: str = Field(default="", description="Resource type: 'cpu', 'memory', or 'ephemeral'")
    requests: float | None = Field(default=None, description="Total requested units (vCPUs or GiB)")
    usage: float | None = Field(default=None, description="Total actual usage units (from Cloud Monitoring); None if unavailable")
    request_cost_usd: float | None = Field(default=None, description="Monthly cost based on requests (USD)")
    usage_cost_usd: float | None = Field(default=None, description="Monthly cost based on actual usage (USD); None if unavailable")
    waste_cost_usd: float | None = Field(default=None, description="Estimated monthly waste (request_cost - usage_cost); None if unavailable")


class GKEContainerCost(BaseModel):
    """Cost breakdown for a single container within a workload."""

    model_config = ConfigDict(extra="ignore")

    container_name: str = Field(default="", description="Container name within the pod spec")
    resource_costs: list[GKEResourceCost] = Field(default_factory=list)
    total_request_cost_usd: float | None = Field(default=None, description="Monthly request cost for this container (USD)")
    total_usage_cost_usd: float | None = Field(default=None, description="Monthly usage cost for this container (USD); None if unavailable")
    total_waste_usd: float | None = Field(default=None, description="Estimated monthly waste for this container (USD); None if unavailable")


class GKENamespaceSummary(BaseModel):
    """Aggregated cost summary for all workloads in a namespace."""

    model_config = ConfigDict(extra="ignore")

    namespace: str = Field(default="", description="Kubernetes namespace name")
    total_request_cost_usd: float = Field(default=0.0, description="Sum of all workload request costs in the namespace (USD)")
    total_usage_cost_usd: float | None = Field(default=None, description="Sum of all workload usage costs in the namespace (USD); None if any workload is missing usage data")
    total_waste_usd: float | None = Field(default=None, description="Estimated total monthly waste in the namespace (USD); None if usage data unavailable")


class GKEWorkloadCost(BaseModel):
    """Cost estimate for a single Kubernetes workload (Deployment/StatefulSet)."""

    model_config = ConfigDict(extra="ignore")

    namespace: str = Field(default="")
    workload_name: str = Field(default="")
    resource_costs: list[GKEResourceCost] = Field(default_factory=list)
    total_request_cost_usd: float | None = Field(default=None)
    total_usage_cost_usd: float | None = Field(default=None)
    total_waste_usd: float | None = Field(default=None)
    containers: list[GKEContainerCost] = Field(default_factory=list, description="Per-container cost breakdown; empty if container-level data unavailable")
    partial_metrics: bool = Field(
        default=False,
        description=(
            "True when usage cost is computed from only a subset of resource dimensions "
            "(e.g. CPU available but memory unavailable). Consumers should display a "
            "'partial data' indicator alongside usage cost figures."
        ),
    )


class GKECostReport(BaseModel):
    """Top-level GKE workload cost estimation report."""

    model_config = ConfigDict(extra="ignore")

    cluster_type: str = Field(
        default="unknown",
        description="'autopilot', 'standard', or 'unknown'",
    )
    region: str = Field(default="", description="GCP region used for pricing, e.g. 'northamerica-northeast1'")
    supported: bool = Field(
        default=False,
        description="True if cost estimation is supported (Autopilot cluster in known region)",
    )
    unsupported_reason: str | None = Field(
        default=None,
        description="Human-readable reason if supported=False",
    )
    workloads: list[GKEWorkloadCost] = Field(default_factory=list)
    total_request_cost_usd: float | None = Field(default=None, description="Sum of all workload request costs")
    total_usage_cost_usd: float | None = Field(default=None, description="Sum of all workload usage costs; None if no workload had any usage data")
    total_savings_usd: float | None = Field(default=None, description="Estimated total monthly savings potential; None if no usage data available")
    namespace_summaries: dict[str, GKENamespaceSummary] = Field(default_factory=dict, description="Per-namespace aggregated cost summaries keyed by namespace name")
    monitoring_status: str | None = Field(
        default=None,
        description=(
            "Status of the Cloud Monitoring usage query. "
            "None means monitoring was not explicitly checked or status was not set. "
            "'ok' means metrics were successfully fetched. "
            "'no_data' means monitoring was queried but returned empty results. "
            "Other string values indicate errors, e.g. 'PermissionDenied: ...'"
        ),
    )
    workloads_with_full_metrics: int = Field(
        default=0,
        description="Number of workloads for which all resource dimensions (cpu + memory) had usage data",
    )
    workloads_with_partial_metrics: int = Field(
        default=0,
        description="Number of workloads for which only some resource dimensions had usage data",
    )
    workloads_without_metrics: int = Field(
        default=0,
        description="Number of workloads for which no usage data was available from Cloud Monitoring",
    )


class CostMetrics(BaseModel):
    """Cost and token usage metrics for the report generation run."""

    model_config = ConfigDict(extra="ignore")

    run_cost_usd: float | None = Field(default=None, description="Estimated cost in USD for this run")
    total_tokens: int | None = Field(default=None, description="Total tokens consumed (prompt + completion)")
    estimated_cost: str | None = Field(
        default=None,
        description="Pre-formatted cost string (e.g. '$0.001234'). None when cost is zero or unknown.",
    )


class ToolUsageSummary(BaseModel):
    """Summary of tool calls made during the orchestrated pipeline run."""

    model_config = ConfigDict(extra="ignore")

    tool_counts: dict[str, int] | None = Field(
        default=None,
        description="Per-tool call counts, e.g. {'kubectl_get': 4, 'get_events': 2}",
    )
    tool_calls: int | None = Field(default=None, description="Total number of tool calls executed")


class ReportMetadata(BaseModel):
    """Metadata about how and when the report was generated."""

    model_config = ConfigDict(extra="ignore")

    generated_at: str = Field(default="", description="ISO 8601 timestamp")
    cluster_name: str = Field(default="")
    project_id: str = Field(default="")
    model_used: str = Field(default="")
    skill_version: str = Field(default="")
    cost_metrics: CostMetrics | None = Field(default=None, description="Cost and token usage for this run")
    tool_usage: ToolUsageSummary | None = Field(default=None, description="Tool call statistics for this run")
    gke_cost: GKECostReport | None = Field(default=None, description="GKE workload cost estimation (Autopilot only)")


# ── Dependency graph models ───────────────────────────────────


class DependencyEdge(BaseModel):
    """A single dependency relationship between two services."""

    model_config = ConfigDict(extra="ignore")

    source: str = Field(description="Service that depends on target")
    target: str = Field(description="Service being depended upon")
    evidence: str = Field(default="", description="How this was discovered (e.g., 'env:DATABASE_URL', 'istio:VirtualService')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="0.0-1.0 confidence score")
    method: str = Field(default="manual", description="Discovery method: 'env_var', 'istio', 'datadog', 'manual'")


class DependencyNode(BaseModel):
    """A node in the dependency graph (a service/resource)."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Service or resource name")
    node_type: str = Field(default="service", description="'service', 'database', 'external', 'cache', 'queue'")
    namespace: str = Field(default="", description="Kubernetes namespace if applicable")


class DependencyGraph(BaseModel):
    """Complete dependency graph for a service ecosystem."""

    model_config = ConfigDict(extra="ignore")

    nodes: list[DependencyNode] = Field(default_factory=list)
    edges: list[DependencyEdge] = Field(default_factory=list)
    root_service: str = Field(default="", description="The primary service being investigated")
    generated_at: str = Field(default="", description="ISO 8601 timestamp")


# ── Root model ───────────────────────────────────────────────


class HealthReport(BaseModel):
    """Root model for a structured service health report.

    Mirrors the mandatory report structure defined in the reporter
    prompt (``HEALTH_REPORTER_PROMPT``).  Every section in the prompt
    maps to a field here.
    """

    model_config = ConfigDict(extra="ignore")

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
    dependencies: DependencyGraph | None = Field(default=None, description="Structured dependency graph for the service ecosystem")
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)

    # ── Serialisation helpers ────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict serialisation (delegates to Pydantic)."""
        return self.model_dump()

    # ── Compact summary ────────────────────────────────────

    def to_summary(self) -> str:
        """Return a compact 3-5 line summary for ``--summary`` mode.

        Includes overall status with emoji, issue counts, and the
        first recommended action (if any).
        """
        es = self.executive_summary
        emoji = _OVERALL_STATUS_EMOJI.get(es.overall_status, "❓")
        first_rec = self.recommendations[0].title if self.recommendations else "None"
        lines = [
            f"{emoji} Status: {es.overall_status.value}",
            f"Issues: {es.issues_found} ({es.critical_count} critical, {es.warning_count} warning)",
            f"Scope: {es.scope}",
            f"Top recommendation: {first_rec}",
        ]
        return "\n".join(lines)

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

        # Argo Rollouts enrichment — render when any rollout field is present
        rollout_svcs = [s for s in self.service_statuses if s.rollout_strategy or s.rollout_status or s.hpa_conditions]
        if rollout_svcs:
            parts.append("### Rollout Details")
            parts.append("")
            parts.append("| Service | Namespace | Strategy | Rollout Status | HPA Conditions |")
            parts.append("|---------|-----------|----------|----------------|----------------|")
            for svc in rollout_svcs:
                hpa = "; ".join(svc.hpa_conditions) if svc.hpa_conditions else "—"
                parts.append(
                    f"| {svc.service} | {svc.namespace or '—'} | {svc.rollout_strategy or 'N/A'} "
                    f"| {svc.rollout_status or 'N/A'} | {hpa} |"
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
                    parts.append(line)
                    if f.evidence:
                        self._render_evidence_subbullets(parts, f.evidence)
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
                        parts.append("- **Evidence**:")
                        self._render_evidence_subbullets(parts, f.evidence)
                    parts.append(f"- **Confidence**: {f.confidence.value}")
                    if f.impact:
                        parts.append(f"- **Impact**: {f.impact}")
                    if f.affected_resources:
                        parts.append(f"- **Affected Resources**: {', '.join(f.affected_resources)}")
                    parts.append("")

    @staticmethod
    def _render_evidence_subbullets(parts: list[str], evidence: list[str]) -> None:
        """Render evidence items as sub-bullets, wrapping multi-line items in code blocks."""
        for item in evidence:
            if "\n" in item:
                parts.append("  - Multi-line evidence:")
                parts.append("    ```text")
                for line in item.split("\n"):
                    parts.append(f"    {line}")
                parts.append("    ```")
            else:
                parts.append(f"  - {item}")

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
            lang = CONTENT_TYPE_FENCE_MAP.get(ev.content_type, "text")
            if ev.evidence_text:
                parts.append(f"```{lang}")
                parts.append(ev.evidence_text)
                parts.append("```")
            if ev.corrected_text:
                parts.append("")
                parts.append("**Corrected**:")
                parts.append(f"```{lang}")
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
                if action.description and action.description.strip():
                    parts.append(f"   {action.description}")
                command_str = action.command.strip() if action.command else ""
                if command_str:
                    parts.append("   ```")
                    parts.append(f"   {command_str}")
                    parts.append("   ```")
                if action.expected_output and action.expected_output.strip():
                    parts.append("   - Expected output:")
                    parts.append("     ```")
                    for line in action.expected_output.strip().splitlines():
                        parts.append(f"     {line}")
                    parts.append("     ```")
                if action.interpretation and action.interpretation.strip():
                    parts.append(f"   - Interpretation: {action.interpretation}")
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

        # Collapse repeated events BEFORE deciding layout
        collapsed = _collapse_repeated_events(self.timeline)

        # Decide: grouped-by-service vs flat table (based on original events)
        events_with_service = sum(1 for ev in self.timeline if ev.service)
        use_grouping = len(self.timeline) > 0 and (
            events_with_service / len(self.timeline) >= 0.5
        )

        if use_grouping:
            self._render_timeline_grouped(parts, collapsed)
        else:
            self._render_timeline_flat(parts, collapsed, show_service=events_with_service > 0)

    def _render_timeline_grouped(self, parts: list[str], collapsed: list[_CollapsedEvent]) -> None:
        """Render timeline events grouped by service with sub-headings."""
        groups: dict[str, list[_CollapsedEvent]] = defaultdict(list)
        for ce in collapsed:
            key = ce.service if ce.service else "General"
            groups[key].append(ce)

        # Sort group names alphabetically, but "General" goes last
        sorted_keys = sorted(
            groups.keys(), key=lambda k: (k == "General", k)
        )

        for key in sorted_keys:
            parts.append(f"### {key}")
            parts.append("| Time | Event | Severity |")
            parts.append("|------|-------|----------|")
            for ce in groups[key]:
                parts.append(f"| {ce.display_time} | {ce.display_event} | {ce.severity.value} |")
            parts.append("")

    def _render_timeline_flat(
        self, parts: list[str], collapsed: list[_CollapsedEvent], *, show_service: bool
    ) -> None:
        """Render timeline as a flat table, optionally with a Service column."""
        if show_service:
            parts.append("| Time | Service | Event | Severity |")
            parts.append("|------|---------|-------|----------|")
            for ce in collapsed:
                parts.append(
                    f"| {ce.display_time} | {ce.service} | {ce.display_event} | {ce.severity.value} |"
                )
        else:
            parts.append("| Time | Event | Severity |")
            parts.append("|------|-------|----------|")
            for ce in collapsed:
                parts.append(f"| {ce.display_time} | {ce.display_event} | {ce.severity.value} |")
        parts.append("")

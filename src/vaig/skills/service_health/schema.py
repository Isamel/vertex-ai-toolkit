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
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode

from vaig.core.memory.models import RecurrenceSignal

logger = logging.getLogger(__name__)

# ── Text-normalisation regexes (for smart timeline collapse) ──

_POD_HASH_RE = re.compile(r'-[a-z0-9]{5,10}-[a-z0-9]{4,7}\b')   # strip -59967f9ccc-4zdx6

# ── quick_remediation placeholder guard ──────────────────────
# Patterns that match vague "investigate" TODO text — rejected by field_validator.

_BANNED_QUICK_REMEDIATION_PATTERNS: tuple[str, ...] = (
    r"^\s*investig(a|a\w*|ate|ating)\b",          # ES "investigar" / EN "investigate"
    r"^\s*look\s+into\b",
    r"^\s*check\s+the\s+(cause|root|issue)\b",
    r"^\s*revisa(r)?\s+la\s+causa\b",
    r"^\s*analyze\s+the\s+issue\b",
)
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


class EvidenceGap(BaseModel):
    """A signal source that was not checked, errored, or returned empty.

    Populated by sub-gatherers via structured output and merged by the
    analyzer into the root ``HealthReport``.  The ``reason`` field uses a
    controlled vocabulary so downstream rendering can apply visual treatment
    without string parsing.
    """

    model_config = ConfigDict(extra="ignore")

    source: str = Field(description="Name of the tool or signal source (e.g. 'deployment_metrics')")
    reason: str = Field(
        description="Why this source did not produce evidence: 'not_called', 'error', or 'empty_result'"
    )
    details: str | None = Field(
        default=None,
        description="Optional human-readable detail (e.g. error message or skip reason)",
    )


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
    degraded_reason: str | None = Field(
        default=None,
        max_length=160,
        description=(
            "One-line summary of why this specific service is DEGRADED or FAILED. "
            "Always populated when status != HEALTHY. "
            "Example: '15.79% APM error rate over last 15 min.'"
        ),
    )

    @model_validator(mode="after")
    def _warn_missing_degraded_reason(self) -> ServiceStatus:
        """Warn when a non-healthy service has no degraded_reason."""
        if self.status != ServiceHealthStatus.HEALTHY and not self.degraded_reason:
            logger.warning(
                "ServiceStatus %r has status=%s but degraded_reason is empty",
                self.service,
                self.status.value,
            )
        return self


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
    quick_remediation: str | None = Field(
        default=None,
        description=(
            "One-line actionable command or '(see Recommended Actions section)'. "
            "NEVER use placeholder text such as 'Investigate' or 'Look into'."
        ),
    )

    @field_validator("remediation", "quick_remediation")
    @classmethod
    def _reject_placeholder_remediations(cls, v: str | None) -> str | None:
        """Reject vague placeholder remediation text in remediation fields."""
        if v is None or v == "":
            return v
        for pattern in _BANNED_QUICK_REMEDIATION_PATTERNS:
            if re.search(pattern, v, flags=re.IGNORECASE):
                raise ValueError(
                    "remediation fields must be actionable and must not contain "
                    "placeholder text such as 'Investigate' or 'Look into'. "
                    f"Received placeholder text: {v!r}"
                )
        return v

    caused_by: list[str] = Field(
        default_factory=list,
        description="Finding.id slugs of upstream causes (findings that caused this one)",
    )
    causes: list[str] = Field(
        default_factory=list,
        description="Finding.id slugs of downstream effects (findings caused by this one)",
    )
    recurrence: RecurrenceSignal | None = Field(
        default=None,
        description=(
            "Populated post-Gemini by the RecurrenceAnalyzer.  None during the "
            "Gemini call — excluded from response_schema to avoid confusing the model."
        ),
        exclude=True,
    )


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


class CostDataQuality(BaseModel):
    """Summary of the cost data quality for a GKE cost report.

    Captures which source provided usage metrics, how fresh the data is,
    how many workloads have real (non-estimated) data, and whether multiple
    sources disagreed on usage figures.
    """

    model_config = ConfigDict(extra="ignore")

    primary_source: str = Field(
        default="unknown",
        description=(
            "Dominant metrics source across all workloads. "
            "Values: 'cloud_monitoring_60m', 'cloud_monitoring_24h', "
            "'metrics_server', 'datadog', 'requests_fallback', 'unknown'."
        ),
    )
    freshness: str = Field(
        default="unknown",
        description=(
            "Dominant freshness descriptor. "
            "Values: '60m_avg', '24h_avg', 'snapshot', '1h_avg', 'unknown'."
        ),
    )
    coverage_count: int = Field(
        default=0,
        description="Number of workloads with non-fallback (real) usage data.",
    )
    total_count: int = Field(
        default=0,
        description="Total number of workloads in the report.",
    )
    confidence: str = Field(
        default="low",
        description=(
            "Overall confidence in cost figures. "
            "'high' = all workloads have real L1/L2 data; "
            "'medium' = some workloads use L3 (Datadog) or 24h window; "
            "'low' = one or more workloads fell back to request-based estimates."
        ),
    )
    fallback_count: int = Field(
        default=0,
        description="Number of workloads using requests_fallback (L4) — no real usage data.",
    )
    cross_check_discrepancies: list[str] = Field(
        default_factory=list,
        description=(
            "Workload names where two sources disagreed by more than 10%% "
            "on CPU or memory usage. Best-effort; may be empty even when sources differ."
        ),
    )


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
    metrics_estimated: bool = Field(
        default=False,
        description=(
            "True when usage metrics were unavailable from Cloud Monitoring and the "
            "usage cost was estimated using resource requests (100%% utilization assumption). "
            "Consumers should display an '(estimated)' indicator alongside cost figures."
        ),
    )
    metrics_source: str | None = Field(
        default=None,
        description=(
            "The source that provided usage metrics for this workload. "
            "Values: 'cloud_monitoring_60m', 'cloud_monitoring_24h', "
            "'metrics_server', 'datadog', 'requests_fallback'. "
            "None when no usage data was attempted."
        ),
    )
    metrics_freshness: str | None = Field(
        default=None,
        description=(
            "Freshness descriptor for the usage metrics. "
            "Values: '60m_avg', '24h_avg', 'snapshot', '1h_avg'. "
            "None when metrics_source is None."
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
    total_usage_cost_usd: float | None = Field(default=None, description="Sum of all workload usage costs; may include estimated values when metrics_estimated is True")
    total_savings_usd: float | None = Field(default=None, description="Estimated total monthly savings potential; may include estimated values when metrics_estimated is True")
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
    pricing_source: str = Field(
        default="hardcoded_fallback",
        description=(
            "Source of the Autopilot pricing rates used for cost estimation. "
            "'billing_api' when rates were fetched from the Cloud Billing Catalog API; "
            "'hardcoded_fallback' when using the static pricing table."
        ),
    )
    metrics_estimated: bool = Field(
        default=False,
        description=(
            "True when one or more workloads had no monitoring data and usage costs were "
            "estimated using resource requests (100%% utilization assumption). "
            "The report totals include these estimated values."
        ),
    )
    cost_data_quality: CostDataQuality | None = Field(
        default=None,
        description=(
            "Summary of cost data quality: which source provided metrics, "
            "confidence level, coverage, and any cross-source discrepancies. "
            "None when cost estimation was not run."
        ),
    )


class MetricTrend(BaseModel):
    """Trend data for a single metric of a single service.

    Captures the rate-of-change between a current observation window and a
    historical baseline, along with a severity classification and an optional
    projection of how many days until a resource limit is reached.
    """

    model_config = ConfigDict(extra="ignore")

    metric: str = Field(description="Metric identifier: 'cpu_usage', 'memory_usage', or 'restart_count'")
    service_name: str = Field(description="Workload / service name")
    namespace: str = Field(default="", description="Kubernetes namespace")
    direction: str = Field(description="Trend direction: 'increasing', 'decreasing', 'stable', or 'new'")
    rate_of_change_percent: float | None = Field(
        default=None,
        description="Percentage change from baseline. None when baseline is zero (new service).",
    )
    current_value: float | None = Field(default=None, description="Current window average value")
    baseline_value: float | None = Field(default=None, description="Baseline window average value")
    baseline_window_days: int = Field(default=7, description="Number of days in the baseline window")
    days_to_threshold: float | None = Field(
        default=None,
        description="Projected days until resource limit is reached. None if stable/decreasing.",
    )
    severity: str = Field(default="info", description="Severity: 'critical', 'warning', or 'info'")


class TrendAnalysis(BaseModel):
    """Aggregated trend analysis results across all services and metrics."""

    model_config = ConfigDict(extra="ignore")

    trends: list[MetricTrend] = Field(default_factory=list, description="Individual metric trends")
    analyzed_at: str = Field(default="", description="ISO 8601 timestamp of analysis")
    baseline_windows: list[int] = Field(default_factory=lambda: [7], description="Baseline window sizes in days")
    services_analyzed: int = Field(default=0, description="Number of services analyzed")
    anomalies_detected: int = Field(default=0, description="Count of warning + critical trends")


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
    successful_calls: int | None = Field(
        default=None,
        description="Number of tool calls that returned useful (non-error, non-empty) data",
    )
    failed_calls: int | None = Field(
        default=None,
        description="Number of tool calls that returned an error or empty result",
    )


# ── AUDIT-15: Post-hoc field population telemetry ────────────


class PostHocFieldStatus(BaseModel):
    """Status record for a single post-hoc populated field."""

    field_name: str = Field(description="Name of the post-hoc field being tracked")
    populated: bool = Field(description="True when the field was successfully populated")
    reason: str | None = Field(
        default=None,
        description="Status detail: 'populated', 'skipped', or 'error:<details truncated to 160 chars>'",
    )


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
    trends: TrendAnalysis | None = Field(default=None, description="Anomaly trend detection results")
    # ── AUDIT-07: Run determinism metadata ────────────────────
    run_seed: int | None = Field(
        default=None,
        description="Deterministic seed used for any sampling; None when unused.",
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of agent-name → model-id-version actually used, resolved at "
            "runtime (not the config default). Example: "
            '\'{"health_analyzer": "gemini-2.5-pro-002"}\'.'
        ),
    )
    pipeline_version: str = Field(
        default="unknown",
        description=(
            "Version identifier of the vaig package that produced this "
            "report: git commit short SHA when available, otherwise the "
            'installed package version, else "unknown".'
        ),
    )
    autonomous_enabled: bool = Field(
        default=False,
        description="True when investigation.enabled was True for this run.",
    )
    autonomous_steps_executed: int | None = Field(
        default=None,
        description="Number of InvestigationAgent steps completed (None when autonomous disabled).",
    )
    autonomous_replan_iterations: int | None = Field(
        default=None,
        description="Number of re-plan iterations executed by the InvestigationAgent (None when autonomous disabled).",
    )
    # ── AUDIT-15: Post-hoc field population telemetry ─────────
    post_hoc_field_status: list[PostHocFieldStatus] = Field(
        default_factory=list,
        description=(
            "Telemetry entries for each post-hoc populated field. "
            "populated=False entries indicate fields that were expected but left empty."
        ),
    )


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


# ── External deep-link models ────────────────────────────────


class ExternalLink(BaseModel):
    """A single external deep-link associated with the report.

    Each link targets a specific system (GCP Console, Datadog, or ArgoCD)
    and carries a human-readable label, a fully-formed URL, and an optional
    SVG icon string.
    """

    model_config = ConfigDict(extra="ignore")

    label: str = Field(description="Human-readable link label")
    url: str = Field(description="Fully-formed target URL")
    system: str = Field(
        description="Originating system — one of 'gcp', 'datadog', or 'argocd'"
    )
    icon: str = Field(default="", description="Inline SVG icon string (optional)")


class ExternalLinks(BaseModel):
    """Grouped external deep-links for a health report.

    Each attribute is a list of :class:`ExternalLink` objects for the
    corresponding system.  All groups default to empty so the model is
    safe to instantiate without any links present.
    """

    model_config = ConfigDict(extra="ignore")

    gcp: list[ExternalLink] = Field(default_factory=list, description="GCP Console links")
    datadog: list[ExternalLink] = Field(default_factory=list, description="Datadog links")
    argocd: list[ExternalLink] = Field(default_factory=list, description="ArgoCD links")


# ── Change correlation model ─────────────────────────────────


class ChangeEvent(BaseModel):
    """A deployment or configuration change event correlated to an issue."""

    model_config = ConfigDict(extra="ignore")

    timestamp: str = Field(description="ISO 8601 timestamp of the change event")
    type: str = Field(description="Event type: 'deployment', 'config_change', or 'hpa_scaling'")
    description: str = Field(description="Human-readable description of the change")
    correlation_to_issue: str = Field(description="Analyst note on how this change correlates to the reported issue")


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
    recent_changes: list[ChangeEvent] = Field(
        default_factory=list,
        description=(
            "Deployment, config change, or HPA scaling events that may correlate with the "
            "reported issue. Populated by the analyzer separately from findings."
        ),
    )
    dependencies: DependencyGraph | None = Field(default=None, description="Structured dependency graph for the service ecosystem")
    external_links: ExternalLinks | None = Field(
        default=None,
        description=(
            "External deep-links populated from investigation context "
            "(GCP Console, Datadog, ArgoCD).  None when no context keys were available."
        ),
    )
    metadata: ReportMetadata = Field(
        default_factory=ReportMetadata,
        description=(
            "Post-hoc metadata: cost, tokens, GKE cost, trends.  "
            "Populated by the pipeline after generation."
        ),
    )
    causal_graph_mermaid: str | None = Field(
        default=None,
        description="Mermaid graph TD diagram string describing causal relationships between findings",
    )
    evidence_gaps: list[EvidenceGap] = Field(
        default_factory=list,
        description=(
            "Signal sources that were not checked, errored, or returned empty data. "
            "Populated by sub-gatherers and merged by the analyzer."
        ),
    )
    investigation_coverage: str | None = Field(
        default=None,
        description="Human-readable coverage summary, e.g. '9/12 signal sources checked'.",
    )
    overall_severity_reason: str | None = Field(
        default=None,
        max_length=240,
        description=(
            "Human-readable one-line justification for overall_severity. "
            "Populated by the analyzer. Example: "
            "'DEGRADED because 1 HIGH finding with confidence=CONFIRMED "
            "and 2 MEDIUM findings in distinct namespaces.'"
        ),
    )
    investigation_evidence: list[InvestigationEvidenceSnapshot] = Field(
        default_factory=list,
        description=(
            "Per-step evidence records rendered in the Autonomous Investigation section. "
            "Populated post-hoc from the EvidenceLedger; excluded from the Gemini schema."
        ),
    )

    @property
    def root_causes(self) -> list[Finding]:
        """Return findings that are true root causes (no upstream causes).

        A finding is a root cause when its ``caused_by`` list is empty,
        meaning no other finding caused it.  This is a ``@property`` (not a
        ``@computed_field``) so it is excluded from Pydantic serialisation
        and does not interfere with Gemini's ``response_schema`` generation.
        """
        return [f for f in self.findings if not f.caused_by]

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
        Manual Investigation Required → Timeline → Causal Graph →
        Recent Changes → Quick Links → Evidence Gaps → Investigation Coverage.
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
        self._render_causal_graph(parts)
        self._render_recent_changes(parts)
        self._render_external_links(parts)
        self._render_evidence_gaps(parts)
        self._render_investigation_coverage(parts)
        return "\n".join(parts)

    # ── Private rendering helpers ────────────────────────────

    def _render_executive_summary(self, parts: list[str]) -> None:
        es = self.executive_summary
        parts.append("## Executive Summary")
        parts.append(f"- **Status**: {es.overall_status.value}")
        if self.overall_severity_reason:
            parts.append(f"  *{self.overall_severity_reason}*")
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
            status_cell = f"{emoji}"
            if svc.status != ServiceHealthStatus.HEALTHY and svc.degraded_reason:
                status_cell = f"{emoji} *{svc.degraded_reason}*"
            parts.append(
                f"| {svc.service} | {svc.namespace} | {status_cell} | {svc.pods_ready} "
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

    def _render_recent_changes(self, parts: list[str]) -> None:
        """Render the What Changed Recently section."""
        if not self.recent_changes:
            return
        parts.append("## What Changed Recently")
        parts.append("")
        for change in self.recent_changes:
            parts.append(f"- **[{change.timestamp}]** `{change.type}`: {change.description}")
            if change.correlation_to_issue:
                parts.append(f"  - *Correlation*: {change.correlation_to_issue}")
        parts.append("")

    def _render_external_links(self, parts: list[str]) -> None:
        """Render the Quick Links section grouped by system (GCP/Datadog/ArgoCD)."""
        if self.external_links is None:
            return
        system_groups: list[tuple[str, list[ExternalLink]]] = [
            ("GCP", self.external_links.gcp),
            ("Datadog", self.external_links.datadog),
            ("ArgoCD", self.external_links.argocd),
        ]
        has_any = any(links for _, links in system_groups)
        if not has_any:
            return
        parts.append("## Quick Links")
        parts.append("")
        for system_name, links in system_groups:
            if not links:
                continue
            parts.append(f"### {system_name}")
            for link in links:
                parts.append(f"- [{link.label}]({link.url})")
            parts.append("")

    def _render_evidence_gaps(self, parts: list[str]) -> None:
        """Render the Evidence Gaps section."""
        _REASON_LABELS: dict[str, str] = {
            "not_called": "Not Called",
            "error": "Error",
            "empty_result": "Empty Result",
        }
        if not self.evidence_gaps:
            return
        parts.append("## Evidence Gaps")
        parts.append("")
        for gap in self.evidence_gaps:
            human_reason = _REASON_LABELS.get(gap.reason, gap.reason.replace("_", " ").title())
            line = f"- **{gap.source}** ({human_reason})"
            if gap.details:
                line += f": {gap.details}"
            parts.append(line)
        parts.append("")

    def _render_investigation_coverage(self, parts: list[str]) -> None:
        """Render the Investigation Coverage percentage."""
        if not self.investigation_coverage:
            return
        parts.append("## Investigation Coverage")
        parts.append("")
        parts.append(f"{self.investigation_coverage}")
        parts.append("")

    def _render_causal_graph(self, parts: list[str]) -> None:
        """Render the causal graph Mermaid block when present."""
        if self.causal_graph_mermaid is None:
            return
        if not self.causal_graph_mermaid.strip():
            return
        parts.append("## Causal Graph")
        parts.append("")
        parts.append("```mermaid")
        parts.append(self.causal_graph_mermaid)
        parts.append("```")
        parts.append("")


# ── Slim Gemini schema (excludes post-hoc fields) ────────────


def _collect_reachable_defs(
    schema: dict[str, object],
    excluded_names: set[str],
) -> set[str]:
    """Return the set of ``$defs`` keys reachable from the *retained* properties.

    We perform a multi-pass reachability walk:

    1. Start from the top-level ``properties`` that are **not** excluded and
       from any other top-level schema keys (``allOf``, ``anyOf``, etc.)
       — excluding the ``$defs`` block itself.
    2. For each ``$def`` key found reachable, add its definition body to the
       frontier so we transitively follow nested references.
    3. Repeat until no new keys are discovered (fixed point).

    This correctly handles defs that are only referenced *from* other defs
    (e.g. ``Confidence`` referenced inside the ``Finding`` definition body).
    """
    import json

    raw_defs = schema.get("$defs")
    defs: dict[str, object] = raw_defs if isinstance(raw_defs, dict) else {}

    # Seed: serialise the schema WITHOUT $defs and WITHOUT excluded properties
    seed: dict[str, object] = {}
    for k, v in schema.items():
        if k == "$defs":
            continue
        if k == "properties" and isinstance(v, dict):
            seed[k] = {pk: pv for pk, pv in v.items() if pk not in excluded_names}
        else:
            seed[k] = v

    reachable: set[str] = set()
    frontier_str = json.dumps(seed)

    while True:
        newly_found: set[str] = set()
        for key in defs:
            if key not in reachable and f'"#/$defs/{key}"' in frontier_str:
                newly_found.add(key)
        if not newly_found:
            break
        reachable |= newly_found
        # Expand frontier: add serialised bodies of newly reachable defs
        frontier_str += json.dumps({k: defs[k] for k in newly_found})

    return reachable


class HealthReportGeminiSchema(HealthReport):
    """Reduced schema passed as ``response_schema`` to Gemini structured output.

    Excludes fields that are populated POST-HOC by the pipeline (not by the
    reporter LLM) to avoid the ``400 INVALID_ARGUMENT: The specified schema
    produces a constraint that has too many states for serving`` error.

    The full :class:`HealthReport` class is still used for
    serialisation / deserialisation — ``HealthReport.model_validate_json()``
    accepts any JSON that validates against the full schema (extra fields from
    the slim schema are ignored, missing post-hoc fields use their defaults).

    Post-hoc fields excluded from the Gemini schema:

    * ``metadata`` — populated by the pipeline after generation
    * ``evidence_gaps`` — populated by sub-gatherers / analyzer
    * ``recent_changes`` — populated by the analyzer
    * ``external_links`` — populated by the link-builder post-hoc
    * ``investigation_coverage`` — populated by the analyzer

    **Exclusion mechanism** — post-hoc fields are listed in the
    ``_GEMINI_EXCLUDED_FIELDS`` frozenset.  The ``model_json_schema()``
    classmethod override reads that set and strips the matching properties
    (and any orphaned ``$defs``) at call time, without touching
    ``.model_dump()`` or ``.model_dump_json()``.  This keeps the HTML report
    template working (it reads ``REPORT_DATA.metadata.project_id`` and
    friends from the full serialised output).
    """

    # Fields to strip from the Gemini JSON schema (but NOT from model_dump).
    # Do NOT use Pydantic's ``exclude=True`` — that also strips from model_dump,
    # which breaks the HTML report template.
    _GEMINI_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "metadata",
            "evidence_gaps",
            "recent_changes",
            "external_links",
            "investigation_coverage",
            "investigation_evidence",
        }
    )

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
        *,
        union_format: Literal["any_of", "primitive_type_array"] = "any_of",
    ) -> dict[str, Any]:
        """Return a pruned JSON schema that omits post-hoc fields.

        The google-genai SDK sends the full JSON schema to Gemini for
        structured output.  Post-hoc fields must be removed here to stay
        under the "too many states" constraint.

        The fields to prune are listed in ``_GEMINI_EXCLUDED_FIELDS``.
        We do NOT use Pydantic's ``exclude=True`` for this purpose because
        ``exclude=True`` also strips fields from ``.model_dump()`` /
        ``.model_dump_json()``, which would break the HTML report template
        that reads ``REPORT_DATA.metadata.project_id`` and friends.

        Post-hoc fields excluded from the Gemini schema:

        * ``metadata`` — populated by the pipeline after generation
        * ``evidence_gaps`` — populated by sub-gatherers / analyzer
        * ``recent_changes`` — populated by the analyzer
        * ``external_links`` — populated by the link-builder post-hoc
        * ``investigation_coverage`` — populated by the analyzer

        Note: nested excluded fields (e.g. within non-excluded sub-models)
        are not yet pruned. This is a known limitation for a future enhancement.
        """
        schema: dict[str, Any] = super().model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
            union_format=union_format,
        )

        # Identify fields to drop via the class-level constant
        excluded_names = cls._GEMINI_EXCLUDED_FIELDS

        # Strip from properties & required
        props = schema.get("properties")
        if isinstance(props, dict):
            for name in excluded_names:
                props.pop(name, None)
        req = schema.get("required")
        if isinstance(req, list):
            schema["required"] = [r for r in req if r not in excluded_names]

        # Remove orphaned $defs (types only referenced by excluded fields)
        defs = schema.get("$defs")
        if isinstance(defs, dict):
            reachable = _collect_reachable_defs(schema, set(excluded_names))
            orphans = [key for key in list(defs) if key not in reachable]
            for key in orphans:
                del defs[key]

        return schema


# ══════════════════════════════════════════════════════════════
# Review / Approval Models
# ══════════════════════════════════════════════════════════════


class ReviewStatus(StrEnum):
    """Status of a report or finding review."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class FindingReview(BaseModel):
    """Review decision for a single finding within a report."""

    model_config = ConfigDict(extra="ignore")

    finding_id: str
    status: ReviewStatus = ReviewStatus.DRAFT
    reviewer_comment: str = ""
    reviewed_at: datetime | None = None


class ReportReview(BaseModel):
    """Aggregate review for an entire health report run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    status: ReviewStatus = ReviewStatus.DRAFT
    reviewer: str = ""
    finding_reviews: list[FindingReview] = Field(default_factory=list)
    overall_comment: str = ""
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── K-03: Investigation Plan models (SPEC-SH-01) ─────────────────────────────


class StepStatus(StrEnum):
    """Lifecycle status for a single investigation step."""

    pending = "pending"
    running = "running"
    complete = "complete"
    skipped = "skipped"


class InvestigationStep(BaseModel):
    """A single step in an investigation plan.

    Each step targets one root-cause finding and carries enough metadata
    for the InvestigationAgent to execute the right tool without further
    LLM reasoning.
    """

    model_config = ConfigDict(frozen=True)

    step_id: str
    """Unique identifier for this step within the plan (e.g. ``step-0``)."""
    target: str
    """Kubernetes resource being investigated (e.g. ``pod/my-service-abc``)."""
    tool_hint: str
    """Suggested tool name from the HypothesisLibrary (e.g. ``kubectl_describe``)."""
    hypothesis: str
    """Human-readable hypothesis text (mapped from RootCauseHypothesis.what_would_confirm)."""
    priority: int = 1
    """Step priority: 1 = highest, 5 = lowest.  Root-cause steps get priority 1."""
    depends_on: list[str] = Field(default_factory=list)
    """List of ``step_id`` values that must complete before this step runs."""
    status: StepStatus = StepStatus.pending
    """Current lifecycle status of the step."""
    budget_usd: float = 0.0
    """Budget allocated to this step in USD."""


class InvestigationPlan(BaseModel):
    """A complete investigation plan produced by the health_planner agent."""

    model_config = ConfigDict(frozen=True)

    plan_id: str
    """Unique plan identifier (typically the run_id)."""
    steps: list[InvestigationStep] = Field(default_factory=list)
    """Ordered list of investigation steps (root-cause steps first)."""
    created_from: str
    """Report fingerprint or run_id that triggered this plan."""
    total_budget_allocated: float = 0.0
    """Sum of budget_usd across all steps."""


# ── GH-02: Config Drift Finding ──────────────────────────────


# ── AUDIT-11: Autonomous investigation evidence snapshot ─────────────────────


class InvestigationEvidenceSnapshot(BaseModel):
    """Compact per-step evidence record rendered in the HTML report.

    Populated post-hoc from the EvidenceLedger by the pipeline skill after
    the InvestigationAgent completes.  Excluded from the Gemini schema.
    """

    model_config = ConfigDict(frozen=True)

    step_id: str
    """Unique identifier for this step within the plan (e.g. ``step-0``)."""
    target: str
    """Kubernetes resource being investigated."""
    tool_name: str
    """Name of the tool that was called."""
    hypothesis: str
    """Human-readable hypothesis text."""
    verdict: Literal["CONFIRMED", "CONTRADICTED", "INCONCLUSIVE", "SKIPPED"]
    """Outcome classification for the step."""
    answer_preview: str = Field(default="", max_length=320)
    """Truncated summary of the tool output (max 320 chars)."""
    iteration: int = 0
    """Re-plan iteration index: 0 = initial plan, 1+ = re-plans."""
    memory_recall_hit: bool = False
    """True when the answer was served from the PatternMemoryStore cache."""


class ConfigDriftFinding(BaseModel):
    """Finding when recent commits correlate with an observed anomaly.

    Produced when the live investigation pipeline correlates a service
    anomaly with recent commits in the linked GitHub repository.
    Requires ``--repo`` to be provided on the CLI.
    """

    model_config = ConfigDict(extra="ignore")

    finding_type: Literal["config_drift"] = "config_drift"
    commit_sha: str = Field(description="Git commit SHA that introduced the suspected change")
    changed_files: list[str] = Field(
        default_factory=list,
        description="List of file paths changed in the commit",
    )
    suspected_config_keys: list[str] = Field(
        default_factory=list,
        description="Config keys / environment variables suspected to cause the drift",
    )
    severity: Severity = Severity.HIGH

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: object) -> object:
        return _make_enum_coercer(Severity, Severity.HIGH)(v)

    description: str = Field(default="", description="Human-readable explanation of the suspected config drift")

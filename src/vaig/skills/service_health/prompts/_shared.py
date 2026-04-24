"""Shared constants and helper builders for the service health skill prompts.

Contains tool reference tables, the priority hierarchy, the Datadog API step
block, and the builder functions that assemble them into prompt fragments.
"""

from __future__ import annotations

import re
from typing import Final

from vaig.core.prompt_defense import wrap_untrusted_content

# ── Attachment context prefix ─────────────────────────────────────────────────

ATTACHMENT_HEADER: Final[str] = "## Attached Context\n"


def _prefix_attachment_context(
    system_instruction: str,
    attachment_context: str | None,
) -> str:
    """Append attachment context to a system instruction string.

    The attachment content originates from EXTERNAL, UNTRUSTED sources (user-
    provided files, git repositories, URLs).  To defend against prompt injection
    attacks, the content is:

    1. Wrapped with :func:`vaig.core.prompt_defense.wrap_untrusted_content`
       (which neutralizes any forged delimiter markers and encloses the text
       between untrusted-data sentinels).
    2. Placed AFTER the ``system_instruction`` — the trusted instructions
       (including the ``ANTI_INJECTION_RULE``) are read first, so by the time
       the model reaches the attachment it has already been told to treat its
       contents as data, not instructions.

    When *attachment_context* is falsy (``None`` or empty string) the original
    *system_instruction* is returned unchanged — preserving byte-for-byte
    compatibility with callers that omit attachments.

    Args:
        system_instruction: The base system prompt string (trusted content;
            must include the anti-injection rule before invoking this helper).
        attachment_context: Rendered attachment text from external sources,
            or ``None``.

    Returns:
        ``system_instruction`` with the attachment appended under
        :data:`ATTACHMENT_HEADER` and wrapped in untrusted-data delimiters
        when *attachment_context* is truthy; otherwise *system_instruction*
        as-is.
    """
    if not attachment_context:
        return system_instruction
    wrapped = wrap_untrusted_content(attachment_context)
    return f"{system_instruction}\n\n{ATTACHMENT_HEADER}{wrapped}"


_CORE_TOOLS_TABLE = """\
| `kubectl_get` | `resource` | `name`, `namespace`, `output`, `label_selector`, `field_selector` |
| `kubectl_describe` | `resource_type`, `name` | `namespace` |
| `kubectl_logs` | `pod` | `namespace`, `container`, `tail_lines`, `since` |
| `kubectl_top` | | `resource_type`, `name`, `namespace` |
| `get_events` | | `namespace`, `event_type`, `involved_object_name`, `involved_object_kind`, `limit` |
| `get_rollout_status` | `name` | `namespace` |
| `get_node_conditions` | | `name` |
| `get_container_status` | `name` | `namespace` |
| `get_rollout_history` | `name`, `namespace` | `revision` |
| `exec_command` | `pod_name`, `namespace`, `command` | `container`, `timeout` |
| `check_rbac` | `verb`, `resource`, `namespace` | `service_account`, `resource_name` |
| `gcloud_logging_query` | `filter_expr` | `project`, `limit`, `order_by` |
| `gcloud_monitoring_query` | `metric_type` | `project`, `interval_minutes`, `aggregation`, `filter_str` |
| `kubectl_get_labels` | `resource_type` | `namespace`, `name`, `label_filter`, `annotation_filter` |
| `get_scaling_status` | `name` | `namespace` |
| `check_metrics_api_health` | | |
| `query_custom_metrics` | | `metric_name`, `namespace` |
| `query_external_metrics` | `metric_name` | `namespace` |
| `get_datadog_config` | | `namespace`, `deployment` |
| `get_pod_metrics` | `namespace`, `pod_name_prefix` | `window_minutes`, `metric_type` |
| `discover_dependencies` | `service_name` | `namespace`, `force_refresh` |"""

_HELM_TOOLS_TABLE = """\
| `helm_list_releases` | | `namespace`, `force_refresh` |
| `helm_release_status` | `release_name` | `namespace`, `force_refresh` |
| `helm_release_history` | `release_name` | `namespace`, `force_refresh` |
| `helm_release_values` | `release_name` | `namespace`, `all_values`, `force_refresh` |"""

_ARGOCD_TOOLS_TABLE = """\
| `argocd_list_applications` | | `namespace` |
| `argocd_app_status` | `app_name` | `namespace` |
| `argocd_app_history` | `app_name` | `namespace` |
| `argocd_app_diff` | `app_name` | `namespace` |
| `argocd_app_managed_resources` | `app_name` | `namespace` |"""

_DATADOG_API_TOOLS_TABLE = """\
| `query_datadog_metrics` | `cluster_name` | `metric`, `from_ts`, `to_ts`, `service`, `env` |
| `get_datadog_monitors` | | `cluster_name`, `state`, `service`, `env` |
| `get_datadog_service_catalog` | | `env`, `cluster_name`, `service_name` |
| `get_datadog_apm_services` | | `service_name` (optional — returns guidance if omitted), `env` — auto-tries web/gRPC/Kafka metric families |
| `get_datadog_service_dependencies` | `service_name` | — returns upstream (called_by) and downstream (calls) services plus structured DependencyEdge data |
| `query_datadog_error_spans` | `service`, `env`, `start_time`, `end_time` | Drill into error spans: endpoint, status code, exception class, upstream service |"""

_PRIORITY_HIERARCHY = """\
1. Kubernetes cluster data is the ABSOLUTE source of truth for deployment status.
2. If K8s shows a service/deployment exists and is running, it IS deployed — regardless of Datadog results.
3. While Kubernetes health is the primary signal, APM metrics with CRITICAL or HIGH severity (error rate > 5%, avg latency > 1s, throughput drops > 50%) generate independent findings that MUST be reported — even when K8s shows healthy pods.
4. Empty Datadog results mean "monitoring not configured" NOT "service not deployed".
5. NEVER conclude a service is "not deployed" or "doesn't exist" based on Datadog tool results."""

_DATADOG_API_STEP = (
    """\

### Step 12 — Datadog API Correlation (real-time metrics & monitors) — MANDATORY

**PRIORITY HIERARCHY — READ THIS FIRST:**
"""
    + _PRIORITY_HIERARCHY
    + """

You MUST complete calls 19–21 below. They are NOT optional — skipping them means the
investigation is incomplete and the report will be missing real-time observability data.
Note that ``query_datadog_metrics`` is called twice with different metric arguments
(once for CPU, once for memory). Calls 22–23 (``get_datadog_service_catalog`` and
``get_datadog_apm_services``) MUST also always be attempted — the tools accept an empty
or absent ``service_name`` and will return guidance on how to proceed if resolution fails.
Calls 19–21 are high priority but should not block the analysis if they fail.

**LABEL-AWARE FILTERING — MANDATORY**: Before making these calls, resolve the service
identity from Kubernetes data you have already gathered. Use this priority order:
- ``tags.datadoghq.com/service`` pod label → store as ``<dd_service>``
- ``app.kubernetes.io/name`` pod label → store as ``<dd_service>`` (if above absent)
- ``app`` pod label → store as ``<dd_service>`` (if above absent)
- Deployment or Service name from Kubernetes → store as ``<dd_service>`` (fallback)
- ``tags.datadoghq.com/env`` pod label or ``DD_ENV`` env var → store as ``<dd_env>``
You MUST pass these values as ``service=`` and ``env=`` parameters in calls 19–21 below.
For calls 22–23, the parameter name is ``service_name=`` (not ``service=``) — see calls
22–23 for the full resolution rules. If a value was NOT found, omit that parameter
(do NOT pass None or empty string — simply leave the parameter out).

19. You MUST call ``query_datadog_metrics(cluster_name="<cluster>", metric="cpu",
    service="<dd_service>", env="<dd_env>")``  [include service/env only if resolved
    above] — CPU usage time-series scoped to this service when labels are present,
    or cluster-wide when they are absent. Correlate the returned series with the pods
    and services discovered in earlier steps.
    Example with labels: ``query_datadog_metrics(cluster_name="prod", metric="cpu",
    service="my-api", env="production")``
    Example without labels: ``query_datadog_metrics(cluster_name="prod", metric="cpu")``
20. You MUST call ``query_datadog_metrics(cluster_name="<cluster>", metric="memory",
    service="<dd_service>", env="<dd_env>")``  [include service/env only if resolved
    above] — Memory usage time-series (same service/env scope as call 19).
21. You MUST call ``get_datadog_monitors(cluster_name="<cluster>", service="<dd_service>",
    env="<dd_env>")``  [include service/env only if resolved above]
    — Monitor alerts scoped to this service when labels are present, or all cluster
    monitors when they are absent. Note any alerts in Alert or Warn state.

**Service identity resolution (applies to calls 22 AND 23):**

- **Tier 1 — Datadog Unified Service Tagging labels** (check pod/deployment YAML
  output from earlier kubectl calls):
  - ``tags.datadoghq.com/service`` → use as ``service_name``
  - ``tags.datadoghq.com/env``     → use as ``env``
  - ``tags.datadoghq.com/version`` → note for context only

- **Tier 2 — Kubernetes identity** (if Tier 1 labels are absent):
  - ``app.kubernetes.io/name`` or ``app`` label → use as ``service_name``
  - Deployment or Service name → use as ``service_name`` (last resort)

- **Tier 3 — Call without service_name** (if NEITHER Tier 1 nor Tier 2 yields a value):
  - Call the tool without ``service_name`` — it will return guidance on how to
    resolve the service identity. Record the guidance in Raw Findings.

22. ALWAYS call ``get_datadog_service_catalog`` — attempt it even if ``service_name``
    cannot be resolved. The tool handles empty service_name gracefully and returns
    guidance on resolution. Use the service identity resolved via the shared rules above.

    When ``service_name`` IS resolved: call
    ``get_datadog_service_catalog(service_name="<resolved>", env="<resolved>")``
    — check if monitoring data is available and fetch ownership metadata
    (team, language, tier).  This tool returns service *definition* metadata, NOT live
    latency or error-rate metrics.
    Example: ``get_datadog_service_catalog(service_name="my-api", env="production")``

23. ALWAYS call ``get_datadog_apm_services`` — attempt it even if ``service_name``
    cannot be resolved. The tool handles empty service_name gracefully and returns
    guidance. This tool queries LIVE APM trace data (throughput, error rate,
    avg latency) for the configured default lookback window (default: 4 hours),
    scoped to the resolved service and env.
    It complements call 22: call 22 gives ownership metadata, call 23 gives real-time
    performance signals. Use the service identity resolved via the shared rules above.

    When ``service_name`` IS resolved: call
    ``get_datadog_apm_services(service_name="<resolved>", env="<resolved>")``
    — fetch live throughput, error rate, and latency.
    For low-traffic services or when no data is returned, pass ``hours_back`` to
    increase the lookback window (e.g. ``hours_back=8`` for 8 hours).
    Example: ``get_datadog_apm_services(service_name="my-api", env="production")``

Report findings as a "## Raw Findings (Datadog API)" section with:
- Whether data is **service-filtered** (service/env were passed as params) or
  **cluster-wide** (no service labels found — data covers all services in the cluster).
  ALWAYS state which scope applies so the reporter can interpret the data correctly.
- Any monitors currently in Alert or Warn state (name, status, query)
- CPU/memory trends that contradict or confirm the kubectl_top data
- Whether the service was found in the Datadog service catalog (team, language, tier ownership metadata)
- Live APM trace metrics: throughput (req/s), error rate (%), avg latency (ms) — if available
- If the service is absent from the catalog: note that Datadog monitoring may not be configured
- If no issues found: "No active Datadog monitors or APM anomalies detected."
"""
)


def _build_datadog_api_step(enabled: bool) -> str:
    """Return the Datadog API step block when *enabled*, otherwise empty string."""
    if not enabled:
        return ""
    return _DATADOG_API_STEP


def _build_tool_reference_table(
    *,
    helm_enabled: bool = True,
    argocd_enabled: bool = True,
    datadog_api_enabled: bool = False,
) -> str:
    """Assemble the tool reference table from enabled sections.

    Only includes Helm, ArgoCD, and Datadog API tool rows when the
    corresponding integration is enabled, keeping the prompt lean and
    within Vertex AI's recommended 10-20 active tools guideline.
    """
    header = (
        "| Tool | Required Parameters | Optional Parameters |\n|------|---------------------|---------------------|"
    )
    sections = [header, _CORE_TOOLS_TABLE]
    if helm_enabled:
        sections.append(_HELM_TOOLS_TABLE)
    if argocd_enabled:
        sections.append(_ARGOCD_TOOLS_TABLE)
    if datadog_api_enabled:
        sections.append(_DATADOG_API_TOOLS_TABLE)
    return "\n".join(sections)


# ── Resource-triggered tool gating (SPEC-SH-10) ──────────────────────────────

_RESOURCE_TRIGGERED_TOOLS: Final[list[tuple[re.Pattern[str], list[str]]]] = [
    (
        re.compile(r"istio|ingressgateway|virtualservice|destinationrule|mesh", re.IGNORECASE),
        ["istio_get_virtual_services", "istio_get_destination_rules", "check_mtls_status"],
    ),
    (
        re.compile(r"argocd|gitops|application\.argoproj\.io", re.IGNORECASE),
        ["argocd_list_applications", "argocd_app_status", "argocd_app_diff"],
    ),
    (
        re.compile(r"helm|helmrelease|chart", re.IGNORECASE),
        ["helm_list_releases", "helm_release_status", "helm_release_history"],
    ),
    (
        re.compile(r"argo-rollout|rollout|canary|bluegreen|blue-green", re.IGNORECASE),
        ["get_rollout_status", "get_rollout_history", "kubectl_get_analysisrun"],
    ),
]


def _build_mandatory_tools_section(query: str = "", namespace: str = "") -> str:
    """Return a mandatory-tools hint block when the query/namespace matches known patterns.

    Checks *query* and *namespace* against :data:`_RESOURCE_TRIGGERED_TOOLS` and
    returns a formatted ``⚠️ MANDATORY TOOLS`` section listing every tool group
    that matched.  Returns ``""`` when there are no matches or both inputs are empty.

    Args:
        query: User query string — used to detect technology-specific patterns.
        namespace: Kubernetes namespace — also checked for pattern matches.

    Returns:
        Formatted multi-line hint string or ``""`` when no match.
    """
    combined = f"{query} {namespace}".strip()
    if not combined:
        return ""

    matched: list[tuple[str, list[str]]] = []
    for pattern, tools in _RESOURCE_TRIGGERED_TOOLS:
        if pattern.search(combined):
            matched.append((pattern.pattern, tools))

    if not matched:
        return ""

    lines = [
        "⚠️ MANDATORY TOOLS — resource/namespace pattern detected:",
        "",
    ]
    for _pattern, tools in matched:
        for tool in tools:
            lines.append(f"- You MUST call `{tool}` during this investigation.")
    lines.append("")
    return "\n".join(lines)

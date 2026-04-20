"""Sub-gatherer prompt builders for the service health skill parallel pipeline.

These builders are used by ``get_parallel_agents_config()`` in skill.py to
assemble the ``parallel_sequential`` pipeline (Phase 3).  Each builder
targets a focused subset of the full investigation so 4+ agents can run
concurrently instead of one monolithic gatherer running all steps sequentially.

Bug fixes applied vs. the original monolithic prompts.py:
- Added ANTI_HALLUCINATION_RULES to all 5 sub-gatherers (node, workload,
  event, logging, datadog) — previously only ANTI_INJECTION_RULE was present.
- Replaced inline _PRIORITY_HIERARCHY copy in build_workload_gatherer_prompt()
  with the shared constant from _shared.py.
"""

from vaig.core.prompt_defense import (
    ANTI_HALLUCINATION_RULES,
    ANTI_INJECTION_RULE,
    _sanitize_namespace,
    wrap_untrusted_content,
)

from ._shared import (
    _DATADOG_API_TOOLS_TABLE,
    _PRIORITY_HIERARCHY,
    _build_mandatory_tools_section,
)

# ── Evidence gap tracking instructions ───────────────────────
# Appended to every sub-gatherer system prompt so the LLM knows to
# populate an ``evidence_gaps`` list in its structured output for any
# tool that was not called, errored, or returned empty data.

TOOL_TRACKING_INSTRUCTIONS = """
## Evidence Gap Tracking

At the end of your response, you MUST report which signal sources produced
data and which did not.  Use the following structured output field:

```
evidence_gaps: list of objects with fields:
  - source  (str)  — tool or signal name, e.g. "deployment_metrics"
  - reason  (str)  — one of: "not_called", "error", "empty_result"
  - details (str, optional) — error message or skip reason
```

### Rules
- For every tool you were instructed to call but SKIPPED: add an entry with reason="not_called".
- For every tool call that raised an error or returned an API failure: add an entry with reason="error"
  and include the error message in `details`.
- For every tool call that succeeded but returned zero / empty results: add an entry with reason="empty_result".
- If a tool returned useful data, do NOT add an entry for it.
- An empty evidence_gaps list means all expected signal sources were checked and produced data.
"""


def _node_step3_instruction(prefetched_node_metrics: str) -> str:
    """Return the Step 3 instruction text for the node gatherer.

    When pre-fetched node metrics are available, instructs the agent to
    use the pre-gathered data instead of calling ``kubectl_top`` again.
    Otherwise, emits the standard ``kubectl_top`` tool call instruction.
    """
    if prefetched_node_metrics:
        return (
            "kubectl_top data for nodes was pre-gathered — see the "
            "PRE-GATHERED METRICS DATA section below.  Do NOT call "
            "``kubectl_top(resource_type=\"nodes\")`` again."
        )
    return "``kubectl_top(resource_type=\"nodes\")`` — CPU/memory utilisation per node"


def _workload_step2_instruction(prefetched_pod_metrics: str, ns: str) -> str:
    """Return the Step 2 kubectl_top instruction text for the workload gatherer.

    When pre-fetched pod metrics are available, instructs the agent to
    use the pre-gathered data instead of calling ``kubectl_top`` again.
    Otherwise, emits the standard MANDATORY ``kubectl_top`` tool call instruction.
    """
    if prefetched_pod_metrics:
        return (
            f"kubectl_top data for pods in namespace ``{ns}`` was pre-gathered — see the "
            "PRE-GATHERED METRICS DATA section below.  Do NOT call "
            "``kubectl_top`` again.  The reporter can use this data directly for the "
            "Service Status CPU/Memory columns."
        )
    return (
        f"``kubectl_top(resource_type=\"pods\", namespace=\"{ns}\")`` — real-time CPU/memory usage per pod\n"
        "    This is MANDATORY — the reporter needs real CPU and memory values for the Service Status table.\n"
        "    If this call fails, record the error and note \"kubectl_top unavailable\" — do NOT fabricate values."
    )


def build_node_gatherer_prompt(
    is_autopilot: bool = False,
    prefetched_node_metrics: str = "",
) -> str:
    """Build the system instruction for the ``node_gatherer`` sub-agent.

    The ``node_gatherer`` is responsible for **Step 1** of the standard SRE
    investigation checklist: Cluster Overview & Node Health.  It runs in
    parallel with :func:`build_workload_gatherer_prompt`,
    :func:`build_event_gatherer_prompt`, and
    :func:`build_logging_gatherer_prompt` inside a
    ``parallel_sequential`` pipeline.

    **Scope** — cluster-level resources only:

    * All node objects with their current conditions (``Ready``,
      ``MemoryPressure``, ``DiskPressure``, ``PIDPressure``, ``NetworkUnavailable``)
    * CPU and memory utilisation per node via ``kubectl top nodes``
    * Health of kube-system pods (control-plane components)
    * Namespace inventory
    * System-level events from the ``kube-system`` namespace

    **Output section produced**: ``## Cluster Overview``

    The section header is required by downstream agents and by
    :meth:`~vaig.skills.base.BaseSkill.get_required_output_sections` for
    validation.

    Args:
        is_autopilot: When ``True``, returns a lightweight 2-tool-call
            Autopilot-specific prompt that skips per-node investigation
            (nodes are managed by Google and not actionable).  When ``False``
            (default), returns the full 6-step Standard prompt unchanged.
        prefetched_node_metrics: Pre-gathered ``kubectl_top nodes`` output
            from :meth:`ServiceHealthSkill._pre_fetch_metrics`.  When
            non-empty, the prompt instructs the agent to use this data
            directly instead of calling ``kubectl_top`` again.

    Returns:
        A formatted system-instruction string injecting
        :data:`~vaig.core.prompt_defense.ANTI_INJECTION_RULE`,
        :data:`~vaig.core.prompt_defense.ANTI_HALLUCINATION_RULES`,
        and the appropriate node-gatherer task description.
    """
    if is_autopilot:
        return f"""{ANTI_INJECTION_RULE}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}

## Your Scope: Step 1 — Cluster Overview (Autopilot Mode)

This is a GKE Autopilot cluster. Nodes are managed by Google — do NOT
do deep per-node investigation. Your job is to establish CONTEXT only.

### Tools to use:
1. kubectl_get(resource="nodes") — list nodes (count, status summary ONLY)
2. kubectl_get(resource="namespaces") — namespace inventory

### What NOT to do:
- Do NOT call get_node_conditions for each node individually — in Autopilot,
  node conditions are Google's responsibility and not actionable
- Do NOT call kubectl_top(resource_type="nodes") — not available on Autopilot
- Do NOT call get_events(namespace="kube-system") — not actionable on Autopilot
- Do NOT report NotReady nodes as findings — Google recycles Autopilot nodes
  routinely, transient NotReady is NORMAL

### MANDATORY OUTPUT FORMAT
## Cluster Overview
- Cluster type: GKE Autopilot (managed nodes)
- Node count: <print the actual count> (<actual Ready count> Ready, <actual NotReady count> NotReady — NotReady is normal on Autopilot)
- Namespaces: <print the actual namespace list>
- Note: Node-level metrics and conditions are not investigated on Autopilot
  clusters because they are managed by Google and not actionable.

IMPORTANT: Replace all angle-bracket instructions above with actual values from
your tool call results. Do NOT output literal placeholders or brackets.
""" + TOOL_TRACKING_INSTRUCTIONS

    prompt = f"""{ANTI_INJECTION_RULE}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}

You are a focused Kubernetes diagnostic agent. Your ONLY responsibility is to
collect **cluster and node health data** (Step 1 of the standard SRE
investigation checklist).  Do NOT perform pod analysis, log queries, or any
work outside this scope.

## Your Scope: Step 1 — Cluster Overview & Node Health

### Tools to use (in order):
1. ``kubectl_get(resource="nodes", output="wide")`` — list all nodes with status
2. ``get_node_conditions(name="<each node>")`` — for every node returned
   in step 1, call this to retrieve the full condition set
3. {_node_step3_instruction(prefetched_node_metrics)}
4. ``kubectl_get(resource="pods", namespace="kube-system", output="wide")`` — system component health
5. ``kubectl_get(resource="namespaces")`` — cluster namespace inventory
6. ``get_events(namespace="kube-system", event_type="Warning")`` — system-level warning events
   NOTE: DO NOT use ``kubectl_get`` for events — ``events`` is a blocked resource; always use ``get_events`` instead.

### Data collection rules:
- Call ``get_node_conditions`` for EVERY node, not just unhealthy ones.
- If ``kubectl_top nodes`` fails (e.g. Autopilot cluster), note "kubectl_top nodes
  not available (Autopilot)" and continue — do NOT fabricate CPU/memory values.
- If a tool call fails, record the error verbatim and move on — NEVER invent data.

### MANDATORY OUTPUT FORMAT

Produce exactly this section at the end of your response:

## Cluster Overview

### Node Inventory
| Node | Status | Roles | Age | Version | CPU Usage | Memory Usage |
|------|--------|-------|-----|---------|-----------|--------------|
(one row per node; use "N/A" for any value not returned by tools)

### Node Conditions
(For each node: list all conditions from get_node_conditions — include True/False/Unknown, reason, message)

### Resource Pressure
(List any nodes with DiskPressure, MemoryPressure, PIDPressure, or NotReady conditions)

### System Components (kube-system)
(List non-Running pods in kube-system; if all healthy write "All kube-system pods Running and Ready")

### Namespaces
(List all namespaces and their status)

### Key Findings
(Bullet list of significant node-level observations — if none write "No node-level issues detected")

### CRITICAL RULES:
- ONLY report data returned by the tools above.  NEVER fabricate node names,
  IP addresses, kernel versions, CPU percentages, or memory values.
- If tool output is empty or unavailable, state "No data returned by tool." —
  do NOT invent substitute data.
- Every value in the Node Inventory table MUST come from an actual tool call.
"""

    # ── Inject pre-fetched node metrics when available ────────────────
    if prefetched_node_metrics:
        wrapped_metrics = wrap_untrusted_content(prefetched_node_metrics)
        prompt += f"""
## PRE-GATHERED METRICS DATA — kubectl_top nodes

The following ``kubectl_top(resource_type="nodes")`` output was gathered
programmatically BEFORE your execution started.  Use this data directly
for the Node Inventory CPU/Memory columns.  Do NOT call ``kubectl_top``
again — the data is already here.

{wrapped_metrics}
"""

    prompt += TOOL_TRACKING_INSTRUCTIONS
    return prompt


def build_workload_gatherer_prompt(
    namespace: str = "",
    datadog_api_enabled: bool = False,  # noqa: ARG001 — deprecated; Batch 2 will remove callers
    argo_rollouts_enabled: bool = False,
    prefetched_pod_metrics: str = "",
    user_query: str = "",
) -> str:
    """Build the system instruction for the ``workload_gatherer`` sub-agent.

    The ``workload_gatherer`` is responsible for **Steps 2, 4, 5, 6** of the
    standard SRE investigation checklist.  It runs in parallel with the other
    sub-gatherers in the ``parallel_sequential`` pipeline.

    **Scope** — pod and workload-level resources:

    * **Step 2** — Pod Status Analysis: running/failed pods, container statuses,
      restart counts, CrashLoopBackOff investigation via logs and describe.
    * **Step 4** — Deployment Deep-Dive: rollout status, rollout history,
      unhealthy deployment conditions, management annotations (ArgoCD, Helm,
      Flux, operators).
    * **Step 5** — Service & Endpoint Health: ClusterIP/NodePort/LoadBalancer
      services, endpoint readiness, missing endpoint backends.
    * **Step 6** — HPA / Autoscaling: HorizontalPodAutoscaler targets vs
      current replicas, scaling events.

    Datadog API correlation (real-time metrics and monitors) is handled by the
    dedicated ``datadog_gatherer`` running in the same parallel group.

    **Output sections produced**: ``## Service Status``, ``## Raw Findings``
    (workload portion).

    Args:
        namespace: Kubernetes namespace to investigate.
        datadog_api_enabled: Deprecated — no longer used.  Datadog API
            correlation has moved to ``build_datadog_gatherer_prompt()``.
            This parameter is retained for backward-compatibility until
            all callers are updated in Batch 2.
        argo_rollouts_enabled: When ``True``, injects additional guidance
            for Argo Rollouts workloads — the ``Rollout → ReplicaSet → Pod``
            ownership chain alongside standard Deployments, and instructs
            the agent to call ``kubectl_get_rollout`` and
            ``kubectl_get_analysisrun`` as part of Step 4.
        prefetched_pod_metrics: Pre-gathered ``kubectl_top pods`` output
            from :meth:`ServiceHealthSkill._pre_fetch_metrics`.  When
            non-empty, the prompt instructs the agent to use this data
            directly instead of calling ``kubectl_top`` again.

    Returns:
        A formatted system-instruction string injecting
        :data:`~vaig.core.prompt_defense.ANTI_INJECTION_RULE`,
        :data:`~vaig.core.prompt_defense.ANTI_HALLUCINATION_RULES`,
        and the full workload-gatherer task description.
    """
    # Safe namespace for embedding in tool call examples — validated K8s name.
    # Falls back to "default" when the input is empty or invalid (injection attempt).
    ns = _sanitize_namespace(namespace) or "default"
    # Derive ns_context from ns (not from raw namespace) so the context message
    # is consistent with the fallback value used in tool call examples.
    ns_context = (
        f"Target namespace: {ns}.  "
        "Also check any other non-system namespaces if relevant."
        if namespace and _sanitize_namespace(namespace)
        else (
            "If no explicit namespace is given in the query, investigate ALL non-system namespaces found in\n"
            "the cluster. Prioritise namespaces with the highest pod count."
        )
    )
    # ── SPEC-SH-10: inject mandatory tools section ────────────────────────
    mandatory_tools_section = _build_mandatory_tools_section(query=user_query, namespace=namespace)
    prompt = f"""{ANTI_INJECTION_RULE}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}

You are a focused Kubernetes diagnostic agent. Your ONLY responsibility is to
collect **workload health data** (Steps 2, 4, 5, 6 of the standard SRE
investigation checklist).  Do NOT collect node data, events, or Cloud Logging
— those are handled by other agents running in parallel.

### Target namespace:
{ns_context}

## Your Scope

### Step 2 — Pod Status Analysis
1. ``kubectl_get(resource="pods", namespace="{ns}", output="wide")`` — all pods
2. {_workload_step2_instruction(prefetched_pod_metrics, ns)}
    For **historical trends** (past 1h by default), also call ``get_pod_metrics(namespace="{ns}", pod_name_prefix="<workload-prefix>")``
    to retrieve Cloud Monitoring time-series data (Avg/Max/Latest/Trend per pod). Use ``kubectl_top``
    for current-state values in the Service Status table and ``get_pod_metrics`` for trend context.
    When populating the **Service Status** table (one row per workload: Deployment, StatefulSet,
    DaemonSet, Job, or CronJob), aggregate these per-pod metrics at the workload level:
    - Associate each pod with its owning workload using ownerReferences or standard labels
      (app, app.kubernetes.io/name, or controller-specific labels).
    - For each workload: use the **TOTAL across all pods** (CPU total, memory total across all pods).
      ALWAYS include pod count and per-pod average, e.g. "0.38 cores (10 pods, 0.038/pod avg)".
      Without per-pod context, totals across many pods look misleading to readers.
    - Include a pod count per workload (ready/total pods).
    - Never report raw per-pod values in the Service Status table — always use aggregated per-workload values.

    After collecting per-container metrics from kubectl_top:
    - Step 1: Calculate per-POD totals by summing all containers in each pod
    - Step 2: Calculate per-DEPLOYMENT TOTALS by summing across all pods (do NOT average)
    - Present a summary like:
      Deployment: payment-svc | Pods: 10 | CPU Total: 0.563 cores | Mem Total: 42.7Gi | CPU/Pod avg: 0.056 cores | Mem/Pod avg: 4.3Gi
    - The per-DEPLOYMENT TOTAL is what the reporter needs for cpu_usage/memory_usage fields.
    - CPU values are always expressed as decimal cores (e.g. "0.563 cores") — NEVER millicore "m" notation.
    - Memory uses decimal Gi notation (e.g. "42.7Gi", "0.5Gi") if ≥ 1 GiB, or Mi notation (e.g. "105Mi", "512Mi") if < 1 GiB.
    - Per-container breakdown goes in Raw Findings for detailed analysis.
3. For any pod NOT in Running/Succeeded state: ``get_container_status(name="<pod>", namespace="{ns}")``
4. For pods with restart count > 3: ``kubectl_logs(pod="<pod>", namespace="{ns}")``
5. For CrashLoopBackOff pods: ``kubectl_describe(resource_type="pod", name="<pod>", namespace="{ns}")``

### Step 4 — Workload Deep-Dive
6. ``kubectl_get(resource="deployments", namespace="{ns}", output="wide")`` — all deployments
7. ``get_rollout_status(name="<name>", namespace="{ns}")`` for each deployment
8. For unhealthy deployments: ``kubectl_describe(resource_type="deployment", name="<name>", namespace="{ns}")``
9. ``get_rollout_history(name="<name>", namespace="{ns}")`` for recently changed deployments
10. Check ``kubectl_get_labels`` equivalent via ``kubectl_describe`` to detect management annotations
   (ArgoCD: ``argocd.argoproj.io/``, Flux: ``fluxcd.io/``, Helm: ``app.kubernetes.io/managed-by: Helm``,
   OwnerReferences for operator-managed resources, ``.spec.template.metadata.annotations`` for
   webhook injection annotations) — report these management indicators for the reporter

### Step 4b — Management Context Detection (extends Step 4; MANDATORY)
11. Resolve which **workload controller types** are present from Step 2's pod
    ``ownerReferences`` analysis.  Typical ownership chains:
    - ``Pod → ReplicaSet → Deployment`` → the top-level owner is a **Deployment**
    - ``Pod → ReplicaSet`` (no Deployment owner) → standalone **ReplicaSet**
    - ``Pod → StatefulSet`` → **StatefulSet**
    - ``Pod → DaemonSet`` → **DaemonSet**

    Call ``kubectl_get_labels`` for **each distinct top-level workload type** found:
    - ``kubectl_get_labels(resource_type="deployments", namespace="{ns}")`` — if Deployments exist
    - ``kubectl_get_labels(resource_type="replicasets", namespace="{ns}")`` — if standalone ReplicaSets exist (not owned by a Deployment)
    - ``kubectl_get_labels(resource_type="statefulsets", namespace="{ns}")`` — if StatefulSets exist
    - ``kubectl_get_labels(resource_type="daemonsets", namespace="{ns}")`` — if DaemonSets exist

    If Step 2 has not yet been analyzed or ownerReferences are unclear,
    default to ``resource_type="deployments"`` first; if that returns no data,
    fall back to ``"statefulsets"``, ``"daemonsets"``, ``"replicasets"`` in order.

    From the output, look for these management indicators:

    Look for these Helm indicators:
    - Label ``app.kubernetes.io/managed-by: Helm`` → resource is Helm-managed
    - Label ``helm.sh/chart`` → chart name and version
    - Annotation ``meta.helm.sh/release-name`` → Helm release name
    - Annotation ``meta.helm.sh/release-namespace`` → namespace of the release

    **``meta.helm.sh/release-name`` (CRITICAL — record explicitly)**:
    If annotation ``meta.helm.sh/release-name`` is present on a workload, record its
    exact value as ``<helm_release_name>`` for that workload. This value is used
    directly in Step 9 (Helm assessment) — it is the Helm release name and allows
    skipping ``helm_list_releases()``.

    Look for these ArgoCD indicators:
    - Annotation ``argocd.argoproj.io/managed-by`` → ArgoCD app name
    - Label ``app.kubernetes.io/instance`` → release/app instance name

    **Datadog labels (record for completeness)**:
    Look for the following in workload labels (from ``kubectl_get_labels`` output)
    and in ``.spec.template.spec.containers[].env`` (from any workload YAML retrieved
    in Step 4 deep-dive):
    - Label ``tags.datadoghq.com/service`` or env var ``DD_SERVICE``
      → If found, record its exact value as ``<dd_service>`` for that workload.
    - Label ``tags.datadoghq.com/env`` or env var ``DD_ENV``
      → If found, record its exact value as ``<dd_env>`` for that workload.
    Record these prominently in the Management Indicators section — the
    ``datadog_gatherer`` (running in parallel) will use them for Datadog API
    correlation.

    For each Helm-managed workload, report in the Management Indicators section:
    ``"Managed by: Helm (release: <release-name>, chart: <chart-name>)"``
    For each ArgoCD-managed workload, report:
    ``"Managed by: ArgoCD (app: <app-name>)"``

### Step 4c — Pod Ownership Chain Tracing (MANDATORY for CrashLoopBackOff / Error pods)

For **every** pod that is in ``CrashLoopBackOff``, ``Error``, ``OOMKilled``, or ``ImagePullBackOff``
state, trace the full ownership chain before moving on:

a. **Identify owning ReplicaSet** — inspect the pod's ``ownerReferences`` field from the
   ``kubectl_describe(resource_type="pod", name="<pod>", namespace="{ns}")`` output.
    For ``CrashLoopBackOff`` pods this was already collected in Step 2 (item 5).
    For ``Error``, ``OOMKilled``, or ``ImagePullBackOff`` pods it may not yet be available —
    **run** ``kubectl_describe(resource_type="pod", name="<pod>", namespace="{ns}")`` now if you
   have not already done so.  Look for ``kind: ReplicaSet`` and record its ``name``.

b. **Identify owning Deployment** — inspect the ReplicaSet's ``ownerReferences`` via
   ``kubectl_describe(resource_type="replicaset", name="<rs-name>", namespace="{ns}")``
   and record the owning Deployment name.

c. **Correlate with rollout revision** — call
   ``get_rollout_history(name="<deployment>", namespace="{ns}")`` to list all revisions.
   Match the failing ReplicaSet to its revision by comparing the ``pod-template-hash``
   label on the ReplicaSet against the revision entries in the history.
   Record the revision number that introduced the failure.

d. **Report** in the Deployment Issues section:
   ``"Pod <pod-name> → RS <rs-name> → Deployment <deploy-name> (revision <N>)"``

If ``get_rollout_history`` is not in your available tools list, SKIP sub-step (c) and
mark it as ``SKIPPED — tool unavailable`` in the Deployment Issues section.

### Step 5 — Service & Endpoint Connectivity
12. ``kubectl_get(resource="services", namespace="{ns}", output="wide")``
13. ``kubectl_get(resource="endpoints", namespace="{ns}")``
14. For services with 0 endpoints: ``kubectl_describe(resource_type="service", name="<svc>", namespace="{ns}")``

### Step 6 — HPA & Scaling Status
15. ``kubectl_get(resource="hpa", namespace="{ns}", output="wide")``
16. For HPA at maxReplicas: ``kubectl_describe(resource_type="hpa", name="<hpa>", namespace="{ns}")``
17. ``gcloud_monitoring_query(...)`` if HPA uses custom metrics and metric fetch is failing

### Step 6b — Scaling Deep-Dive (HPA + VPA)
For each deployment that has an HPA or that has a ``VerticalPodAutoscaler`` resource:
18. Call ``get_scaling_status(name="<deployment_name>", namespace="{ns}")``
    - Note: **ceiling-hit** — when ``current_replicas == max_replicas`` and the workload is
      still under load.  This means the HPA cannot scale further and the service is at risk.
    - Note: **VPA-vs-HPA conflicts** — if both VPA and HPA are present and VPA is in ``Auto``
      or ``Recreate`` mode while HPA scales on CPU/memory, they may fight each other.
      Report any mismatch between VPA recommended limits and HPA-driven replica counts.
    - If ``get_scaling_status`` is not in your available tools list, SKIP this sub-step and
      mark it as SKIPPED in the Investigation Checklist.

### Step 6c — Metrics API Health (MANDATORY when HPAs exist)
19. **ALWAYS** call ``check_metrics_api_health()`` when Step 6 found ANY HPA — no parameters needed.
    Do NOT skip this step when the tool is available. It is essential for diagnosing HPA scaling failures.
    - This checks whether metrics.k8s.io, custom.metrics.k8s.io, and external.metrics.k8s.io
      API groups are registered and healthy.
    - If ANY HPA references custom or external metrics and the corresponding API
      group is unavailable, this is a **root cause** for HPA scaling failure — flag it as CRITICAL.
    - If ``check_metrics_api_health`` is not in your available tools list, SKIP this sub-step and
      mark it as SKIPPED in the Investigation Checklist.
    - After completing this step, you MUST immediately proceed to Step 6d below.
      Do NOT emit your final response until you have checked whether the HPA uses custom or external metrics.

### Step 6d — Custom / External Metric Queries (MANDATORY when HPAs use custom/external metrics)
20. If an HPA references **custom metrics** (e.g. ``type: Pods`` or ``type: Object``
    with ``custom.metrics.k8s.io``), you **MUST** call ``query_custom_metrics(metric_name="<metric>",
    namespace="{ns}")`` to verify the metric exists and has data.
    Do NOT skip this when the tool is available — missing metric data is the #1 cause of HPA scaling failures.
    - If the metric returns **no data**, this is likely the root cause for
      ``FailedGetCustomMetric`` or ``ScalingLimited`` HPA conditions.
    - Call ``query_custom_metrics()`` (no metric_name) to **list all available**
      custom metrics when you need to verify what metrics the adapter exposes.

**How to recognize external metrics in HPA output**: Look for ``type: External`` in the
HPA spec, OR any metric name containing pipes (``|``) such as ``istio.io|service|server|request_count``
or ``pubsub.googleapis.com|subscription|num_undelivered_messages``. In GKE with Stackdriver adapter,
pipe-separated metric names typically indicate external metrics from Cloud Monitoring.

21. **ALWAYS CHECK**: If an HPA references **external metrics** (e.g. ``type: External``
    with ``external.metrics.k8s.io``), you **MUST** call ``query_external_metrics(metric_name="<metric>",
    namespace="{ns}")`` to verify the metric exists and has data.
    Do NOT skip this when the tool is available — missing metric data is the #1 cause of HPA scaling failures.
    - Common external metrics: ``pubsub.googleapis.com|subscription|num_undelivered_messages``,
      ``custom.googleapis.com/*``.
    - If the metric returns **no data** or a 404, this is likely the root cause for
      ``FailedGetExternalMetric`` HPA conditions.
    - If ``query_custom_metrics`` or ``query_external_metrics`` are not in your available
      tools list, SKIP this sub-step and mark it as SKIPPED in the Investigation Checklist.

### Step 6e — Adapter Pod Health (when HPA external metrics are unknown or stale)
22. If external metrics show ``?/?`` or stale values, call ``get_pods`` filtering for adapter
    pods named ``prometheus-adapter``, ``datadog-cluster-agent``, or ``custom-metrics-apiserver``
    in the ``kube-system`` and ``monitoring`` namespaces.
    Report: pod status, restart count, and any OOM-kill events.

### Envoy / Istio Sidecar Diagnostics
If the pod has a container named `istio-proxy` or `envoy`, or the annotation
`sidecar.istio.io/inject: "true"`, treat it as mesh-enrolled and run:

1. exec_command: curl -s http://localhost:15000/clusters
   → captures upstream cluster health and endpoint status
2. exec_command: curl -s "http://localhost:15000/stats?filter=upstream"
   → captures upstream retry/error counters

If localhost:15000 is unreachable, note the failure and continue — do not abort.
Include Envoy findings in the returned findings dict under key "envoy_admin".

### Data collection rules:
- For kubectl_logs: use ``pod="<name>"`` — NOT ``pod_name``. No ``previous`` parameter.
- For get_container_status: use ``name="<name>"`` — NOT ``pod_name``.
- If a tool call fails, record the error and continue — NEVER invent substitute output.
- Collect management annotations (ArgoCD, Flux, Helm, OwnerReferences) for every failing workload.

### PRIORITY HIERARCHY — Kubernetes vs Datadog
{_PRIORITY_HIERARCHY}

{mandatory_tools_section}### MANDATORY OUTPUT FORMAT

Produce exactly these sections at the end of your response:

## Service Status

| Service/Deployment | Namespace | Status | Ready | Restarts | CPU Usage | Memory Usage | Issue |
|--------------------|-----------|--------|-------|----------|-----------|--------------|-------|
(one row per workload; Status: Running/Degraded/Failed/Unknown; use "N/A" for missing values)

## Raw Findings (Workload)

### Unhealthy Pods
(For each pod not in Running/Succeeded: name, namespace, status, restart count, last error from logs/events)

### Deployment Issues
(For each deployment not fully available: name, desired vs ready replicas, rollout status, last revision)

### Service Connectivity Gaps
(For each service with 0 ready endpoints: service name, selector, reason if determinable)

### HPA Issues
(For each HPA at maxReplicas or with unknown metrics: name, current/desired/max replicas, metric status)

### Management Indicators
(For ALL deployments: detected management method — GitOps/Helm/Operator/Manual — with evidence.
Report even for healthy deployments, as management context affects remediation recommendations.)

### CRITICAL RULES:
- ONLY report data returned by tool calls.  NEVER fabricate pod names, restart counts,
  replica counts, or error messages.
- If a namespace has no workload issues, write "No workload issues detected in <namespace>."
- A shorter, 100% accurate report is always better than a longer report with invented data.
"""
    if argo_rollouts_enabled:
        argo_rollouts_section = f"""
### Step 4d — Argo Rollouts Workloads (MANDATORY when Argo Rollouts is detected)

Argo Rollouts replaces standard Kubernetes Deployments with ``Rollout`` CRDs that implement
canary and blueGreen strategies.  The ownership chain is:

    Rollout → ReplicaSet → Pod

This is analogous to ``Deployment → ReplicaSet → Pod`` but uses the Argo Rollouts controller.

**IMPORTANT — Deployment stubs**: When Argo Rollouts manages a workload, it deliberately sets
``spec.replicas=0`` on the corresponding Kubernetes Deployment (making it a passive stub).
The ``get_rollout_status`` tool will return ``Overall Status: Managed by Argo Rollout`` for
these Deployments.  A ``Managed by Argo Rollout`` status MUST NOT be reported as "scaled to
zero", "unavailable", or any other health issue.  It is the expected and correct state.
The actual replica counts and health information live in the Rollout object, not the Deployment.

**IMPORTANT — ``## Service Status`` table with Argo Rollouts**: For the ``## Service Status``
table, emit **one row per Argo Rollout** instead of one row per Deployment stub.  When a
Deployment is a stub (``spec.replicas=0``, managed by Argo Rollout), **omit that Deployment
row entirely** and replace it with a row for the corresponding Rollout.  Use the Rollout's
actual replica counts (``ready/desired``) for the ``Ready`` column — NEVER the stub
Deployment's ``0/0``.  Map Rollout phase to the ``Status`` column as follows:
``Healthy → Running``, ``Progressing → Degraded``, ``Degraded → Failed``, ``Error → Failed``,
``Paused → Unknown``, ``Unknown → Unknown``.  For any unrecognized phase, default ``Status`` to
``Unknown``.  This ensures the table reflects real service health, not the intentional stub state.

For each namespace, perform:

a. ``kubectl_get_rollout(namespace="{ns}")`` — list all Rollouts, note phase, strategy,
   and replica counts (desired/ready/available/updated).  A phase of ``Degraded``, ``Paused``,
   or ``Error`` is a finding that must be reported.

b. For any Rollout **not** in ``Healthy`` phase:
   ``kubectl_get_rollout(namespace="{ns}", name="<rollout-name>")`` — get full detail.

c. ``kubectl_get_analysisrun(namespace="{ns}")`` — list recent AnalysisRuns.  A phase of
   ``Failed`` or ``Error`` is a root-cause signal (the rollout was aborted because analysis failed).

d. For any failed AnalysisRun, report: run name, phase, and which metric(s) failed.

e. For pod ownership tracing (Step 4c): when a failing pod's owning ReplicaSet is controlled
   by a ``Rollout`` (check ``ownerReferences.kind == "Rollout"``), report:
   ``"Pod <pod-name> → RS <rs-name> → Rollout <rollout-name> (canary|blueGreen)"``
   instead of the standard Deployment chain.

f. Add a ``### Rollout Issues`` sub-section to ``## Raw Findings (Workload)`` if any Rollouts
   are not in Healthy phase.  Format:
   ``"Rollout <name> in <namespace>: phase=<phase>, strategy=<canary|blueGreen>, <detail>"``

g. **Rollout strategy extraction** — for EVERY Rollout found in step (a), extract the strategy
   type from the Rollout spec:
   - If ``spec.strategy.canary`` is present → strategy is ``canary``
   - If ``spec.strategy.blueGreen`` is present → strategy is ``blue-green``
   The ``kubectl_get_rollout`` output includes a "Strategy:" line — use it.
   Include the strategy in the per-service output as:
   ``"rollout_strategy: canary"`` or ``"rollout_strategy: blue-green"``

   **IMPORTANT — camelCase normalization**: the tool output may show ``blueGreen`` (camelCase).
   ALWAYS normalize to ``blue-green`` (hyphenated) in your output. Emit ``canary`` as-is.

h. **Rollout status for output schema** — map the Rollout phase to one of these values and
   include it in the per-service output as ``"rollout_status: <value>"``:
   - ``Healthy`` → ``"Healthy"``
   - ``Progressing`` → ``"Progressing"``
   - ``Paused`` → ``"Paused"``
   - ``Degraded`` or ``Error`` → ``"Degraded"``

i. **HPA conditions for Rollout-managed workloads** — when collecting HPA data (Step 4b),
   check whether the HPA's ``spec.scaleTargetRef.kind`` is ``"Rollout"`` (not ``"Deployment"``).
   If so, extract all HPA ``status.conditions`` messages and report them as:
   ``"hpa_conditions: [<condition-message-1>, <condition-message-2>, ...]"``
   This list may be empty if the HPA has no conditions.
   These conditions are CRITICAL context for diagnosing scaling failures on Argo-managed workloads.

j. **Rollout Details section** — inside ``## Raw Findings (Workload)``, add a
    ``### Rollout Details`` subsection (NOT a top-level section, NOT a subsection of
    ``## Service Status``) for every service that has rollout data.
    Use ``#### <service-name>`` for per-service entries.  Use this format per service:

    ``#### <service-name>``
    - rollout_strategy: <blue-green|canary|N/A>
    - rollout_status: <Healthy|Progressing|Paused|Degraded|N/A>
    - hpa_conditions: <condition1; condition2|none>
"""
        prompt = prompt + argo_rollouts_section

    # ── Inject pre-fetched pod metrics when available ─────────────────
    if prefetched_pod_metrics:
        wrapped_metrics = wrap_untrusted_content(prefetched_pod_metrics)
        prompt += f"""
## PRE-GATHERED METRICS DATA — kubectl_top pods

The following ``kubectl_top(resource_type="pods")`` output was gathered
programmatically BEFORE your execution started.  Use this data directly
for the Service Status CPU/Memory columns.  Do NOT call ``kubectl_top``
again — the data is already here.  You still need ``get_pod_metrics``
    for historical trends if required.

{wrapped_metrics}
"""

    prompt += TOOL_TRACKING_INSTRUCTIONS
    return prompt


def build_datadog_gatherer_prompt(
    namespace: str = "",
    cluster_name: str = "",
    datadog_api_enabled: bool = False,
    dd_service_name: str = "",
    dd_env: str = "",
    dd_resource_type: str = "",
) -> str:
    """Build the system instruction for the ``datadog_gatherer`` sub-agent.

    The ``datadog_gatherer`` is responsible for **Datadog API Correlation**
    (equivalent to the former Step 12 of the monolithic workload_gatherer
    prompt).  It runs in parallel with the other sub-gatherers in the
    ``parallel_sequential`` pipeline.

    Because all gatherers run in parallel, this agent CANNOT depend on output
    from ``workload_gatherer``.  It resolves ``dd_service`` / ``dd_env``
    labels independently by calling ``kubectl_get_labels`` as Step 0 before
    making any Datadog API calls.

    When ``dd_service_name`` is provided (pre-resolved programmatically in
    ``get_parallel_agents_config``), Step 0 is replaced with a directive that
    uses the pre-resolved name directly — eliminating LLM non-determinism.

    **Scope** — Datadog API correlation only:

    * **Step 0** — Label Resolution: call ``kubectl_get_labels`` to discover
      ``tags.datadoghq.com/service`` and ``tags.datadoghq.com/env`` values from
      pod/deployment labels and annotations (environment variables such as
      ``DD_SERVICE``/``DD_ENV`` are NOT available via this tool).
    * **Calls 1–2** — ``query_datadog_metrics`` (CPU, then memory).
    * **Call 3** — ``get_datadog_monitors`` for active alerts.
    * **Call 4** — ``get_datadog_service_catalog`` for ownership metadata.
    * **Call 5** — ``get_datadog_apm_services`` for live APM trace metrics.

    If ``datadog_api_enabled`` is ``False`` the function returns an empty
    string — the caller (``get_parallel_agents_config``) is expected to skip
    adding this agent when Datadog is disabled.

    **Output section produced**: ``## Raw Findings (Datadog API)``.

    Args:
        namespace: Kubernetes namespace to investigate.
        cluster_name: GKE cluster name, embedded verbatim in Datadog tool call
            examples (``cluster_name=`` parameter for ``query_datadog_metrics``
            and ``get_datadog_monitors``).
        datadog_api_enabled: When ``False``, returns an empty string so the
            caller can gate the agent on the runtime Datadog setting.
        dd_service_name: Pre-resolved Datadog service name from
            ``_resolve_dd_service_name``.  When non-empty, Step 0 is replaced
            with a directive to use this value directly.
        dd_env: Pre-resolved ``tags.datadoghq.com/env`` value.  When both
            ``dd_service_name`` and ``dd_env`` are pre-resolved, Step 0 tells
            the LLM to skip ``kubectl_get_labels`` entirely.
        dd_resource_type: The K8s resource type the winning workload came from
            (e.g. ``"deployments"``).  Used in the Step 0 fallback when only
            ``dd_service_name`` is pre-resolved but ``dd_env`` is not.

    Returns:
        A formatted system-instruction string, or an empty string when Datadog
        API integration is disabled.
    """
    if not datadog_api_enabled:
        return ""

    # Safe namespace for embedding in tool call examples — validated K8s name.
    # Falls back to "default" when the input is empty or invalid (injection attempt).
    ns = _sanitize_namespace(namespace) or "default"
    ns_context = (
        f"Target namespace: {ns}."
        if namespace and _sanitize_namespace(namespace)
        else "No explicit namespace given — query all non-system namespaces or use cluster-wide scope."
    )
    # Embed the actual cluster name in tool call examples so the LLM produces
    # correct calls.  Fall back to the generic placeholder when not supplied.
    cluster = cluster_name if cluster_name else "<cluster>"

    # Build a focused tool reference table.  kubectl_get_labels is ALWAYS
    # included — even when the DD service name is pre-resolved, the LLM may
    # still need it for dd_env resolution or other label lookups.
    tool_reference_table = (
        "| Tool | Required Parameters | Optional Parameters |\n"
        "|------|---------------------|---------------------|\n"
        '| `kubectl_get_labels` | `resource_type` | `namespace`, `name`, `label_filter`, `annotation_filter` |\n'
        + _DATADOG_API_TOOLS_TABLE
    )

    # ── Step 0 block: fully pre-resolved / partially pre-resolved / LLM-based ──
    if dd_service_name and dd_env:
        # Both service and env pre-resolved → skip kubectl_get_labels entirely.
        step_0_block = f"""### Step 0 — Service Identity (PRE-RESOLVED)

The Datadog service name AND environment have been pre-resolved programmatically
from Kubernetes workload labels.  Do NOT call ``kubectl_get_labels`` to resolve
the service name or environment — use these values directly:

- ``<dd_service>``: **{dd_service_name}**
- ``<dd_env>``: **{dd_env}**

Use ``<dd_service>`` = ``{dd_service_name}`` and ``<dd_env>`` = ``{dd_env}``
in ALL subsequent Datadog API calls (Calls 1–5) as the ``service`` and ``env``
parameters respectively."""
    elif dd_service_name:
        # Service pre-resolved but env is not — need kubectl_get_labels for env only.
        # Use the correct resource_type from resolution instead of hardcoded "deployments".
        env_resource = dd_resource_type or "deployments"
        step_0_block = f"""### Step 0 — Service Identity (PRE-RESOLVED)

The Datadog service name has been pre-resolved programmatically from Kubernetes
workload labels.  Do NOT call ``kubectl_get_labels`` to resolve the service name
— use this value directly:

- ``<dd_service>``: **{dd_service_name}**
- ``<dd_env>``: Not pre-resolved — call ``kubectl_get_labels(resource_type="{env_resource}", namespace="{ns}")``
  ONLY to resolve ``tags.datadoghq.com/env``.  If the call fails, omit the ``env`` parameter.

Use ``<dd_service>`` = ``{dd_service_name}`` in ALL subsequent Datadog API calls
(Calls 1–5) as the ``service`` parameter."""
    else:
        step_0_block = f"""### Step 0 — Independent Label Resolution (MANDATORY — do this FIRST)

Because this agent runs in parallel with the workload_gatherer, you CANNOT
access its output.  Resolve the Datadog service identity independently:

1. Call ``kubectl_get_labels(resource_type="deployments", namespace="{ns}")``
   to retrieve labels and annotations for all deployments in the target namespace.
2. **If no deployments are found** (empty output or "No deployments found"), the
   namespace may use a different workload type.  Try each in order until you get
   label data:
   - ``kubectl_get_labels(resource_type="statefulsets", namespace="{ns}")``
   - ``kubectl_get_labels(resource_type="daemonsets", namespace="{ns}")``
   - ``kubectl_get_labels(resource_type="replicasets", namespace="{ns}")``
   (ReplicaSets are tried last because they are usually owned by Deployments;
   standalone ReplicaSets without a Deployment owner are uncommon.)
3. From the output, identify the Datadog Unified Service Tagging values using
   this priority order (``kubectl_get_labels`` returns labels and annotations ONLY —
   container environment variables such as ``DD_SERVICE``/``DD_ENV`` are NOT available):
   - **Tier 1** — Datadog UST pod/deployment labels:
     - ``tags.datadoghq.com/service`` → store as ``<dd_service>``
     - ``tags.datadoghq.com/env``     → store as ``<dd_env>``
   - **Tier 2** — Generic Kubernetes identity labels (if Tier 1 absent):
     - ``app.kubernetes.io/name`` → store as ``<dd_service>``
     - ``app`` → store as ``<dd_service>``
     - Workload name (deployment/statefulset/daemonset/replicaset name) → store as ``<dd_service>`` (last resort)
4. If all ``kubectl_get_labels`` calls fail or return no useful data, proceed with
   cluster-wide Datadog queries (omit ``service=`` and ``env=`` parameters).
   Record the failure reason in Raw Findings."""

    prompt = f"""{ANTI_INJECTION_RULE}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}

You are a focused Kubernetes diagnostic agent. Your ONLY responsibility is to
collect **Datadog API correlation data** for the target namespace.  Do NOT
collect pod logs, node data, events, or workload status — those are handled by
other agents running in parallel.

### Target namespace:
{ns_context}

## Tool Call Reference — EXACT Parameter Names

Use ONLY these parameter names when calling tools. Using wrong names causes runtime errors.

{tool_reference_table}

## PRIORITY HIERARCHY — Kubernetes vs Datadog

{_PRIORITY_HIERARCHY}

## EXECUTION ORDER — FOLLOW THIS EXACT SEQUENCE

Step 0 (label resolution) → Calls 1–2 (metrics) → Call 3 (monitors) → Call 4 (service catalog) → Call 5 (APM services)

You MUST complete all 5 calls (1–5).  Step 0 is mandatory preparation — do NOT skip it.

## Data Collection Procedure

{step_0_block}

### Calls 1–2 — Datadog Metrics (CPU and Memory)

**Call 1** — CPU metrics:
You MUST call ``query_datadog_metrics(cluster_name="{cluster}", metric="cpu")``
[add ``service="<dd_service>", env="<dd_env>"`` ONLY if resolved in Step 0].
- Correlate the returned CPU time-series with the namespace under investigation.
- Example with labels: ``query_datadog_metrics(cluster_name="{cluster}", metric="cpu", service="my-api", env="production")``
- Example without labels: ``query_datadog_metrics(cluster_name="{cluster}", metric="cpu")``

**Call 2** — Memory metrics:
You MUST call ``query_datadog_metrics(cluster_name="{cluster}", metric="memory")``
[add ``service=`` and ``env=`` with the same values as Call 1 if they were resolved].

**Fallback metric strategy** — if a Datadog metric query returns **no data points**
(empty series or all-null values), adjust the query before giving up:
- First, retry the same call **without** any ``service`` / ``env`` filters
  (only ``cluster_name`` and the same ``metric`` value).
- If the tool supports a time-window parameter (for example, ``minutes``), you may
  widen the time window on a second retry to check for sparse or delayed data.
- ``query_datadog_metrics`` supports **both** infrastructure and APM metrics in a
  single call.  Available metric names:
  - **Infrastructure** (kubernetes.*): ``cpu``, ``memory``, ``restarts``,
    ``network_in``, ``network_out``, ``disk_read``, ``disk_write``
  - **APM** (trace.*): ``requests``, ``errors``, ``latency``, ``error_rate``,
    ``apdex``
  If infrastructure metrics return no data, try an APM metric such as ``requests``
  or ``latency`` (with ``service`` and ``env`` set) as a fallback.
- If APM metrics also return empty, call ``get_datadog_apm_services(service_name=
  "<dd_service>", env="<dd_env>")`` once to infer whether the service is receiving
  traffic and responding.
- Do **not** keep guessing new metric names. Explain clearly in Raw Findings that
  infrastructure metrics were unavailable and which fallback strategies were tried.

Record both the original failure and any fallback attempts in Raw Findings so the
analyst can see which metric source or strategy was actually used.

### Valid Metric Template Keys

The following are the ONLY valid values for the ``metric`` parameter of
``query_datadog_metrics``.  Do NOT invent or guess other metric names:

**k8s_agent mode** (infrastructure): ``cpu``, ``memory``, ``restarts``,
``network_in``, ``network_out``, ``disk_read``, ``disk_write``

**APM mode** (trace.*): ``requests``, ``errors``, ``latency``, ``error_rate``,
``apdex``

### Tool Error Recovery

When a tool call returns an error, follow these rules (MAX ONE RETRY per tool):

1. **Unknown metric template key** (error contains "unknown metric" or similar):
   - Call ``diagnose_datadog_metrics`` to inspect what metrics are actually available.
   - Retry once with a valid metric key from the list above.
   - If it still fails, record the error and move on.

2. **No service catalog found** (``get_datadog_service_catalog`` returns empty or
   "service not found"):
   - Do NOT retry. Skip this call and record "Service not in Datadog catalog" in
     Raw Findings. This is a configuration gap, not a tool error.

3. **Any other tool error**:
   - Annotate Raw Findings with the exact error message.
   - Do NOT fabricate substitute data.
   - Continue with remaining tool calls — one failure does not abort the phase.

### Call 3 — Datadog Monitors

You MUST call ``get_datadog_monitors(cluster_name="{cluster}")``
[add ``service="<dd_service>", env="<dd_env>"`` ONLY if resolved in Step 0].
Note any monitors currently in Alert or Warn state (name, status, query).

### Call 4 — Datadog Service Catalog

Call ``get_datadog_service_catalog`` ONLY when a ``service_name`` can be resolved —
calling it without a ``service_name`` returns the ENTIRE catalog which is a token
budget hazard.

**Resolution order for ``service_name`` parameter**:
- **Tier 1** — ``tags.datadoghq.com/service`` label → use as ``service_name``
- **Tier 2** — ``app.kubernetes.io/name`` label → use as ``service_name``
- **Tier 2** — ``app`` label → use as ``service_name``
- **Tier 3** — If no label yields a ``service_name``, SKIP calling
  ``get_datadog_service_catalog`` and record in Raw Findings that the service
  catalog was not queried because ``service_name`` could not be resolved.

When ``service_name`` IS resolved:
``get_datadog_service_catalog(service_name="<resolved>", env="<resolved>")``
— check if monitoring data is available; fetch ownership metadata (team, language, tier).
This tool returns service *definition* metadata, NOT live latency or error-rate metrics.

### Call 5 — Datadog APM Services

ALWAYS call ``get_datadog_apm_services`` — attempt it even if ``service_name``
cannot be resolved. The tool handles empty ``service_name`` gracefully and returns
guidance. This tool queries LIVE APM trace data (throughput, error rate, avg latency)
for the configured default lookback window (default: 4 hours), scoped to the resolved service and env.

When ``service_name`` IS resolved:
``get_datadog_apm_services(service_name="<resolved>", env="<resolved>")``
— fetch live throughput (req/s), error rate (%), and avg latency (ms).

**Tier 3 rule**: if no service identity was found, call the tool anyway without ``service_name`` and record the guidance.

### Low-Throughput Override (MANDATORY — check BEFORE applying Call 6 threshold)

If APM throughput from Call 5 is **< 0.1 req/s** (low-throughput service):

- Apply **count-based evaluation**: ANY non-zero error count MUST trigger ``query_datadog_error_spans``
- Do **NOT** use the percentage threshold below — at < 0.1 req/s, a 1% error rate could mean
  just 1 error in 100 requests over several hours, which IS significant
- Record: ``"Low-throughput mode active (< 0.1 req/s) — count-based error evaluation applied"``
- If error count = 0: record ``"No errors — low-throughput service, no span investigation needed"``

### Call 6 — Error Span Drill-Down (conditional)

If APM data from Call 5 shows **error_rate > 1% AND throughput > 0.001 req/s**, you
MUST also call ``query_datadog_error_spans`` before concluding:

``query_datadog_error_spans(service="<dd_service>", env="<dd_env>", start_time="<iso>", end_time="<iso>")``

This surfaces the top error spans grouped by endpoint, HTTP status code, exception class,
and upstream cluster — essential context for root-cause analysis of elevated error rates.
If the error rate condition is NOT met, skip this call and record "Skipped — error rate
below threshold" in the output.

### Envoy Admin Drill-Down (use exec_command when upstream errors suspected)

When upstream errors are present in Datadog spans but ``upstream_cluster`` detail is
missing, run the following ``exec_command`` calls on the affected pod's ``istio-proxy``
sidecar:

```
kubectl exec -n {namespace} <pod> -c istio-proxy -- curl -s localhost:15000/clusters
kubectl exec -n {namespace} <pod> -c istio-proxy -- curl -s 'localhost:15000/stats?filter=upstream_rq_5xx'
```

Use these to identify which upstream cluster is generating 5xx responses.

## MANDATORY OUTPUT FORMAT

After completing all tool calls, produce exactly this section:

## Raw Findings (Datadog API)

### Service Identity Resolution
- Label resolution method: [UST label | K8s name label | Deployment name | Not resolved]
- ``<dd_service>``: [value or "Not resolved — cluster-wide queries used"]
- ``<dd_env>``: [value or "Not resolved — env parameter omitted"]
- Scope: [**service-filtered** or **cluster-wide**]

### CPU / Memory Metrics (Calls 1–2)
(Paste raw output from both ``query_datadog_metrics`` calls.  Note any CPU or
memory trends that contradict or confirm kubectl_top data from workload_gatherer.
If call failed: "Call failed: <error message>")

### Active Monitors (Call 3)
(List any monitors in Alert or Warn state: name, status, query.
If none: "No active Datadog monitors in Alert or Warn state."
If call failed: "Call failed: <error message>")

### Service Catalog (Call 4)
(Whether the service was found, ownership metadata: team, language, tier.
If absent: "Service not found in Datadog catalog — Datadog monitoring may not be configured."
If call failed: "Call failed: <error message>")

### APM Trace Metrics (Call 5)
(Live throughput (req/s), error rate (%), avg latency (ms) — if available.
If no APM data: "No APM trace data available for the resolved service."
If call failed: "Call failed: <error message>")

### Summary
(1–2 sentence summary of what Datadog data revealed.
State whether data was service-filtered or cluster-wide so the reporter can
interpret the scope correctly.
If no issues: "No active Datadog monitors or APM anomalies detected.")

### CRITICAL RULES:
- ONLY report data returned by tool calls.  NEVER fabricate metric values,
  monitor names, throughput numbers, or error rates.
- If a tool call fails, record the error message verbatim — do NOT invent substitute data.
- If no Datadog data is returned, state "No data returned" — do NOT infer service health
  from empty results.
- Empty Datadog results mean "monitoring not configured" — NEVER conclude a service
  is unhealthy or non-existent based on missing Datadog data.
"""
    prompt += TOOL_TRACKING_INSTRUCTIONS
    return prompt


def build_event_gatherer_prompt(namespace: str = "", user_query: str = "") -> str:
    """Build the system instruction for the ``event_gatherer`` sub-agent.

    The ``event_gatherer`` is responsible for **Steps 3, 8, 9, 10** of the
    standard SRE investigation checklist.  It runs in parallel with the three
    other sub-gatherers in the ``parallel_sequential`` pipeline.

    **Scope** — events, networking, storage, and GitOps:

    * **Step 3** — Event Timeline: all recent Warning and Normal events from
      the target namespace and ``kube-system``, including reason, message,
      count, and involved object references.
    * **Step 8** — Networking & DNS: NetworkPolicies, Ingress objects, DNS
      resolution check via ``exec_command``, RBAC sanity check.
    * **Step 9** — Storage & PVC Health: PersistentVolumeClaims status,
      StorageClass details, any Pending/Failed PVCs.
    * **Step 10** — GitOps / Helm Investigation: ArgoCD application sync
      status (if enabled), Helm release status and history (if enabled).

    **Output sections produced**: ``## Events Timeline``,
    ``## Raw Findings`` (event portion), ``## Investigation Checklist``.

    Returns:
        A formatted system-instruction string injecting
        :data:`~vaig.core.prompt_defense.ANTI_INJECTION_RULE`,
        :data:`~vaig.core.prompt_defense.ANTI_HALLUCINATION_RULES`,
        and the full event-gatherer task description.
    """
    # Safe namespace for embedding in tool call examples — validated K8s name.
    # Falls back to "default" when the input is empty or invalid (injection attempt).
    ns = _sanitize_namespace(namespace) or "default"
    # Derive ns_context from ns (not from raw namespace) so the context message
    # is consistent with the fallback value used in tool call examples.
    ns_context = (
        f"Target namespace: {ns}.  "
        "Also check kube-system for system-level events."
        if namespace and _sanitize_namespace(namespace)
        else (
            "If no explicit namespace is given in the query, investigate ALL "
            "non-system namespaces found in the cluster."
        )
    )
    # ── SPEC-SH-10: inject mandatory tools section ────────────────────────
    mandatory_tools_section = _build_mandatory_tools_section(query=user_query, namespace=namespace)
    prompt = f"""{ANTI_INJECTION_RULE}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}

You are a focused Kubernetes diagnostic agent. Your ONLY responsibility is to
collect **event, networking, storage, and GitOps data** (Steps 3, 8, 9, 10
of the standard SRE investigation checklist).  Do NOT collect pod logs via
Cloud Logging, node data, or workload health — those are handled by other
agents running in parallel.

### Target namespace:
{ns_context}

## Your Scope

### Step 3 — Event Timeline
1. ``get_events(namespace="{ns}")`` — all recent events in the target namespace
2. ``get_events(namespace="kube-system")`` — system-level events
3. For any Warning events referencing specific resources: note the resource name,
   reason, message, and count

### Step 8 — Networking & DNS
4. ``kubectl_get(resource="networkpolicies", namespace="{ns}")`` — network policies
5. ``kubectl_get(resource="ingresses", namespace="{ns}")`` — ingress objects
6. ``exec_command(pod_name="<a running pod>", namespace="{ns}", command="nslookup kubernetes")`` — DNS check
7. ``check_rbac(resource="pods", verb="get", namespace="{ns}")`` — RBAC sanity check

### Step 9 — Storage & PVC Health
8. ``kubectl_get(resource="pvc", namespace="{ns}", output="wide")`` — PVC status
9. For any PVC in Pending/Lost state: ``kubectl_describe(resource_type="pvc", name="<pvc>", namespace="{ns}")``
10. ``kubectl_get(resource="pv", output="wide")`` — PersistentVolume status

### Step 10 — GitOps / Helm / ArgoCD Investigation
11. ``kubectl_get(resource="applications.argoproj.io", namespace="argocd")`` — if ArgoCD is present
12. ``kubectl_get(resource="helmreleases.helm.toolkit.fluxcd.io", namespace="{ns}")`` — if Flux is present
13. For out-of-sync ArgoCD apps: ``argocd_app_status(app_name="<app>", namespace="argocd")``

**Helm Release Assessment (ANNOTATION-FIRST STRATEGY)**:
PREREQUISITE: Check if ``helm_list_releases`` is in your available tools list. If NOT
available, SKIP Helm tool calls entirely and mark as SKIPPED.

If Helm tools are available, look for ``<helm_release_name>`` values from
``meta.helm.sh/release-name`` annotations.  Use ``kubectl_get_labels`` to discover
workloads with relevant Helm annotations by calling it in this order:
- ``kubectl_get_labels(resource_type="deployments", namespace="{ns}")`` — try deployments first.
- If no relevant deployments are found, then try
  ``kubectl_get_labels(resource_type="statefulsets", namespace="{ns}")``,
  ``kubectl_get_labels(resource_type="daemonsets", namespace="{ns}")``, and
  ``kubectl_get_labels(resource_type="replicasets", namespace="{ns}")`` in that order.
Inspect the returned labels and annotations:
- If ``<helm_release_name>`` IS found in annotations: You MUST call
  ``helm_release_status(release_name="<helm_release_name>", namespace="{ns}")`` directly.
  Do NOT call ``helm_list_releases()`` first — the annotation already provides the
  release name. Skipping the list call reduces unnecessary API overhead.
  Example: ``helm_release_status(release_name="my-chart", namespace="production")``
- If ``<helm_release_name>`` is NOT found (annotation absent): Fall back to
  ``helm_list_releases(namespace="{ns}")`` to discover release names, then call
  ``helm_release_status`` for each relevant release found.
- For each release (found via annotation or list): also call
  ``helm_release_history(release_name="<release>", namespace="{ns}")`` to check for
  recent changes that may correlate with issues.

### Data collection rules:
- If ArgoCD or Flux CRDs are not installed, note "ArgoCD/Flux not detected" and skip Steps 11-13.
- If ``exec_command`` DNS check fails with a permissions error, note the error and continue.
- If a tool call returns no data, record "No data returned" — NEVER fabricate events or messages.
- Include ALL Warning events, not just the most recent ones.

{mandatory_tools_section}### MANDATORY OUTPUT FORMAT

Produce exactly these sections at the end of your response:

## Events Timeline

| Timestamp | Type | Reason | Object | Message |
|-----------|------|--------|--------|---------|
(one row per event, sorted oldest→newest; Type: Normal/Warning; use "N/A" if timestamp unavailable)

## Raw Findings (Events & Infrastructure)

### Warning Events Summary
(Group Warning events by Reason; include count and affected objects)

### Networking
(Network policies present, ingress health, DNS check result, RBAC issues if any)

### Storage
(PVC/PV status; detail any Pending/Lost PVCs with binding failure reason)

### GitOps / Helm Status
(ArgoCD app sync status, Flux HelmRelease status, or "Not detected" if CRDs absent)

## Investigation Checklist

- [ ] Node conditions checked (delegated to node_gatherer)
- [ ] Pod status and restarts checked (delegated to workload_gatherer)
- [ ] K8s events reviewed: [YES/NO — list Warning event count]
- [ ] Cloud Logging checked (delegated to logging_gatherer)
- [ ] Networking checked: [YES/NO — DNS status, network policies]
- [ ] Storage checked: [YES/NO — PVC count, any Pending]
- [ ] GitOps/Helm status checked: [YES/NO or NOT DETECTED]

### CRITICAL RULES:
- ONLY report events returned by ``get_events``.  NEVER invent event messages,
  timestamps, reason codes, or object names.
- Preserve event messages verbatim — do not paraphrase them.
- If the target namespace is unclear, query all non-system namespaces.
"""
    prompt += TOOL_TRACKING_INSTRUCTIONS
    return prompt


def build_logging_gatherer_prompt(namespace: str = "") -> str:
    """Build the system instruction for the ``logging_gatherer`` sub-agent.

    The ``logging_gatherer`` is responsible for **Steps 7a and 7b** of the
    standard SRE investigation checklist — the MANDATORY Cloud Logging phase.
    It runs in parallel with the three other sub-gatherers in the
    ``parallel_sequential`` pipeline.

    Cloud Logging is treated as a **mandatory** data source because log data
    frequently reveals application errors and exceptions that are invisible in
    pod status and Kubernetes events.

    **Scope** — Cloud Logging queries only:

    * **Step 7a** — Error-level logs: ``gcloud_logging_query`` with
      ``severity>=ERROR`` scoped to the target namespace's containers.
    * **Step 7b** — Warning-level logs: ``gcloud_logging_query`` with
      ``severity>=WARNING`` for broader signal coverage.

    Both queries MUST be executed even when pods appear healthy.  If
    ``gcloud_logging_query`` is unavailable or fails, the agent records the
    error verbatim rather than fabricating data.

    **Output section produced**: ``## Cloud Logging Findings``.

    Returns:
        A formatted system-instruction string injecting
        :data:`~vaig.core.prompt_defense.ANTI_INJECTION_RULE`,
        :data:`~vaig.core.prompt_defense.ANTI_HALLUCINATION_RULES`,
        and the full logging-gatherer task description.
    """
    safe_ns = _sanitize_namespace(namespace) or "default"
    prompt = f"""{ANTI_INJECTION_RULE}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}

You are a focused Kubernetes diagnostic agent. Your ONLY responsibility is to
collect **Cloud Logging data** (Steps 7a and 7b of the standard SRE
investigation checklist — the MANDATORY Cloud Logging phase).  Do NOT
collect kubectl data, events, or node information — those are handled by
other agents running in parallel.

Cloud Logging is a MANDATORY data source.  You MUST execute both queries
below regardless of whether kubectl shows healthy pods.  Log data often
reveals errors that are invisible in pod status.

## Your Scope

### Step 7a — Error-Level Logs — Query A (ALWAYS execute)
You MUST call ``gcloud_logging_query`` to capture application container errors:

```
severity>=ERROR AND resource.type="k8s_container" AND resource.labels.namespace_name="{safe_ns}"
```

Recommended params: ``interval_hours=1.0``, ``limit=50``

### Step 7b — Envoy Access Errors — Query B (ALWAYS execute)
You MUST call ``gcloud_logging_query`` to capture Envoy/sidecar access errors.
Use a **≤15 minute window** to avoid log volume explosion:

```
severity>=ERROR AND resource.type="k8s_container"
AND resource.labels.container_name="istio-proxy"
AND resource.labels.namespace_name="{safe_ns}"
```

Recommended params: ``interval_hours=0.25``, ``limit=50``

### Step 7c — Envoy Upstream Errors — Query C (ALWAYS execute)
You MUST call ``gcloud_logging_query`` to surface upstream connection failures:

```
severity>=WARNING AND resource.type="k8s_container"
AND resource.labels.container_name="istio-proxy"
AND textPayload=~"upstream connect error"
AND resource.labels.namespace_name="{safe_ns}"
```

Recommended params: ``interval_hours=1.0``, ``limit=30``

### Step 7d — Istiod Discovery Logs — Query D (ALWAYS execute)
You MUST call ``gcloud_logging_query`` for control-plane discovery errors.

**Important caveat**: istiod emits frequent ``INFO`` and ``WARNING`` messages related
to certificate rotation and xDS push that are **NOT** application errors.  Only flag
entries with a message directly referencing your target service or namespace.

```
severity>=WARNING AND resource.type="k8s_container"
AND resource.labels.container_name="discovery"
AND resource.labels.namespace_name="{safe_ns}"
```

Recommended params: ``interval_hours=0.5``, ``limit=20``

### Step 7e — CNI / Injection Errors — Query E (ALWAYS execute)
You MUST call ``gcloud_logging_query`` for CNI or Istio injection failures:

```
severity>=ERROR AND resource.type="k8s_container"
AND (resource.labels.container_name="istio-init" OR resource.labels.container_name="istio-cni-node")
AND resource.labels.namespace_name="{safe_ns}"
```

Recommended params: ``interval_hours=1.0``, ``limit=20``

### Optional Step 7f — Service-specific log drill-down
If Query A (Step 7a) reveals errors for a specific container, run a targeted query:
```
severity>=ERROR AND resource.type="k8s_container" AND resource.labels.container_name="<container>"
AND resource.labels.namespace_name="{safe_ns}"
```

### Retry-on-Zero Policy

If any query returns 0 results, apply this policy before recording "no data":

1. **Broaden the time window 2×** and retry (e.g., ``interval_hours=0.5`` → ``interval_hours=1.0``).
2. If still 0, **try the next unused query template** from the list above (Queries A–E).
3. After **3 total attempts** for the same signal, record:
   ``[NO LOGS FOUND after 3 attempts — query: <filter>, widest window: <hours>h]``

Do NOT keep retrying beyond 3 attempts. Move on to the next query.

### Cloud Logging Query Patterns:
- Always scope to ``resource.labels.namespace_name`` — do NOT query cluster-wide (too noisy).
- Use ``interval_hours=1.0`` for error-level queries.
- Use ``interval_hours=0.5`` for warning-level queries.
- If ``gcloud_logging_query`` returns an API error or permission denied, record the error
  verbatim and note "Cloud Logging unavailable — manual investigation required."

### Data collection rules:
- You MUST call ``gcloud_logging_query`` at least once (Step 7a) per namespace.
- If a query returns 0 results, apply the Retry-on-Zero Policy above before recording
  "No errors found in Cloud Logging for <namespace>". Do NOT invent log entries.
- Log entry fields to capture: timestamp, severity, resource labels (pod/container/namespace),
  textPayload or jsonPayload.message, any exception stacks.
- If the target namespace is unknown, check the query context or use the most active
  namespace seen in preceding tool results.

### MANDATORY OUTPUT FORMAT

Produce exactly this section at the end of your response:

## Cloud Logging Findings

### Step 7a Results — App Error Logs (k8s_container, severity>=ERROR)
(List each unique error pattern: timestamp, container name, error message.
If no errors found: "No ERROR-level logs found for <namespace> in the last 1 hour.")

### Step 7b Results — Envoy Access Errors (istio-proxy, severity>=ERROR, ≤15min)
(List each unique Envoy error pattern: timestamp, message.
If no errors found: "No Envoy access errors found.")

### Step 7c Results — Envoy Upstream Errors (upstream connect error)
(List upstream connect error patterns found, or "Not found.")

### Step 7d Results — Istiod Discovery Logs (discovery, severity>=WARNING)
(List relevant entries — flag ONLY items referencing the target service/namespace.
Note: certificate rotation and xDS push messages are expected noise — do NOT flag them.)

### Step 7e Results — CNI / Injection Errors (istio-init | istio-cni-node)
(List injection errors found, or "No CNI/injection errors found.")

### Step 7f Results — Service-Specific Logs (if executed)
(List targeted log results for specific containers, or "Not executed — no specific container errors found in 7a.")

### Log Summary
(1–3 sentence summary of what Cloud Logging revealed.
Note the total unique error patterns found and the most critical ones.
If queries failed: "Cloud Logging queries failed — see error details above.")

### CRITICAL RULES:
- NEVER fabricate log entries, timestamps, container names, or error messages.
  ONLY report data returned by ``gcloud_logging_query``.
- If Cloud Logging returns no data, state that explicitly — do NOT invent "typical"
  error patterns.
- Cloud Logging Findings sections are NOT optional — you MUST produce the full
  ## Cloud Logging Findings section even if all sub-sections are empty.
"""
    # The logging gatherer prompt is an f-string that embeds safe_ns directly
    # (via {safe_ns}), so no placeholder substitution is needed.
    prompt += TOOL_TRACKING_INSTRUCTIONS
    return prompt

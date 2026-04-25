"""Service Health Skill — live Kubernetes service health assessment.

A 4-agent sequential pipeline with two-pass verification that demonstrates
the ToolAwareAgent + Orchestrator integration.  The first agent uses live
tools to collect cluster health data; the second analyzes patterns; the
third verifies findings with targeted tool calls; the fourth produces a
structured JSON report (rendered to Markdown).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
from typing import Any, NamedTuple

from pydantic import ValidationError

from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from vaig.core.quality import QualityIssue
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.service_health.contradiction_validator import apply_contradiction_rules, detect_contradictions
from vaig.skills.service_health.prompts import (
    ANALYZER_AUTONOMOUS_OVERLAY,
    HEALTH_ANALYZER_PROMPT,
    HEALTH_INVESTIGATOR_PROMPT,
    HEALTH_PLANNER_PROMPT,
    HEALTH_VERIFIER_PROMPT,
    INVESTIGATOR_AUTONOMOUS_OVERLAY,
    PHASE_PROMPTS,
    SYSTEM_INSTRUCTION,
    build_attachment_seeded_section,
    build_datadog_gatherer_prompt,
    build_event_gatherer_prompt,
    build_gatherer_prompt,
    build_logging_gatherer_prompt,
    build_node_gatherer_prompt,
    build_reporter_prompt,
    build_verifier_ratification_section,
    build_workload_gatherer_prompt,
)
from vaig.skills.service_health.prompts._shared import _prefix_attachment_context
from vaig.skills.service_health.schema import Finding, HealthReport, HealthReportGeminiSchema, OperatingMode
from vaig.skills.service_health.schema import apply_ratification as _apply_ratification
from vaig.skills.service_health.schema import render_attachment_sections as _render_attachment_sections
from vaig.tools.base import ToolResult
from vaig.tools.gke._clients import ensure_client_initialized
from vaig.utils.json_cleaner import clean_llm_json

logger = logging.getLogger(__name__)


# ── Finding dedup helpers (SPEC-RP-01) ───────────────────────────────────────


def _finding_fingerprint(finding: Finding) -> str:
    """Return a SHA-1 fingerprint for *finding* used for duplicate detection.

    The fingerprint is computed over title, category, and the kind/name of
    the first affected resource (normalised to lowercase, stripped of
    whitespace).  It is intentionally NOT a security hash — the ``# noqa:
    S324`` comment suppresses the bandit warning.
    """
    kind = ""
    name = ""
    if finding.affected_resources:
        parts = finding.affected_resources[0].split("/", 1)
        if len(parts) == 2:  # noqa: PLR2004
            kind, name = parts[0], parts[1]
        else:
            kind = parts[0]
    raw = "|".join(
        [
            finding.title.lower().strip(),
            finding.category.lower().strip(),
            kind,
            name,
        ]
    )
    return hashlib.sha1(raw.encode()).hexdigest()  # noqa: S324


def _dedup_findings(findings: list[Finding]) -> list[Finding]:
    """Return *findings* with duplicates removed (first occurrence wins).

    Duplicates are detected by fingerprint (see :func:`_finding_fingerprint`).
    The input list is NOT mutated.  On any unexpected exception the original
    list is returned unchanged so dedup never breaks the pipeline.
    """
    try:
        seen: dict[str, bool] = {}
        result: list[Finding] = []
        for f in findings:
            fp = _finding_fingerprint(f)
            if fp in seen:
                logger.warning(
                    "Duplicate finding dropped: fingerprint=%.8s title=%r",
                    fp,
                    f.title,
                )
            else:
                seen[fp] = True
                result.append(f)
        return result
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        logger.error("_dedup_findings failed — returning original list", exc_info=True)
        return findings


# ── Label priority for Datadog service name resolution ───────────────────
# Used by ``_resolve_dd_service_name`` to extract the DD service identity
# from K8s workload objects.

_DD_SERVICE_LABEL = "tags.datadoghq.com/service"
_DD_ENV_LABEL = "tags.datadoghq.com/env"
_K8S_NAME_LABEL = "app.kubernetes.io/name"
_APP_LABEL = "app"


class DDResolutionResult(NamedTuple):
    """Result of pre-resolving Datadog identity from K8s workload labels.

    Returned by :meth:`ServiceHealthSkill._resolve_dd_service_name`.

    Attributes:
        dd_service_name: Resolved Datadog service name (empty if unresolved).
        dd_env: Resolved ``tags.datadoghq.com/env`` value (empty if unresolved).
        dd_resource_type: The K8s resource type the winning workload came from
            (e.g. ``"deployments"``, ``"statefulsets"``).  Empty if unresolved.
    """

    dd_service_name: str = ""
    dd_env: str = ""
    dd_resource_type: str = ""


def _extract_dd_service(item: Any) -> str:
    """Extract the Datadog service name from a K8s workload object.

    Follows the label priority order defined in the exploration:
    1. Pod template labels ``tags.datadoghq.com/service``
    2. Workload-level labels ``tags.datadoghq.com/service``
    3. Pod template labels ``app.kubernetes.io/name``
    4. Pod template labels ``app``
    5. ``metadata.name`` (workload name — last resort)
    """
    # --- Pod template labels (spec.template.metadata.labels) ---
    pod_labels: dict[str, str] = {}
    try:
        _tmpl = getattr(item.spec, "template", None)
        _tmpl_meta = getattr(_tmpl, "metadata", None) if _tmpl else None
        _tmpl_labels = getattr(_tmpl_meta, "labels", None) if _tmpl_meta else None
        pod_labels = _tmpl_labels or {}
    except Exception:  # noqa: BLE001
        pod_labels = {}

    # Tier 1a: DD UST label on pod template
    if _DD_SERVICE_LABEL in pod_labels:
        return pod_labels[_DD_SERVICE_LABEL]

    # --- Workload-level labels (metadata.labels) ---
    wl_labels: dict[str, str] = getattr(getattr(item, "metadata", None), "labels", None) or {}

    # Tier 1b: DD UST label on workload
    if _DD_SERVICE_LABEL in wl_labels:
        return wl_labels[_DD_SERVICE_LABEL]

    # Tier 2: standard K8s name label (pod template)
    if _K8S_NAME_LABEL in pod_labels:
        return pod_labels[_K8S_NAME_LABEL]

    # Tier 3: legacy app label (pod template)
    if _APP_LABEL in pod_labels:
        return pod_labels[_APP_LABEL]

    # Tier 4: workload name as last resort
    return getattr(getattr(item, "metadata", None), "name", "") or ""


def _extract_dd_env(item: Any) -> str:
    """Extract the Datadog environment tag from a K8s workload object.

    Follows the same label priority as ``_extract_dd_service`` but only
    looks for ``tags.datadoghq.com/env``:

    1. Pod template labels ``tags.datadoghq.com/env``
    2. Workload-level labels ``tags.datadoghq.com/env``

    Returns an empty string when the label is not found.
    """
    # --- Pod template labels (spec.template.metadata.labels) ---
    pod_labels: dict[str, str] = {}
    try:
        _tmpl = getattr(item.spec, "template", None)
        _tmpl_meta = getattr(_tmpl, "metadata", None) if _tmpl else None
        _tmpl_labels = getattr(_tmpl_meta, "labels", None) if _tmpl_meta else None
        pod_labels = _tmpl_labels or {}
    except Exception:  # noqa: BLE001
        pod_labels = {}

    if _DD_ENV_LABEL in pod_labels:
        return pod_labels[_DD_ENV_LABEL]

    # --- Workload-level labels (metadata.labels) ---
    wl_labels: dict[str, str] = getattr(getattr(item, "metadata", None), "labels", None) or {}

    if _DD_ENV_LABEL in wl_labels:
        return wl_labels[_DD_ENV_LABEL]

    return ""


class ServiceHealthSkill(BaseSkill):
    """Service health assessment skill using live Kubernetes tools.

    Implements a 4-agent sequential pipeline with two-pass verification:

    1. **health_gatherer** (``requires_tools=True``):
       Uses live kubectl tools to collect pod status, resource usage,
       logs, events, and deployment state.

    2. **health_analyzer** (``requires_tools=False``):
       Text-only agent that receives gathered data and performs SRE-style
       pattern analysis — degraded services, resource pressure, error
       rate spikes, cross-service correlations.

    3. **health_verifier** (``requires_tools=True``):
       Tool-aware agent that makes targeted verification calls specified
       in the analyzer's Verification Gap fields.  Confirms, upgrades,
       or downgrades finding confidence levels before reporting.

    4. **health_reporter** (``requires_tools=False``):
       Text-only agent that synthesizes verified findings into a
        structured JSON report (rendered to Markdown) with severity
        classification, root-cause hypotheses, and actionable
        remediation commands.

    The pipeline strategy is **sequential**: each agent's output feeds
    as context into the next agent.
    """

    # Maximum lines of kubectl_top output to inject into prompts.
    # Beyond this threshold the output is truncated with an informational
    # notice so that very large namespaces don't bloat the prompt context.
    _MAX_KUBECTL_TOP_LINES = 100

    def __init__(self) -> None:
        super().__init__()
        # Pre-fetched metrics are populated by ``pre_execute_parallel()``
        # and consumed by ``get_parallel_agents_config()``.  This ensures
        # that inspecting the config (dry-run, ``vaig skills info``) does
        # NOT trigger live K8s API calls — only the execution path does.
        self._prefetched_metrics: dict[str, str] = {"pods": "", "nodes": ""}
        # Pre-resolved Datadog service identity.  Populated by
        # ``get_parallel_agents_config()`` via ``_resolve_dd_service_name()``.
        # Empty result means resolution failed or was not attempted —
        # the prompt builder falls back to LLM-based Step 0 resolution.
        self._prefetched_dd_resolution: DDResolutionResult = DDResolutionResult()
        # Lazy-initialized enrichment resources (SH-08).
        # Created on first call to ``_enrich_report_recommendations``; reused on
        # subsequent calls.  Single-worker pool — used sequentially, never
        # concurrently.  Call ``close()`` to release resources explicitly.
        self._enrichment_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._gemini_client: Any | None = None  # GeminiClient, lazy-imported
        # SPEC-ATT-10 §6.5.5 — operating mode signals.  Set by the CLI before
        # the pipeline runs so that post_process_report can detect the mode
        # without receiving extra kwargs through the entire call chain.
        self._offline_mode: bool = False
        self._attachments_present: bool = False

    def close(self) -> None:
        """Shut down lazy-initialized resources. Idempotent.

        Safe to call multiple times or when resources were never created.
        After ``close()``, the next call to ``_enrich_report_recommendations``
        will re-create a fresh pool and client.
        """
        if self._enrichment_pool is not None:
            self._enrichment_pool.shutdown(wait=False, cancel_futures=True)
            self._enrichment_pool = None
        self._gemini_client = None

    @staticmethod
    def _get_api_client(gke_config: Any) -> Any | None:
        """Extract the Kubernetes ``ApiClient`` from *gke_config*, or ``None``."""
        from vaig.tools.base import ToolResult as _ToolResult  # noqa: PLC0415
        from vaig.tools.gke import _clients as _gke_clients  # noqa: PLC0415

        clients = _gke_clients._create_k8s_clients(gke_config)
        if isinstance(clients, _ToolResult):
            return None
        return clients[3]  # ApiClient is the 4th element

    def _detect_argocd(self, namespace: str, gke_config: Any) -> bool:
        """Resolve the 3-state ``argocd_enabled`` flag to a concrete bool.

        - ``False``  → skip detection, return ``False``
        - ``True``   → force-enable, return ``True``
        - ``None``   → perform a live CRD probe via the GKE k8s client and
                       return the result of :func:`~vaig.tools.gke.argocd.detect_argocd`
        """
        setting = gke_config.argocd_enabled
        if setting is not None:
            return bool(setting)
        from vaig.tools.gke.argocd import detect_argocd  # noqa: PLC0415

        return detect_argocd(namespace=namespace, api_client=self._get_api_client(gke_config))

    def _detect_argo_rollouts(self, namespace: str, gke_config: Any) -> bool:
        """Resolve the 3-state ``argo_rollouts_enabled`` flag to a concrete bool.

        - ``False``  → skip detection, return ``False``
        - ``True``   → force-enable, return ``True``
        - ``None``   → perform a live CRD probe via the GKE k8s client and
                       return the result of
                       :func:`~vaig.tools.gke.argo_rollouts.detect_argo_rollouts`
        """
        setting = gke_config.argo_rollouts_enabled
        if setting is not None:
            return bool(setting)
        from vaig.tools.gke.argo_rollouts import detect_argo_rollouts  # noqa: PLC0415

        return detect_argo_rollouts(namespace=namespace, api_client=self._get_api_client(gke_config))

    # ── SPEC-ATT-10 §6.5.5 ───────────────────────────────────────────────────

    @staticmethod
    def _detect_operating_mode(
        *,
        offline_mode: bool,
        attachments_present: bool,
    ) -> OperatingMode:
        """Determine the pipeline operating mode from run-time signals.

        Args:
            offline_mode: ``True`` when the caller passed ``--offline-mode``
                (or equivalent) to suppress live GKE tool calls.
            attachments_present: ``True`` when at least one attachment adapter
                was resolved (i.e. ``bool(attachment_adapters)``).

        Returns:
            The :class:`~vaig.skills.service_health.schema.OperatingMode` that
            should govern this pipeline run.
        """
        if offline_mode:
            return OperatingMode.ATTACHMENT_ONLY
        if attachments_present:
            return OperatingMode.LIVE_PLUS_ATTACHMENTS
        return OperatingMode.LIVE_ONLY

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="service-health",
            display_name="Service Health Assessment",
            description="Live Kubernetes service health check with tool-backed data collection",
            version="1.0.0",
            tags=["sre", "live", "health", "kubernetes"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.EXECUTE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_required_output_sections(self) -> list[str]:
        """Mandatory sections the gatherer (first agent) must produce.

        These correspond to the MANDATORY OUTPUT FORMAT defined in
        ``HEALTH_GATHERER_PROMPT``.  The orchestrator validates the gatherer's
        output against these sections and retries once if any are missing.
        """
        return [
            "Cluster Overview",
            "Service Status",
            "Events Timeline",
            "Raw Findings",
            "Cloud Logging Findings",
            "Investigation Checklist",
        ]

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def pre_execute_parallel(self, query: str) -> None:
        """Pre-warm the K8s client cache and pre-fetch kubectl_top metrics.

        This hook runs **once**, sequentially, before any parallel gatherer
        threads are launched by the orchestrator.  It performs two tasks:

        1. **Client pre-warming** — The K8s client cache
           (``_CLIENT_CACHE``) is not thread-safe on first write because
           :func:`~vaig.tools.gke._clients._suppress_stderr` mutates
           ``sys.stdout`` and OS fd 2.  Warming the cache here ensures the
           client is fully constructed before concurrent threads start.

        2. **Metrics pre-fetch** — Calls ``_pre_fetch_metrics`` to gather
           ``kubectl_top`` output programmatically.  The results are stored
           on ``self._prefetched_metrics`` and consumed later by
           :meth:`get_parallel_agents_config` when it builds prompts.
           Moving the pre-fetch here (instead of inside
           ``get_parallel_agents_config``) ensures that callers that only
           *inspect* the config (e.g. ``vaig skills info``, dry-run) do
           NOT trigger live K8s API calls.

        Both steps are **best-effort** — any failure is swallowed silently.
        """
        try:
            from vaig.core.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            ensure_client_initialized(settings.gke)
        except Exception:  # noqa: BLE001
            logger.debug(
                "K8s client pre-warm skipped (non-fatal): see ensure_client_initialized logs",
                exc_info=True,
            )

        # ── Pre-fetch kubectl_top metrics ────────────────────────────────
        # Uses the same effective config resolution as get_parallel_agents_config
        # so the namespace / location / cluster_name overrides are respected.
        try:
            from vaig.core.config import get_settings  # noqa: PLC0415
            from vaig.tools.gke._clients import detect_autopilot  # noqa: PLC0415

            settings = get_settings()
            effective_gke = settings.gke
            is_autopilot = bool(detect_autopilot(effective_gke))
            self._prefetched_metrics = self._pre_fetch_metrics(
                gke_config=effective_gke,
                namespace=effective_gke.default_namespace,
                is_autopilot=is_autopilot,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "kubectl_top pre-fetch skipped (non-fatal)",
                exc_info=True,
            )

    @staticmethod
    def _pre_fetch_metrics(
        gke_config: Any,
        namespace: str,
        is_autopilot: bool,
    ) -> dict[str, str]:
        """Call ``kubectl_top`` programmatically and return pre-gathered metrics.

        This removes the dependency on LLM non-determinism for basic resource
        metrics.  The results are injected into the sub-gatherer prompts so
        the agents can use real data immediately without spending tool-call
        budget on ``kubectl_top``.

        The method is **best-effort** — any failure is captured as an
        informational error string so the pipeline never crashes here.

        **Sentinel detection** — ``kubectl_top`` can return non-metric
        informational strings with ``error=False`` (e.g. *"No metrics data
        available"*).  Genuine metric output always contains a header line
        with ``CPU`` and ``MEMORY`` column labels.  Results that lack these
        headers are treated as unavailable with an informational message.

        **Truncation** — on namespaces with many pods/containers the output
        can be very large.  Results are truncated to
        ``_MAX_KUBECTL_TOP_LINES`` lines with a notice.

        Args:
            gke_config: Effective :class:`~vaig.core.config.GKEConfig` with
                CLI overrides already applied.
            namespace: Target namespace for pod metrics.
            is_autopilot: Whether the cluster is GKE Autopilot.

        Returns:
            A dict with ``"pods"`` and ``"nodes"`` keys.  Each value is
            either the formatted ``kubectl_top`` output string or an
            informational error message.
        """
        from vaig.tools.gke.kubectl import kubectl_top  # noqa: PLC0415

        result: dict[str, str] = {"pods": "", "nodes": ""}

        max_lines = ServiceHealthSkill._MAX_KUBECTL_TOP_LINES

        def _validate_and_truncate(output: str, resource_type: str) -> str:
            """Return validated, potentially truncated output or a sentinel message."""
            # Sentinel detection — genuine kubectl_top output has CPU/MEMORY headers
            upper = output.upper()
            if "CPU" not in upper or "MEMORY" not in upper:
                return f"[kubectl_top {resource_type} unavailable: {output.strip()}]"
            # Truncation for large outputs
            lines = output.splitlines()
            if len(lines) > max_lines:
                truncated = "\n".join(lines[:max_lines])
                return f"{truncated}\n[... truncated — {len(lines)} total lines]"
            return output

        # ── Pod metrics ──────────────────────────────────────────────────
        try:
            pod_result: ToolResult = kubectl_top(
                "pods",
                gke_config=gke_config,
                namespace=namespace or "default",
            )
            if pod_result.error:
                result["pods"] = f"[kubectl_top pods failed: {pod_result.output}]"
            else:
                result["pods"] = _validate_and_truncate(pod_result.output, "pods")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pre-fetch kubectl_top pods failed (non-fatal): %s", exc)
            result["pods"] = f"[kubectl_top pods unavailable: {exc}]"

        # ── Node metrics ─────────────────────────────────────────────────
        if is_autopilot:
            result["nodes"] = (
                "GKE Autopilot cluster detected — kubectl top nodes is not available. "
                "Node infrastructure is managed by Google on Autopilot."
            )
        else:
            try:
                node_result: ToolResult = kubectl_top(
                    "nodes",
                    gke_config=gke_config,
                )
                if node_result.error:
                    result["nodes"] = f"[kubectl_top nodes failed: {node_result.output}]"
                else:
                    result["nodes"] = _validate_and_truncate(node_result.output, "nodes")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Pre-fetch kubectl_top nodes failed (non-fatal): %s", exc)
                result["nodes"] = f"[kubectl_top nodes unavailable: {exc}]"

        return result

    @staticmethod
    def _resolve_dd_service_name(
        gke_config: Any,
        namespace: str,
        user_query: str,
    ) -> DDResolutionResult:
        """Pre-resolve Datadog identity from K8s workload labels.

        Queries the Kubernetes API directly via :func:`_list_resource` to
        discover workload labels and extract the Datadog service identity,
        environment tag, and the resource type of the winning workload.

        **Workload probe order** — deployments → statefulsets → daemonsets →
        replicasets (standalone only — deployment-managed RS are skipped).

        **Label priority** (per workload):

        1. ``spec.template.metadata.labels["tags.datadoghq.com/service"]``
        2. ``metadata.labels["tags.datadoghq.com/service"]``
        3. ``spec.template.metadata.labels["app.kubernetes.io/name"]``
        4. ``spec.template.metadata.labels["app"]``
        5. ``metadata.name`` (workload name — last resort)

        When multiple workloads are found the method matches against
        ``user_query`` — exact match first, then substring containment.
        If a single workload exists it is returned directly.

        Args:
            gke_config: Effective :class:`~vaig.core.config.GKEConfig` with
                CLI overrides already applied.
            namespace: Target namespace to probe.
            user_query: Original user query, used to disambiguate when
                multiple workloads are found.

        Returns:
            A :class:`DDResolutionResult` with the resolved service name,
            environment tag, and resource type.  All fields are empty strings
            when resolution fails (the prompt builder falls back to LLM-based
            Step 0).
        """
        from vaig.tools.gke._clients import _create_k8s_clients  # noqa: PLC0415
        from vaig.tools.gke._resources import _list_resource  # noqa: PLC0415

        _empty = DDResolutionResult()

        clients = _create_k8s_clients(gke_config)
        if isinstance(clients, ToolResult):
            logger.debug("DD service resolution skipped — K8s clients unavailable: %s", clients)
            return _empty

        core_v1, apps_v1, custom_api, _api_client = clients
        ns = namespace or "default"

        # Collect workload_name → (dd_service, dd_env, resource_type).
        # First-hit semantics: if the same workload name appears in multiple
        # resource types, keep the first one found (probe order wins).
        workload_services: dict[str, tuple[str, str, str]] = {}
        resource_types = ("deployments", "statefulsets", "daemonsets", "replicasets")

        for resource_type in resource_types:
            try:
                result = _list_resource(core_v1, apps_v1, custom_api, resource_type, ns)
            except Exception:  # noqa: BLE001
                logger.debug("DD service resolution: failed to list %s (non-fatal)", resource_type)
                continue

            # ToolResult means the resource type is not supported or API error.
            if isinstance(result, ToolResult):
                continue

            items = getattr(result, "items", None)
            if not items:
                continue

            for item in items:
                metadata = getattr(item, "metadata", None)
                if not metadata:
                    continue

                # Skip deployment-managed ReplicaSets — only keep standalone
                # ones (e.g. from Argo Rollouts).
                if resource_type == "replicasets":
                    owner_refs = getattr(metadata, "owner_references", None) or []
                    if any(getattr(ref, "kind", "") == "Deployment" for ref in owner_refs):
                        continue

                workload_name: str = getattr(metadata, "name", "") or ""
                if not workload_name:
                    continue

                # First-hit semantics — keep the first result per workload name.
                if workload_name not in workload_services:
                    dd_service = _extract_dd_service(item)
                    dd_env = _extract_dd_env(item)
                    workload_services[workload_name] = (dd_service, dd_env, resource_type)

        if not workload_services:
            return _empty

        # Single workload → return directly.
        if len(workload_services) == 1:
            svc, env, rtype = next(iter(workload_services.values()))
            return DDResolutionResult(dd_service_name=svc, dd_env=env, dd_resource_type=rtype)

        # Multiple workloads → match against user_query.
        query_lower = user_query.lower()

        # Pass 1: exact match on workload name.
        for wl_name, (svc, env, rtype) in workload_services.items():
            if wl_name.lower() == query_lower:
                return DDResolutionResult(dd_service_name=svc, dd_env=env, dd_resource_type=rtype)

        # Pass 2: substring containment (workload name in query or query in workload name).
        for wl_name, (svc, env, rtype) in workload_services.items():
            wl_lower = wl_name.lower()
            if wl_lower in query_lower or query_lower in wl_lower:
                return DDResolutionResult(dd_service_name=svc, dd_env=env, dd_resource_type=rtype)

        # No match — return empty so the prompt falls back to LLM-based resolution.
        return _empty

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return the default pipeline configuration — the parallel pipeline config.

        Delegates to :meth:`get_parallel_agents_config` so that the
        ``parallel_sequential`` execution strategy is used by default.  The
        Orchestrator auto-detects the ``parallel_group`` key in the returned
        configs and upgrades from ``"sequential"`` to ``"parallel_sequential"``
        automatically — no call-site changes required.

        For the legacy 4-agent sequential pipeline (single monolithic
        ``health_gatherer`` agent), use
        :meth:`get_sequential_agents_config` directly.

        Args:
            **kwargs: Caller-supplied keyword arguments.  The orchestrator
                passes ``namespace``, ``location``, and ``cluster_name``.
                These are extracted and forwarded to
                :meth:`get_parallel_agents_config`.

        Returns:
            The parallel pipeline: 4–5 parallel sub-gatherers
            (``node_gatherer``, ``workload_gatherer``, ``event_gatherer``,
            ``logging_gatherer``, and optionally ``datadog_gatherer``)
            followed by the unchanged sequential tail
            (``health_analyzer`` → ``health_verifier`` → ``health_reporter``).
        """
        namespace: str = kwargs.get("namespace", "")
        location: str = kwargs.get("location", "")
        cluster_name: str = kwargs.get("cluster_name", "")
        user_query: str = kwargs.get("user_query", "")
        attachment_context: str | None = kwargs.get("attachment_context")
        attachment_priors_json: str | None = kwargs.get("attachment_priors_json")
        return self.get_parallel_agents_config(
            namespace=namespace,
            location=location,
            cluster_name=cluster_name,
            user_query=user_query,
            attachment_context=attachment_context,
            attachment_priors_json=attachment_priors_json,
        )

    def get_sequential_agents_config(
        self,
        *,
        namespace: str = "",
        location: str = "",  # noqa: ARG002 — reserved for future cluster routing
        cluster_name: str = "",  # noqa: ARG002 — reserved for future cluster routing
        attachment_context: str | None = None,
        attachment_priors_json: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the legacy 4-agent sequential pipeline configuration.

        This is the original pre-Phase-3 configuration: a single monolithic
        ``health_gatherer`` agent (``gemini-2.5-pro``, 25 iterations) that
        runs all 10 investigation steps sequentially, followed by the
        analyzer → verifier → reporter tail.

        Use this method when you need the sequential fallback path (e.g. in
        tests that explicitly verify backward-compat sequential behaviour).

        Agent roles:

        * Agent 1 (``health_gatherer``): ``requires_tools=True`` — instantiated
          as :class:`~vaig.agents.tool_aware.ToolAwareAgent`.
        * Agent 2 (``health_analyzer``): ``requires_tools=False`` —
          :class:`~vaig.agents.specialist.SpecialistAgent`.
        * Agent 3 (``health_verifier``): ``requires_tools=True`` — targeted
          verification tool calls with a fast model.
        * Agent 4 (``health_reporter``): ``requires_tools=False`` —
          structured JSON output (``response_schema=HealthReport``).

        The gatherer prompt's tool-reference table is built dynamically from
        ``gke.helm_enabled`` / ``gke.argocd_enabled`` settings so disabled
        tools are excluded from the prompt (R4).

        Args:
            namespace: Override namespace from CLI.  Empty string falls back
                to ``settings.gke.default_namespace``.
            location: Reserved for future cluster-routing support.
            cluster_name: Reserved for future cluster-routing support.
        """
        from vaig.core.config import get_settings

        settings = get_settings()
        namespace = namespace or settings.gke.default_namespace

        # Resolve the 3-state ArgoCD flag via helper (SH-07).
        effective_gke = settings.gke.model_copy(update={"default_namespace": namespace})
        argocd_active = self._detect_argocd(namespace, effective_gke)

        gatherer_prompt = build_gatherer_prompt(
            helm_enabled=settings.gke.helm_enabled,
            argocd_enabled=argocd_active,
            datadog_api_enabled=settings.datadog.enabled,
            attachment_context=attachment_context,
        )
        gatherer_tool_categories = ["kubernetes", "helm", "scaling", "mesh", "datadog", "logging"]
        if argocd_active:
            gatherer_tool_categories.append("argocd")
        return [
            {
                "name": "health_gatherer",
                "role": "Health Data Gatherer",
                "requires_tools": True,
                "tool_categories": gatherer_tool_categories,
                "system_instruction": gatherer_prompt,
                "model": "gemini-2.5-pro",
                "temperature": 0.0,  # Deterministic — gatherer follows a procedure, no creativity needed
                "frequency_penalty": 0.3,  # Discourage repetitive tool call patterns
                # Mandatory Cloud Logging (Steps 7a-7d) requires ~20 iterations
                "max_iterations": 25,
            },
            {
                "name": "health_analyzer",
                "role": "Health Pattern Analyzer",
                "requires_tools": False,
                "system_instruction": _prefix_attachment_context(
                    (
                        HEALTH_ANALYZER_PROMPT + ANALYZER_AUTONOMOUS_OVERLAY
                        if settings.investigation.enabled is True
                        else HEALTH_ANALYZER_PROMPT
                    )
                    + build_attachment_seeded_section(attachment_priors_json or ""),
                    attachment_context,
                ),
                "model": "gemini-2.5-flash",
                "temperature": 0.2,  # Low temp for precise analysis
            },
            {
                "name": "health_verifier",
                "role": "Health Finding Verifier",
                "requires_tools": True,
                "tool_categories": ["kubernetes", "scaling", "mesh", "datadog"],
                "system_instruction": _prefix_attachment_context(
                    HEALTH_VERIFIER_PROMPT + build_verifier_ratification_section(attachment_priors_json),
                    attachment_context,
                ),
                "model": "gemini-2.5-flash",
                "max_iterations": 15,
                "temperature": 0.2,  # Low temp for precise verification
            },
            {
                "name": "health_reporter",
                "role": "Health Report Generator",
                "requires_tools": False,
                "system_instruction": build_reporter_prompt(
                    namespace=namespace,
                    datadog_api_enabled=settings.datadog.enabled,
                    attachment_context=attachment_context,
                ),
                "model": "gemini-2.5-flash",
                "temperature": 0.3,  # Slightly higher for natural writing
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "response_schema": HealthReportGeminiSchema,
                "response_mime_type": "application/json",
            },
        ]

    def get_parallel_agents_config(
        self,
        *,
        namespace: str = "",
        location: str = "",
        cluster_name: str = "",
        attachment_context: str | None = None,
        attachment_priors_json: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the parallel-then-sequential pipeline configuration.

        Phase 3 of the parallel-gatherer design replaces the single monolithic
        ``health_gatherer`` with focused sub-gatherers that run concurrently
        via ``_execute_parallel_then_sequential`` in the Orchestrator.

        Pipeline structure:
        - **Parallel group** (``parallel_group="gather"``): 4–5 sub-gatherers,
          each covering a focused subset of the investigation checklist.
          Core gatherers use ``gemini-2.5-pro``; ``datadog_gatherer`` uses
          ``gemini-2.5-flash`` (structured API calls only).
        - **Sequential tail** (unchanged): health_analyzer → health_verifier →
          health_reporter.  The merged output of the parallel group is passed
          as context to the analyzer.

        Sub-gatherer scope:
        - ``node_gatherer``: Step 1 — cluster overview & node health
        - ``workload_gatherer``: Steps 2, 4, 5, 6 — pods, deployments, services, HPA
        - ``event_gatherer``: Steps 3, 8, 9, 10 — events, networking, storage, GitOps
        - ``logging_gatherer``: Steps 7a, 7b — Cloud Logging error & warning queries
        - ``datadog_gatherer`` (optional): Datadog API correlation — only included
          when ``settings.datadog.enabled`` is ``True``

        Requires orchestrator strategy ``"parallel_sequential"`` (implemented in
        Phase 1 via ``_execute_parallel_then_sequential``).

        Args:
            namespace: Override namespace from CLI (takes precedence over
                ``settings.gke.default_namespace``).  Empty string falls back
                to the configured default.
            location: Override GKE location.  When provided, overrides
                ``settings.gke.location`` for Autopilot detection.
            cluster_name: Override cluster name.  When provided, overrides
                ``settings.gke.cluster_name`` for Autopilot detection.
        """
        from vaig.core.config import get_settings  # noqa: PLC0415
        from vaig.tools.gke._clients import detect_autopilot  # noqa: PLC0415

        settings = get_settings()

        # Build an effective GKEConfig that merges CLI overrides with file defaults.
        # This ensures detect_autopilot uses the cluster the user actually targeted.
        # Use model_copy so ALL fields from settings.gke are preserved
        # (proxy_url, impersonate_sa, exec_enabled, argocd_*, etc.) and only
        # the CLI-supplied overrides are replaced.
        effective_gke = settings.gke.model_copy(
            update={
                "cluster_name": cluster_name or settings.gke.cluster_name,
                "location": location or settings.gke.location,
                "default_namespace": namespace or settings.gke.default_namespace,
            }
        )
        is_autopilot = bool(detect_autopilot(effective_gke))
        effective_namespace = effective_gke.default_namespace

        # ── Read pre-fetched kubectl_top metrics ─────────────────────────
        # The actual kubectl_top calls happen in ``pre_execute_parallel()``
        # which the orchestrator invokes before threads launch.  Here we
        # only *consume* the results.  If ``pre_execute_parallel()`` was
        # never called (e.g. dry-run, ``vaig skills info``, direct test
        # call), the dict defaults to empty strings — prompts simply omit
        # the pre-gathered section and agents call kubectl_top themselves.
        prefetched = self._prefetched_metrics

        # ── Pre-resolve Datadog service identity ─────────────────────────
        # Uses the effective_gke (with CLI overrides) so that --namespace,
        # --cluster, --location flags are respected.  Resolution happens
        # here (not in pre_execute_parallel) because CLI overrides are only
        # available at this point.
        dd_resolution = self._prefetched_dd_resolution
        if settings.datadog.enabled and not dd_resolution.dd_service_name:
            try:
                dd_resolution = self._resolve_dd_service_name(
                    gke_config=effective_gke,
                    namespace=effective_namespace,
                    user_query=kwargs.get("user_query", ""),
                )
                self._prefetched_dd_resolution = dd_resolution
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Datadog service identity pre-resolution skipped (non-fatal)",
                    exc_info=True,
                )

        # Resolve the 3-state Argo Rollouts flag via helper (SH-07).
        argo_rollouts_active = self._detect_argo_rollouts(effective_namespace, effective_gke)

        # Resolve the 3-state ArgoCD flag via helper (SH-07).
        argocd_active = self._detect_argocd(effective_namespace, effective_gke)

        agents: list[dict[str, Any]] = [
            # ── Parallel group: core sub-gatherers ───────────────────────
            {
                "name": "node_gatherer",
                "role": "Cluster & Node Health Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "tool_categories": ["kubernetes"],
                "capabilities": [
                    "node",
                    "nodes",
                    "cluster",
                    "cpu",
                    "memory",
                    "disk",
                    "capacity",
                    "allocatable",
                    "pressure",
                    "resource",
                    "taint",
                    "cordon",
                    "drain",
                ],
                "system_instruction": build_node_gatherer_prompt(
                    is_autopilot=is_autopilot,
                    prefetched_node_metrics=prefetched["nodes"],
                    attachment_context=attachment_context,
                ),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 4 if is_autopilot else 15,
            },
            {
                "name": "workload_gatherer",
                "role": "Workload Health Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "injectable_agents": ["node_gatherer"],
                "tool_categories": (
                    ["kubernetes", "scaling", "argo_rollouts"] if argo_rollouts_active else ["kubernetes", "scaling"]
                ),
                "capabilities": [
                    "pod",
                    "pods",
                    "deployment",
                    "workload",
                    "restart",
                    "crash",
                    "crashloop",
                    "oom",
                    "container",
                    "replicas",
                    "replicaset",
                    "statefulset",
                    "daemonset",
                    "hpa",
                    "scaling",
                    "pending",
                    "evicted",
                    "oomkilled",
                ],
                "system_instruction": build_workload_gatherer_prompt(
                    namespace=effective_namespace,
                    argo_rollouts_enabled=argo_rollouts_active,
                    prefetched_pod_metrics=prefetched["pods"],
                    user_query=kwargs.get("user_query", ""),
                    attachment_context=attachment_context,
                ),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 20,
            },
            {
                "name": "event_gatherer",
                "role": "Events & Infrastructure Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "tool_categories": (["kubernetes", "helm", "argocd"] if argocd_active else ["kubernetes", "helm"]),
                "capabilities": [
                    "event",
                    "events",
                    "network",
                    "networking",
                    "dns",
                    "service",
                    "endpoint",
                    "ingress",
                    "connectivity",
                    "storage",
                    "pvc",
                    "volume",
                    "argocd",
                    "gitops",
                    "helm",
                    "configmap",
                    "secret",
                    "infrastructure",
                ],
                "system_instruction": build_event_gatherer_prompt(
                    namespace=effective_namespace,
                    user_query=kwargs.get("user_query", ""),
                    attachment_context=attachment_context,
                ),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 10,
            },
            {
                "name": "logging_gatherer",
                "role": "Cloud Logging Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "tool_categories": ["logging"],
                "capabilities": [
                    "log",
                    "logs",
                    "logging",
                    "error",
                    "errors",
                    "warning",
                    "warnings",
                    "stacktrace",
                    "exception",
                    "stderr",
                    "stdout",
                    "cloud",
                    "gcp",
                    "cloudlogging",
                ],
                "system_instruction": build_logging_gatherer_prompt(
                    namespace=effective_namespace,
                    attachment_context=attachment_context,
                ),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 8,
            },
        ]

        if settings.datadog.enabled:
            agents.append(
                {
                    "name": "datadog_gatherer",
                    "role": "Datadog API Correlation Gatherer",
                    "requires_tools": True,
                    "parallel_group": "gather",
                    "tool_categories": ["datadog", "kubernetes"],
                    "capabilities": [
                        "datadog",
                        "apm",
                        "trace",
                        "traces",
                        "latency",
                        "error-rate",
                        "throughput",
                        "monitoring",
                        "metric",
                        "metrics",
                        "dashboard",
                        "slo",
                        "alert",
                        "service-map",
                    ],
                    "system_instruction": build_datadog_gatherer_prompt(
                        namespace=effective_namespace,
                        cluster_name=effective_gke.cluster_name,
                        datadog_api_enabled=True,
                        dd_service_name=dd_resolution.dd_service_name,
                        dd_env=dd_resolution.dd_env,
                        dd_resource_type=dd_resolution.dd_resource_type,
                        attachment_context=attachment_context,
                    ),
                    "model": "gemini-2.5-flash",
                    "temperature": 0.0,
                    "max_iterations": 12,
                }
            )

        agents += [
            # ── Sequential tail: unchanged from get_agents_config() ──────
            {
                "name": "health_analyzer",
                "role": "Health Pattern Analyzer",
                "requires_tools": False,
                "system_instruction": _prefix_attachment_context(
                    (
                        HEALTH_ANALYZER_PROMPT + ANALYZER_AUTONOMOUS_OVERLAY
                        if settings.investigation.enabled is True
                        else HEALTH_ANALYZER_PROMPT
                    )
                    + build_attachment_seeded_section(attachment_priors_json or ""),
                    attachment_context,
                ),
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
            },
        ]

        # ── Optional investigation phase (gated by feature flag) ─────────
        if settings.investigation.enabled is True:
            investigator_kwargs: dict[str, Any] = {
                "name": "health_investigator",
                "role": "Hypothesis Investigator",
                "requires_tools": True,
                "tool_categories": ["kubernetes", "scaling", "mesh", "logging", "datadog"],
                "system_instruction": HEALTH_INVESTIGATOR_PROMPT + INVESTIGATOR_AUTONOMOUS_OVERLAY,
                "max_iterations": settings.investigation.max_iterations,
                "temperature": 0.1,
                "agent_class": "InvestigationAgent",
            }
            if settings.investigation.autonomous_mode is True:
                from vaig.core.global_budget import GlobalBudgetManager  # noqa: PLC0415
                from vaig.core.memory.pattern_store import PatternMemoryStore  # noqa: PLC0415
                from vaig.core.self_correction import SelfCorrectionController  # noqa: PLC0415

                investigator_kwargs["self_correction"] = SelfCorrectionController
                if settings.investigation.budget_per_run_usd > 0.0:
                    from vaig.core.config import GlobalBudgetConfig  # noqa: PLC0415

                    investigator_kwargs["budget_manager"] = GlobalBudgetManager(
                        config=GlobalBudgetConfig(max_cost_usd=settings.investigation.budget_per_run_usd)
                    )
                investigator_kwargs["pattern_store"] = PatternMemoryStore(base_dir=".vaig/memory/patterns")

            agents += [
                {
                    "name": "health_planner",
                    "role": "Investigation Planner",
                    "requires_tools": False,
                    "system_instruction": HEALTH_PLANNER_PROMPT,
                    "model": "gemini-2.5-flash",
                    "temperature": 0.1,
                    "agent_class": "SpecialistAgent",
                },
                investigator_kwargs,
            ]

        agents += [
            {
                "name": "health_verifier",
                "role": "Health Finding Verifier",
                "requires_tools": True,
                "tool_categories": ["kubernetes", "scaling", "mesh", "datadog", "logging", "monitoring"],
                "system_instruction": _prefix_attachment_context(
                    HEALTH_VERIFIER_PROMPT + build_verifier_ratification_section(attachment_priors_json),
                    attachment_context,
                ),
                "model": "gemini-2.5-flash",
                "max_iterations": 15,
                "temperature": 0.2,
            },
            {
                "name": "health_reporter",
                "role": "Health Report Generator",
                "requires_tools": False,
                "system_instruction": build_reporter_prompt(
                    namespace=effective_namespace,
                    datadog_api_enabled=settings.datadog.enabled,
                    attachment_context=attachment_context,
                ),
                "model": "gemini-2.5-flash",
                "temperature": 0.3,
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "response_schema": HealthReportGeminiSchema,
                "response_mime_type": "application/json",
            },
        ]

        return agents

    def post_process_report(
        self,
        content: str,
        run_quality: list[QualityIssue] | None = None,
    ) -> str:
        """Convert the reporter's structured JSON output to Markdown.

        Gemini's structured output mode returns a JSON string conforming to
        :class:`~vaig.skills.service_health.schema.HealthReport`.  This method
        validates the JSON via ``HealthReport.model_validate_json()`` and
        renders it as Markdown via ``report.to_markdown()``.

        After parsing, a **two-pass recommendation enrichment** step makes
        focused LLM calls (one per recommendation) to replace the generic
        ``expected_output`` and ``interpretation`` fields with specific,
        expert-level debugging guidance — mimicking the quality of interactive
        chat mode.  Enrichment is best-effort: if it fails for any
        recommendation, the original values are preserved.

        Before parsing, the raw content is passed through
        :func:`~vaig.utils.json_cleaner.clean_llm_json` to strip common LLM
        artefacts such as markdown code fences or conversational preamble.

        If JSON parsing or schema validation fails after cleaning, a visible
        warning is prepended to the raw content so the failure is never silent.

        If *run_quality* contains any issues, a ``## Run Quality`` table is
        prepended to the Markdown output.
        """
        cleaned = clean_llm_json(content)
        try:
            report = HealthReport.model_validate_json(cleaned)

            # ── SPEC-RP-01: Dedup findings by fingerprint ─────────────────
            if report.findings:
                deduped = _dedup_findings(report.findings)
                if len(deduped) != len(report.findings):
                    logger.info(
                        "Finding dedup: %d → %d findings (%d duplicates removed)",
                        len(report.findings),
                        len(deduped),
                        len(report.findings) - len(deduped),
                    )
                    report = report.model_copy(update={"findings": deduped})

            # ── SPEC-SH-13: Contradiction detection ───────────────────────
            contradiction_findings = detect_contradictions(report)
            if contradiction_findings:
                report = report.model_copy(update={"findings": report.findings + contradiction_findings})

            # ── SPEC-V2-AUDIT-10: Causal correlation rules ─────────────────
            causal_findings = apply_contradiction_rules(report)
            if causal_findings:
                report = report.model_copy(update={"findings": report.findings + causal_findings})

            # ── SPEC-ATT-10 §6.5.3: Verifier ratification pass ─────────────
            if report.ratification_json:
                try:
                    report = _apply_ratification(report, report.ratification_json)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Verifier ratification pass failed — findings unchanged",
                        exc_info=True,
                    )

            # ── SPEC-ATT-10 §6.5.4: Reporter attachment sections ────────────
            if report.attachment_priors is not None or report.ratification_json:
                try:
                    sections_md = _render_attachment_sections(report)
                    if sections_md:
                        report = report.model_copy(update={"attachment_sections_md": sections_md})
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Attachment sections rendering failed — skipping",
                        exc_info=True,
                    )

            # ── SPEC-ATT-10 §6.5.5: Operating mode detection ───────────────
            operating_mode = self._detect_operating_mode(
                offline_mode=self._offline_mode,
                attachments_present=self._attachments_present,
            )
            report = report.model_copy(update={"operating_mode": operating_mode})
            if operating_mode == OperatingMode.ATTACHMENT_ONLY and report.findings:
                # All findings in attachment-only mode are sourced exclusively
                # from attachment evidence; override source_support accordingly.
                patched: list[Finding] = []
                for f in report.findings:
                    patched.append(f.model_copy(update={"source_support": "attachment_only"}))
                report = report.model_copy(update={"findings": patched})
                logger.info(
                    "ATTACHMENT_ONLY mode: tagged %d findings with source_support='attachment_only'",
                    len(patched),
                )


            # Warn if report has no meaningful data
            if not report.findings and not report.service_statuses:
                logger.warning(
                    "Reporter produced a report with no findings and no service statuses. "
                    "This may indicate data was lost in the agent pipeline."
                )

            # Two-pass enrichment: enhance recommendations with focused LLM
            # calls using the default model (gemini-2.5-pro) for higher quality.
            if report.findings and report.recommendations:
                try:
                    report = self._enrich_report_recommendations(report)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Recommendation enrichment failed, using original values",
                        exc_info=True,
                    )

            return self._render_run_quality_section(run_quality) + report.to_markdown()
        except (ValueError, ValidationError):
            logger.warning(
                "Failed to parse reporter JSON as HealthReport, "
                "returning raw content with warning. Input starts with: %.100s",
                content,
                exc_info=True,
            )
            return "⚠️ Report parsing failed — showing raw output\n\n" + content

    @staticmethod
    def _render_run_quality_section(
        run_quality: list[QualityIssue] | None,
    ) -> str:
        """Return a Markdown Run Quality table for runs with issues, or empty string.

        If *run_quality* is None or empty, returns an empty string so nothing
        is prepended to the report.

        Otherwise returns a ``## Run Quality`` section with a Markdown table
        listing every issue and a suggested action line.
        """
        if not run_quality:
            return ""
        issues = list(run_quality)
        n = len(issues)
        label = "issue" if n == 1 else "issues"
        lines: list[str] = [
            f"## Run Quality ⚠ ({n} {label})\n",
            "| Issue | Where | Consequence |",
            "|---|---|---|",
        ]
        for issue in issues:
            lines.append(f"| {issue.kind} | {issue.where} | {issue.consequence} |")
        lines.append("")
        lines.append(
            "Suggested action: re-run during a lower-quota window, "
            "or pass `--model gemini-2.5-flash` for the whole run to avoid 429s on gemini-2.5-pro quota."
        )
        return "\n".join(lines) + "\n\n"

    def _enrich_report_recommendations(
        self,
        report: HealthReport,
        *,
        overall_timeout: float = 120.0,
    ) -> HealthReport:
        """Run async recommendation enrichment from a synchronous context.

        Lazily initialises a :class:`~vaig.core.client.GeminiClient` (cached on
        ``self._gemini_client``) and delegates to :func:`enrich_recommendations`.
        Uses a :class:`~concurrent.futures.ThreadPoolExecutor` with a hard
        *overall_timeout* to guarantee the enrichment never blocks the report
        pipeline — even when GCP auth probes hang (e.g. inside CI or
        environments without credentials).

        Args:
            report: The parsed HealthReport to enrich.
            overall_timeout: Maximum seconds to wait for the entire enrichment
                pass.  Defaults to 120 s (enough for ~10 recommendations at
                30 s each with concurrency).  Set lower in tests or CI.
        """
        import asyncio

        from vaig.core.client import GeminiClient
        from vaig.core.config import get_settings
        from vaig.skills.service_health.recommendation_enricher import (
            enrich_recommendations,
        )

        settings = get_settings()
        # Lazy-init: reuse client and pool across calls (SH-08).
        if self._gemini_client is None:
            self._gemini_client = GeminiClient(settings, fallback_model=settings.models.fallback)
        client = self._gemini_client
        if self._enrichment_pool is None:
            self._enrichment_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pool = self._enrichment_pool

        logger.info(
            "Starting two-pass recommendation enrichment for %d recommendations using %s (timeout=%ss)",
            len(report.recommendations),
            settings.models.default,
            overall_timeout,
        )

        # Run enrichment in a dedicated thread with a hard timeout so a
        # hanging credential probe (GCE metadata, etc.) can never block the
        # report pipeline.
        import sys  # noqa: PLC0415

        from vaig.core.log import _make_console  # noqa: PLC0415

        future = pool.submit(asyncio.run, enrich_recommendations(report, client))
        try:
            if sys.stderr.isatty():
                console = _make_console(stderr=True)
                with console.status(
                    "✨ Enriching recommendations with detailed context...",
                    spinner="dots",
                ):
                    enriched_report = future.result(timeout=overall_timeout)
            else:
                logger.info("Enriching recommendations with detailed context...")
                enriched_report = future.result(timeout=overall_timeout)
            return enriched_report
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Recommendation enrichment exceeded hard timeout of %ss; returning original report without enrichment.",
                overall_timeout,
            )
            future.cancel()
            return report

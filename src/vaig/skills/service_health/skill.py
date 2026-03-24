"""Service Health Skill — live Kubernetes service health assessment.

A 4-agent sequential pipeline with two-pass verification that demonstrates
the ToolAwareAgent + Orchestrator integration.  The first agent uses live
tools to collect cluster health data; the second analyzes patterns; the
third verifies findings with targeted tool calls; the fourth produces a
structured JSON report (rendered to Markdown).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from vaig.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.service_health.prompts import (
    HEALTH_ANALYZER_PROMPT,
    HEALTH_VERIFIER_PROMPT,
    PHASE_PROMPTS,
    SYSTEM_INSTRUCTION,
    build_datadog_gatherer_prompt,
    build_event_gatherer_prompt,
    build_gatherer_prompt,
    build_logging_gatherer_prompt,
    build_node_gatherer_prompt,
    build_reporter_prompt,
    build_workload_gatherer_prompt,
)
from vaig.skills.service_health.schema import HealthReport
from vaig.tools.gke._clients import ensure_client_initialized
from vaig.utils.json_cleaner import clean_llm_json

logger = logging.getLogger(__name__)


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

    def pre_execute_parallel(self, query: str) -> None:  # noqa: ARG002
        """Pre-warm the K8s client cache before parallel gatherers launch.

        The K8s client cache (``_CLIENT_CACHE``) is not thread-safe on first
        write because :func:`~vaig.tools.gke._clients._suppress_stderr` mutates
        ``sys.stdout`` and OS fd 2.  Calling this hook once, sequentially,
        ensures the client is fully constructed and stored in the cache before
        any concurrent threads start.

        Errors are swallowed silently — pre-warming is best-effort.  If the
        client cannot be initialized (e.g. no kubeconfig, missing package),
        the individual tools will surface the error through their normal
        :class:`~vaig.tools.base.ToolResult` mechanism.
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
        return self.get_parallel_agents_config(
            namespace=namespace,
            location=location,
            cluster_name=cluster_name,
        )

    def get_sequential_agents_config(
        self,
        *,
        namespace: str = "",
        location: str = "",  # noqa: ARG002 — reserved for future cluster routing
        cluster_name: str = "",  # noqa: ARG002 — reserved for future cluster routing
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
        gatherer_prompt = build_gatherer_prompt(
            helm_enabled=settings.gke.helm_enabled,
            argocd_enabled=settings.gke.argocd_enabled,
            datadog_api_enabled=settings.datadog.enabled,
        )
        namespace = namespace or settings.gke.default_namespace
        return [
            {
                "name": "health_gatherer",
                "role": "Health Data Gatherer",
                "requires_tools": True,
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
                "system_instruction": HEALTH_ANALYZER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.2,  # Low temp for precise analysis
            },
            {
                "name": "health_verifier",
                "role": "Health Finding Verifier",
                "requires_tools": True,
                "system_instruction": HEALTH_VERIFIER_PROMPT,
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
                ),
                "model": "gemini-2.5-flash",
                "temperature": 0.3,  # Slightly higher for natural writing
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "response_schema": HealthReport,
                "response_mime_type": "application/json",
            },
        ]

    def get_parallel_agents_config(
        self,
        *,
        namespace: str = "",
        location: str = "",
        cluster_name: str = "",
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

        agents: list[dict[str, Any]] = [
            # ── Parallel group: core sub-gatherers ───────────────────────
            {
                "name": "node_gatherer",
                "role": "Cluster & Node Health Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "capabilities": [
                    "node", "nodes", "cluster", "cpu", "memory", "disk",
                    "capacity", "allocatable", "pressure", "resource",
                    "taint", "cordon", "drain",
                ],
                "system_instruction": build_node_gatherer_prompt(is_autopilot=is_autopilot),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 4 if is_autopilot else 15,
            },
            {
                "name": "workload_gatherer",
                "role": "Workload Health Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "capabilities": [
                    "pod", "pods", "deployment", "workload", "restart",
                    "crash", "crashloop", "oom", "container", "replicas",
                    "replicaset", "statefulset", "daemonset", "hpa",
                    "scaling", "pending", "evicted", "oomkilled",
                ],
                "system_instruction": build_workload_gatherer_prompt(
                    namespace=effective_namespace,
                ),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 12,
            },
            {
                "name": "event_gatherer",
                "role": "Events & Infrastructure Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "capabilities": [
                    "event", "events", "network", "networking", "dns",
                    "service", "endpoint", "ingress", "connectivity",
                    "storage", "pvc", "volume", "argocd", "gitops",
                    "helm", "configmap", "secret", "infrastructure",
                ],
                "system_instruction": build_event_gatherer_prompt(namespace=effective_namespace),
                "model": "gemini-2.5-pro",
                "temperature": 0.0,
                "max_iterations": 10,
            },
            {
                "name": "logging_gatherer",
                "role": "Cloud Logging Gatherer",
                "requires_tools": True,
                "parallel_group": "gather",
                "capabilities": [
                    "log", "logs", "logging", "error", "errors", "warning",
                    "warnings", "stacktrace", "exception", "stderr",
                    "stdout", "cloud", "gcp", "cloudlogging",
                ],
                "system_instruction": build_logging_gatherer_prompt(namespace=effective_namespace),
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
                    "capabilities": [
                        "datadog", "apm", "trace", "traces", "latency",
                        "error-rate", "throughput", "monitoring", "metric",
                        "metrics", "dashboard", "slo", "alert", "service-map",
                    ],
                    "system_instruction": build_datadog_gatherer_prompt(
                        namespace=effective_namespace,
                        cluster_name=effective_gke.cluster_name,
                        datadog_api_enabled=True,
                    ),
                    "model": "gemini-2.5-flash",
                    "temperature": 0.0,
                    "max_iterations": 8,
                }
            )

        agents += [
            # ── Sequential tail: unchanged from get_agents_config() ──────
            {
                "name": "health_analyzer",
                "role": "Health Pattern Analyzer",
                "requires_tools": False,
                "system_instruction": HEALTH_ANALYZER_PROMPT,
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
            },
            {
                "name": "health_verifier",
                "role": "Health Finding Verifier",
                "requires_tools": True,
                "system_instruction": HEALTH_VERIFIER_PROMPT,
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
                ),
                "model": "gemini-2.5-flash",
                "temperature": 0.3,
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "response_schema": HealthReport,
                "response_mime_type": "application/json",
            },
        ]

        return agents

    def post_process_report(self, content: str) -> str:
        """Convert the reporter's structured JSON output to Markdown.

        Gemini's structured output mode returns a JSON string conforming to
        :class:`~vaig.skills.service_health.schema.HealthReport`.  This method
        validates the JSON via ``HealthReport.model_validate_json()`` and
        renders it as Markdown via ``report.to_markdown()``.

        Before parsing, the raw content is passed through
        :func:`~vaig.utils.json_cleaner.clean_llm_json` to strip common LLM
        artefacts such as markdown code fences or conversational preamble.

        If JSON parsing or schema validation fails after cleaning, a visible
        warning is prepended to the raw content so the failure is never silent.
        """
        cleaned = clean_llm_json(content)
        try:
            report = HealthReport.model_validate_json(cleaned)

            # Warn if report has no meaningful data
            if not report.findings and not report.service_statuses:
                logger.warning(
                    "Reporter produced a report with no findings and no service statuses. "
                    "This may indicate data was lost in the agent pipeline."
                )

            return report.to_markdown()
        except (ValueError, ValidationError):
            logger.warning(
                "Failed to parse reporter JSON as HealthReport, "
                "returning raw content with warning. Input starts with: %.100s",
                content,
                exc_info=True,
            )
            return (
                "⚠️ Report parsing failed — showing raw output\n\n"
                + content
            )

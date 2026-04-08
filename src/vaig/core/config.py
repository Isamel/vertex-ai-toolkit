"""Core configuration — Pydantic settings with layered config loading."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ── Shared constants ─────────────────────────────────────────
# Centralised defaults that are referenced across the codebase to avoid
# magic numbers.  Import from here instead of hardcoding.

DEFAULT_CHARS_PER_TOKEN: float = 4.0
"""Conservative estimate of characters per token for quick size checks.

The ``ChunkingConfig.chars_per_token`` (default ``2.0``) is more conservative
and is used for budget calculations.  This constant is the *fallback* used by
``load_file`` and ``count_tokens_safe`` when no settings object is available.
"""

DEFAULT_MAX_OUTPUT_TOKENS: int = 65_536
"""Default max output tokens for Gemini models."""

DEFAULT_CONTEXT_WINDOW: int = 1_048_576
"""Default context window size in tokens for Gemini models."""


class AuthMode(StrEnum):
    """Authentication mode for Vertex AI."""

    ADC = "adc"
    IMPERSONATE = "impersonate"


class ProjectEntry(BaseModel):
    """A GCP project the user has access to.

    Used to maintain a catalog of available projects in config,
    with optional description and role annotation.
    """

    project_id: str
    description: str = ""
    role: str = ""  # e.g., "vertex-ai", "gke", "both"


class GCPConfig(BaseModel):
    """GCP project configuration."""

    project_id: str = ""
    location: str = "us-central1"
    fallback_location: str = "us-central1"
    available_projects: list[ProjectEntry] = Field(default_factory=list)

    def model_post_init(self, _context: Any) -> None:
        """Warn when project_id is unset (will fall back to ADC default)."""
        if not self.project_id:
            logger.debug(
                "GCP project_id is empty — will use Application Default Credentials "
                "project. Set gcp.project_id in config or GOOGLE_CLOUD_PROJECT env var "
                "to make it explicit."
            )


class AuthConfig(BaseModel):
    """Authentication configuration."""

    mode: AuthMode = AuthMode.ADC
    impersonate_sa: str = ""


class ThinkingConfig(BaseModel):
    """Configuration for Gemini thinking mode.

    When ``enabled`` is True and the model supports thinking (e.g.
    ``gemini-2.5-flash``, ``gemini-2.5-pro``), the ``thinking_config``
    parameter is included in the ``GenerateContentConfig`` sent to the API.

    Thinking mode is opt-in — disabled by default — so existing behaviour
    is unchanged unless explicitly enabled via config or CLI.
    """

    enabled: bool = False
    budget_tokens: int | None = None
    """Token budget for thinking. ``None`` means "use model default".

    Set to ``0`` to disable thinking, ``-1`` for automatic budget,
    or a positive integer for a fixed budget.
    """
    include_thoughts: bool = True
    """Whether to include thought content in the response.

    When True, thought parts are returned alongside the output.
    """


# ── Thinking-capable model detection ─────────────────────────

THINKING_CAPABLE_MODELS: frozenset[str] = frozenset(
    {
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    }
)
"""Model name prefixes that support thinking mode."""


def supports_thinking(model_name: str) -> bool:
    """Check if a model supports thinking mode.

    Uses prefix matching so versioned variants (e.g.
    ``gemini-2.5-flash-001``) are also detected.
    """
    return any(model_name.startswith(prefix) for prefix in THINKING_CAPABLE_MODELS)


class GenerationConfig(BaseModel):
    """Model generation parameters."""

    temperature: float = 0.7
    max_output_tokens: int = 16384
    top_p: float = 0.95
    top_k: int = 40
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    description: str = ""
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_window: int = DEFAULT_CONTEXT_WINDOW


class ModelsConfig(BaseModel):
    """Model selection configuration."""

    default: str = "gemini-2.5-pro"
    fallback: str = "gemini-2.5-flash"
    available: list[ModelInfo] = Field(default_factory=list)


class SessionConfig(BaseModel):
    """Session persistence configuration."""

    db_path: str = "~/.vaig/sessions.db"
    repl_history_path: str = "~/.vaig/repl_history"
    auto_save: bool = True
    max_history_messages: int = 100
    max_history_tokens: int = 28_000
    """Conservative token budget for conversation history.

    When the estimated token count of the in-memory history approaches
    this limit (controlled by ``summarization_threshold``), older messages
    are summarized into a single compact message to free context window
    space for new turns.
    """
    summarization_threshold: float = 0.8
    """Fraction of ``max_history_tokens`` at which summarization triggers.

    A value of ``0.8`` means summarization runs when the rough token
    estimate reaches 80 % of ``max_history_tokens``.
    """
    summary_target_tokens: int = 4_000
    """Target size in tokens for the summary message that replaces
    the older portion of history."""


class SkillsConfig(BaseModel):
    """Skills configuration."""

    enabled: list[str] = Field(
        default_factory=lambda: [
            "rca",
            "anomaly",
            "migration",
            "log-analysis",
            "error-triage",
            "config-audit",
            "slo-review",
            "postmortem",
            "code-review",
            "iac-review",
            "cost-analysis",
            "capacity-planning",
            "test-generation",
            "compliance-check",
            "api-design",
            "runbook-generator",
            "dependency-audit",
            "db-review",
            "pipeline-review",
            "perf-analysis",
            "threat-model",
            "change-risk",
            "alert-tuning",
            "resilience-review",
            "incident-comms",
            "toil-analysis",
            "network-review",
            "adr-generator",
            "service-health",
            "discovery",
        ]
    )
    custom_dir: str | None = None
    auto_routing: bool = True
    auto_routing_threshold: float = 1.5


class AgentsConfig(BaseModel):
    """Multi-agent configuration."""

    max_concurrent: int = 3
    orchestrator_model: str = "gemini-2.5-pro"
    specialist_model: str = "gemini-2.5-flash"
    max_iterations_retry: int = 15
    parallel_tool_calls: bool = True
    """Execute independent tool calls in parallel (async path only).

    When Gemini returns multiple function calls in a single response,
    they are independent by API contract.  Enabling this executes them
    concurrently via ``asyncio.gather`` instead of sequentially.
    The sync path always runs sequentially regardless of this setting.
    """
    max_concurrent_tool_calls: int = 5
    """Maximum number of tool calls to execute concurrently.

    Acts as a semaphore limit to prevent API throttling when the model
    requests many tool calls at once.  Only applies when
    ``parallel_tool_calls`` is enabled.
    """
    max_failures_before_fallback: int = 2
    """Number of consecutive rate-limit or connection failures on a single agent
    before the orchestrator switches it to ``settings.models.fallback``.

    Set to ``0`` to disable model fallback entirely.
    """
    min_inter_call_delay: float = Field(default=0.0, ge=0.0)
    """Seconds to sleep between LLM API calls in the tool loop (RPM throttle).

    Set to ``0`` (default) to disable throttling.  A value of ``1.0`` limits
    the loop to ~60 calls/min which avoids 429s on quota-limited projects.
    """


class CodingConfig(BaseModel):
    """Coding agent configuration."""

    workspace_root: str = "."
    max_tool_iterations: int = 25
    confirm_actions: bool = True
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    pipeline_mode: bool = Field(
        default=False,
        description=(
            "When True, routes `vaig code` through CodingSkillOrchestrator "
            "(Planner→Implementer→Verifier) instead of the single-agent CodingAgent"
        ),
    )
    denied_commands: list[str] = Field(
        default_factory=lambda: [
            # Destructive disk / filesystem operations
            r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/\s*$",  # rm -rf /
            r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/\w",  # rm -rf /anything
            r"\bmkfs\b",
            r"\bdd\b\s+",
            r"\bformat\b",
            r"\bfdisk\b",
            # Fork bomb patterns
            r":\(\)\s*\{",
            # Insecure permission patterns
            r"\bchmod\s+(-\w+\s+)*777\b",
            # Piped remote execution
            r"\bcurl\b.*\|\s*(sh|bash|zsh)\b",
            r"\bwget\b.*\|\s*(sh|bash|zsh)\b",
            # Generic pipe-to-shell (catch-all for remote execution)
            r"\|\s*sh\b",
            r"\|\s*bash\b",
            # System control
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bhalt\b",
            r"\bpoweroff\b",
            r"\binit\s+[0-6]",
            # Privilege escalation
            r"^\s*sudo\b",
            # Direct device writes
            r">\s*/dev/sd",
        ],
        description=(
            "Regex patterns for commands that should NEVER be executed. "
            "Checked against the full command string before execution. "
            "Extend this list in config to add project-specific denials."
        ),
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "WARNING"
    show_path: bool = False
    file_enabled: bool = True
    file_path: str = "~/.vaig/logs/vaig.log"
    file_level: str = "DEBUG"
    file_max_bytes: int = 5_242_880  # 5 MB
    file_backup_count: int = 3
    tool_results: bool = True
    tool_results_dir: str = "~/.vaig"


class RetryConfig(BaseModel):
    """Retry and backoff configuration for API calls."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    rate_limit_initial_delay: float = Field(default=8.0, ge=0.0)
    """Longer initial backoff (seconds) used when a 429 rate-limit error is
    detected.  Applied instead of ``initial_delay`` so the client waits
    longer before retrying quota-exhausted requests."""
    retryable_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
    )


class ContextConfig(BaseModel):
    """Context and file handling configuration."""

    max_file_size_mb: int = 50
    supported_extensions: dict[str, list[str]] = Field(default_factory=dict)
    ignore_patterns: list[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    """Configuration for chunked file processing (Map-Reduce for large files)."""

    chunk_overlap_ratio: float = 0.1
    token_safety_margin: float = 0.1
    chars_per_token: float = 2.0
    inter_chunk_delay: float = 2.0


class HelmConfig(BaseModel):
    """Helm integration configuration."""

    enabled: bool = True  # Helm tools available by default


class ArgoCDConfig(BaseModel):
    """ArgoCD integration configuration."""

    enabled: bool = False
    server: str = Field(default="", repr=False)
    token: str = Field(default="", repr=False)
    context: str = ""
    namespace: str = ""
    verify_ssl: bool = True


class DatadogLabelConfig(BaseModel):
    """Configurable tag/label names for Datadog API queries.

    Override these to match custom Datadog tag naming conventions used
    in your organisation.  All defaults match the standard Kubernetes
    integration tag names shipped by the Datadog agent.
    """

    cluster_name: str = "kube_cluster_name"
    service: str = "service"
    env: str = "env"
    version: str = "version"
    namespace: str = "kube_namespace"
    deployment: str = "kube_deployment"
    pod_name: str = "pod_name"
    custom: dict[str, str] = Field(default_factory=dict)


class DatadogDetectionConfig(BaseModel):
    """Configurable annotation/label prefixes for Datadog detection in K8s.

    Override these when your cluster uses non-standard annotation or
    label prefixes to signal Datadog instrumentation.
    """

    annotation_prefixes: list[str] = Field(
        default_factory=lambda: [
            "ad.datadoghq.com/",
            "admission.datadoghq.com/",
        ]
    )
    label_prefix: str = "tags.datadoghq.com/"
    env_vars: list[str] = Field(
        default_factory=lambda: [
            "DD_AGENT_HOST",
            "DD_TRACE_AGENT_URL",
            "DD_SERVICE",
            "DD_ENV",
            "DD_VERSION",
            "DD_TRACE_ENABLED",
            "DD_PROFILING_ENABLED",
            "DD_LOGS_INJECTION",
            "DD_RUNTIME_METRICS_ENABLED",
        ]
    )


class DatadogAPIConfig(BaseModel):
    """Datadog REST API integration configuration."""

    enabled: bool = False
    api_key: str = Field(default="", repr=False)
    app_key: str = Field(default="", repr=False)
    site: str = "datadoghq.com"
    timeout: int = 30
    ssl_verify: bool | str = Field(
        default=True,
        description=(
            "SSL certificate verification for Datadog API requests. "
            "True (default) = standard SSL verification. "
            "False = disable SSL verification (not recommended; use for debugging only). "
            "str = path to a custom CA bundle file (e.g. '/etc/ssl/certs/corporate-ca.crt'). "
            "The REQUESTS_CA_BUNDLE environment variable is also respected by the requests "
            "library and takes effect when ssl_verify=True."
        ),
    )
    labels: DatadogLabelConfig = Field(default_factory=DatadogLabelConfig)
    detection: DatadogDetectionConfig = Field(default_factory=DatadogDetectionConfig)
    custom_metrics: dict[str, str] = Field(default_factory=dict)
    metric_mode: Literal["k8s_agent", "apm", "both", "auto"] = Field(
        default="auto",
        description=(
            "Metric source: 'k8s_agent' for kubernetes.* metrics (requires DaemonSet Agent), "
            "'apm' for trace.* metrics (APM-only setups), "
            "'both' for combined kubernetes.* and trace.* metrics, "
            "'auto' (default) tries k8s_agent first — if all queries return 0 data points, "
            "falls back to apm templates automatically"
        ),
    )
    cluster_name_override: str = Field(
        default="",
        description=(
            "Override the cluster_name tag value used in Datadog queries. "
            "When empty, uses the GKE cluster name."
        ),
    )
    default_lookback_hours: float = Field(
        default=4.0,
        description=(
            "Default lookback window (hours) for APM trace queries. "
            "Increase for low-traffic services."
        ),
    )
    apm_operation: str = Field(
        default="auto",
        description=(
            "APM operation name for trace.* metrics (e.g. 'servlet.request', "
            "'grpc.server'). When set to 'auto' (default), the system probes "
            "common operation names to find the one with data."
        ),
    )

    @field_validator("ssl_verify", mode="before")
    @classmethod
    def _validate_ssl_verify(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError(
                "ssl_verify must be True, False, or a non-empty path to a CA bundle"
            )
        return v

    @field_validator("default_lookback_hours")
    @classmethod
    def _validate_default_lookback_hours(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"default_lookback_hours must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _auto_enable_or_disable(self) -> DatadogAPIConfig:
        """Auto-enable when both keys are present; disable when enabled=True but keys missing."""
        has_keys = bool(self.api_key and self.app_key)
        if not self.enabled and has_keys:
            # Both keys provided — enable automatically so users just need to set the keys.
            self.enabled = True
        elif self.enabled and not has_keys:
            # Requested enabled=True but keys are missing — disable and warn.
            logger.warning(
                "datadog.enabled=True but api_key or app_key is missing — "
                "disabling Datadog API tools. Set VAIG_DATADOG__API_KEY and "
                "VAIG_DATADOG__APP_KEY (or datadog.api_key/app_key in config)."
            )
            self.enabled = False
        return self


class GKEConfig(BaseModel):
    """GKE live-cluster connection and query configuration."""

    cluster_name: str = ""
    project_id: str = ""
    location: str = ""
    default_namespace: str = "default"
    kubeconfig_path: str = ""
    context: str = ""
    log_limit: int = 100
    metrics_interval_minutes: int = 60
    # Proxy URL for private GKE clusters. The kubernetes Python client
    # ignores the ``proxy-url`` field in kubeconfig; this setting lets
    # users provide an explicit override via config/CLI.
    proxy_url: str = ""
    # Service account email to impersonate for GKE/GCP observability APIs
    # (Cloud Logging, Cloud Monitoring, GKE cluster API).  When set, this
    # overrides ``auth.impersonate_sa`` for GKE tools only, enabling
    # dual-auth scenarios where Vertex AI and GKE use different SAs.
    impersonate_sa: str = ""
    # Allow exec_command tool to execute diagnostic commands inside containers.
    # Disabled by default for security — must be explicitly opted-in.
    exec_enabled: bool = False
    # Helm integration — enabled by default.
    helm_enabled: bool = True
    # ArgoCD integration — None means auto-detect via CRD + annotation fallback,
    # True forces enable, False disables entirely.
    argocd_enabled: bool | None = None  # None=auto-detect, True=force-enable, False=disable
    argocd_server: str = Field(default="", repr=False)
    argocd_token: str = Field(default="", repr=False)
    argocd_context: str = ""
    argocd_namespace: str = ""
    argocd_verify_ssl: bool = True
    # Argo Rollouts integration — None means auto-detect via CRD presence,
    # True forces enable, False disables entirely.
    argo_rollouts_enabled: bool | None = None
    # Timeout in seconds for Kubernetes API calls.  Applied via ``_request_timeout``
    # on all CustomObjectsApi and AppsV1Api requests.  Prevents indefinite hangs
    # when the cluster is unreachable or slow (connect timeout=None bug).
    request_timeout: int = 30
    # Short timeout (seconds) for CRD existence checks.  The CRD probe runs
    # before each ArgoCD/Argo Rollouts tool invocation on a potentially different
    # cluster (e.g. Argo on a separate cluster).  A short timeout here prevents
    # the tool from hanging ~84s (urllib3 retries × TCP timeout) when the
    # apiextensions endpoint is unreachable.
    crd_check_timeout: int = 5
    # Timeout (seconds) for Argo Rollouts tool API calls.  Argo Rollouts
    # typically lives on a *separate* cluster; a shorter timeout prevents
    # the 5 tool functions from hanging ~84s when that cluster is
    # unreachable, while still being generous enough for normal queries.
    argo_request_timeout: int = 10


class OllamaConfig(BaseModel):
    """Ollama-compatible proxy configuration.

    When ``enabled`` is True, the Ollama-compatible API endpoints
    (``/api/generate``, ``/api/chat``, ``/api/tags``) are registered
    in the FastAPI application, allowing Ollama clients (VS Code
    Continue, Cody, CLI tools) to use VAIG's Vertex AI backend.
    """

    enabled: bool = False


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    description: str = ""


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) integration configuration."""

    enabled: bool = False
    auto_register: bool = False
    servers: list[MCPServerConfig] = Field(default_factory=list)


class PluginConfig(BaseModel):
    """Plugin tool registration configuration.

    Allows users to register custom tools from Python modules
    without modifying vaig source code.
    """

    enabled: bool = False
    directories: list[str] = Field(default_factory=list)


class SafetySettingConfig(BaseModel):
    """A single safety setting mapping a harm category to a block threshold.

    Attributes:
        category: Harm category name (e.g. ``HARM_CATEGORY_HARASSMENT``).
        threshold: Block threshold (e.g. ``BLOCK_MEDIUM_AND_ABOVE``).
    """

    category: str
    threshold: str = "BLOCK_MEDIUM_AND_ABOVE"


class SafetyConfig(BaseModel):
    """Safety settings for Gemini API content filtering.

    When ``enabled`` is True and ``settings`` is non-empty, the configured
    safety thresholds are passed to every ``GenerateContentConfig``.
    """

    enabled: bool = True
    settings: list[SafetySettingConfig] = Field(default_factory=list)


class ContextWindowConfig(BaseModel):
    """Context window monitoring configuration.

    Controls thresholds for warning and error states when the model's
    prompt token usage approaches the context window limit.
    """

    warn_threshold_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    """Percentage of context window usage that triggers a WARNING log."""
    error_threshold_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    """Percentage of context window usage that triggers an ERROR log."""
    context_window_size: int = DEFAULT_CONTEXT_WINDOW
    """Default context window size in tokens (overridden per model via models.available)."""

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> ContextWindowConfig:
        """Ensure warn_threshold_pct <= error_threshold_pct."""
        if self.warn_threshold_pct > self.error_threshold_pct:
            raise ValueError(
                f"warn_threshold_pct ({self.warn_threshold_pct}) must be <= "
                f"error_threshold_pct ({self.error_threshold_pct})"
            )
        return self


class PlatformConfig(BaseModel):
    """Platform authentication and management configuration.

    When ``enabled`` is True, the CLI operates in platform mode with
    centralized auth, config enforcement, and admin management via a
    backend API.  When False (the default), all platform features are
    disabled and the CLI behaves as a standalone tool.
    """

    enabled: bool = False
    backend_url: str = ""
    org_id: str = ""

    @model_validator(mode="after")
    def _validate_backend_url_when_enabled(self) -> PlatformConfig:
        """Require ``backend_url`` when platform mode is enabled."""
        if self.enabled and not self.backend_url:
            raise ValueError(
                "platform.backend_url is required when platform.enabled is True"
            )
        return self


class BudgetConfig(BaseModel):
    """Token budget tracking configuration.

    Controls session-level cost tracking and enforcement.
    """

    enabled: bool = False
    max_cost_usd: float = 5.0
    warn_threshold: float = 0.8  # 80% of max_cost_usd
    action: Literal["warn", "stop"] = "warn"
    max_cost_per_run: float = 0.0
    """Maximum USD cost allowed for a single orchestrator pipeline run.

    When the accumulated cost of all agent steps within one
    :meth:`~vaig.agents.orchestrator.Orchestrator.execute_with_tools` (or its
    async / parallel-sequential variants) call exceeds this value, the pipeline
    halts immediately and returns partial results with
    ``OrchestratorResult.budget_exceeded=True``.

    Set to ``0.0`` (the default) to disable the per-run cost circuit breaker.
    """


class CacheConfig(BaseModel):
    """Response caching configuration.

    Controls the in-memory LRU cache for non-streaming, non-tool-use
    Gemini API responses.  Disabled by default because LLM responses
    often depend on conversation context — enable only when appropriate
    (e.g. repeated stateless queries).
    """

    enabled: bool = False
    max_size: int = 128
    ttl_seconds: int = 300


class TelemetryConfig(BaseModel):
    """Local usage telemetry configuration.

    Controls the in-process event collector that persists usage data
    to a local SQLite database for analytics and self-diagnostics.
    """

    enabled: bool = True
    buffer_size: int = 50


class AuditConfig(BaseModel):
    """Audit logging configuration — sends events to BigQuery + Cloud Logging."""

    enabled: bool = False
    bigquery_dataset: str = "vaig_audit"
    bigquery_table: str = "audit_events"
    cloud_logging_log_name: str = "vaig-audit"
    buffer_size: int = 20
    flush_interval_seconds: int = 30


class WebhookServerConfig(BaseModel):
    """Webhook server configuration — receives Datadog alert webhooks.

    The webhook server runs as a FastAPI application that:
    1. Receives Datadog Monitor webhook payloads
    2. Runs vaig health analysis on affected services (background task)
    3. Dispatches results via PagerDuty + Google Chat

    Auto-enables when ``hmac_secret`` is provided.
    """

    enabled: bool = False
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8080
    hmac_secret: str = Field(default="", repr=False)
    max_analyses_per_day: int = 50
    dedup_cooldown_seconds: int = 300
    analysis_timeout_seconds: int = 600

    @model_validator(mode="after")
    def _auto_enable(self) -> WebhookServerConfig:
        """Auto-enable when hmac_secret is provided."""
        if self.hmac_secret and not self.enabled:
            self.enabled = True
        return self


class JiraConfig(BaseModel):
    """Jira Cloud REST API v3 integration configuration.

    Used by :class:`~vaig.integrations.jira.JiraClient` for exporting
    health-report findings as Jira issues.  Auto-enables when
    ``base_url`` is set (same pattern as :class:`PagerDutyConfig`).
    """

    enabled: bool = False
    base_url: str = ""
    email: str = Field(default="", repr=False)
    api_token: SecretStr = Field(default=SecretStr(""), repr=False)
    project_key: str = ""
    issue_type: str = "Bug"
    severity_field_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "CRITICAL": "Highest",
            "HIGH": "High",
            "MEDIUM": "Medium",
            "LOW": "Low",
            "INFO": "Lowest",
        }
    )

    @model_validator(mode="after")
    def _auto_enable(self) -> JiraConfig:
        """Auto-enable when base_url is provided unless explicitly disabled."""
        has_url = bool(self.base_url)
        if not self.enabled and has_url and "enabled" not in self.model_fields_set:
            self.enabled = True
        elif self.enabled and not has_url:
            logger.warning(
                "Jira integration is enabled but base_url is empty; "
                "disabling Jira integration."
            )
            self.enabled = False
        return self


class PagerDutyConfig(BaseModel):
    """PagerDuty Events API v2 + REST API v2 integration configuration.

    The ``routing_key`` is required for triggering incidents via the
    Events API v2.  The ``api_token`` is optional and enables enrichment
    features like adding incident notes and searching incidents.
    """

    enabled: bool = False
    routing_key: str = Field(default="", repr=False)
    api_token: str = Field(default="", repr=False)
    service_id: str = ""
    base_url: str = "https://api.pagerduty.com"
    auto_create_incident: bool = True
    severity_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "critical": "critical",
            "high": "error",
            "medium": "warning",
            "info": "info",
        }
    )

    # Alert correlation tool fields (read-only fetch from PD REST API)
    alert_service_ids: list[str] = Field(default_factory=list)
    alert_fetch_limit: int = 25

    @model_validator(mode="after")
    def _auto_enable(self) -> PagerDutyConfig:
        """Auto-enable when routing_key is provided unless explicitly disabled."""
        if self.enabled and not self.routing_key:
            logger.warning(
                "PagerDuty integration is enabled but no routing_key is configured; "
                "disabling PagerDuty integration."
            )
            self.enabled = False
            return self

        if self.routing_key and "enabled" not in self.model_fields_set:
            self.enabled = True
        return self


class GoogleChatConfig(BaseModel):
    """Google Chat incoming webhook integration configuration.

    When ``webhook_url`` is set, the integration auto-enables and will
    send Card v2 notifications for severities listed in ``notify_on``.
    """

    enabled: bool = False
    webhook_url: str = Field(default="", repr=False)
    notify_on: list[str] = Field(default_factory=lambda: ["critical", "high"])

    @model_validator(mode="after")
    def _auto_enable(self) -> GoogleChatConfig:
        """Normalize enabled state based on webhook_url presence."""
        if self.webhook_url and not self.enabled:
            self.enabled = True
        elif self.enabled and not self.webhook_url:
            logger.warning(
                "Google Chat integration is enabled but webhook_url is empty; "
                "disabling integration."
            )
            self.enabled = False
        return self


class SlackConfig(BaseModel):
    """Slack incoming webhook integration configuration.

    When ``webhook_url`` is set, the integration auto-enables and will
    send Block Kit notifications for severities listed in ``notify_on``.
    The ``bot_token`` is used separately by alert-correlation tools to
    read channel history via the conversations.history API.
    """

    enabled: bool = False
    webhook_url: str = Field(default="", repr=False)
    notify_on: list[str] = Field(default_factory=lambda: ["critical", "high"])

    # Alert correlation tool fields (read-only Slack conversations.history)
    bot_token: SecretStr = Field(default=SecretStr(""), repr=False)

    @model_validator(mode="after")
    def _auto_enable(self) -> SlackConfig:
        """Normalize enabled state based on webhook_url presence."""
        if self.webhook_url and not self.enabled:
            self.enabled = True
        elif self.enabled and not self.webhook_url:
            logger.warning(
                "Slack integration is enabled but webhook_url is empty; "
                "disabling integration."
            )
            self.enabled = False
        return self


class OpsGenieConfig(BaseModel):
    """OpsGenie v2 API integration configuration for alert correlation.

    When ``api_key`` is set, the integration auto-enables and the
    ``list_opsgenie_alerts`` tool becomes available in ``vaig live``.
    Use ``base_url`` to switch between US and EU regions.
    """

    enabled: bool = False
    api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    base_url: str = "https://api.opsgenie.com"
    team_ids: list[str] = Field(default_factory=list)
    alert_fetch_limit: int = 25

    @model_validator(mode="after")
    def _auto_enable(self) -> OpsGenieConfig:
        """Auto-enable when api_key is provided; disable when missing."""
        has_key = bool(self.api_key.get_secret_value())
        if not self.enabled and has_key:
            self.enabled = True
        elif self.enabled and not has_key:
            logger.warning(
                "OpsGenie integration is enabled but api_key is empty; "
                "disabling integration."
            )
            self.enabled = False
        return self


class EmailConfig(BaseModel):
    """SMTP email notification configuration.

    Auto-enables when ``smtp_host``, ``from_address``, and ``recipients``
    are all provided.  Uses stdlib ``smtplib`` — no new dependencies.
    """

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = Field(default="", repr=False)
    password: str = Field(default="", repr=False)
    from_address: str = ""
    recipients: list[str] = Field(default_factory=list)
    use_tls: bool = True
    timeout: int = 30
    notify_on: list[str] = Field(default_factory=lambda: ["critical", "high"])

    @model_validator(mode="after")
    def _auto_enable(self) -> EmailConfig:
        """Auto-enable when credentials are present; disable when missing."""
        has_credentials = bool(self.smtp_host and self.from_address and self.recipients)
        if not self.enabled and has_credentials:
            self.enabled = True
        elif self.enabled and not has_credentials:
            logger.warning(
                "Email integration is enabled but smtp_host, from_address, or "
                "recipients is missing; disabling integration."
            )
            self.enabled = False
        return self


class RateLimitConfig(BaseModel):
    """Rate limiting configuration — quota policy loaded from GCS."""

    enabled: bool = False
    policy_gcs_bucket: str = ""
    policy_gcs_path: str = "vaig/quota-policy.yaml"
    cache_ttl_seconds: int = 300


class EffectivenessConfig(BaseModel):
    """Tool effectiveness learning configuration.

    When ``enabled`` is True, the system scores tools based on historical
    call data and automatically skips, deprioritizes, or boosts tools
    per configurable failure-rate and duration thresholds.

    Disabled by default — enable explicitly with
    ``effectiveness.enabled = true`` in config or
    ``VAIG_EFFECTIVENESS__ENABLED=true``.
    """

    enabled: bool = False
    skip_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    """Failure rate above which a tool is assigned SKIP tier (not executed)."""
    deprioritize_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Failure rate above which a tool is assigned DEPRIORITIZE tier (warning logged)."""
    boost_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    """Failure rate below which a reliable tool is assigned BOOST tier."""
    slow_tool_threshold_s: float = Field(default=10.0, ge=0.0)
    """Average duration (seconds) above which a tool is DEPRIORITIZE tier."""
    min_calls_for_scoring: int = Field(default=3, ge=1)
    """Minimum historical calls required before a tool can be scored."""
    lookback_days: int = Field(default=7, ge=1)
    """Number of days of history to consider when computing scores."""
    cache_ttl_seconds: int = Field(default=300, ge=0)
    """Seconds to cache computed scores before recomputing."""


class ExportConfig(BaseModel):
    """RAG data pipeline export configuration.

    Controls export of vaig telemetry, tool calls, and health reports
    to GCP (BigQuery and GCS).  Disabled by default — enable explicitly
    with ``export.enabled = true`` in config or ``VAIG_EXPORT__ENABLED=true``.

    Requires the ``[rag]`` optional dependency group:
    ``pip install 'vertex-ai-toolkit[rag]'``
    """

    enabled: bool = False
    gcp_project_id: str = Field(
        default="",
        validation_alias=AliasChoices("gcp_project_id", "bigquery_project"),
    )
    bigquery_dataset: str = "vaig_analytics"
    gcs_bucket: str = ""
    gcs_prefix: str = "rag_data/"
    auto_export_reports: bool = False
    auto_export_telemetry: bool = False
    vertex_rag_corpus_id: str = Field(
        default="",
        validation_alias=AliasChoices("vertex_rag_corpus_id", "rag_corpus_name"),
    )
    rag_enabled: bool = False
    rag_chunk_size: int = 1024
    rag_chunk_overlap: int = 200
    org_id: str = ""

    @model_validator(mode="after")
    def _normalize_export_fields(self) -> ExportConfig:
        """Normalize legacy aliases and ensure stable path formatting."""
        self.gcp_project_id = self.gcp_project_id.strip()
        self.gcs_bucket = self.gcs_bucket.strip()
        self.vertex_rag_corpus_id = self.vertex_rag_corpus_id.strip()

        prefix = self.gcs_prefix.strip()
        if prefix:
            self.gcs_prefix = prefix.rstrip("/") + "/"
        else:
            self.gcs_prefix = ""

        if self.rag_chunk_size <= 0:
            raise ValueError("rag_chunk_size must be a positive integer")
        if self.rag_chunk_overlap < 0:
            raise ValueError("rag_chunk_overlap must be non-negative")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                f"rag_chunk_overlap ({self.rag_chunk_overlap}) must be less "
                f"than rag_chunk_size ({self.rag_chunk_size})"
            )

        return self

    @property
    def bigquery_project(self) -> str:
        """Backward-compatible alias for the configured GCP project ID."""
        return self.gcp_project_id

    @bigquery_project.setter
    def bigquery_project(self, value: str) -> None:
        """Backward-compatible setter for legacy callers."""
        self.gcp_project_id = value

    @property
    def effective_gcs_prefix(self) -> str:
        """GCS prefix with optional org_id segment for per-org isolation."""
        if self.org_id:
            return f"{self.gcs_prefix}{self.org_id}/"
        return self.gcs_prefix

    @property
    def rag_corpus_name(self) -> str:
        """Backward-compatible alias for the Vertex AI RAG corpus ID."""
        return self.vertex_rag_corpus_id

    @rag_corpus_name.setter
    def rag_corpus_name(self, value: str) -> None:
        """Backward-compatible setter for legacy callers."""
        self.vertex_rag_corpus_id = value


class TrainingConfig(BaseModel):
    """Fine-tuning pipeline configuration.

    Controls extraction of high-quality rated examples from BigQuery,
    transformation into Gemini supervised-tuning JSONL, and submission of
    fine-tuning jobs via Vertex AI.

    Disabled by default — enable explicitly with ``training.enabled = true``
    in config or ``VAIG_TRAINING__ENABLED=true``.

    Requires the ``[rag]`` optional dependency group:
    ``pip install 'vertex-ai-toolkit[rag]'``
    """

    enabled: bool = False
    base_model: str = "gemini-2.5-flash"
    min_examples: int = 50
    max_examples: int = 10000
    min_rating: int = 4
    output_dir: Path = Path("training_data")
    epochs: int = 3
    learning_rate_multiplier: float = 1.0
    gcs_staging_prefix: str = "training_data/"

    @model_validator(mode="after")
    def _validate_training_bounds(self) -> TrainingConfig:
        """Validate numeric constraints for training configuration."""
        if self.min_examples < 10:
            raise ValueError("min_examples must be >= 10")
        if self.max_examples > 100000:
            raise ValueError("max_examples must be <= 100000")
        if self.max_examples < self.min_examples:
            raise ValueError("max_examples must be >= min_examples")
        if self.min_rating < 1 or self.min_rating > 5:
            raise ValueError("min_rating must be between 1 and 5")
        if self.epochs < 1 or self.epochs > 10:
            raise ValueError("epochs must be between 1 and 10")
        if self.learning_rate_multiplier <= 0:
            raise ValueError("learning_rate_multiplier must be > 0")

        prefix = self.gcs_staging_prefix.strip()
        if prefix:
            self.gcs_staging_prefix = prefix.rstrip("/") + "/"
        else:
            self.gcs_staging_prefix = ""

        return self


def _strip_empty_strings(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove keys whose value is an empty string.

    This prevents YAML defaults like ``project_id: ""`` from shadowing
    environment variables in pydantic-settings (which treats explicit
    init args as higher priority than env vars).

    Also recurses into lists of dicts (e.g. ``models.available``) so that
    empty strings inside list items are cleaned as well.
    """
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = _strip_empty_strings(value)
            if nested:  # only add if dict still has content
                cleaned[key] = nested
        elif isinstance(value, list):
            cleaned[key] = _strip_empty_strings_in_list(value)
        elif value != "":
            cleaned[key] = value
    return cleaned


def _strip_empty_strings_in_list(items: list[Any]) -> list[Any]:
    """Clean empty strings inside a list, recursing into nested dicts and lists."""
    cleaned: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            cleaned.append(_strip_empty_strings(item))
        elif isinstance(item, list):
            cleaned.append(_strip_empty_strings_in_list(item))
        elif item != "":
            cleaned.append(item)
    return cleaned


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, with override values winning.

    For dict-valued keys that appear in both, the dicts are merged recursively
    rather than replaced wholesale.  All other types (including lists) are
    replaced by the override value.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class RemediationConfig(BaseModel):
    """Remediation engine configuration — safety-tiered command execution.

    Controls the runbook execution engine that classifies recommended
    commands into SAFE/REVIEW/BLOCKED tiers and enforces approval
    workflows before execution.

    Feature-flagged: disabled by default (``enabled=False``).
    Enable explicitly with ``remediation.enabled = true`` in config
    or ``VAIG_REMEDIATION__ENABLED=true``.
    """

    enabled: bool = False
    """Master feature flag — entire engine is disabled until explicitly enabled."""
    auto_approve_safe: bool = False
    """When True, SAFE-tier commands execute without user confirmation."""
    blocked_commands: list[str] = Field(default_factory=list)
    """Regex patterns for commands that should ALWAYS be BLOCKED.

    Checked against the full command string before tier assignment.
    Reuses the same regex pattern approach as
    :attr:`CodingConfig.denied_commands`.
    """
    timeout: int = 30
    """Execution timeout in seconds per command."""
    dry_run: bool = False
    """When True, all commands show execution plan without side effects."""
    tier_overrides: dict[str, str] = Field(default_factory=dict)
    """Map of command regex pattern → tier name (safe/review/blocked).

    Allows SREs to promote or demote specific commands without code
    changes.  Patterns are matched against the full command string.
    Example: ``{"kubectl rollout restart": "safe"}`` promotes rollout
    restarts from REVIEW to SAFE.
    """


class ScheduleTarget(BaseModel):
    """A single GKE cluster/namespace to scan on a schedule."""

    cluster_name: str
    namespace: str = ""
    all_namespaces: bool = False
    skip_healthy: bool = True


class ScheduleConfig(BaseModel):
    """Scheduled health-scan configuration.

    Auto-enables when ``targets`` is non-empty, following the same
    pattern as :class:`WebhookServerConfig` and :class:`GoogleChatConfig`.
    """

    enabled: bool = False
    default_interval_minutes: int = 30
    cron_expression: str | None = None
    targets: list[ScheduleTarget] = Field(default_factory=list)
    alert_severity_threshold: str = "HIGH"
    daily_max_analyses: int = 48
    per_schedule_max_analyses: int | None = None
    max_concurrent_scans: int = 1
    store_results: bool = True
    misfire_grace_time: int = 900
    db_path: str = "~/.vaig/scheduler.db"

    @model_validator(mode="after")
    def _auto_enable(self) -> ScheduleConfig:
        """Auto-enable when targets are configured."""
        if self.targets and not self.enabled:
            self.enabled = True
        return self


class FleetCluster(BaseModel):
    """A single GKE cluster in a fleet scan configuration."""

    name: str = Field(description="Display name (required)")
    cluster_name: str = Field(description="GKE cluster name (required)")
    project_id: str = Field(default="", description="GCP project; falls back to gcp.project_id")
    location: str = Field(default="", description="GKE location; falls back to gcp.location")
    namespace: str = ""
    all_namespaces: bool = False
    skip_healthy: bool = True
    kubeconfig_path: str = ""
    context: str = ""
    impersonate_sa: str = ""


class FleetConfig(BaseModel):
    """Multi-cluster fleet scanning configuration.

    Auto-enables when ``clusters`` is non-empty, following the same
    pattern as :class:`ScheduleConfig`.
    """

    enabled: bool = False
    clusters: list[FleetCluster] = Field(default_factory=list)
    parallel: bool = False
    max_workers: int = Field(default=4, ge=1)
    daily_budget_usd: float = 0.0

    @model_validator(mode="after")
    def _auto_enable(self) -> FleetConfig:
        """Auto-enable when clusters are configured."""
        if self.clusters and not self.enabled:
            self.enabled = True
        return self


class Settings(BaseSettings):
    """Root application settings — merges env vars, YAML, and CLI overrides."""

    model_config = SettingsConfigDict(
        env_prefix="VAIG_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    language: str = Field(
        default="en",
        description=(
            "Preferred output language (BCP-47 code, e.g. 'es', 'pt', 'ja'). "
            "When set to a non-'en' value, ALL agent output is produced in this "
            "language regardless of the query language.  When 'en' (the default), "
            "language is auto-detected from the user query at runtime."
        ),
    )

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, v: Any) -> str:
        """Strip whitespace and lowercase the language code."""
        if isinstance(v, str):
            return v.strip().lower()
        return str(v)

    gcp: GCPConfig = Field(default_factory=GCPConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    coding: CodingConfig = Field(default_factory=CodingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    gke: GKEConfig = Field(default_factory=GKEConfig)
    helm: HelmConfig = Field(default_factory=HelmConfig)
    argocd: ArgoCDConfig = Field(default_factory=ArgoCDConfig)
    datadog: DatadogAPIConfig = Field(default_factory=DatadogAPIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    context_window: ContextWindowConfig = Field(default_factory=ContextWindowConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    effectiveness: EffectivenessConfig = Field(default_factory=EffectivenessConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    webhook_server: WebhookServerConfig = Field(default_factory=WebhookServerConfig)
    jira: JiraConfig = Field(default_factory=JiraConfig)
    pagerduty: PagerDutyConfig = Field(default_factory=PagerDutyConfig)
    google_chat: GoogleChatConfig = Field(default_factory=GoogleChatConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    opsgenie: OpsGenieConfig = Field(default_factory=OpsGenieConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    fleet: FleetConfig = Field(default_factory=FleetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    remediation: RemediationConfig = Field(default_factory=RemediationConfig)

    @model_validator(mode="after")
    def _bridge_platform_org_id(self) -> Settings:
        """Copy platform.org_id → export.org_id when platform is enabled."""
        if self.platform.enabled and self.platform.org_id and not self.export.org_id:
            self.export.org_id = self.platform.org_id
        return self

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        """Load settings from YAML config, env vars, and defaults.

        Priority (highest wins): env vars > explicit config_path > vaig.yaml
        (cwd) > ~/.vaig/config.yaml > config/default.yaml (project defaults).

        All applicable config files are loaded and deep-merged in order from
        lowest to highest priority, so that user or project-specific files can
        override individual keys in the project defaults without repeating
        every setting.

        Empty strings in YAML are stripped so they don't shadow env vars
        (pydantic-settings treats explicit init args as higher priority
        than environment variables).
        """
        # Ordered from lowest priority to highest priority.
        # Each file found is deep-merged over the accumulated result so that
        # higher-priority files override lower-priority ones.
        paths_by_priority: list[Path | None] = [
            Path.cwd() / "config" / "default.yaml",   # project defaults (lowest)
            Path.home() / ".vaig" / "config.yaml",     # user home config
            Path.cwd() / "vaig.yaml",                  # project-specific override
            Path(config_path).expanduser() if config_path is not None else None,  # explicit (highest)
        ]

        yaml_data: dict[str, Any] = {}
        for p in paths_by_priority:
            if p is not None:
                resolved = Path(p).expanduser()
                if resolved.exists():
                    file_data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
                    yaml_data = _deep_merge(yaml_data, file_data)

        # Strip empty strings so env vars can take precedence
        yaml_data = _strip_empty_strings(yaml_data)

        # Merge env vars over yaml (pydantic-settings handles env automatically)
        return cls(**yaml_data)

    @classmethod
    def from_overrides(cls, base_path: str | Path | None = None, **overrides: Any) -> Settings:
        """Load base config and overlay keyword overrides for web requests.

        Uses :meth:`load` for the layered YAML loading, then maps
        well-known shorthand keys to their nested config paths and
        returns a copy with overrides applied.

        - ``project`` → ``gcp.project_id``
        - ``model`` → ``models.default``
        - ``temperature`` → ``generation.temperature``
        - ``max_tokens`` → ``generation.max_output_tokens``
        - ``region`` → ``gcp.location``

        Any unknown ``**overrides`` are silently ignored so that callers
        can forward form data without pre-filtering.

        Args:
            base_path: Optional explicit YAML config path (same as ``load()``).
            **overrides: Shorthand key-value pairs from a web form.

        Returns:
            A fresh :class:`Settings` instance with overrides applied.
        """
        # 1. Reuse the canonical layered loader
        base = cls.load(base_path)

        # 2. Map shorthand overrides → nested model updates
        _OVERRIDE_MAP: dict[str, tuple[str, str]] = {
            "project": ("gcp", "project_id"),
            "model": ("models", "default"),
            "temperature": ("generation", "temperature"),
            "max_tokens": ("generation", "max_output_tokens"),
            "region": ("gcp", "location"),
        }

        # Build a dict of nested model updates: {"gcp": {"project_id": "..."}, ...}
        nested_updates: dict[str, dict[str, Any]] = {}
        for key, value in overrides.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            mapping = _OVERRIDE_MAP.get(key)
            if mapping is not None:
                section, field_name = mapping
                if section not in nested_updates:
                    nested_updates[section] = {}
                nested_updates[section][field_name] = value

        if not nested_updates:
            return base

        # 3. Apply overrides via model_copy (Pydantic v2 pattern)
        top_level_updates: dict[str, Any] = {}
        for section, fields in nested_updates.items():
            current_sub_model = getattr(base, section)
            top_level_updates[section] = current_sub_model.model_copy(update=fields)

        return base.model_copy(update=top_level_updates)

    @property
    def db_path_resolved(self) -> Path:
        """Resolve the session DB path (expand ~)."""
        return Path(self.session.db_path).expanduser()

    @property
    def current_model(self) -> str:
        """Get the current default model ID."""
        return self.models.default

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model."""
        return next((m for m in self.models.available if m.id == model_id), None)


# ── Singleton ──────────────────────────────────────────────
_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None) -> Settings:
    """Get or create the global settings instance."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings.load(config_path)
    return _settings


def reset_settings() -> None:
    """Reset settings (for testing)."""
    global _settings  # noqa: PLW0603
    _settings = None

"""Core configuration — Pydantic settings with layered config loading."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
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


class GenerationConfig(BaseModel):
    """Model generation parameters."""

    temperature: float = 0.7
    max_output_tokens: int = 16384
    top_p: float = 0.95
    top_k: int = 40


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


class SkillsConfig(BaseModel):
    """Skills configuration."""

    enabled: list[str] = Field(default_factory=lambda: ["rca", "anomaly", "migration", "log-analysis", "error-triage", "config-audit", "slo-review", "postmortem", "code-review", "iac-review", "cost-analysis", "capacity-planning", "test-generation", "compliance-check", "api-design", "runbook-generator", "dependency-audit", "db-review", "pipeline-review", "perf-analysis", "threat-model", "change-risk", "alert-tuning", "resilience-review", "incident-comms", "toil-analysis", "network-review", "adr-generator", "service-health"])
    custom_dir: str | None = None
    auto_routing: bool = True
    auto_routing_threshold: float = 1.5


class AgentsConfig(BaseModel):
    """Multi-agent configuration."""

    max_concurrent: int = 3
    orchestrator_model: str = "gemini-2.5-pro"
    specialist_model: str = "gemini-2.5-flash"
    max_iterations_retry: int = 10


class CodingConfig(BaseModel):
    """Coding agent configuration."""

    workspace_root: str = "."
    max_tool_iterations: int = 25
    confirm_actions: bool = True
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    denied_commands: list[str] = Field(
        default_factory=lambda: [
            # Destructive disk / filesystem operations
            r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/\s*$",  # rm -rf /
            r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/\w",     # rm -rf /anything
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
    server: str = ""
    token: str = ""
    context: str = ""
    namespace: str = "argocd"
    verify_ssl: bool = True


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
    # ArgoCD integration — disabled by default, requires explicit opt-in.
    argocd_enabled: bool = False
    argocd_server: str = ""
    argocd_token: str = ""
    argocd_context: str = ""
    argocd_namespace: str = "argocd"
    argocd_verify_ssl: bool = True


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


class BudgetConfig(BaseModel):
    """Token budget tracking configuration.

    Controls session-level cost tracking and enforcement.
    """

    enabled: bool = False
    max_cost_usd: float = 5.0
    warn_threshold: float = 0.8  # 80% of max_cost_usd
    action: Literal["warn", "stop"] = "warn"


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


class Settings(BaseSettings):
    """Root application settings — merges env vars, YAML, and CLI overrides."""

    model_config = SettingsConfigDict(
        env_prefix="VAIG_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

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
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

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

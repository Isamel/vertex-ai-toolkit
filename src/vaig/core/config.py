"""Core configuration — Pydantic settings with layered config loading."""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

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


class GCPConfig(BaseModel):
    """GCP project configuration."""

    project_id: str = ""
    location: str = "us-central1"
    fallback_location: str = "us-central1"

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
    auto_save: bool = True
    max_history_messages: int = 100


class SkillsConfig(BaseModel):
    """Skills configuration."""

    enabled: list[str] = Field(default_factory=lambda: ["rca", "anomaly", "migration", "log-analysis", "error-triage", "config-audit", "slo-review", "postmortem", "code-review", "iac-review", "cost-analysis", "capacity-planning", "test-generation", "compliance-check", "api-design", "runbook-generator", "dependency-audit", "db-review", "pipeline-review", "perf-analysis", "threat-model", "change-risk", "alert-tuning", "resilience-review", "incident-comms", "toil-analysis", "network-review", "adr-generator", "service-health"])
    custom_dir: str | None = None


class AgentsConfig(BaseModel):
    """Multi-agent configuration."""

    max_concurrent: int = 3
    orchestrator_model: str = "gemini-2.5-pro"
    specialist_model: str = "gemini-2.5-flash"


class CodingConfig(BaseModel):
    """Coding agent configuration."""

    workspace_root: str = "."
    max_tool_iterations: int = 25
    confirm_actions: bool = True
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "WARNING"
    show_path: bool = False


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


class GKEConfig(BaseModel):
    """GKE live-cluster connection and query configuration."""

    cluster_name: str = ""
    project_id: str = ""
    default_namespace: str = "default"
    kubeconfig_path: str = ""
    context: str = ""
    log_limit: int = 100
    metrics_interval_minutes: int = 60
    # Proxy URL for private GKE clusters. The kubernetes Python client
    # ignores the ``proxy-url`` field in kubeconfig; this setting lets
    # users provide an explicit override via config/CLI.
    proxy_url: str = ""


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

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        """Load settings from YAML config, env vars, and defaults.

        Priority: env vars > yaml config > defaults.

        Empty strings in YAML are stripped so they don't shadow env vars
        (pydantic-settings treats explicit init args as higher priority
        than environment variables).
        """
        yaml_data: dict[str, Any] = {}

        # Try loading YAML config
        paths_to_try = [
            config_path,
            Path.cwd() / "config" / "default.yaml",
            Path.cwd() / "vaig.yaml",
            Path.home() / ".vaig" / "config.yaml",
        ]

        for p in paths_to_try:
            if p is not None:
                resolved = Path(p).expanduser()
                if resolved.exists():
                    yaml_data = yaml.safe_load(resolved.read_text()) or {}
                    break

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

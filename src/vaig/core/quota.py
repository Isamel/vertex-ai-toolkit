"""Rate-limit quota enforcement via GCS-hosted YAML policy.

Tracks daily per-user usage (requests, tokens, executions) in-memory and
enforces limits loaded from a YAML policy file in GCS.  Counters reset at
midnight UTC automatically.

This module uses **lazy imports** for ``google-cloud-storage`` so that the
``[audit]`` extras are only required when rate limiting is actually enabled.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from vaig.core.exceptions import QuotaExceededError

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

__all__ = [
    "QuotaChecker",
]


# ── Dependency guard ─────────────────────────────────────────


def _require_audit_deps() -> None:
    """Raise ImportError with install instructions if [audit] deps are missing."""
    try:
        import google.cloud.storage  # noqa: F401
    except ImportError:
        raise ImportError(
            "Rate-limit features require the [audit] extras. "
            "Install with: pip install 'vertex-ai-toolkit[audit]'"
        ) from None


# ── Data structures ──────────────────────────────────────────


@dataclass
class DailyUsage:
    """In-memory daily usage counters for a single user key."""

    date: date = field(default_factory=lambda: datetime.now(UTC).date())
    requests: int = 0
    tokens: int = 0
    executions: int = 0


@dataclass
class QuotaPolicy:
    """Parsed quota policy from the GCS YAML file."""

    default_requests_per_day: int = 500
    default_tokens_per_day: int = 2_000_000
    default_executions_per_day: int = 50
    user_overrides: dict[str, dict[str, int]] = field(default_factory=dict)


# ── QuotaChecker ─────────────────────────────────────────────


class QuotaChecker:
    """Enforces per-user rate limits using a GCS-hosted YAML policy.

    Designed for single-process CLI usage — counters are in-memory and
    reset at midnight UTC.  The policy is cached with a configurable TTL.

    Args:
        settings: Application settings (provides ``rate_limit`` config).
        credentials: GCP credentials for GCS access.
        storage_client: Optional pre-built GCS client (for testing).
    """

    def __init__(
        self,
        settings: Settings,
        credentials: Credentials | None = None,
        *,
        storage_client: Any = None,
    ) -> None:
        _require_audit_deps()
        self._settings = settings
        self._credentials = credentials
        self._storage_client = storage_client

        # Policy cache
        self._policy: QuotaPolicy | None = None
        self._policy_loaded_at: float = 0.0

        # Per-user daily counters: composite_key → DailyUsage
        self._usage: dict[str, DailyUsage] = {}

    # ── Public API ───────────────────────────────────────────

    def check_and_consume(
        self,
        user_key: str,
        estimated_tokens: int,
        *,
        is_execution: bool = False,
    ) -> None:
        """Check quota and consume usage if within limits.

        Must be called BEFORE the actual Gemini API call.

        Args:
            user_key: Composite user key ``"{os_user}:{gcp_email}"``.
            estimated_tokens: Estimated token count for the upcoming call.
            is_execution: ``True`` when the call originates from ``diagnose``/``live``.

        Raises:
            QuotaExceededError: If any dimension would be exceeded.
        """
        policy = self._get_policy()
        usage = self._get_or_create_usage(user_key)
        limits = self._resolve_limits(user_key, policy)

        # Check all three dimensions BEFORE consuming
        new_requests = usage.requests + 1
        new_tokens = usage.tokens + estimated_tokens
        new_executions = usage.executions + (1 if is_execution else 0)

        if new_requests > limits["requests_per_day"]:
            raise QuotaExceededError(
                dimension="requests_per_day",
                used=new_requests,
                limit=limits["requests_per_day"],
                user_key=user_key,
            )

        if new_tokens > limits["tokens_per_day"]:
            raise QuotaExceededError(
                dimension="tokens_per_day",
                used=new_tokens,
                limit=limits["tokens_per_day"],
                user_key=user_key,
            )

        if is_execution and new_executions > limits["executions_per_day"]:
            raise QuotaExceededError(
                dimension="executions_per_day",
                used=new_executions,
                limit=limits["executions_per_day"],
                user_key=user_key,
            )

        # All checks passed — consume
        usage.requests = new_requests
        usage.tokens = new_tokens
        if is_execution:
            usage.executions = new_executions

    # ── Policy loading ───────────────────────────────────────

    def _get_policy(self) -> QuotaPolicy:
        """Return the cached policy, reloading from GCS if TTL expired."""
        now = time.monotonic()
        ttl = self._settings.rate_limit.cache_ttl_seconds

        if self._policy is not None and (now - self._policy_loaded_at) < ttl:
            return self._policy

        # Try to reload from GCS
        try:
            raw = self._load_policy_from_gcs()
            self._policy = self._parse_policy(raw)
            self._policy_loaded_at = now
            return self._policy
        except Exception as exc:
            if self._policy is not None:
                # Stale cache — use it with a warning
                logger.warning(
                    "Failed to reload quota policy from GCS, using stale cache: %s",
                    exc,
                )
                return self._policy
            # No cache at all — fail closed
            msg = (
                "Could not load quota policy from GCS and no cached policy exists. "
                "Check GCS connectivity and bucket configuration. "
                f"Bucket: '{self._settings.rate_limit.policy_gcs_bucket}', "
                f"Path: '{self._settings.rate_limit.policy_gcs_path}'"
            )
            raise RuntimeError(msg) from exc

    def _load_policy_from_gcs(self) -> dict[str, Any]:
        """Download and parse the YAML policy file from GCS."""
        import google.cloud.storage as gcs  # lazy import

        client = self._storage_client
        if client is None:
            client = gcs.Client(credentials=self._credentials)
            self._storage_client = client

        bucket_name = self._settings.rate_limit.policy_gcs_bucket
        blob_path = self._settings.rate_limit.policy_gcs_path

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        content = blob.download_as_text()

        import yaml as _yaml  # already a project dependency

        data = _yaml.safe_load(content)
        if not isinstance(data, dict):
            msg = f"Quota policy file is not a valid YAML mapping: {type(data)}"
            raise ValueError(msg)
        return data

    @staticmethod
    def _parse_policy(raw: dict[str, Any]) -> QuotaPolicy:
        """Validate and parse a raw YAML dict into a QuotaPolicy."""
        # Schema version check
        version = raw.get("schema_version")
        if version != 1:
            msg = (
                f"Unsupported quota policy schema_version: {version!r}. "
                "Only schema_version 1 is supported."
            )
            raise ValueError(msg)

        # Default policy
        defaults = raw.get("default_policy", {})
        if not isinstance(defaults, dict):
            msg = "default_policy must be a mapping"
            raise ValueError(msg)

        policy = QuotaPolicy(
            default_requests_per_day=int(defaults.get("requests_per_day", 500)),
            default_tokens_per_day=int(defaults.get("tokens_per_day", 2_000_000)),
            default_executions_per_day=int(defaults.get("executions_per_day", 50)),
        )

        # User overrides
        overrides = raw.get("user_overrides", {})
        if isinstance(overrides, dict):
            for key, vals in overrides.items():
                if isinstance(vals, dict):
                    policy.user_overrides[str(key)] = {
                        k: int(v) for k, v in vals.items() if isinstance(v, (int, float))
                    }

        return policy

    # ── Limit resolution ─────────────────────────────────────

    @staticmethod
    def _resolve_limits(user_key: str, policy: QuotaPolicy) -> dict[str, int]:
        """Resolve the applicable limits for a user key.

        Lookup order:
        1. Exact composite key match in user_overrides
        2. GCP email only (part after ':') match in user_overrides
        3. Fall back to default_policy
        """
        defaults = {
            "requests_per_day": policy.default_requests_per_day,
            "tokens_per_day": policy.default_tokens_per_day,
            "executions_per_day": policy.default_executions_per_day,
        }

        # 1. Exact composite key match
        if user_key in policy.user_overrides:
            merged = {**defaults, **policy.user_overrides[user_key]}
            return merged

        # 2. GCP email only (after ':')
        parts = user_key.split(":", 1)
        if len(parts) == 2:
            email = parts[1]
            if email in policy.user_overrides:
                merged = {**defaults, **policy.user_overrides[email]}
                return merged

        # 3. Default policy
        return defaults

    # ── Usage tracking ───────────────────────────────────────

    def _get_or_create_usage(self, user_key: str) -> DailyUsage:
        """Return the current day's usage for a user, resetting if date changed."""
        today = datetime.now(UTC).date()
        usage = self._usage.get(user_key)

        if usage is None or usage.date != today:
            # New user or midnight reset
            usage = DailyUsage(date=today)
            self._usage[user_key] = usage

        return usage

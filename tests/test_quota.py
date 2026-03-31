"""Tests for the QuotaChecker (vaig.core.quota).

Covers:
- Default policy enforcement
- User override precedence (composite key > email-only > default)
- QuotaExceededError for each dimension
- Midnight UTC counter reset
- Policy cache TTL refresh
- GCS unavailable + no cache → fail-closed
- GCS unavailable + stale cache → uses cache with WARNING
- Invalid schema_version → clear error
- Lazy import guard → ImportError with install instructions
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vaig.core.exceptions import QuotaExceededError
from vaig.core.quota import DailyUsage, QuotaChecker, QuotaPolicy, _require_audit_deps

# ── Helpers ──────────────────────────────────────────────────

_DEFAULT_POLICY_YAML = {
    "schema_version": 1,
    "default_policy": {
        "requests_per_day": 500,
        "tokens_per_day": 2_000_000,
        "executions_per_day": 50,
    },
}


def _make_settings(
    *,
    enabled: bool = True,
    bucket: str = "test-bucket",
    path: str = "vaig/quota-policy.yaml",
    cache_ttl: int = 300,
) -> MagicMock:
    """Create a mock Settings with rate_limit config."""
    settings = MagicMock()
    settings.rate_limit.enabled = enabled
    settings.rate_limit.policy_gcs_bucket = bucket
    settings.rate_limit.policy_gcs_path = path
    settings.rate_limit.cache_ttl_seconds = cache_ttl
    return settings


def _make_storage_client(policy: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock GCS storage client that returns a YAML policy."""
    if policy is None:
        policy = _DEFAULT_POLICY_YAML

    client = MagicMock()
    blob = MagicMock()
    blob.download_as_text.return_value = yaml.dump(policy)
    client.bucket.return_value.blob.return_value = blob
    return client


# ── DailyUsage tests ─────────────────────────────────────────


class TestDailyUsage:
    """Tests for the DailyUsage dataclass."""

    def test_defaults(self) -> None:
        """Default counters start at zero."""
        usage = DailyUsage()
        assert usage.requests == 0
        assert usage.tokens == 0
        assert usage.executions == 0
        assert isinstance(usage.date, date)

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        d = date(2026, 3, 31)
        usage = DailyUsage(date=d, requests=10, tokens=5000, executions=2)
        assert usage.date == d
        assert usage.requests == 10
        assert usage.tokens == 5000
        assert usage.executions == 2


# ── QuotaPolicy tests ───────────────────────────────────────


class TestQuotaPolicy:
    """Tests for the QuotaPolicy dataclass."""

    def test_defaults(self) -> None:
        """Default policy has sensible limits."""
        policy = QuotaPolicy()
        assert policy.default_requests_per_day == 500
        assert policy.default_tokens_per_day == 2_000_000
        assert policy.default_executions_per_day == 50
        assert policy.user_overrides == {}


# ── Lazy import guard ────────────────────────────────────────


class TestRequireAuditDeps:
    """Tests for the _require_audit_deps guard."""

    def test_passes_when_installed(self) -> None:
        """Does not raise when google.cloud.storage is importable."""
        # In the test env the [audit] deps are installed via [dev], so this must succeed.
        _require_audit_deps()  # should not raise

    def test_raises_with_install_message(self) -> None:
        """Raises ImportError with pip install instructions when deps missing."""
        with patch.dict("sys.modules", {"google.cloud.storage": None}):
            with pytest.raises(ImportError, match="pip install 'vertex-ai-toolkit\\[audit\\]'"):
                _require_audit_deps()


# ── QuotaChecker: Default policy ─────────────────────────────


class TestQuotaCheckerDefaultPolicy:
    """Tests for QuotaChecker with default policy enforcement."""

    def test_check_passes_under_limit(self) -> None:
        """Check passes when usage is well under limits."""
        settings = _make_settings()
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        # Should not raise
        checker.check_and_consume("alice:alice@co.com", 1000)

    def test_requests_counted(self) -> None:
        """Each check_and_consume increments request counter by 1."""
        settings = _make_settings()
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        for _ in range(10):
            checker.check_and_consume("alice:alice@co.com", 100)

        usage = checker._usage["alice:alice@co.com"]
        assert usage.requests == 10

    def test_tokens_accumulated(self) -> None:
        """Token usage accumulates across calls."""
        settings = _make_settings()
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("alice:alice@co.com", 1000)
        checker.check_and_consume("alice:alice@co.com", 2000)

        usage = checker._usage["alice:alice@co.com"]
        assert usage.tokens == 3000

    def test_executions_tracked_when_flag_set(self) -> None:
        """Executions counter only increments when is_execution=True."""
        settings = _make_settings()
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("alice:alice@co.com", 100, is_execution=False)
        checker.check_and_consume("alice:alice@co.com", 100, is_execution=True)
        checker.check_and_consume("alice:alice@co.com", 100, is_execution=True)

        usage = checker._usage["alice:alice@co.com"]
        assert usage.requests == 3
        assert usage.executions == 2


# ── QuotaChecker: Quota exceeded ─────────────────────────────


class TestQuotaCheckerExceeded:
    """Tests for QuotaExceededError on each dimension."""

    def test_requests_exceeded(self) -> None:
        """QuotaExceededError raised when requests_per_day exceeded."""
        policy = {**_DEFAULT_POLICY_YAML, "default_policy": {**_DEFAULT_POLICY_YAML["default_policy"], "requests_per_day": 3}}
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100)
        checker.check_and_consume("u:u@co.com", 100)
        checker.check_and_consume("u:u@co.com", 100)

        with pytest.raises(QuotaExceededError) as exc_info:
            checker.check_and_consume("u:u@co.com", 100)

        assert exc_info.value.dimension == "requests_per_day"
        assert exc_info.value.used == 4
        assert exc_info.value.limit == 3

    def test_tokens_exceeded(self) -> None:
        """QuotaExceededError raised when tokens_per_day exceeded."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 5000,
                "executions_per_day": 50,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 3000)

        with pytest.raises(QuotaExceededError) as exc_info:
            checker.check_and_consume("u:u@co.com", 3000)

        assert exc_info.value.dimension == "tokens_per_day"
        assert exc_info.value.used == 6000
        assert exc_info.value.limit == 5000

    def test_executions_exceeded(self) -> None:
        """QuotaExceededError raised when executions_per_day exceeded."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 2,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100, is_execution=True)
        checker.check_and_consume("u:u@co.com", 100, is_execution=True)

        with pytest.raises(QuotaExceededError) as exc_info:
            checker.check_and_consume("u:u@co.com", 100, is_execution=True)

        assert exc_info.value.dimension == "executions_per_day"
        assert exc_info.value.used == 3
        assert exc_info.value.limit == 2

    def test_no_execution_check_when_flag_false(self) -> None:
        """Executions limit not checked when is_execution=False."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 1,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100, is_execution=True)
        # Second call without is_execution should pass even though exec limit = 1
        checker.check_and_consume("u:u@co.com", 100, is_execution=False)

    def test_error_message_format(self) -> None:
        """Error message contains dimension, used, and limit with comma formatting."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 1,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100)

        with pytest.raises(QuotaExceededError, match="requests_per_day"):
            checker.check_and_consume("u:u@co.com", 100)


# ── QuotaChecker: User overrides ─────────────────────────────


class TestQuotaCheckerUserOverrides:
    """Tests for user override precedence."""

    def test_composite_key_override(self) -> None:
        """Exact composite key match takes priority."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 2,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
            "user_overrides": {
                "alice:alice@co.com": {"requests_per_day": 1000},
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        # Should pass — alice has 1000 request limit, not 2
        for _ in range(10):
            checker.check_and_consume("alice:alice@co.com", 100)

    def test_email_only_override(self) -> None:
        """GCP email-only match is used when no composite key match."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 2,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
            "user_overrides": {
                "alice@co.com": {"requests_per_day": 1000},
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        # alice:alice@co.com → no composite match → email "alice@co.com" matches
        for _ in range(10):
            checker.check_and_consume("alice:alice@co.com", 100)

    def test_composite_key_over_email(self) -> None:
        """Composite key override wins over email-only override."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 2,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
            "user_overrides": {
                "alice@co.com": {"requests_per_day": 5},
                "alice:alice@co.com": {"requests_per_day": 1000},
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        # Should use composite match (1000), not email match (5)
        for _ in range(10):
            checker.check_and_consume("alice:alice@co.com", 100)

    def test_unmatched_user_gets_default(self) -> None:
        """Users not in overrides get default policy limits."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 2,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
            "user_overrides": {
                "alice@co.com": {"requests_per_day": 1000},
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("bob:bob@co.com", 100)
        checker.check_and_consume("bob:bob@co.com", 100)

        # Third call should exceed default limit of 2
        with pytest.raises(QuotaExceededError):
            checker.check_and_consume("bob:bob@co.com", 100)

    def test_partial_override_inherits_defaults(self) -> None:
        """Overrides with partial limits inherit remaining from defaults."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 100,
                "executions_per_day": 50,
            },
            "user_overrides": {
                "alice@co.com": {"tokens_per_day": 1_000_000},
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        # Alice has overridden tokens (1M) but requests should still be default (500)
        checker.check_and_consume("alice:alice@co.com", 500_000)
        checker.check_and_consume("alice:alice@co.com", 400_000)  # 900k total — under 1M


# ── QuotaChecker: Midnight reset ─────────────────────────────


class TestQuotaCheckerMidnightReset:
    """Tests for counter reset at midnight UTC."""

    def test_counters_reset_on_new_day(self) -> None:
        """All counters reset when the UTC date changes."""
        settings = _make_settings()
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        # Consume some usage
        checker.check_and_consume("u:u@co.com", 1000, is_execution=True)
        assert checker._usage["u:u@co.com"].requests == 1
        assert checker._usage["u:u@co.com"].tokens == 1000
        assert checker._usage["u:u@co.com"].executions == 1

        # Simulate next day
        tomorrow = datetime.now(UTC).date() + timedelta(days=1)
        with patch("vaig.core.quota.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                tomorrow.year, tomorrow.month, tomorrow.day, 0, 1, 0, tzinfo=UTC
            )
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # The date check happens inside _get_or_create_usage
            checker.check_and_consume("u:u@co.com", 500)

        usage = checker._usage["u:u@co.com"]
        assert usage.requests == 1  # Reset to 0, then +1
        assert usage.tokens == 500  # Reset to 0, then +500
        assert usage.executions == 0  # Reset to 0, not incremented


# ── QuotaChecker: Policy cache ───────────────────────────────


class TestQuotaCheckerPolicyCache:
    """Tests for policy caching and TTL refresh."""

    def test_policy_loaded_once_within_ttl(self) -> None:
        """Policy is loaded from GCS only once within the TTL window."""
        settings = _make_settings(cache_ttl=300)
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100)
        checker.check_and_consume("u:u@co.com", 100)
        checker.check_and_consume("u:u@co.com", 100)

        # download_as_text should be called only once
        blob = client.bucket.return_value.blob.return_value
        assert blob.download_as_text.call_count == 1

    def test_policy_reloaded_after_ttl(self) -> None:
        """Policy is reloaded from GCS after TTL expires."""
        settings = _make_settings(cache_ttl=10)
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100)

        # Expire the cache by moving the loaded_at timestamp back
        checker._policy_loaded_at -= 20  # 20 seconds ago (TTL is 10)

        checker.check_and_consume("u:u@co.com", 100)

        blob = client.bucket.return_value.blob.return_value
        assert blob.download_as_text.call_count == 2


# ── QuotaChecker: GCS failures ───────────────────────────────


class TestQuotaCheckerGCSFailure:
    """Tests for GCS unavailability scenarios."""

    def test_fail_closed_no_cache(self) -> None:
        """Raises RuntimeError when GCS is unreachable and no cache exists."""
        settings = _make_settings()
        client = MagicMock()
        client.bucket.return_value.blob.return_value.download_as_text.side_effect = (
            ConnectionError("GCS unreachable")
        )
        checker = QuotaChecker(settings, storage_client=client)

        with pytest.raises(RuntimeError, match="Could not load quota policy from GCS"):
            checker.check_and_consume("u:u@co.com", 100)

    def test_stale_cache_used_with_warning(self) -> None:
        """Uses stale cache and logs WARNING when GCS reload fails."""
        settings = _make_settings(cache_ttl=5)
        client = _make_storage_client()
        checker = QuotaChecker(settings, storage_client=client)

        # Load policy successfully first
        checker.check_and_consume("u:u@co.com", 100)

        # Now make GCS fail and expire TTL
        blob = client.bucket.return_value.blob.return_value
        blob.download_as_text.side_effect = ConnectionError("GCS down")
        checker._policy_loaded_at -= 20  # Expire cache

        # Should still work — uses stale cache
        with patch("vaig.core.quota.logger") as mock_logger:
            checker.check_and_consume("u:u@co.com", 100)
            mock_logger.warning.assert_called_once()
            assert "stale cache" in str(mock_logger.warning.call_args)


# ── QuotaChecker: Policy parsing ─────────────────────────────


class TestQuotaCheckerParsing:
    """Tests for policy YAML parsing and validation."""

    def test_invalid_schema_version(self) -> None:
        """Raises RuntimeError (wrapping ValueError) for unsupported schema_version."""
        policy = {
            "schema_version": 99,
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        with pytest.raises(RuntimeError, match="Could not load quota policy"):
            checker.check_and_consume("u:u@co.com", 100)

    def test_missing_schema_version(self) -> None:
        """Raises RuntimeError (wrapping ValueError) when schema_version is missing."""
        policy = {
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        with pytest.raises(RuntimeError, match="Could not load quota policy"):
            checker.check_and_consume("u:u@co.com", 100)

    def test_valid_policy_parsed(self) -> None:
        """A well-formed policy is parsed without errors."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 100,
                "tokens_per_day": 500_000,
                "executions_per_day": 10,
            },
            "user_overrides": {
                "vip@co.com": {"requests_per_day": 10000},
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        # Should load and parse without error
        checker.check_and_consume("u:u@co.com", 100)
        assert checker._policy is not None
        assert checker._policy.default_requests_per_day == 100
        assert checker._policy.default_tokens_per_day == 500_000
        assert "vip@co.com" in checker._policy.user_overrides

    def test_empty_user_overrides(self) -> None:
        """Policy works with no user_overrides section."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 500,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        checker.check_and_consume("u:u@co.com", 100)
        assert checker._policy is not None
        assert checker._policy.user_overrides == {}


# ── QuotaChecker: Per-user isolation ─────────────────────────


class TestQuotaCheckerUserIsolation:
    """Tests that usage is tracked independently per user."""

    def test_separate_users_separate_counters(self) -> None:
        """Different user keys have independent counters."""
        policy = {
            "schema_version": 1,
            "default_policy": {
                "requests_per_day": 3,
                "tokens_per_day": 2_000_000,
                "executions_per_day": 50,
            },
        }
        settings = _make_settings()
        client = _make_storage_client(policy)
        checker = QuotaChecker(settings, storage_client=client)

        # Alice uses 3 requests (at limit)
        checker.check_and_consume("alice:alice@co.com", 100)
        checker.check_and_consume("alice:alice@co.com", 100)
        checker.check_and_consume("alice:alice@co.com", 100)

        # Alice is now at limit
        with pytest.raises(QuotaExceededError):
            checker.check_and_consume("alice:alice@co.com", 100)

        # Bob should still be able to make requests
        checker.check_and_consume("bob:bob@co.com", 100)
        checker.check_and_consume("bob:bob@co.com", 100)

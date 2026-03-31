"""Tests for QuotaExceededError and QuotaExceeded event."""

from __future__ import annotations

import dataclasses

import pytest

from vaig.core.events import QuotaExceeded
from vaig.core.exceptions import QuotaExceededError, VAIGError

# ══════════════════════════════════════════════════════════════
# QuotaExceededError
# ══════════════════════════════════════════════════════════════


class TestQuotaExceededError:
    """Verify QuotaExceededError hierarchy, message, and properties."""

    def test_inherits_from_vaig_error(self) -> None:
        assert issubclass(QuotaExceededError, VAIGError)

    def test_message_format_with_comma_separated_numbers(self) -> None:
        exc = QuotaExceededError(
            dimension="tokens",
            used=1_500_000,
            limit=2_000_000,
            user_key="alice:alice@example.com",
        )
        expected = (
            "Quota exceeded for 'alice:alice@example.com': "
            "tokens usage 1,500,000/2,000,000 per day"
        )
        assert str(exc) == expected

    def test_message_format_small_numbers(self) -> None:
        exc = QuotaExceededError(
            dimension="requests",
            used=101,
            limit=100,
            user_key="bob:bob@example.com",
        )
        assert str(exc) == (
            "Quota exceeded for 'bob:bob@example.com': "
            "requests usage 101/100 per day"
        )

    def test_dimension_property(self) -> None:
        exc = QuotaExceededError(
            dimension="executions",
            used=50,
            limit=50,
            user_key="u:e",
        )
        assert exc.dimension == "executions"

    def test_used_property(self) -> None:
        exc = QuotaExceededError(
            dimension="requests",
            used=42,
            limit=100,
            user_key="u:e",
        )
        assert exc.used == 42

    def test_limit_property(self) -> None:
        exc = QuotaExceededError(
            dimension="requests",
            used=42,
            limit=100,
            user_key="u:e",
        )
        assert exc.limit == 100

    def test_user_key_property(self) -> None:
        exc = QuotaExceededError(
            dimension="tokens",
            used=0,
            limit=1000,
            user_key="deploy-bot:sa@proj.iam.gserviceaccount.com",
        )
        assert exc.user_key == "deploy-bot:sa@proj.iam.gserviceaccount.com"

    def test_can_be_caught_as_vaig_error(self) -> None:
        with pytest.raises(VAIGError):
            raise QuotaExceededError(
                dimension="requests",
                used=10,
                limit=5,
                user_key="u:e",
            )


# ══════════════════════════════════════════════════════════════
# QuotaExceeded Event
# ══════════════════════════════════════════════════════════════


class TestQuotaExceededEvent:
    """Verify QuotaExceeded event fields, types, and immutability."""

    def test_event_type_is_quota_exceeded(self) -> None:
        evt = QuotaExceeded(
            user_key="alice:alice@example.com",
            dimension="tokens",
            used=500,
            limit=1000,
        )
        assert evt.event_type == "quota.exceeded"

    def test_timestamp_auto_populated(self) -> None:
        evt = QuotaExceeded(
            user_key="u:e",
            dimension="requests",
            used=1,
            limit=10,
        )
        assert isinstance(evt.timestamp, str)
        assert len(evt.timestamp) > 0

    def test_fields_have_correct_types(self) -> None:
        evt = QuotaExceeded(
            user_key="alice:alice@example.com",
            dimension="executions",
            used=25,
            limit=50,
        )
        assert isinstance(evt.user_key, str)
        assert isinstance(evt.dimension, str)
        assert isinstance(evt.used, int)
        assert isinstance(evt.limit, int)

    def test_fields_store_correct_values(self) -> None:
        evt = QuotaExceeded(
            user_key="bob:bob@corp.com",
            dimension="tokens",
            used=999_999,
            limit=1_000_000,
        )
        assert evt.user_key == "bob:bob@corp.com"
        assert evt.dimension == "tokens"
        assert evt.used == 999_999
        assert evt.limit == 1_000_000

    def test_event_is_frozen(self) -> None:
        evt = QuotaExceeded(
            user_key="u:e",
            dimension="requests",
            used=1,
            limit=10,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.used = 999  # type: ignore[misc]

    def test_event_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(QuotaExceeded)

    def test_default_values(self) -> None:
        evt = QuotaExceeded()
        assert evt.event_type == "quota.exceeded"
        assert evt.user_key == ""
        assert evt.dimension == ""
        assert evt.used == 0
        assert evt.limit == 0

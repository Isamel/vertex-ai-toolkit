"""Tests for the review gate in RemediationExecutor.

Verifies that remediation is blocked when review is required but not
approved, allowed when approved, and bypassed when review is disabled
or when no ReviewStore is provided.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from vaig.core.config import RemediationConfig, ReviewConfig
from vaig.core.remediation import ClassifiedCommand, RemediationExecutor, SafetyTier

# ── Helpers ───────────────────────────────────────────────────


def _make_classified(
    tier: SafetyTier = SafetyTier.SAFE,
    command: str = "kubectl annotate pod/foo note=ok",
) -> ClassifiedCommand:
    return ClassifiedCommand(
        tool="kubectl",
        subcommand="annotate",
        args=("pod/foo", "note=ok"),
        tier=tier,
        raw_command=command,
    )


def _make_action() -> MagicMock:
    action = MagicMock()
    action.title = "Test action"
    action.description = "Test description"
    action.risk = "low"
    action.command = "kubectl annotate pod/foo note=ok"
    return action


def _make_gke_config() -> MagicMock:
    gke = MagicMock()
    gke.cluster_name = "test-cluster"
    gke.project = "test-project"
    gke.region = "us-central1"
    return gke


def _make_bus() -> MagicMock:
    return MagicMock()


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ── 1. Blocked when review required but NOT approved ─────────


def test_gate_blocks_when_not_approved():
    review_store = MagicMock()
    review_store.is_approved.return_value = False

    review_config = ReviewConfig(enabled=True, require_review_for_remediation=True)
    rem_config = RemediationConfig()
    bus = _make_bus()

    executor = RemediationExecutor(
        rem_config,
        bus,
        review_store=review_store,
        review_config=review_config,
    )

    result = _run(
        executor.execute(
            _make_action(),
            _make_classified(),
            _make_gke_config(),
            approved=True,
            run_id="run-123",
        )
    )

    assert result.error is True
    assert "Review not approved" in result.output
    assert "run-123" in result.output
    review_store.is_approved.assert_called_once_with("run-123")


# ── 2. Allowed when review IS approved ───────────────────────


def test_gate_allows_when_approved():
    review_store = MagicMock()
    review_store.is_approved.return_value = True

    review_config = ReviewConfig(enabled=True, require_review_for_remediation=True)
    rem_config = RemediationConfig(auto_approve_safe=True)
    bus = _make_bus()

    executor = RemediationExecutor(
        rem_config,
        bus,
        review_store=review_store,
        review_config=review_config,
    )

    # Patch _dispatch to avoid real execution
    with patch.object(executor, "_dispatch") as mock_dispatch:
        mock_result = MagicMock()
        mock_result.output = "success"
        mock_result.error = False
        mock_dispatch.return_value = mock_result

        result = _run(
            executor.execute(
                _make_action(),
                _make_classified(tier=SafetyTier.SAFE),
                _make_gke_config(),
                approved=True,
                run_id="run-456",
            )
        )

    # Gate passed — command proceeds to execution (or dry_run/plan)
    assert result.error is not True or "Review not approved" not in result.output
    review_store.is_approved.assert_called_once_with("run-456")


# ── 3. Bypassed when config disabled ─────────────────────────


def test_gate_bypassed_when_disabled():
    review_store = MagicMock()
    # is_approved should NOT be called when review is disabled
    review_store.is_approved.return_value = False

    review_config = ReviewConfig(enabled=False, require_review_for_remediation=True)
    rem_config = RemediationConfig(auto_approve_safe=True)
    bus = _make_bus()

    executor = RemediationExecutor(
        rem_config,
        bus,
        review_store=review_store,
        review_config=review_config,
    )

    with patch.object(executor, "_dispatch") as mock_dispatch:
        mock_result = MagicMock()
        mock_result.output = "success"
        mock_result.error = False
        mock_dispatch.return_value = mock_result

        result = _run(
            executor.execute(
                _make_action(),
                _make_classified(tier=SafetyTier.SAFE),
                _make_gke_config(),
                approved=True,
                run_id="run-789",
            )
        )

    # Gate should not have fired — is_approved never called
    review_store.is_approved.assert_not_called()
    assert "Review not approved" not in result.output


# ── 4. Bypassed when no ReviewStore provided ─────────────────


def test_gate_bypassed_when_no_store():
    """Backward compat: no review_store → gate is skipped entirely."""
    rem_config = RemediationConfig(auto_approve_safe=True)
    bus = _make_bus()

    # No review_store and no review_config → original behavior
    executor = RemediationExecutor(rem_config, bus)

    with patch.object(executor, "_dispatch") as mock_dispatch:
        mock_result = MagicMock()
        mock_result.output = "success"
        mock_result.error = False
        mock_dispatch.return_value = mock_result

        result = _run(
            executor.execute(
                _make_action(),
                _make_classified(tier=SafetyTier.SAFE),
                _make_gke_config(),
                approved=True,
                run_id="run-999",
            )
        )

    assert "Review not approved" not in result.output


# ── 5. Fail-closed when no run_id provided ──────────────────


def test_gate_fails_closed_when_no_run_id():
    """If execute() is called without run_id, gate fails closed → error."""
    review_store = MagicMock()
    review_config = ReviewConfig(enabled=True, require_review_for_remediation=True)
    rem_config = RemediationConfig(auto_approve_safe=True)
    bus = _make_bus()

    executor = RemediationExecutor(
        rem_config,
        bus,
        review_store=review_store,
        review_config=review_config,
    )

    with patch.object(executor, "_dispatch") as mock_dispatch:
        mock_result = MagicMock()
        mock_result.output = "success"
        mock_result.error = False
        mock_dispatch.return_value = mock_result

        result = _run(
            executor.execute(
                _make_action(),
                _make_classified(tier=SafetyTier.SAFE),
                _make_gke_config(),
                approved=True,
                # no run_id
            )
        )

    review_store.is_approved.assert_not_called()
    assert result.error is True
    assert "no run_id provided" in result.output


# ── 6. Fail-closed when no store wired but review required ───


def test_gate_fails_closed_when_no_store():
    """If review is required but ReviewStore is not wired → fail closed."""
    review_config = ReviewConfig(enabled=True, require_review_for_remediation=True)
    rem_config = RemediationConfig(auto_approve_safe=True)
    bus = _make_bus()

    # No review_store but review IS required
    executor = RemediationExecutor(
        rem_config,
        bus,
        review_store=None,
        review_config=review_config,
    )

    with patch.object(executor, "_dispatch") as mock_dispatch:
        mock_result = MagicMock()
        mock_result.output = "success"
        mock_result.error = False
        mock_dispatch.return_value = mock_result

        result = _run(
            executor.execute(
                _make_action(),
                _make_classified(tier=SafetyTier.SAFE),
                _make_gke_config(),
                approved=True,
                run_id="run-with-no-store",
            )
        )

    assert result.error is True
    assert "ReviewStore is not configured" in result.output

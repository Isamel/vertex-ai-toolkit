"""Tests for pod status noise reduction bugs.

Bug 4: Orphaned Terminating pods from old ReplicaSets trigger false positives.
       _pod_status() should return 'Terminating rollout' for pods terminating
       less than 10 minutes, and 'Terminating stuck' for pods terminating
       10 minutes or more.

Bug 5: Istio sidecar containers inflate the total container count, making a
       fully-ready pod appear as '1/2' instead of '1/1'.
       _pod_ready_count() should annotate with '[app: X/Y]' when sidecars are
       present and not all app containers are ready, allowing the LLM to
       distinguish between an unhealthy app and a normal unready sidecar.
       When both ratios are equivalent (e.g. all containers ready), no annotation
       is added.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from vaig.tools.gke._formatters import _pod_ready_count, _pod_status

# ── Helpers ──────────────────────────────────────────────────


def _make_pod(
    deletion_timestamp: datetime | None = None,
    phase: str = "Running",
    container_specs: list[str] | None = None,
    container_statuses: list[dict] | None = None,
) -> MagicMock:
    """Create a minimal mock pod object.

    Args:
        deletion_timestamp: When the pod started terminating (UTC).
        phase: Pod phase string.
        container_specs: List of container names in spec.containers.
        container_statuses: List of dicts with keys 'name' and 'ready'.
    """
    pod = MagicMock()
    pod.metadata.deletion_timestamp = deletion_timestamp

    pod.status = MagicMock()
    pod.status.phase = phase
    pod.status.container_statuses = []

    if container_specs is None:
        container_specs = ["app"]

    containers = []
    for name in container_specs:
        c = MagicMock()
        c.name = name
        containers.append(c)
    pod.spec.containers = containers

    if container_statuses is not None:
        cs_mocks = []
        for cs_dict in container_statuses:
            cs = MagicMock()
            cs.name = cs_dict["name"]
            cs.ready = cs_dict["ready"]
            cs.state = MagicMock()
            cs.state.waiting = None
            cs.state.terminated = None
            cs_mocks.append(cs)
        pod.status.container_statuses = cs_mocks

    return pod


# ── Bug 4: Terminating pod classification ────────────────────


class TestTerminatingPodClassification:
    """_pod_status() classifies terminating pods as rollout vs stuck."""

    def test_recently_terminating_pod_is_rollout(self) -> None:
        """Pod terminating for less than 10 minutes → 'Terminating rollout'."""
        deletion_ts = datetime.now(UTC) - timedelta(minutes=3)
        pod = _make_pod(deletion_timestamp=deletion_ts)
        status = _pod_status(pod)
        assert status == "Terminating rollout"

    def test_just_started_terminating_pod_is_rollout(self) -> None:
        """Pod terminating for under 1 minute → 'Terminating rollout'."""
        deletion_ts = datetime.now(UTC) - timedelta(seconds=30)
        pod = _make_pod(deletion_timestamp=deletion_ts)
        status = _pod_status(pod)
        assert status == "Terminating rollout"

    def test_near_boundary_is_rollout(self) -> None:
        """Pod terminating for 590 s (clearly < 10 min) → 'Terminating rollout'."""
        deletion_ts = datetime.now(UTC) - timedelta(seconds=590)
        pod = _make_pod(deletion_timestamp=deletion_ts)
        status = _pod_status(pod)
        assert status == "Terminating rollout"

    def test_exactly_at_threshold_is_stuck(self) -> None:
        """Pod terminating for exactly 600 s → 'Terminating stuck' (>= threshold)."""
        frozen_now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        deletion_ts = frozen_now - timedelta(seconds=600)
        pod = _make_pod(deletion_timestamp=deletion_ts)
        with patch("vaig.tools.gke._formatters.datetime") as mock_dt:
            mock_dt.now.return_value = frozen_now
            status = _pod_status(pod)
        assert status == "Terminating stuck"

    def test_long_terminating_pod_is_stuck(self) -> None:
        """Pod terminating for more than 10 minutes → 'Terminating stuck'."""
        deletion_ts = datetime.now(UTC) - timedelta(minutes=15)
        pod = _make_pod(deletion_timestamp=deletion_ts)
        status = _pod_status(pod)
        assert status == "Terminating stuck"

    def test_very_long_terminating_pod_is_stuck(self) -> None:
        """Pod terminating for hours → 'Terminating stuck'."""
        deletion_ts = datetime.now(UTC) - timedelta(hours=2)
        pod = _make_pod(deletion_timestamp=deletion_ts)
        status = _pod_status(pod)
        assert status == "Terminating stuck"

    def test_terminating_pod_with_naive_timestamp(self) -> None:
        """Pod with timezone-naive deletion_timestamp → still classified correctly."""
        # Some K8s client versions return naive datetimes
        deletion_ts = datetime.utcnow() - timedelta(minutes=20)  # naive, > 10 min
        pod = _make_pod(deletion_timestamp=deletion_ts)
        status = _pod_status(pod)
        assert status == "Terminating stuck"

    def test_running_pod_not_affected(self) -> None:
        """Pod without deletion_timestamp → normal phase-based status."""
        pod = _make_pod(deletion_timestamp=None, phase="Running")
        status = _pod_status(pod)
        assert status == "Running"


# ── Bug 5: Sidecar-aware ready count ─────────────────────────


class TestSidecarAwareReadyCount:
    """_pod_ready_count() annotates with [app: X/Y] when sidecars mask app health."""

    def test_app_ready_sidecar_not_ready_shows_app_annotation(self) -> None:
        """App ready, istio-proxy not ready → shows [app: 1/1] annotation."""
        pod = _make_pod(
            container_specs=["my-app", "istio-proxy"],
            container_statuses=[
                {"name": "my-app", "ready": True},
                {"name": "istio-proxy", "ready": False},
            ],
        )
        result = _pod_ready_count(pod)
        # Total: 1/2 but app is fine — annotation shows [app: 1/1]
        assert result == "1/2 [app: 1/1]"

    def test_app_not_ready_sidecar_ready_shows_app_annotation(self) -> None:
        """App not ready, sidecar ready → shows [app: 0/1] annotation."""
        pod = _make_pod(
            container_specs=["my-app", "istio-proxy"],
            container_statuses=[
                {"name": "my-app", "ready": False},
                {"name": "istio-proxy", "ready": True},
            ],
        )
        result = _pod_ready_count(pod)
        assert result == "1/2 [app: 0/1]"

    def test_both_ready_no_annotation_needed(self) -> None:
        """All containers (including sidecar) ready → no annotation, clean display."""
        pod = _make_pod(
            container_specs=["my-app", "istio-proxy"],
            container_statuses=[
                {"name": "my-app", "ready": True},
                {"name": "istio-proxy", "ready": True},
            ],
        )
        result = _pod_ready_count(pod)
        # All app containers ready → no noise annotation
        assert result == "2/2"

    def test_no_sidecar_plain_ready_count(self) -> None:
        """Pod with no sidecar → standard 'ready/total' format."""
        pod = _make_pod(
            container_specs=["my-app"],
            container_statuses=[{"name": "my-app", "ready": True}],
        )
        result = _pod_ready_count(pod)
        assert result == "1/1"

    def test_multiple_app_containers_with_sidecar(self) -> None:
        """Multiple app containers + sidecar → annotation shows app totals."""
        pod = _make_pod(
            container_specs=["frontend", "backend", "istio-proxy"],
            container_statuses=[
                {"name": "frontend", "ready": True},
                {"name": "backend", "ready": False},
                {"name": "istio-proxy", "ready": False},
            ],
        )
        result = _pod_ready_count(pod)
        # 1 ready out of 3 total; [app: 1/2] — backend is the problem
        assert result == "1/3 [app: 1/2]"

    def test_datadog_agent_sidecar_treated_as_sidecar(self) -> None:
        """datadog-agent is recognized as sidecar → annotation shown."""
        pod = _make_pod(
            container_specs=["my-app", "datadog-agent"],
            container_statuses=[
                {"name": "my-app", "ready": True},
                {"name": "datadog-agent", "ready": False},
            ],
        )
        result = _pod_ready_count(pod)
        assert result == "1/2 [app: 1/1]"

    def test_envoy_sidecar_treated_as_sidecar(self) -> None:
        """envoy is recognized as sidecar → annotation shown when app is ready."""
        pod = _make_pod(
            container_specs=["my-app", "envoy"],
            container_statuses=[
                {"name": "my-app", "ready": True},
                {"name": "envoy", "ready": False},
            ],
        )
        result = _pod_ready_count(pod)
        assert result == "1/2 [app: 1/1]"

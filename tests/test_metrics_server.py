"""Tests for the Kubernetes Metrics Server adapter (Layer 2 in cost pipeline)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import GKEConfig


def _make_gke_config(**kwargs: object) -> GKEConfig:
    defaults = {
        "cluster_name": "test-cluster",
        "project_id": "test-project",
        "location": "us-central1",
        "default_namespace": "default",
        "kubeconfig_path": "",
        "context": "",
        "log_limit": 100,
        "metrics_interval_minutes": 60,
        "proxy_url": "",
    }
    defaults.update(kwargs)
    return GKEConfig(**defaults)


def _make_custom_api(items: list[dict]) -> MagicMock:
    """Build a mock custom_objects_api that returns the given pod metric items."""
    api = MagicMock()
    api.list_namespaced_custom_object.return_value = {"items": items}
    return api


def _mock_clients_tuple(custom_api: MagicMock) -> tuple:
    return (MagicMock(), MagicMock(), custom_api, MagicMock())


# ── Basic success path ────────────────────────────────────────


@patch("vaig.tools.gke.metrics_server._clients._create_k8s_clients")
def test_returns_workload_metrics_on_success(mock_create: MagicMock) -> None:
    """Should return WorkloadUsageMetrics when Metrics Server returns pod data."""
    from vaig.tools.gke.metrics_server import get_metrics_server_usage

    items = [
        {
            "metadata": {"name": "my-app-abc12"},
            "containers": [
                {"name": "app", "usage": {"cpu": "100m", "memory": "128Mi"}},
            ],
        }
    ]
    custom_api = _make_custom_api(items)
    mock_create.return_value = _mock_clients_tuple(custom_api)

    workload_pod_names = {"my-app": ["my-app-abc12"]}
    result = get_metrics_server_usage("default", workload_pod_names, gke_config=_make_gke_config())

    assert "my-app" in result
    wl = result["my-app"]
    assert "app" in wl.containers
    ctr = wl.containers["app"]
    assert ctr.avg_cpu_cores == pytest.approx(0.1)
    assert ctr.avg_memory_gib == pytest.approx(128 / 1024)


# ── 404: Metrics Server not installed ─────────────────────────


@patch("vaig.tools.gke.metrics_server._clients._create_k8s_clients")
def test_returns_empty_on_404(mock_create: MagicMock) -> None:
    """Should silently return {} when Metrics Server API returns 404."""
    from kubernetes.client.exceptions import ApiException

    from vaig.tools.gke.metrics_server import get_metrics_server_usage

    custom_api = MagicMock()
    custom_api.list_namespaced_custom_object.side_effect = ApiException(status=404)
    mock_create.return_value = _mock_clients_tuple(custom_api)

    result = get_metrics_server_usage("default", {"my-app": ["pod-1"]}, gke_config=_make_gke_config())
    assert result == {}


# ── 403: RBAC denied ──────────────────────────────────────────


@patch("vaig.tools.gke.metrics_server._clients._create_k8s_clients")
def test_returns_empty_on_403(mock_create: MagicMock) -> None:
    """Should silently return {} when RBAC denies access (403)."""
    from kubernetes.client.exceptions import ApiException

    from vaig.tools.gke.metrics_server import get_metrics_server_usage

    custom_api = MagicMock()
    custom_api.list_namespaced_custom_object.side_effect = ApiException(status=403)
    mock_create.return_value = _mock_clients_tuple(custom_api)

    result = get_metrics_server_usage("default", {"my-app": ["pod-1"]}, gke_config=_make_gke_config())
    assert result == {}


# ── Generic exception ─────────────────────────────────────────


@patch("vaig.tools.gke.metrics_server._clients._create_k8s_clients")
def test_returns_empty_on_generic_exception(mock_create: MagicMock) -> None:
    """Should return {} and not raise on any unexpected error."""
    from vaig.tools.gke.metrics_server import get_metrics_server_usage

    custom_api = MagicMock()
    custom_api.list_namespaced_custom_object.side_effect = RuntimeError("boom")
    mock_create.return_value = _mock_clients_tuple(custom_api)

    result = get_metrics_server_usage("default", {"my-app": ["pod-1"]}, gke_config=_make_gke_config())
    assert result == {}


# ── Pod not in workload_pod_names mapping ─────────────────────


@patch("vaig.tools.gke.metrics_server._clients._create_k8s_clients")
def test_ignores_unknown_pods(mock_create: MagicMock) -> None:
    """Pods not in workload_pod_names should be silently skipped."""
    from vaig.tools.gke.metrics_server import get_metrics_server_usage

    items = [
        {
            "metadata": {"name": "unknown-pod-xyz"},
            "containers": [{"name": "app", "usage": {"cpu": "200m", "memory": "256Mi"}}],
        }
    ]
    custom_api = _make_custom_api(items)
    mock_create.return_value = _mock_clients_tuple(custom_api)

    result = get_metrics_server_usage("default", {"my-app": ["my-app-abc12"]}, gke_config=_make_gke_config())
    assert result == {}


# ── Multiple pods aggregated ──────────────────────────────────


@patch("vaig.tools.gke.metrics_server._clients._create_k8s_clients")
def test_aggregates_multiple_pods(mock_create: MagicMock) -> None:
    """CPU and memory should be summed across multiple pods of the same workload."""
    from vaig.tools.gke.metrics_server import get_metrics_server_usage

    items = [
        {
            "metadata": {"name": "my-app-pod1"},
            "containers": [{"name": "app", "usage": {"cpu": "100m", "memory": "128Mi"}}],
        },
        {
            "metadata": {"name": "my-app-pod2"},
            "containers": [{"name": "app", "usage": {"cpu": "200m", "memory": "256Mi"}}],
        },
    ]
    custom_api = _make_custom_api(items)
    mock_create.return_value = _mock_clients_tuple(custom_api)

    workload_pod_names = {"my-app": ["my-app-pod1", "my-app-pod2"]}
    result = get_metrics_server_usage("default", workload_pod_names, gke_config=_make_gke_config())

    assert "my-app" in result
    ctr = result["my-app"].containers["app"]
    assert ctr.avg_cpu_cores == pytest.approx(0.3)   # 100m + 200m
    assert ctr.avg_memory_gib == pytest.approx(384 / 1024)  # 128Mi + 256Mi


# ── CPU parsing ───────────────────────────────────────────────


def test_parse_cpu_millicore() -> None:
    from vaig.tools.gke.metrics_server import _parse_cpu_safe
    assert _parse_cpu_safe("500m") == pytest.approx(0.5)


def test_parse_cpu_nanocores() -> None:
    from vaig.tools.gke.metrics_server import _parse_cpu_safe
    assert _parse_cpu_safe("500000000n") == pytest.approx(0.5)


def test_parse_cpu_full() -> None:
    from vaig.tools.gke.metrics_server import _parse_cpu_safe
    assert _parse_cpu_safe("2") == pytest.approx(2.0)


def test_parse_cpu_none() -> None:
    from vaig.tools.gke.metrics_server import _parse_cpu_safe
    assert _parse_cpu_safe(None) is None


def test_parse_cpu_invalid() -> None:
    from vaig.tools.gke.metrics_server import _parse_cpu_safe
    assert _parse_cpu_safe("bad") is None


# ── Memory parsing ────────────────────────────────────────────


def test_parse_memory_mib() -> None:
    from vaig.tools.gke.metrics_server import _parse_memory_gib_safe
    assert _parse_memory_gib_safe("1024Mi") == pytest.approx(1.0)


def test_parse_memory_gib() -> None:
    from vaig.tools.gke.metrics_server import _parse_memory_gib_safe
    assert _parse_memory_gib_safe("2Gi") == pytest.approx(2.0)


def test_parse_memory_kib() -> None:
    from vaig.tools.gke.metrics_server import _parse_memory_gib_safe
    assert _parse_memory_gib_safe("1048576Ki") == pytest.approx(1.0)


def test_parse_memory_bytes() -> None:
    from vaig.tools.gke.metrics_server import _parse_memory_gib_safe
    assert _parse_memory_gib_safe("1073741824") == pytest.approx(1.0)


def test_parse_memory_none() -> None:
    from vaig.tools.gke.metrics_server import _parse_memory_gib_safe
    assert _parse_memory_gib_safe(None) is None

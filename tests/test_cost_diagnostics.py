"""Tests for the GKE cost metrics diagnostic tool (cost_diagnostics.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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


# ── _check_iam_monitoring_viewer ──────────────────────────────


def test_iam_check_ok_when_list_succeeds() -> None:
    """IAM check returns 'ok' when list_time_series succeeds."""
    from vaig.tools.gke.cost_diagnostics import _check_iam_monitoring_viewer

    with (
        patch("google.cloud.monitoring_v3.MetricServiceClient") as mock_client_cls,
        patch("google.cloud.monitoring_v3.TimeInterval"),
        patch("google.cloud.monitoring_v3.ListTimeSeriesRequest"),
        patch("google.protobuf.timestamp_pb2.Timestamp"),
        patch("time.time", return_value=1_000_000),
    ):
        mock_client = MagicMock()
        mock_client.list_time_series.return_value = iter([])
        mock_client_cls.return_value = mock_client

        result = _check_iam_monitoring_viewer(_make_gke_config())

    assert result["status"] == "ok"
    assert result["name"] == "iam_monitoring_viewer"


def test_iam_check_error_on_permission_denied() -> None:
    """IAM check returns 'error' on PermissionDenied."""
    from google.api_core.exceptions import PermissionDenied

    from vaig.tools.gke.cost_diagnostics import _check_iam_monitoring_viewer

    with patch("google.cloud.monitoring_v3.MetricServiceClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.list_time_series.side_effect = PermissionDenied("denied")
        mock_client_cls.return_value = mock_client

        result = _check_iam_monitoring_viewer(_make_gke_config())

    assert result["status"] == "error"
    assert "denied" in result["detail"]


# ── _check_metrics_server_installed ──────────────────────────


def test_metrics_server_ok_when_api_responds() -> None:
    """Metrics Server check returns 'ok' when custom API call succeeds."""
    from vaig.tools.gke.cost_diagnostics import _check_metrics_server_installed

    custom_api = MagicMock()
    custom_api.list_namespaced_custom_object.return_value = {"items": []}
    mock_clients = (MagicMock(), MagicMock(), custom_api, MagicMock())

    with patch("vaig.tools.gke._clients._create_k8s_clients", return_value=mock_clients):
        result = _check_metrics_server_installed(_make_gke_config())

    assert result["status"] == "ok"


def test_metrics_server_warning_on_404() -> None:
    """Metrics Server check returns 'warning' on 404 (not installed)."""
    from kubernetes.client.exceptions import ApiException

    from vaig.tools.gke.cost_diagnostics import _check_metrics_server_installed

    custom_api = MagicMock()
    custom_api.list_namespaced_custom_object.side_effect = ApiException(status=404)
    mock_clients = (MagicMock(), MagicMock(), custom_api, MagicMock())

    with patch("vaig.tools.gke._clients._create_k8s_clients", return_value=mock_clients):
        result = _check_metrics_server_installed(_make_gke_config())

    assert result["status"] == "warning"
    assert "404" in result["detail"] or "not installed" in result["detail"].lower()


def test_metrics_server_warning_on_403() -> None:
    """Metrics Server check returns 'warning' on 403 (RBAC denied)."""
    from kubernetes.client.exceptions import ApiException

    from vaig.tools.gke.cost_diagnostics import _check_metrics_server_installed

    custom_api = MagicMock()
    custom_api.list_namespaced_custom_object.side_effect = ApiException(status=403)
    mock_clients = (MagicMock(), MagicMock(), custom_api, MagicMock())

    with patch("vaig.tools.gke._clients._create_k8s_clients", return_value=mock_clients):
        result = _check_metrics_server_installed(_make_gke_config())

    assert result["status"] == "warning"
    assert "403" in result["detail"] or "rbac" in result["detail"].lower()


# ── diagnose_gke_cost_metrics (integration) ───────────────────


def test_diagnose_returns_tool_result_with_json() -> None:
    """diagnose_gke_cost_metrics returns a ToolResult with valid JSON output."""
    from vaig.tools.gke.cost_diagnostics import diagnose_gke_cost_metrics

    ok_check = {"name": "test", "status": "ok", "detail": "ok"}

    with (
        patch(
            "vaig.tools.gke.cost_diagnostics._check_iam_monitoring_viewer",
            return_value={**ok_check, "name": "iam_monitoring_viewer"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_cluster_name_match",
            return_value={**ok_check, "name": "cluster_name_match"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_metrics_server_installed",
            return_value={**ok_check, "name": "metrics_server_installed"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_system_metrics_enabled",
            return_value={**ok_check, "name": "system_metrics_enabled"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_window_coverage",
            return_value={**ok_check, "name": "window_coverage"},
        ),
    ):
        result = diagnose_gke_cost_metrics(gke_config=_make_gke_config())

    assert result.error is False
    parsed = json.loads(result.output)
    assert "checks" in parsed
    assert len(parsed["checks"]) == 5
    assert "recommendation" in parsed
    assert "All checks passed" in parsed["recommendation"]


def test_diagnose_error_true_when_any_check_errors() -> None:
    """diagnose returns error=True when at least one check has status='error'."""
    from vaig.tools.gke.cost_diagnostics import diagnose_gke_cost_metrics

    ok_check = {"status": "ok", "detail": "ok"}
    error_check = {"status": "error", "detail": "Permission denied"}

    with (
        patch(
            "vaig.tools.gke.cost_diagnostics._check_iam_monitoring_viewer",
            return_value={**error_check, "name": "iam_monitoring_viewer"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_cluster_name_match",
            return_value={**ok_check, "name": "cluster_name_match"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_metrics_server_installed",
            return_value={**ok_check, "name": "metrics_server_installed"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_system_metrics_enabled",
            return_value={**ok_check, "name": "system_metrics_enabled"},
        ),
        patch(
            "vaig.tools.gke.cost_diagnostics._check_window_coverage",
            return_value={**ok_check, "name": "window_coverage"},
        ),
    ):
        result = diagnose_gke_cost_metrics(gke_config=_make_gke_config())

    assert result.error is True
    parsed = json.loads(result.output)
    assert any(c["status"] == "error" for c in parsed["checks"])

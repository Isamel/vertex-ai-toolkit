"""Tests for pod status misreporting bugs.

Bug 1: HPA formatter missing — `_format_hpa_table()` not registered in
       `_format_items()` dispatch dict, so HPA resources fall through to the
       generic formatter (only NAME/NAMESPACE/AGE shown).

Bug 2: `spec.replicas=None` incorrectly triggers "Scaled to zero" in
       `get_rollout_status()`. When HPA manages a deployment, K8s sets
       spec.replicas=None which was treated as 0.

Bug 3: `currentReplicas=None` silently becomes 0 via `or 0` in
       `_format_hpa_section()`, masking HPA status not yet populated.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from vaig.core.config import GKEConfig
from vaig.tools.base import ToolResult

# ── Helpers ──────────────────────────────────────────────────


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


def _make_hpa_mock(
    name: str = "my-hpa",
    current_replicas: int | None = 5,
    desired_replicas: int = 5,
    min_replicas: int = 2,
    max_replicas: int = 10,
    spec_metrics: list | None = None,
    current_metrics: list | None = None,
    scale_target_kind: str = "Deployment",
    scale_target_name: str = "my-deployment",
    created_at: datetime | None = None,
) -> MagicMock:
    """Create a mock HPA K8s object (typed API object, not dict-backed)."""
    hpa = MagicMock()
    hpa.metadata.name = name
    hpa.metadata.creation_timestamp = created_at or datetime(
        2025, 1, 1, tzinfo=UTC
    )

    hpa.spec.min_replicas = min_replicas
    hpa.spec.max_replicas = max_replicas
    hpa.spec.scale_target_ref.kind = scale_target_kind
    hpa.spec.scale_target_ref.name = scale_target_name
    hpa.spec.metrics = spec_metrics if spec_metrics is not None else []

    hpa.status.current_replicas = current_replicas
    hpa.status.desired_replicas = desired_replicas
    hpa.status.conditions = []
    hpa.status.current_metrics = current_metrics if current_metrics is not None else []

    return hpa


def _make_metric_spec(
    target_type: str = "Utilization",
    average_utilization: int | None = 80,
) -> MagicMock:
    """Build a minimal mock spec.metrics entry with to_dict() support."""
    m = MagicMock()
    m.to_dict.return_value = {
        "type": "Resource",
        "resource": {
            "name": "cpu",
            "target": {
                "type": target_type,
                "averageUtilization": average_utilization,
            },
        },
    }
    return m


def _make_current_metric(average_utilization: int = 65) -> MagicMock:
    """Build a minimal mock status.currentMetrics entry with to_dict() support."""
    m = MagicMock()
    m.to_dict.return_value = {
        "type": "Resource",
        "resource": {
            "name": "cpu",
            "current": {
                "averageUtilization": average_utilization,
                "averageValue": None,
            },
        },
    }
    return m


# ══════════════════════════════════════════════════════════════
# Bug 1: HPA formatter — _format_hpa_table() must be registered
# ══════════════════════════════════════════════════════════════


class TestHpaFormatterRegistered:
    """Bug 1 — _format_hpa_table must be wired into _format_items() dispatch."""

    def test_hpa_resource_uses_hpa_formatter_not_generic(self) -> None:
        """Calling _format_items with 'horizontalpodautoscalers' must return HPA headers."""
        from vaig.tools.gke._formatters import _format_items

        hpa = _make_hpa_mock(name="web-hpa")
        result = _format_items("horizontalpodautoscalers", [hpa], "table")

        # HPA-specific columns must be present
        assert "REFERENCE" in result, "Expected REFERENCE column in HPA table output"
        assert "MINPODS" in result, "Expected MINPODS column in HPA table output"
        assert "MAXPODS" in result, "Expected MAXPODS column in HPA table output"
        assert "REPLICAS" in result, "Expected REPLICAS column in HPA table output"

    def test_hpa_shortname_alias_works(self) -> None:
        """'hpa' shortname must also resolve to the HPA formatter."""
        from vaig.tools.gke._formatters import _format_items

        hpa = _make_hpa_mock(name="web-hpa")
        result = _format_items("hpa", [hpa], "table")

        assert "REFERENCE" in result
        assert "REPLICAS" in result

    def test_hpa_row_contains_name_and_reference(self) -> None:
        """Each HPA row must contain the HPA name and its scale target reference."""
        from vaig.tools.gke._formatters import _format_items

        hpa = _make_hpa_mock(
            name="frontend-hpa",
            scale_target_kind="Deployment",
            scale_target_name="frontend",
        )
        result = _format_items("horizontalpodautoscalers", [hpa], "table")

        assert "frontend-hpa" in result
        assert "Deployment/frontend" in result

    def test_hpa_row_contains_replica_counts(self) -> None:
        """HPA row must show current replica count."""
        from vaig.tools.gke._formatters import _format_items

        hpa = _make_hpa_mock(current_replicas=7, min_replicas=3, max_replicas=20)
        result = _format_items("horizontalpodautoscalers", [hpa], "table")

        assert "7" in result   # current replicas
        assert "3" in result   # min pods
        assert "20" in result  # max pods

    def test_hpa_row_with_targets_utilization(self) -> None:
        """TARGETS column must show current/target for a CPU utilization metric."""
        from vaig.tools.gke._formatters import _format_items

        hpa = _make_hpa_mock(
            spec_metrics=[_make_metric_spec(average_utilization=80)],
            current_metrics=[_make_current_metric(average_utilization=65)],
        )
        result = _format_items("horizontalpodautoscalers", [hpa], "table")

        # Expected format: "65%/80%"
        assert "65%" in result
        assert "80%" in result

    def test_hpa_empty_list_returns_no_resources_found(self) -> None:
        """Empty HPA list must return 'No resources found.'"""
        from vaig.tools.gke._formatters import _format_items

        result = _format_items("horizontalpodautoscalers", [], "table")
        assert result == "No resources found."

    def test_hpa_none_current_replicas_shows_unknown(self) -> None:
        """When current_replicas is None (status not yet populated), show '<unknown>'."""
        from vaig.tools.gke._formatters import _format_items

        hpa = _make_hpa_mock(current_replicas=None)
        result = _format_items("horizontalpodautoscalers", [hpa], "table")

        assert "<unknown>" in result, (
            "Expected '<unknown>' in REPLICAS column when current_replicas is None"
        )
        # Must NOT show '0' as a stand-in for None
        lines = result.splitlines()
        data_lines = [l for l in lines if l.strip() and "NAME" not in l]
        assert any("<unknown>" in l for l in data_lines), (
            "Data row must contain '<unknown>' not '0' for missing replicas"
        )


# ══════════════════════════════════════════════════════════════
# Bug 2: spec.replicas=None must NOT trigger "Scaled to zero"
# ══════════════════════════════════════════════════════════════


class TestRolloutStatusHpaManaged:
    """Bug 2 — HPA-managed deployments (spec.replicas=None) must not say 'Scaled to zero'."""

    def _make_deployment_mock(
        self,
        spec_replicas: int | None = None,
        status_replicas: int = 110,
        ready_replicas: int = 110,
        updated_replicas: int = 110,
        available_replicas: int = 110,
        unavailable_replicas: int = 0,
        conditions: list | None = None,
    ) -> MagicMock:
        dep = MagicMock()
        dep.spec.replicas = spec_replicas
        dep.status.replicas = status_replicas
        dep.status.ready_replicas = ready_replicas
        dep.status.updated_replicas = updated_replicas
        dep.status.available_replicas = available_replicas
        dep.status.unavailable_replicas = unavailable_replicas
        dep.status.conditions = conditions or []
        return dep

    def _make_available_condition(self) -> MagicMock:
        cond = MagicMock()
        cond.type = "Available"
        cond.status = "True"
        cond.reason = "MinimumReplicasAvailable"
        cond.message = "Deployment has minimum availability."
        return cond

    def _make_progressing_condition(
        self, reason: str = "NewReplicaSetAvailable"
    ) -> MagicMock:
        cond = MagicMock()
        cond.type = "Progressing"
        cond.status = "True"
        cond.reason = reason
        cond.message = ""
        return cond

    @contextmanager
    def _patch_k8s_clients(self, deployment: MagicMock) -> Generator[None, None, None]:
        """Patch K8s client stack and wire the given deployment mock as the API response."""
        with (
            patch("vaig.tools.gke._clients.detect_autopilot", return_value=None),
            patch("vaig.tools.gke._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke._clients._create_k8s_clients") as mock_clients,
        ):
            core_v1 = MagicMock()
            apps_v1 = MagicMock()
            custom_api = MagicMock()
            batch_v1 = MagicMock()
            mock_clients.return_value = (core_v1, apps_v1, custom_api, batch_v1)
            apps_v1.read_namespaced_deployment.return_value = deployment
            yield

    def test_hpa_managed_110_pods_not_scaled_to_zero(self) -> None:
        """Deployment with spec.replicas=None and 110 ready pods must NOT be 'Scaled to zero'."""
        from vaig.tools.gke.diagnostics import get_rollout_status

        dep = self._make_deployment_mock(
            spec_replicas=None,
            status_replicas=110,
            ready_replicas=110,
            conditions=[
                self._make_available_condition(),
                self._make_progressing_condition(),
            ],
        )

        cfg = _make_gke_config()
        with self._patch_k8s_clients(dep):
            result = get_rollout_status("my-app", gke_config=cfg, namespace="production")

        assert isinstance(result, ToolResult)
        assert result.error is False, f"Expected success, got error: {result.output}"
        assert "Scaled to zero" not in result.output, (
            "HPA-managed deployment with 110 pods must NOT report 'Scaled to zero'"
        )

    def test_hpa_managed_shows_status_replicas_as_desired(self) -> None:
        """With spec.replicas=None, the Desired count should fall back to status.replicas."""
        from vaig.tools.gke.diagnostics import get_rollout_status

        dep = self._make_deployment_mock(
            spec_replicas=None,
            status_replicas=5,
            ready_replicas=5,
        )

        cfg = _make_gke_config()
        with self._patch_k8s_clients(dep):
            result = get_rollout_status("my-app", gke_config=cfg, namespace="default")

        assert result.error is False
        # Desired should be shown as 5 (from status), not 0
        assert "Desired:     5" in result.output, (
            f"Expected 'Desired: 5' in output, got:\n{result.output}"
        )

    def test_explicit_zero_replicas_does_report_scaled_to_zero(self) -> None:
        """A deployment with spec.replicas=0 (explicitly scaled down) MUST say 'Scaled to zero'."""
        from vaig.tools.gke.diagnostics import get_rollout_status

        dep = self._make_deployment_mock(
            spec_replicas=0,
            status_replicas=0,
            ready_replicas=0,
            updated_replicas=0,
            available_replicas=0,
        )

        cfg = _make_gke_config()
        with self._patch_k8s_clients(dep):
            result = get_rollout_status("my-app", gke_config=cfg, namespace="default")

        assert result.error is False
        assert "Scaled to zero" in result.output, (
            "Deployment with spec.replicas=0 and no ready pods MUST report 'Scaled to zero'"
        )

    def test_normal_deployment_not_hpa_managed(self) -> None:
        """Standard deployment (spec.replicas=3) with all pods ready must report 'Complete'."""
        from vaig.tools.gke.diagnostics import get_rollout_status

        dep = self._make_deployment_mock(
            spec_replicas=3,
            status_replicas=3,
            ready_replicas=3,
            updated_replicas=3,
            available_replicas=3,
            conditions=[
                self._make_available_condition(),
                self._make_progressing_condition(),
            ],
        )

        cfg = _make_gke_config()
        with self._patch_k8s_clients(dep):
            result = get_rollout_status("my-app", gke_config=cfg, namespace="default")

        assert result.error is False
        assert "Complete" in result.output


# ══════════════════════════════════════════════════════════════
# Bug 3: currentReplicas=None must show "unknown", not 0
# ══════════════════════════════════════════════════════════════


class TestHpaSectionCurrentReplicasNone:
    """Bug 3 — _format_hpa_section must not silently convert None currentReplicas to 0."""

    def test_current_replicas_none_shows_unknown_not_zero(self) -> None:
        """When status.current_replicas is None, output must say 'unknown', not '0'."""
        from vaig.tools.gke.scaling import _format_hpa_section

        hpa = _make_hpa_mock(current_replicas=None, desired_replicas=5)
        output = _format_hpa_section(hpa)

        assert "unknown" in output.lower(), (
            f"Expected 'unknown' when current_replicas is None, got:\n{output}"
        )
        # Must NOT show 0/5 when replicas are genuinely unknown
        assert "0/5" not in output, (
            f"Must NOT show '0/5' when current_replicas is None, got:\n{output}"
        )

    def test_current_replicas_populated_shows_count(self) -> None:
        """When status.current_replicas is set, the actual count must appear."""
        from vaig.tools.gke.scaling import _format_hpa_section

        hpa = _make_hpa_mock(current_replicas=7, desired_replicas=7)
        output = _format_hpa_section(hpa)

        assert "7/7" in output, (
            f"Expected '7/7' in replicas line, got:\n{output}"
        )

    def test_current_replicas_zero_is_valid_not_none(self) -> None:
        """Explicit 0 current_replicas is a valid count (scaling down), must show '0'."""
        from vaig.tools.gke.scaling import _format_hpa_section

        hpa = _make_hpa_mock(current_replicas=0, desired_replicas=3)
        output = _format_hpa_section(hpa)

        assert "0/3" in output, (
            f"Expected '0/3' when explicitly scaled to 0, got:\n{output}"
        )

    def test_no_status_object_shows_unknown(self) -> None:
        """When status is None entirely, must show 'unknown' not crash or show 0."""
        from vaig.tools.gke.scaling import _format_hpa_section

        hpa = MagicMock()
        hpa.metadata.name = "my-hpa"
        hpa.spec.min_replicas = 1
        hpa.spec.max_replicas = 10
        hpa.spec.metrics = []
        hpa.status = None

        output = _format_hpa_section(hpa)

        assert "unknown" in output.lower(), (
            f"Expected 'unknown' when status is None, got:\n{output}"
        )

"""Unit tests for cross-cluster comparison — models, diff, runner."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from vaig.core.compare import (
    CompareMetadata,
    CompareReport,
    CompareRunner,
    DeploymentSnapshot,
    FieldDiff,
    _delta_pct,
    diff_snapshots,
)
from vaig.core.config import FleetCluster

# ── Helpers ───────────────────────────────────────────────────


def _make_snapshot(
    cluster: str = "prod-us",
    *,
    image_tag: str = "v2.1.0",
    replicas_desired: int = 3,
    replicas_ready: int = 3,
    rollout_generation: int = 5,
    hpa_min: int | None = 2,
    hpa_max: int | None = 10,
    cpu_usage_cores: float | None = None,
    memory_usage_gib: float | None = None,
    error_rate_pct: float | None = None,
) -> DeploymentSnapshot:
    """Build a DeploymentSnapshot with sensible defaults."""
    return DeploymentSnapshot(
        cluster_name=cluster,
        namespace="default",
        deployment_name="api-server",
        image_tag=image_tag,
        replicas_desired=replicas_desired,
        replicas_ready=replicas_ready,
        rollout_generation=rollout_generation,
        hpa_min=hpa_min,
        hpa_max=hpa_max,
        cpu_usage_cores=cpu_usage_cores,
        memory_usage_gib=memory_usage_gib,
        error_rate_pct=error_rate_pct,
        collected_at=datetime.now(tz=UTC),
    )


def _make_settings_mock() -> MagicMock:
    """Create a minimal mock Settings object for CompareRunner."""
    settings = MagicMock()
    settings.gcp.project_id = "test-project"
    settings.gcp.location = "us-central1"
    settings.gke.cluster_name = "default-cluster"
    settings.gke.project_id = ""
    settings.gke.default_namespace = "default"
    settings.gke.location = "us-central1"
    settings.gke.kubeconfig_path = ""
    settings.gke.context = ""
    settings.gke.log_limit = 100
    settings.gke.metrics_interval_minutes = 5
    settings.gke.proxy_url = ""
    settings.gke.impersonate_sa = ""
    settings.gke.exec_enabled = False
    settings.helm.enabled = False
    settings.argocd.enabled = False
    settings.argocd.server = ""
    settings.argocd.token = ""
    settings.argocd.context = ""
    settings.argocd.namespace = ""
    settings.argocd.verify_ssl = True
    return settings


# ── DeploymentSnapshot tests ─────────────────────────────────


class TestDeploymentSnapshot:
    """REQ-CMP-01: DeploymentSnapshot construction and validation."""

    def test_basic_construction(self) -> None:
        snap = _make_snapshot()
        assert snap.cluster_name == "prod-us"
        assert snap.namespace == "default"
        assert snap.deployment_name == "api-server"
        assert snap.image_tag == "v2.1.0"
        assert snap.replicas_desired == 3

    def test_none_optional_fields(self) -> None:
        """Optional fields default to None."""
        snap = DeploymentSnapshot(
            cluster_name="test",
            namespace="default",
            deployment_name="web",
        )
        assert snap.hpa_min is None
        assert snap.hpa_max is None
        assert snap.cpu_usage_cores is None
        assert snap.memory_usage_gib is None
        assert snap.error_rate_pct is None

    def test_model_dump_json(self) -> None:
        """Pydantic model_dump with mode=json produces serializable dict."""
        snap = _make_snapshot()
        data = snap.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["cluster_name"] == "prod-us"
        assert data["image_tag"] == "v2.1.0"


# ── FieldDiff / CompareReport tests ─────────────────────────


class TestDataModels:
    """FieldDiff, CompareMetadata, CompareReport."""

    def test_field_diff_construction(self) -> None:
        diff = FieldDiff(
            field="image_tag",
            values={"prod-us": "v2.1", "prod-eu": "v2.0"},
            severity="critical",
        )
        assert diff.field == "image_tag"
        assert diff.severity == "critical"

    def test_compare_metadata_auto_timestamp(self) -> None:
        meta = CompareMetadata(namespace="default", deployment="api")
        assert meta.timestamp  # auto-generated
        assert meta.namespace == "default"

    def test_compare_report_to_dict(self) -> None:
        snap = _make_snapshot()
        report = CompareReport(
            snapshots={"prod-us": snap},
            errors={"staging": "timeout"},
            diffs=[FieldDiff(field="image_tag", values={"prod-us": "v2"}, severity="critical")],
            metadata=CompareMetadata(
                clusters_requested=["prod-us", "staging"],
                namespace="default",
                deployment="api",
            ),
        )
        data = report.to_dict()
        assert "snapshots" in data
        assert "errors" in data
        assert "diffs" in data
        assert "metadata" in data
        assert data["errors"]["staging"] == "timeout"
        assert len(data["diffs"]) == 1
        assert data["diffs"][0]["severity"] == "critical"


# ── _delta_pct tests ─────────────────────────────────────────


class TestDeltaPct:
    """Edge cases for _delta_pct helper."""

    def test_identical_values(self) -> None:
        assert _delta_pct({"a": 3, "b": 3}) == 0.0

    def test_spread(self) -> None:
        # 3 vs 10 → spread=7, max_abs=10 → 70%
        assert _delta_pct({"a": 3, "b": 10}) == 70.0

    def test_single_value(self) -> None:
        assert _delta_pct({"a": 5}) == 0.0

    def test_empty(self) -> None:
        assert _delta_pct({}) == 0.0

    def test_all_zeros(self) -> None:
        assert _delta_pct({"a": 0, "b": 0}) == 0.0

    def test_with_na_sentinel(self) -> None:
        """N/A strings are filtered out — only numerics count."""
        assert _delta_pct({"a": "N/A", "b": 5}) == 0.0

    def test_mixed_na_and_numeric(self) -> None:
        """With one N/A and two numeric values, computes spread on the numerics."""
        result = _delta_pct({"a": "N/A", "b": 3, "c": 10})
        assert result == 70.0


# ── diff_snapshots tests ─────────────────────────────────────


class TestDiffSnapshots:
    """REQ-CMP-04: Diff engine severity classification."""

    def test_no_divergences(self) -> None:
        """SC-03: Identical snapshots produce no diffs."""
        snap1 = _make_snapshot("prod-us")
        snap2 = _make_snapshot("prod-eu")
        diffs = diff_snapshots({"prod-us": snap1, "prod-eu": snap2})
        assert diffs == []

    def test_image_tag_critical(self) -> None:
        """SC-04: Image mismatch is always critical."""
        snap1 = _make_snapshot("prod-us", image_tag="v2.1.0")
        snap2 = _make_snapshot("prod-eu", image_tag="v2.0.3")
        diffs = diff_snapshots({"prod-us": snap1, "prod-eu": snap2})
        image_diffs = [d for d in diffs if d.field == "image_tag"]
        assert len(image_diffs) == 1
        assert image_diffs[0].severity == "critical"
        assert image_diffs[0].values["prod-us"] == "v2.1.0"
        assert image_diffs[0].values["prod-eu"] == "v2.0.3"

    def test_replicas_desired_warning(self) -> None:
        """Small replica difference → warning (<=50% spread)."""
        snap1 = _make_snapshot("a", replicas_desired=3)
        snap2 = _make_snapshot("b", replicas_desired=4)
        diffs = diff_snapshots({"a": snap1, "b": snap2})
        rep_diffs = [d for d in diffs if d.field == "replicas_desired"]
        assert len(rep_diffs) == 1
        assert rep_diffs[0].severity == "warning"

    def test_replicas_desired_critical(self) -> None:
        """Large replica difference → critical (>50% spread)."""
        snap1 = _make_snapshot("a", replicas_desired=1)
        snap2 = _make_snapshot("b", replicas_desired=10)
        diffs = diff_snapshots({"a": snap1, "b": snap2})
        rep_diffs = [d for d in diffs if d.field == "replicas_desired"]
        assert len(rep_diffs) == 1
        assert rep_diffs[0].severity == "critical"

    def test_hpa_missing_on_one_cluster(self) -> None:
        """SC-05: Missing HPA on one cluster → N/A vs value, warning severity."""
        snap1 = _make_snapshot("prod-us", hpa_min=2, hpa_max=10)
        snap2 = _make_snapshot("prod-eu", hpa_min=None, hpa_max=None)
        diffs = diff_snapshots({"prod-us": snap1, "prod-eu": snap2})
        hpa_diffs = [d for d in diffs if d.field.startswith("hpa_")]
        assert len(hpa_diffs) == 2
        for hd in hpa_diffs:
            assert hd.severity == "warning"
            assert hd.values["prod-eu"] == "N/A"

    def test_single_snapshot_no_diffs(self) -> None:
        """Single cluster returns no diffs."""
        snap = _make_snapshot()
        assert diff_snapshots({"prod-us": snap}) == []

    def test_rollout_generation_info(self) -> None:
        """Rollout generation divergence is info severity."""
        snap1 = _make_snapshot("a", rollout_generation=5)
        snap2 = _make_snapshot("b", rollout_generation=7)
        diffs = diff_snapshots({"a": snap1, "b": snap2})
        gen_diffs = [d for d in diffs if d.field == "rollout_generation"]
        assert len(gen_diffs) == 1
        assert gen_diffs[0].severity == "info"


# ── collect_deployment_snapshot tests ─────────────────────────


class TestCollectDeploymentSnapshot:
    """REQ-CMP-02: Snapshot collection with mocked K8s clients."""

    def test_success_path(self) -> None:
        """Collect snapshot from a mocked K8s API."""
        from vaig.core.compare import collect_deployment_snapshot

        mock_gke_config = MagicMock()
        mock_gke_config.cluster_name = "gke-prod"
        mock_gke_config.request_timeout = 30

        # Mock _create_k8s_clients to return a tuple
        mock_core = MagicMock()
        mock_apps = MagicMock()
        mock_custom = MagicMock()
        mock_api_client = MagicMock()

        # Setup mock deployment response
        mock_dep = MagicMock()
        mock_dep.spec.replicas = 3
        mock_dep.spec.template.spec.containers = [MagicMock(image="gcr.io/proj/app:v1.2.3")]
        mock_dep.status.ready_replicas = 3
        mock_dep.status.observed_generation = 5
        mock_apps.read_namespaced_deployment.return_value = mock_dep

        with patch(
            "vaig.tools.gke._clients._create_k8s_clients",
            return_value=(mock_core, mock_apps, mock_custom, mock_api_client),
        ), patch("kubernetes.client.AutoscalingV1Api") as mock_autoscaling_cls:
            # Mock HPA — returns 404
            mock_autoscaler = MagicMock()
            mock_autoscaler.read_namespaced_horizontal_pod_autoscaler.side_effect = Exception("404 Not Found")
            mock_autoscaling_cls.return_value = mock_autoscaler

            snap = collect_deployment_snapshot(mock_gke_config, "default", "api")

        assert snap.cluster_name == "gke-prod"
        assert snap.replicas_desired == 3
        assert snap.replicas_ready == 3
        assert snap.image_tag == "v1.2.3"
        assert snap.rollout_generation == 5
        assert snap.hpa_min is None  # HPA 404

    def test_k8s_client_error(self) -> None:
        """ToolResult from _create_k8s_clients raises RuntimeError."""
        from vaig.core.compare import collect_deployment_snapshot
        from vaig.tools.base import ToolResult

        mock_gke_config = MagicMock()
        mock_gke_config.cluster_name = "gke-fail"

        with patch(
            "vaig.tools.gke._clients._create_k8s_clients",
            return_value=ToolResult(output="auth failed", error=True),
        ):
            try:
                collect_deployment_snapshot(mock_gke_config, "default", "api")
                msg = "Should have raised RuntimeError"
                raise AssertionError(msg)
            except RuntimeError as exc:
                assert "auth failed" in str(exc)


# ── CompareRunner tests ──────────────────────────────────────


class TestCompareRunner:
    """REQ-CMP-03: Runner parallel collection and error handling."""

    def test_run_parallel_all_success(self) -> None:
        """SC-01: All clusters succeed — report contains snapshots + diffs."""
        clusters = [
            FleetCluster(name="prod-us", cluster_name="gke-us"),
            FleetCluster(name="prod-eu", cluster_name="gke-eu"),
        ]
        settings = _make_settings_mock()

        snap_us = _make_snapshot("gke-us", image_tag="v2.1.0")
        snap_eu = _make_snapshot("gke-eu", image_tag="v2.0.3")

        def fake_collect(gke_config: Any, ns: str, dep: str) -> DeploymentSnapshot:
            if gke_config.cluster_name == "gke-us":
                return snap_us
            return snap_eu

        runner = CompareRunner(
            clusters=clusters,
            namespace="default",
            deployment="api",
            settings=settings,
        )

        with patch("vaig.core.compare.collect_deployment_snapshot", side_effect=fake_collect), \
             patch("vaig.core.gke.build_gke_config") as mock_build:
            # make build_gke_config return a mock with the right cluster_name
            def build_side_effect(settings: Any, cluster: str = "", **kw: Any) -> MagicMock:
                m = MagicMock()
                m.cluster_name = cluster
                return m
            mock_build.side_effect = build_side_effect

            report = runner.run_parallel()

        assert len(report.snapshots) == 2
        assert len(report.errors) == 0
        # Image divergence should produce a diff
        image_diffs = [d for d in report.diffs if d.field == "image_tag"]
        assert len(image_diffs) == 1
        assert image_diffs[0].severity == "critical"

    def test_run_parallel_partial_failure(self) -> None:
        """SC-02: One cluster fails — error captured, other succeeds."""
        clusters = [
            FleetCluster(name="prod-us", cluster_name="gke-us"),
            FleetCluster(name="staging", cluster_name="gke-staging"),
        ]
        settings = _make_settings_mock()

        snap_us = _make_snapshot("gke-us")

        call_count = 0

        def fake_collect(gke_config: Any, ns: str, dep: str) -> DeploymentSnapshot:
            nonlocal call_count
            call_count += 1
            if gke_config.cluster_name == "gke-staging":
                msg = "Connection refused"
                raise RuntimeError(msg)
            return snap_us

        runner = CompareRunner(
            clusters=clusters,
            namespace="default",
            deployment="api",
            settings=settings,
        )

        with patch("vaig.core.compare.collect_deployment_snapshot", side_effect=fake_collect), \
             patch("vaig.core.gke.build_gke_config") as mock_build:
            def build_side_effect(settings: Any, cluster: str = "", **kw: Any) -> MagicMock:
                m = MagicMock()
                m.cluster_name = cluster
                return m
            mock_build.side_effect = build_side_effect

            report = runner.run_parallel()

        assert len(report.snapshots) == 1
        assert "prod-us" in report.snapshots
        assert len(report.errors) == 1
        assert "staging" in report.errors
        assert "Connection refused" in report.errors["staging"]

    def test_run_parallel_config_order(self) -> None:
        """Snapshots are sorted by config order, not completion order."""
        clusters = [
            FleetCluster(name="alpha", cluster_name="gke-a"),
            FleetCluster(name="beta", cluster_name="gke-b"),
            FleetCluster(name="gamma", cluster_name="gke-g"),
        ]
        settings = _make_settings_mock()

        def fake_collect(gke_config: Any, ns: str, dep: str) -> DeploymentSnapshot:
            return _make_snapshot(gke_config.cluster_name)

        runner = CompareRunner(
            clusters=clusters,
            namespace="default",
            deployment="api",
            settings=settings,
        )

        with patch("vaig.core.compare.collect_deployment_snapshot", side_effect=fake_collect), \
             patch("vaig.core.gke.build_gke_config") as mock_build:
            def build_side_effect(settings: Any, cluster: str = "", **kw: Any) -> MagicMock:
                m = MagicMock()
                m.cluster_name = cluster
                return m
            mock_build.side_effect = build_side_effect

            report = runner.run_parallel()

        assert list(report.snapshots.keys()) == ["alpha", "beta", "gamma"]

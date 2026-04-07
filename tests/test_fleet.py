"""Tests for fleet scanning — config, runner, correlation, budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from vaig.core.config import FleetCluster, FleetConfig

# ── Fixtures ──────────────────────────────────────────────────


@dataclass
class _FakeFinding:
    """Minimal stand-in for Finding — avoids importing the full schema."""

    title: str = "CrashLoopBackOff"
    category: str = "pod-health"
    severity: str = "HIGH"


@dataclass
class _FakeHealthReport:
    """Minimal stand-in for HealthReport."""

    findings: list[_FakeFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"findings": [{"title": f.title, "category": f.category} for f in self.findings]}


@dataclass
class _FakeOrchestratorResult:
    """Minimal stand-in for OrchestratorResult."""

    skill_name: str = "discovery"
    success: bool = True
    run_cost_usd: float = 0.50
    structured_report: _FakeHealthReport | None = None
    synthesized_output: str = ""
    phase: str = "report"


def _make_cluster(name: str, cluster_name: str = "", **kwargs: Any) -> FleetCluster:
    """Helper to build FleetCluster with sensible defaults."""
    return FleetCluster(
        name=name,
        cluster_name=cluster_name or name,
        **kwargs,
    )


def _make_fleet_config(n_clusters: int = 3, **kwargs: Any) -> FleetConfig:
    """Helper to build FleetConfig with N mock clusters."""
    clusters = [_make_cluster(f"cluster-{i}", f"gke-{i}") for i in range(n_clusters)]
    return FleetConfig(clusters=clusters, **kwargs)


def _make_settings_mock() -> MagicMock:
    """Create a minimal mock Settings object."""
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


# ── T-14: Config model tests ─────────────────────────────────


class TestFleetConfigModels:
    """REQ-FLEET-01: FleetCluster + FleetConfig validation."""

    def test_fleet_cluster_defaults(self) -> None:
        fc = FleetCluster(name="prod", cluster_name="gke-prod")
        assert fc.name == "prod"
        assert fc.cluster_name == "gke-prod"
        assert fc.project_id == ""
        assert fc.location == ""
        assert fc.namespace == ""
        assert fc.all_namespaces is False
        assert fc.skip_healthy is True
        assert fc.kubeconfig_path == ""
        assert fc.context == ""
        assert fc.impersonate_sa == ""

    def test_fleet_config_auto_enable(self) -> None:
        """Auto-enables when clusters is non-empty."""
        config = FleetConfig(
            clusters=[FleetCluster(name="x", cluster_name="y")],
        )
        assert config.enabled is True

    def test_fleet_config_empty_clusters_stays_disabled(self) -> None:
        config = FleetConfig(clusters=[])
        assert config.enabled is False

    def test_fleet_config_defaults(self) -> None:
        config = FleetConfig()
        assert config.parallel is False
        assert config.max_workers == 4
        assert config.daily_budget_usd == 0.0

    def test_fleet_cluster_empty_project_id_accepted(self) -> None:
        """Empty project_id is valid — will inherit from gcp.project_id at runtime."""
        fc = FleetCluster(name="prod", cluster_name="gke-prod", project_id="")
        assert fc.project_id == ""

    def test_fleet_config_explicit_enable_false_overridden(self) -> None:
        """Even if enabled=False is set, auto-enable kicks in when clusters exist."""
        config = FleetConfig(
            enabled=False,
            clusters=[FleetCluster(name="x", cluster_name="y")],
        )
        assert config.enabled is True


# ── T-15: FleetRunner._scan_one tests ────────────────────────


class TestFleetRunnerScanOne:
    """REQ-FLEET-02, REQ-FLEET-08: Single cluster scan + error handling."""

    def test_scan_one_success(self) -> None:
        from vaig.core.fleet import FleetRunner

        fake_result = _FakeOrchestratorResult(
            structured_report=_FakeHealthReport(findings=[_FakeFinding()]),
        )

        runner = FleetRunner()
        settings = _make_settings_mock()
        cluster = _make_cluster("prod", "gke-prod")

        with patch("vaig.core.gke.build_gke_config") as mock_gke, \
             patch("vaig.skills.discovery.skill.DiscoverySkill") as mock_skill, \
             patch("vaig.core.discovery.build_discover_query", return_value="test query"), \
             patch("vaig.core.headless.execute_skill_headless", return_value=fake_result):
            mock_gke.return_value = MagicMock()
            mock_gke.return_value.model_copy = MagicMock(return_value=mock_gke.return_value)
            mock_skill.return_value = MagicMock()

            cr = runner._scan_one(settings, cluster)

        assert cr.status == "success"
        assert cr.display_name == "prod"
        assert cr.cost_usd == 0.50
        assert cr.result is not None

    def test_scan_one_error_captured(self) -> None:
        from vaig.core.fleet import FleetRunner

        runner = FleetRunner()
        settings = _make_settings_mock()
        cluster = _make_cluster("broken", "gke-broken")

        with patch("vaig.core.gke.build_gke_config") as mock_gke, \
             patch("vaig.skills.discovery.skill.DiscoverySkill") as mock_skill, \
             patch("vaig.core.discovery.build_discover_query", return_value="test query"), \
             patch("vaig.core.headless.execute_skill_headless", side_effect=RuntimeError("Connection refused")):
            mock_gke.return_value = MagicMock()
            mock_gke.return_value.model_copy = MagicMock(return_value=mock_gke.return_value)
            mock_skill.return_value = MagicMock()

            cr = runner._scan_one(settings, cluster)

        assert cr.status == "error"
        assert "Connection refused" in (cr.error or "")
        assert cr.result is None

    def test_scan_one_gke_config_inherits_project(self) -> None:
        from vaig.core.fleet import FleetRunner

        fake_result = _FakeOrchestratorResult()

        runner = FleetRunner()
        settings = _make_settings_mock()
        cluster = _make_cluster("prod", "gke-prod", project_id="")

        with patch("vaig.core.gke.build_gke_config") as mock_gke, \
             patch("vaig.skills.discovery.skill.DiscoverySkill"), \
             patch("vaig.core.discovery.build_discover_query", return_value="q"), \
             patch("vaig.core.headless.execute_skill_headless", return_value=fake_result):
            mock_gke.return_value = MagicMock()
            mock_gke.return_value.model_copy = MagicMock(return_value=mock_gke.return_value)

            runner._scan_one(settings, cluster)

        # project_id=None (empty string → None via `or None`)
        mock_gke.assert_called_once()
        call_kwargs = mock_gke.call_args
        assert call_kwargs.kwargs.get("project_id") is None


# ── T-16: Sequential + Parallel run tests ────────────────────


class TestFleetRunnerRun:
    """REQ-FLEET-02, REQ-FLEET-03, SC-01, SC-02."""

    def test_sequential_run_three_clusters(self) -> None:
        from vaig.core.fleet import ClusterResult, FleetRunner

        runner = FleetRunner()
        results = [
            ClusterResult(cluster_name=f"gke-{i}", display_name=f"cluster-{i}", status="success", cost_usd=0.5, duration_s=1.0)
            for i in range(3)
        ]

        with patch.object(runner, "_scan_one", side_effect=results):
            settings = _make_settings_mock()
            config = _make_fleet_config(3)
            report = runner.run(settings, config)

        assert len(report.clusters) == 3
        # Verify order preserved
        assert [cr.display_name for cr in report.clusters] == [
            "cluster-0", "cluster-1", "cluster-2"
        ]

    def test_sequential_run_partial_failure(self) -> None:
        from vaig.core.fleet import ClusterResult, FleetRunner

        runner = FleetRunner()
        results = [
            ClusterResult(cluster_name="gke-0", display_name="cluster-0", status="success", cost_usd=0.5),
            ClusterResult(cluster_name="gke-1", display_name="cluster-1", status="error", error="timeout"),
            ClusterResult(cluster_name="gke-2", display_name="cluster-2", status="success", cost_usd=0.5),
        ]

        with patch.object(runner, "_scan_one", side_effect=results):
            settings = _make_settings_mock()
            config = _make_fleet_config(3)
            report = runner.run(settings, config)

        assert len(report.clusters) == 3
        assert report.clusters[1].status == "error"

    def test_parallel_run_all_complete(self) -> None:
        from vaig.core.fleet import ClusterResult, FleetRunner

        runner = FleetRunner()

        def fake_scan_one(_settings: Any, cluster: Any) -> ClusterResult:
            return ClusterResult(
                cluster_name=cluster.cluster_name,
                display_name=cluster.name,
                status="success",
                cost_usd=0.5,
                duration_s=0.1,
            )

        with patch.object(runner, "_scan_one", side_effect=fake_scan_one):
            settings = _make_settings_mock()
            config = _make_fleet_config(3, parallel=True, max_workers=2)
            report = runner.run_parallel(settings, config)

        assert len(report.clusters) == 3
        assert all(cr.status == "success" for cr in report.clusters)


# ── T-17: Correlation tests ──────────────────────────────────


class TestCorrelation:
    """REQ-FLEET-04, SC-03: Cross-cluster correlation."""

    def test_two_clusters_share_finding(self) -> None:
        from vaig.core.fleet import ClusterResult, correlate

        results = [
            ClusterResult(
                cluster_name="gke-0",
                display_name="prod-us",
                status="success",
                result=_FakeOrchestratorResult(
                    structured_report=_FakeHealthReport(
                        findings=[_FakeFinding(title="ImagePullBackOff", category="pod-health")]
                    ),
                ),
            ),
            ClusterResult(
                cluster_name="gke-1",
                display_name="prod-eu",
                status="success",
                result=_FakeOrchestratorResult(
                    structured_report=_FakeHealthReport(
                        findings=[_FakeFinding(title="ImagePullBackOff", category="pod-health")]
                    ),
                ),
            ),
            ClusterResult(
                cluster_name="gke-2",
                display_name="staging",
                status="success",
                result=_FakeOrchestratorResult(
                    structured_report=_FakeHealthReport(
                        findings=[_FakeFinding(title="OOMKilled", category="resource")]
                    ),
                ),
            ),
        ]

        correlations = correlate(results)
        assert len(correlations) == 1
        assert correlations[0].pattern == "imagepullbackoff"
        assert len(correlations[0].affected_clusters) == 2
        assert set(correlations[0].affected_clusters) == {"prod-us", "prod-eu"}

    def test_no_findings_no_correlations(self) -> None:
        from vaig.core.fleet import ClusterResult, correlate

        results = [
            ClusterResult(
                cluster_name="gke-0",
                display_name="empty",
                status="success",
                result=_FakeOrchestratorResult(structured_report=_FakeHealthReport(findings=[])),
            ),
        ]
        assert correlate(results) == []

    def test_all_unique_findings_no_correlations(self) -> None:
        from vaig.core.fleet import ClusterResult, correlate

        results = [
            ClusterResult(
                cluster_name=f"gke-{i}",
                display_name=f"cluster-{i}",
                status="success",
                result=_FakeOrchestratorResult(
                    structured_report=_FakeHealthReport(
                        findings=[_FakeFinding(title=f"Unique-{i}", category="misc")]
                    ),
                ),
            )
            for i in range(3)
        ]
        assert correlate(results) == []

    def test_error_clusters_excluded(self) -> None:
        from vaig.core.fleet import ClusterResult, correlate

        results = [
            ClusterResult(cluster_name="gke-0", display_name="ok", status="success",
                          result=_FakeOrchestratorResult(
                              structured_report=_FakeHealthReport(
                                  findings=[_FakeFinding(title="X", category="c")]
                              ))),
            ClusterResult(cluster_name="gke-1", display_name="err", status="error"),
        ]
        # Only 1 cluster with findings → no correlations
        assert correlate(results) == []


# ── T-18: Budget tests ───────────────────────────────────────


class TestBudget:
    """REQ-FLEET-09, SC-04: Budget enforcement."""

    def test_budget_exceeded_skips_remaining(self) -> None:
        from vaig.core.fleet import ClusterResult, FleetRunner

        runner = FleetRunner()
        call_count = 0

        def fake_scan(_settings: Any, cluster: Any) -> ClusterResult:
            nonlocal call_count
            call_count += 1
            return ClusterResult(
                cluster_name=cluster.cluster_name,
                display_name=cluster.name,
                status="success",
                cost_usd=0.50,  # $0.50 per cluster
                duration_s=1.0,
            )

        with patch.object(runner, "_scan_one", side_effect=fake_scan):
            settings = _make_settings_mock()
            config = _make_fleet_config(3, daily_budget_usd=1.0)
            report = runner.run(settings, config)

        # Budget is $1.00, each cluster costs $0.50
        # cluster-0: cumulative $0 < $1.00 → scan → $0.50
        # cluster-1: cumulative $0.50 < $1.00 → scan → $1.00
        # cluster-2: cumulative $1.00 >= $1.00 → SKIP
        assert call_count == 2
        assert report.budget_exceeded is True
        assert report.clusters[2].status == "skipped"
        assert report.clusters[2].error == "Budget exceeded"

    def test_no_budget_all_clusters_scanned(self) -> None:
        from vaig.core.fleet import ClusterResult, FleetRunner

        runner = FleetRunner()

        def fake_scan(_settings: Any, cluster: Any) -> ClusterResult:
            return ClusterResult(
                cluster_name=cluster.cluster_name,
                display_name=cluster.name,
                status="success",
                cost_usd=0.50,
                duration_s=1.0,
            )

        with patch.object(runner, "_scan_one", side_effect=fake_scan):
            settings = _make_settings_mock()
            config = _make_fleet_config(3, daily_budget_usd=0.0)
            report = runner.run(settings, config)

        assert len(report.clusters) == 3
        assert all(cr.status == "success" for cr in report.clusters)
        assert report.budget_exceeded is False

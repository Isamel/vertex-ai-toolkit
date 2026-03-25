"""Tests for Argo Rollouts ownership awareness in get_rollout_status() and prompts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Helpers ──────────────────────────────────────────────────


def _make_deployment(
    name: str = "my-svc",
    namespace: str = "production",
    spec_replicas: int | None = 0,
    ready_replicas: int = 0,
    annotations: dict | None = None,
) -> MagicMock:
    """Build a minimal mock Deployment object."""
    dep = MagicMock()
    dep.metadata.name = name
    dep.metadata.namespace = namespace
    dep.metadata.annotations = annotations or {}
    dep.spec.replicas = spec_replicas
    dep.spec.strategy = None
    status = MagicMock()
    status.replicas = 0
    status.ready_replicas = ready_replicas
    status.updated_replicas = 0
    status.available_replicas = 0
    status.unavailable_replicas = 0
    status.conditions = []
    dep.status = status
    return dep


def _make_gke_config(argo_rollouts_enabled: bool | None = True) -> MagicMock:
    cfg = MagicMock()
    cfg.argo_rollouts_enabled = argo_rollouts_enabled
    cfg.default_namespace = "default"
    return cfg


def _make_rollout_obj(name: str = "my-svc", namespace: str = "production") -> dict:
    return {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"replicas": 3, "strategy": {"canary": {}}},
        "status": {
            "phase": "Healthy",
            "readyReplicas": 3,
            "availableReplicas": 3,
            "updatedReplicas": 3,
            "canary": {"currentStepIndex": 0, "weights": {"canary": {"weight": 0}}},
        },
    }


# ── Tests: get_rollout_status() ───────────────────────────────


class TestGetRolloutStatusArgoAwareness:
    """Gap 1: get_rollout_status() must detect Argo-managed deployments."""

    def test_argo_annotations_spec_replicas_zero_returns_managed_state(self) -> None:
        """Deployment with Argo annotation + spec.replicas==0 → 'Managed by Argo Rollout'."""
        dep = _make_deployment(
            spec_replicas=0,
            ready_replicas=0,
            annotations={"rollout.argoproj.io/revision": "5"},
        )
        gke_config = _make_gke_config(argo_rollouts_enabled=False)

        with (
            patch("vaig.tools.gke.diagnostics._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.diagnostics._clients._create_k8s_clients") as mock_clients,
        ):
            mock_apps = MagicMock()
            mock_apps.read_namespaced_deployment.return_value = dep
            mock_custom = MagicMock()
            mock_clients.return_value = (MagicMock(), mock_apps, mock_custom, MagicMock())

            from vaig.tools.gke.diagnostics import get_rollout_status

            result = get_rollout_status("my-svc", gke_config=gke_config, namespace="production")

        assert not result.error
        assert "Managed by Argo Rollout" in result.output
        assert "Scaled to zero" not in result.output

    def test_no_argo_annotations_spec_replicas_zero_returns_scaled_to_zero(self) -> None:
        """Deployment with NO Argo annotations + spec.replicas==0 → 'Scaled to zero'."""
        dep = _make_deployment(
            spec_replicas=0,
            ready_replicas=0,
            annotations={},
        )
        gke_config = _make_gke_config(argo_rollouts_enabled=False)

        with (
            patch("vaig.tools.gke.diagnostics._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.diagnostics._clients._create_k8s_clients") as mock_clients,
        ):
            mock_apps = MagicMock()
            mock_apps.read_namespaced_deployment.return_value = dep
            mock_custom = MagicMock()
            mock_clients.return_value = (MagicMock(), mock_apps, mock_custom, MagicMock())

            from vaig.tools.gke.diagnostics import get_rollout_status

            result = get_rollout_status("my-svc", gke_config=gke_config, namespace="production")

        assert not result.error
        assert "Scaled to zero" in result.output
        assert "Managed by Argo Rollout" not in result.output

    def test_argo_annotations_with_successful_rollout_crossref(self) -> None:
        """Argo-managed deployment with argo_rollouts_enabled → includes Rollout details."""
        dep = _make_deployment(
            spec_replicas=0,
            ready_replicas=0,
            annotations={"rollout.argoproj.io/desired-replicas": "3"},
        )
        gke_config = _make_gke_config(argo_rollouts_enabled=True)
        rollout_obj = _make_rollout_obj()

        with (
            patch("vaig.tools.gke.diagnostics._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.diagnostics._clients._create_k8s_clients") as mock_clients,
        ):
            mock_apps = MagicMock()
            mock_apps.read_namespaced_deployment.return_value = dep
            mock_custom = MagicMock()
            mock_custom.get_namespaced_custom_object.return_value = rollout_obj
            mock_clients.return_value = (MagicMock(), mock_apps, mock_custom, MagicMock())

            from vaig.tools.gke.diagnostics import get_rollout_status

            result = get_rollout_status("my-svc", gke_config=gke_config, namespace="production")

        assert not result.error
        assert "Managed by Argo Rollout" in result.output
        # Cross-reference section should be present
        assert "Argo Rollout details" in result.output
        # _format_rollout output should include phase
        assert "Healthy" in result.output

    def test_argo_annotations_rollout_crossref_with_none_default(self) -> None:
        """argo_rollouts_enabled=None (default/auto-detect) still triggers Rollout cross-reference."""
        dep = _make_deployment(
            spec_replicas=0,
            ready_replicas=0,
            annotations={"rollout.argoproj.io/desired-replicas": "3"},
        )
        gke_config = _make_gke_config(argo_rollouts_enabled=None)
        rollout_obj = _make_rollout_obj()

        with (
            patch("vaig.tools.gke.diagnostics._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.diagnostics._clients._create_k8s_clients") as mock_clients,
        ):
            mock_apps = MagicMock()
            mock_apps.read_namespaced_deployment.return_value = dep
            mock_custom = MagicMock()
            mock_custom.get_namespaced_custom_object.return_value = rollout_obj
            mock_clients.return_value = (MagicMock(), mock_apps, mock_custom, MagicMock())

            from vaig.tools.gke.diagnostics import get_rollout_status

            result = get_rollout_status("my-svc", gke_config=gke_config, namespace="production")

        assert not result.error
        assert "Managed by Argo Rollout" in result.output
        # Cross-reference section should be present even when argo_rollouts_enabled=None
        assert "Argo Rollout details" in result.output
        assert "Healthy" in result.output

    def test_argo_annotations_rollout_fetch_fails_graceful_degrade(self) -> None:
        """Argo-managed deployment + Rollout fetch fails → still returns 'Managed by Argo Rollout'."""
        from kubernetes.client.exceptions import ApiException  # type: ignore[import-untyped]

        dep = _make_deployment(
            spec_replicas=0,
            ready_replicas=0,
            annotations={"rollout.argoproj.io/workload-generation": "2"},
        )
        gke_config = _make_gke_config(argo_rollouts_enabled=True)

        with (
            patch("vaig.tools.gke.diagnostics._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.diagnostics._clients._create_k8s_clients") as mock_clients,
        ):
            mock_apps = MagicMock()
            mock_apps.read_namespaced_deployment.return_value = dep
            mock_custom = MagicMock()
            mock_custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
            mock_clients.return_value = (MagicMock(), mock_apps, mock_custom, MagicMock())

            from vaig.tools.gke.diagnostics import get_rollout_status

            result = get_rollout_status("my-svc", gke_config=gke_config, namespace="production")

        assert not result.error
        assert "Managed by Argo Rollout" in result.output
        # Should NOT have error or crash
        assert "Error" not in result.output
        assert "Argo Rollout details" not in result.output

    def test_managed_by_rollouts_annotation_triggers_argo_state(self) -> None:
        """argo-rollouts.argoproj.io/managed-by-rollouts annotation also triggers Argo state."""
        dep = _make_deployment(
            spec_replicas=0,
            ready_replicas=0,
            annotations={"argo-rollouts.argoproj.io/managed-by-rollouts": "true"},
        )
        gke_config = _make_gke_config(argo_rollouts_enabled=False)

        with (
            patch("vaig.tools.gke.diagnostics._clients._K8S_AVAILABLE", True),
            patch("vaig.tools.gke.diagnostics._clients._create_k8s_clients") as mock_clients,
        ):
            mock_apps = MagicMock()
            mock_apps.read_namespaced_deployment.return_value = dep
            mock_custom = MagicMock()
            mock_clients.return_value = (MagicMock(), mock_apps, mock_custom, MagicMock())

            from vaig.tools.gke.diagnostics import get_rollout_status

            result = get_rollout_status("my-svc", gke_config=gke_config, namespace="production")

        assert not result.error
        assert "Managed by Argo Rollout" in result.output


# ── Tests: prompts ────────────────────────────────────────────


class TestWorkloadGathererPromptArgoPart:
    """Gap 2: build_workload_gatherer_prompt() Step 4d must include stub-deployment guidance."""

    def test_argo_enabled_includes_stub_warning(self) -> None:
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="production",
            datadog_api_enabled=False,
            argo_rollouts_enabled=True,
        )
        assert "Managed by Argo Rollout" in prompt
        assert "MUST NOT be reported as" in prompt or "NEVER flag it" in prompt

    def test_argo_disabled_excludes_step4d(self) -> None:
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="production",
            datadog_api_enabled=False,
            argo_rollouts_enabled=False,
        )
        assert "Step 4d" not in prompt

    def test_argo_enabled_includes_pod_ownership_guidance(self) -> None:
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="production",
            datadog_api_enabled=False,
            argo_rollouts_enabled=True,
        )
        # Pod ownership tracing guidance
        assert "ownerReferences" in prompt or "ownership" in prompt.lower()


class TestHealthAnalyzerPromptArgoPart:
    """Gap 3: HEALTH_ANALYZER_PROMPT must contain Argo Rollouts ownership rules."""

    def test_contains_argo_rollouts_section(self) -> None:
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "Argo Rollouts Ownership Rules" in HEALTH_ANALYZER_PROMPT

    def test_no_scaled_to_zero_false_alarm_rule(self) -> None:
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "NEVER flag it as" in HEALTH_ANALYZER_PROMPT
        assert "scaled to zero" in HEALTH_ANALYZER_PROMPT.lower()

    def test_no_orphaned_pod_false_alarm_rule(self) -> None:
        from vaig.skills.service_health.prompts import HEALTH_ANALYZER_PROMPT

        assert "orphaned" in HEALTH_ANALYZER_PROMPT.lower()
        assert "NEVER flag these pods" in HEALTH_ANALYZER_PROMPT

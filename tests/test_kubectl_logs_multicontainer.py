"""Tests for kubectl_logs multi-container pod handling (Bug 3).

Covers:
- Auto-detection and retry when a single non-sidecar container is found
- Helpful error message when multiple app containers exist
- Known sidecar prefixes are filtered out (istio-, datadog-, linkerd-, envoy-)
- Init containers (-init-) are filtered out
- Existing CrashLoopBackOff path is unaffected
- Non-400 exceptions are unaffected
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kubernetes.client.exceptions import ApiException

from vaig.tools.gke import kubectl

# ── Helpers ──────────────────────────────────────────────────


def _make_gke_config() -> MagicMock:
    """Return a minimal GKEConfig mock."""
    cfg = MagicMock()
    cfg.project = "my-project"
    cfg.location = "us-central1"
    cfg.cluster = "my-cluster"
    cfg.default_namespace = "default"
    cfg.log_limit = None
    return cfg


def _api_exception(status: int, body: str) -> ApiException:
    exc = ApiException(status=status, reason="Bad Request")
    exc.body = body
    return exc


def _multi_container_body(*containers: str) -> str:
    """Simulate the Kubernetes API error body for multi-container pods."""
    names = " ".join(containers)
    return (
        f'{{"message": "a container name must be specified for pod my-pod, '
        f'choose one of: [{names}]"}}'
    )


def _make_clients(core_v1: MagicMock | None = None) -> tuple:
    """Return a 4-tuple of mocks as returned by _create_k8s_clients."""
    if core_v1 is None:
        core_v1 = MagicMock()
    return (core_v1, MagicMock(), MagicMock(), MagicMock())


# ── Tests ────────────────────────────────────────────────────


@patch("vaig.tools.gke.kubectl._clients._K8S_AVAILABLE", True)
@patch("vaig.tools.gke.kubectl._clients._create_k8s_clients")
class TestMultiContainerAutoDetect:
    """Single app container → auto-detect, retry, return logs."""

    def test_single_app_container_retries_and_returns_logs(
        self, mock_create: MagicMock
    ) -> None:
        """When only one non-sidecar container exists, logs are fetched automatically."""
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("istio-proxy", "my-app")),
            "hello from my-app",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "[container: my-app]" in result.output
        assert "hello from my-app" in result.output
        assert core_v1.read_namespaced_pod_log.call_count == 2

    def test_auto_detect_filters_istio_sidecar(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("istio-proxy", "worker")),
            "worker logs",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "[container: worker]" in result.output

    def test_auto_detect_filters_datadog_sidecar(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("datadog-agent", "api")),
            "api logs",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "[container: api]" in result.output

    def test_auto_detect_filters_linkerd_sidecar(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("linkerd-proxy", "backend")),
            "backend logs",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "[container: backend]" in result.output

    def test_auto_detect_filters_envoy_sidecar(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("envoy", "frontend")),
            "frontend logs",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "[container: frontend]" in result.output

    def test_auto_detect_filters_init_container(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("db-init-container", "app")),
            "app logs",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "[container: app]" in result.output

    def test_empty_logs_on_auto_detect_returns_no_logs_message(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("istio-proxy", "app")),
            "",  # empty logs
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "no logs available" in result.output
        assert "app" in result.output

    def test_retry_api_exception_falls_through_to_error_message(
        self, mock_create: MagicMock
    ) -> None:
        """If the retry itself raises ApiException, fall through to the list message."""
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, _multi_container_body("istio-proxy", "app")),
            _api_exception(404, "container not found"),
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert result.error
        assert "my-pod has multiple containers" in result.output


@patch("vaig.tools.gke.kubectl._clients._K8S_AVAILABLE", True)
@patch("vaig.tools.gke.kubectl._clients._create_k8s_clients")
class TestMultiContainerAmbiguous:
    """Multiple app containers → return helpful error listing containers."""

    def test_multiple_app_containers_returns_helpful_error(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = _api_exception(
            400, _multi_container_body("api", "worker", "scheduler")
        )

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert result.error
        assert "my-pod has multiple containers" in result.output
        assert "api" in result.output
        assert "worker" in result.output
        assert "scheduler" in result.output
        assert "Retry with container= parameter" in result.output

    def test_only_sidecars_reports_unknown_app_containers(
        self, mock_create: MagicMock
    ) -> None:
        """Edge case: all containers are sidecars, app_containers list is empty."""
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = _api_exception(
            400, _multi_container_body("istio-proxy", "datadog-agent")
        )

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert result.error
        assert "unknown" in result.output

    def test_sidecars_plus_two_app_containers_returns_likely_list(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = _api_exception(
            400, _multi_container_body("istio-proxy", "api", "worker")
        )

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert result.error
        assert "api" in result.output
        assert "worker" in result.output
        assert "Likely app containers" in result.output


@patch("vaig.tools.gke.kubectl._clients._K8S_AVAILABLE", True)
@patch("vaig.tools.gke.kubectl._clients._create_k8s_clients")
class TestExistingBehaviorUnaffected:
    """Existing CrashLoopBackOff and non-400 paths must not regress."""

    def test_crash_loop_backoff_still_retries_previous(
        self, mock_create: MagicMock
    ) -> None:
        """A 400 without 'container name must be specified' still triggers CrashLoopBackOff path."""
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        crash_body = '{"message": "container is not running"}'
        core_v1.read_namespaced_pod_log.side_effect = [
            _api_exception(400, crash_body),
            "previous container logs",
        ]

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert "previous container logs" in result.output

    def test_404_returns_not_found_error(self, mock_create: MagicMock) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = _api_exception(404, "not found")

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert result.error
        assert "not found" in result.output.lower()

    def test_403_returns_access_denied(self, mock_create: MagicMock) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.side_effect = _api_exception(403, "forbidden")

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert result.error
        assert "Access denied" in result.output

    def test_successful_single_container_returns_logs(
        self, mock_create: MagicMock
    ) -> None:
        core_v1 = MagicMock()
        mock_create.return_value = _make_clients(core_v1)
        core_v1.read_namespaced_pod_log.return_value = "normal log output"

        result = kubectl.kubectl_logs("my-pod", gke_config=_make_gke_config())

        assert not result.error
        assert result.output == "normal log output"

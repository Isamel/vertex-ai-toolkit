"""Unit tests for SH-07: _detect_argocd / _detect_argo_rollouts helpers.

Verifies the 3-state flag resolution logic without making any live k8s calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_skill() -> object:
    """Instantiate ServiceHealthSkill with minimal mocking."""
    from vaig.skills.service_health.skill import ServiceHealthSkill

    return ServiceHealthSkill()


def _make_gke_config(argocd_enabled=None, argo_rollouts_enabled=None):
    cfg = MagicMock()
    cfg.argocd_enabled = argocd_enabled
    cfg.argo_rollouts_enabled = argo_rollouts_enabled
    return cfg


# ── _detect_argocd tests ──────────────────────────────────────────────────────


class TestDetectArgocd:
    def test_false_setting_returns_false_no_probe(self):
        """T-SH07-01: Returns False immediately when setting is False."""
        skill = _make_skill()
        gke = _make_gke_config(argocd_enabled=False)

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            result = skill._detect_argocd("my-ns", gke)

        assert result is False
        mock_create.assert_not_called()

    def test_true_setting_returns_true_no_probe(self):
        """T-SH07-02: Returns True immediately when setting is True."""
        skill = _make_skill()
        gke = _make_gke_config(argocd_enabled=True)

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            result = skill._detect_argocd("my-ns", gke)

        assert result is True
        mock_create.assert_not_called()

    def test_none_setting_calls_detect_argocd_and_returns_result(self):
        """T-SH07-03: Calls detect_argocd with namespace when setting is None."""
        skill = _make_skill()
        gke = _make_gke_config(argocd_enabled=None)

        fake_api_client = MagicMock()
        fake_clients = (None, None, None, fake_api_client)  # index 3 = ApiClient

        with (
            patch(
                "vaig.tools.gke._clients._create_k8s_clients",
                return_value=fake_clients,
            ),
            patch(
                "vaig.tools.gke.argocd.detect_argocd",
                return_value=True,
            ) as mock_detect,
        ):
            result = skill._detect_argocd("prod-ns", gke)

        assert result is True
        mock_detect.assert_called_once_with(namespace="prod-ns", api_client=fake_api_client)

    def test_none_setting_passes_none_api_client_when_k8s_fails(self):
        """T-SH07-04: Passes api_client=None when k8s client creation returns ToolResult."""
        from vaig.tools.base import ToolResult

        skill = _make_skill()
        gke = _make_gke_config(argocd_enabled=None)
        tool_result_error = ToolResult(output="error", error=True)

        with (
            patch(
                "vaig.tools.gke._clients._create_k8s_clients",
                return_value=tool_result_error,
            ),
            patch(
                "vaig.tools.gke.argocd.detect_argocd",
                return_value=False,
            ) as mock_detect,
        ):
            result = skill._detect_argocd("prod-ns", gke)

        assert result is False
        mock_detect.assert_called_once_with(namespace="prod-ns", api_client=None)


# ── _detect_argo_rollouts tests ───────────────────────────────────────────────


class TestDetectArgoRollouts:
    def test_false_setting_returns_false_no_probe(self):
        """T-SH07-05: Returns False immediately when setting is False."""
        skill = _make_skill()
        gke = _make_gke_config(argo_rollouts_enabled=False)

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            result = skill._detect_argo_rollouts("my-ns", gke)

        assert result is False
        mock_create.assert_not_called()

    def test_true_setting_returns_true_no_probe(self):
        """T-SH07-06: Returns True immediately when setting is True."""
        skill = _make_skill()
        gke = _make_gke_config(argo_rollouts_enabled=True)

        with patch("vaig.tools.gke._clients._create_k8s_clients") as mock_create:
            result = skill._detect_argo_rollouts("my-ns", gke)

        assert result is True
        mock_create.assert_not_called()

    def test_none_setting_calls_detect_argo_rollouts_with_namespace(self):
        """T-SH07-07: Calls detect_argo_rollouts with correct namespace when setting is None."""
        skill = _make_skill()
        gke = _make_gke_config(argo_rollouts_enabled=None)

        fake_api_client = MagicMock()
        fake_clients = (None, None, None, fake_api_client)

        with (
            patch(
                "vaig.tools.gke._clients._create_k8s_clients",
                return_value=fake_clients,
            ),
            patch(
                "vaig.tools.gke.argo_rollouts.detect_argo_rollouts",
                return_value=True,
            ) as mock_detect,
        ):
            result = skill._detect_argo_rollouts("staging-ns", gke)

        assert result is True
        mock_detect.assert_called_once_with(namespace="staging-ns", api_client=fake_api_client)

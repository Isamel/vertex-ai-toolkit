"""Tests for SPEC-SH-10 — Resource-triggered mandatory tools section."""
from __future__ import annotations

from vaig.skills.service_health.prompts._shared import _build_mandatory_tools_section


class TestBuildMandatoryToolsSection:
    def test_empty_inputs_returns_empty_string(self) -> None:
        assert _build_mandatory_tools_section() == ""
        assert _build_mandatory_tools_section(query="", namespace="") == ""

    def test_no_pattern_match_returns_empty_string(self) -> None:
        assert _build_mandatory_tools_section(query="check pod status", namespace="default") == ""

    # ── istio / mesh ──────────────────────────────────────────────────────────

    def test_istio_query_triggers_istio_tools(self) -> None:
        result = _build_mandatory_tools_section(query="istio sidecar crash")
        assert "istio_get_virtual_services" in result
        assert "check_mtls_status" in result

    def test_mesh_keyword_triggers_istio_tools(self) -> None:
        result = _build_mandatory_tools_section(query="service mesh issue")
        assert "istio_get_virtual_services" in result

    def test_virtualservice_namespace_triggers_istio_tools(self) -> None:
        result = _build_mandatory_tools_section(namespace="istio-system")
        assert "istio_get_virtual_services" in result

    # ── argocd / gitops ───────────────────────────────────────────────────────

    def test_argocd_query_triggers_argocd_tools(self) -> None:
        result = _build_mandatory_tools_section(query="argocd app out of sync")
        assert "argocd_list_applications" in result
        assert "argocd_app_status" in result

    def test_gitops_keyword_triggers_argocd_tools(self) -> None:
        result = _build_mandatory_tools_section(query="gitops workflow")
        assert "argocd_list_applications" in result

    # ── helm ──────────────────────────────────────────────────────────────────

    def test_helm_query_triggers_helm_tools(self) -> None:
        result = _build_mandatory_tools_section(query="helm chart upgrade failed")
        assert "helm_list_releases" in result
        assert "helm_release_status" in result
        assert "helm_release_history" in result

    def test_chart_keyword_triggers_helm_tools(self) -> None:
        result = _build_mandatory_tools_section(query="chart rollback")
        assert "helm_list_releases" in result

    # ── argo rollouts ──────────────────────────────────────────────────────────

    def test_rollout_query_triggers_rollout_tools(self) -> None:
        result = _build_mandatory_tools_section(query="canary rollout stuck")
        assert "get_rollout_status" in result
        assert "get_rollout_history" in result
        assert "kubectl_get_analysisrun" in result

    def test_bluegreen_triggers_rollout_tools(self) -> None:
        result = _build_mandatory_tools_section(query="blue-green deployment")
        assert "get_rollout_status" in result

    def test_argo_rollout_namespace_triggers_tools(self) -> None:
        result = _build_mandatory_tools_section(namespace="argo-rollout")
        assert "get_rollout_status" in result

    # ── multiple patterns ─────────────────────────────────────────────────────

    def test_multiple_patterns_return_all_tools(self) -> None:
        result = _build_mandatory_tools_section(query="istio mesh with helm chart")
        assert "istio_get_virtual_services" in result
        assert "helm_list_releases" in result

    def test_result_contains_mandatory_header(self) -> None:
        result = _build_mandatory_tools_section(query="argocd gitops")
        assert "MANDATORY TOOLS" in result

    def test_result_contains_must_call_instruction(self) -> None:
        result = _build_mandatory_tools_section(query="istio mesh")
        assert "You MUST call" in result

    def test_case_insensitive_matching(self) -> None:
        result_lower = _build_mandatory_tools_section(query="ISTIO")
        result_upper = _build_mandatory_tools_section(query="istio")
        assert result_lower == result_upper

    def test_only_namespace_match_returns_section(self) -> None:
        """Pattern can match on namespace alone even if query is empty."""
        result = _build_mandatory_tools_section(query="", namespace="istio-system")
        assert "istio_get_virtual_services" in result

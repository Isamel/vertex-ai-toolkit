"""Tests for autopilot-aware-analysis feature.

Verifies that:
- ``build_node_gatherer_prompt(is_autopilot=False)`` returns the standard 6-step prompt
- ``build_node_gatherer_prompt(is_autopilot=True)`` returns a lightweight 2-tool-call prompt
- The Autopilot prompt prohibits ``get_node_conditions`` and ``kubectl_top``
- The Standard prompt includes ``get_node_conditions`` and ``kubectl_top``
- ``build_autopilot_instruction()`` returns the updated directive text
- The Autopilot instruction contains "CONTEXT ONLY", "NotReady", and "WORKLOAD-LEVEL"
- The default value of ``is_autopilot`` is ``False``
- ``_query_autopilot_status`` passes ``timeout=_AUTOPILOT_TIMEOUT`` to ``get_cluster``
- ``detect_autopilot`` returns ``False`` on ``DeadlineExceeded`` timeout
- ``detect_autopilot`` returns ``None`` on generic errors (unchanged behaviour)
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch


class TestBuildNodeGathererPromptStandard:
    """Tests for the standard (non-Autopilot) path of build_node_gatherer_prompt."""

    def test_default_returns_full_standard_prompt(self) -> None:
        """Calling with no args (is_autopilot=False default) returns the standard prompt."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_false_returns_full_standard_prompt(self) -> None:
        """Calling with is_autopilot=False explicitly returns the standard prompt."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert isinstance(result, str)
        assert len(result) > 200

    def test_standard_prompt_contains_get_node_conditions(self) -> None:
        """Standard prompt MUST include get_node_conditions (6-step investigation)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert "get_node_conditions" in result

    def test_standard_prompt_contains_kubectl_top(self) -> None:
        """Standard prompt MUST include kubectl_top for node utilisation."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert "kubectl_top" in result

    def test_standard_prompt_cluster_overview_header(self) -> None:
        """Standard prompt output section must be ## Cluster Overview (downstream dependency)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=False)
        assert "## Cluster Overview" in result

    def test_standard_default_equals_false(self) -> None:
        """build_node_gatherer_prompt() and build_node_gatherer_prompt(False) must be identical."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        assert build_node_gatherer_prompt() == build_node_gatherer_prompt(is_autopilot=False)


class TestBuildNodeGathererPromptAutopilot:
    """Tests for the Autopilot-specific path of build_node_gatherer_prompt."""

    def test_autopilot_returns_lightweight_prompt(self) -> None:
        """Calling with is_autopilot=True returns a prompt (non-empty string)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_autopilot_prompt_prohibits_get_node_conditions(self) -> None:
        """Autopilot prompt MUST explicitly prohibit get_node_conditions per node."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        # The Autopilot prompt mentions get_node_conditions only to PROHIBIT it
        assert "Do NOT call get_node_conditions" in result

    def test_autopilot_prompt_prohibits_kubectl_top(self) -> None:
        """Autopilot prompt MUST explicitly prohibit kubectl_top."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        # The Autopilot prompt mentions kubectl_top only to PROHIBIT it
        assert "Do NOT call kubectl_top" in result

    def test_autopilot_prompt_cluster_overview_header(self) -> None:
        """Autopilot prompt output section must be ## Cluster Overview (downstream dependency)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        assert "## Cluster Overview" in result

    def test_autopilot_prompt_prohibits_get_events_kube_system(self) -> None:
        """Autopilot prompt must explicitly prohibit get_events for kube-system."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        result = build_node_gatherer_prompt(is_autopilot=True)
        assert "kube-system" in result

    def test_autopilot_and_standard_prompts_are_different(self) -> None:
        """The two prompt variants must be distinct."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        standard = build_node_gatherer_prompt(is_autopilot=False)
        autopilot = build_node_gatherer_prompt(is_autopilot=True)
        assert standard != autopilot

    def test_autopilot_prompt_shorter_than_standard(self) -> None:
        """Autopilot prompt (2 tools) must be shorter than standard prompt (6 steps)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        standard = build_node_gatherer_prompt(is_autopilot=False)
        autopilot = build_node_gatherer_prompt(is_autopilot=True)
        assert len(autopilot) < len(standard)


class TestBuildNodeGathererPromptSignature:
    """Tests for the function signature and default values."""

    def test_is_autopilot_default_is_false(self) -> None:
        """is_autopilot parameter MUST default to False."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        sig = inspect.signature(build_node_gatherer_prompt)
        param = sig.parameters.get("is_autopilot")
        assert param is not None, "is_autopilot parameter must exist"
        assert param.default is False

    def test_is_autopilot_annotation_is_bool(self) -> None:
        """is_autopilot parameter MUST be annotated as bool."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        sig = inspect.signature(build_node_gatherer_prompt)
        param = sig.parameters.get("is_autopilot")
        assert param is not None
        assert param.annotation is bool


class TestBuildAutopilotInstruction:
    """Tests for the updated build_autopilot_instruction() in language.py."""

    def test_returns_string(self) -> None:
        """build_autopilot_instruction(True) must return a non-empty string."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_false_returns_empty_string(self) -> None:
        """build_autopilot_instruction(False) must return empty string."""
        from vaig.core.language import build_autopilot_instruction

        assert build_autopilot_instruction(False) == ""

    def test_none_returns_empty_string(self) -> None:
        """build_autopilot_instruction(None) must return empty string."""
        from vaig.core.language import build_autopilot_instruction

        assert build_autopilot_instruction(None) == ""

    def test_contains_context_only(self) -> None:
        """Autopilot instruction must state node data is CONTEXT ONLY."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "CONTEXT ONLY" in result

    def test_contains_notready_is_normal(self) -> None:
        """Autopilot instruction must mention that NotReady is normal."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "NotReady" in result

    def test_contains_workload_level(self) -> None:
        """Autopilot instruction must focus analysis on WORKLOAD-LEVEL health."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "WORKLOAD-LEVEL" in result

    def test_contains_kubectl_top_not_available(self) -> None:
        """Autopilot instruction must state kubectl_top is NOT available."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "kubectl_top" in result
        assert "NOT available" in result

    def test_contains_no_node_level_actions(self) -> None:
        """Autopilot instruction must prohibit node-level actions."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "node-level actions" in result

    def test_contains_resource_requests_mandatory(self) -> None:
        """Autopilot instruction must flag missing resource requests."""
        from vaig.core.language import build_autopilot_instruction

        result = build_autopilot_instruction(True)
        assert "MANDATORY" in result or "mandatory" in result.lower()
        assert "resource requests" in result.lower()


# ---------------------------------------------------------------------------
# Tests: _query_autopilot_status timeout parameter
# ---------------------------------------------------------------------------


class TestQueryAutopilotStatusTimeout:
    """Verify _query_autopilot_status passes timeout to get_cluster."""

    def test_timeout_constant_exists_and_is_positive(self) -> None:
        """_AUTOPILOT_TIMEOUT must be a positive integer."""
        from vaig.tools.gke._clients import _AUTOPILOT_TIMEOUT

        assert isinstance(_AUTOPILOT_TIMEOUT, int)
        assert _AUTOPILOT_TIMEOUT > 0

    def test_timeout_constant_value(self) -> None:
        """_AUTOPILOT_TIMEOUT must be 10 seconds."""
        from vaig.tools.gke._clients import _AUTOPILOT_TIMEOUT

        assert _AUTOPILOT_TIMEOUT == 10

    def test_get_cluster_receives_timeout_kwarg(self) -> None:
        """get_cluster must be called with timeout=_AUTOPILOT_TIMEOUT."""
        from vaig.tools.gke._clients import _AUTOPILOT_TIMEOUT, _query_autopilot_status

        mock_cluster_mgr = MagicMock()
        mock_cluster = MagicMock()
        mock_cluster.autopilot.enabled = True
        mock_cluster_mgr.return_value.get_cluster.return_value = mock_cluster

        with patch(
            "google.cloud.container_v1.ClusterManagerClient",
            mock_cluster_mgr,
        ):
            _query_autopilot_status("my-project", "us-central1", "my-cluster")

        # Verify timeout was passed as kwarg
        mock_cluster_mgr.return_value.get_cluster.assert_called_once()
        call_kwargs = mock_cluster_mgr.return_value.get_cluster.call_args
        assert call_kwargs.kwargs.get("timeout") == _AUTOPILOT_TIMEOUT

    def test_get_cluster_called_with_correct_name(self) -> None:
        """get_cluster must receive the correctly formatted resource name."""
        from vaig.tools.gke._clients import _query_autopilot_status

        mock_cluster_mgr = MagicMock()
        mock_cluster = MagicMock()
        mock_cluster.autopilot.enabled = False
        mock_cluster_mgr.return_value.get_cluster.return_value = mock_cluster

        with patch(
            "google.cloud.container_v1.ClusterManagerClient",
            mock_cluster_mgr,
        ):
            _query_autopilot_status("proj-1", "europe-west1", "cluster-x")

        expected_name = "projects/proj-1/locations/europe-west1/clusters/cluster-x"
        call_kwargs = mock_cluster_mgr.return_value.get_cluster.call_args
        assert call_kwargs.kwargs.get("name") == expected_name


# ---------------------------------------------------------------------------
# Tests: detect_autopilot timeout handling
# ---------------------------------------------------------------------------


class TestDetectAutopilotTimeoutHandling:
    """Verify detect_autopilot returns False on DeadlineExceeded (timeout)."""

    def _make_gke_config(
        self,
        project: str = "test-project",
        location: str = "us-central1",
        cluster: str = "test-cluster",
    ) -> MagicMock:
        cfg = MagicMock()
        cfg.project_id = project
        cfg.location = location
        cfg.cluster_name = cluster
        return cfg

    def test_deadline_exceeded_returns_false(self) -> None:
        """DeadlineExceeded must result in False (assumes Standard cluster)."""
        from vaig.tools.gke._clients import clear_autopilot_cache, detect_autopilot

        clear_autopilot_cache()

        # Simulate google.api_core.exceptions.DeadlineExceeded
        exc = type("DeadlineExceeded", (Exception,), {})("Timeout on get_cluster")
        gke_config = self._make_gke_config()

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
            side_effect=exc,
        ):
            result = detect_autopilot(gke_config)

        assert result is False

        clear_autopilot_cache()

    def test_timeout_error_returns_false(self) -> None:
        """An exception whose type name contains 'Timeout' must return False."""
        from vaig.tools.gke._clients import clear_autopilot_cache, detect_autopilot

        clear_autopilot_cache()

        exc = type("GrpcTimeout", (Exception,), {})("gRPC timed out")
        gke_config = self._make_gke_config()

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
            side_effect=exc,
        ):
            result = detect_autopilot(gke_config)

        assert result is False

        clear_autopilot_cache()

    def test_generic_exception_still_returns_none(self) -> None:
        """Non-timeout exceptions must return None (detection failed)."""
        from vaig.tools.gke._clients import clear_autopilot_cache, detect_autopilot

        clear_autopilot_cache()

        gke_config = self._make_gke_config()

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
            side_effect=RuntimeError("Network unreachable"),
        ):
            result = detect_autopilot(gke_config)

        assert result is None

        clear_autopilot_cache()

    def test_import_error_returns_none(self) -> None:
        """ImportError (missing google-cloud-container) must return None."""
        from vaig.tools.gke._clients import clear_autopilot_cache, detect_autopilot

        clear_autopilot_cache()

        gke_config = self._make_gke_config()

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
            side_effect=ImportError("no module google.cloud.container_v1"),
        ):
            result = detect_autopilot(gke_config)

        assert result is None

        clear_autopilot_cache()

    def test_timeout_result_is_cached_as_false(self) -> None:
        """After a timeout, the cache must store False (not None)."""
        from vaig.tools.gke._clients import (
            _AUTOPILOT_CACHE,
            clear_autopilot_cache,
            detect_autopilot,
        )

        clear_autopilot_cache()

        exc = type("DeadlineExceeded", (Exception,), {})("timed out")
        gke_config = self._make_gke_config()

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
            side_effect=exc,
        ):
            detect_autopilot(gke_config)

        cache_key = (gke_config.project_id, gke_config.location, gke_config.cluster_name)
        assert _AUTOPILOT_CACHE[cache_key] is False

        clear_autopilot_cache()

    def test_missing_config_returns_none_without_calling_api(self) -> None:
        """Missing project_id/location/cluster_name must return None immediately."""
        from vaig.tools.gke._clients import clear_autopilot_cache, detect_autopilot

        clear_autopilot_cache()

        gke_config = self._make_gke_config(project="", location="us-central1", cluster="c1")

        with patch(
            "vaig.tools.gke._clients._query_autopilot_status",
        ) as mock_query:
            result = detect_autopilot(gke_config)

        assert result is None
        mock_query.assert_not_called()

        clear_autopilot_cache()

"""Tests for the programmatic kubectl_top pre-fetch feature.

Verifies that:
- ``_pre_fetch_metrics()`` calls ``kubectl_top`` correctly and returns structured data
- ``_pre_fetch_metrics()`` handles errors gracefully (never crashes)
- ``_pre_fetch_metrics()`` respects Autopilot wrapper for node metrics
- ``_pre_fetch_metrics()`` detects non-metric sentinel outputs (Fix #1)
- ``_pre_fetch_metrics()`` truncates large outputs (Fix #3)
- ``pre_execute_parallel()`` populates ``_prefetched_metrics`` (Fix #2)
- ``get_parallel_agents_config()`` reads from ``_prefetched_metrics`` without calling kubectl_top (Fix #2)
- ``build_node_gatherer_prompt(prefetched_node_metrics=...)`` includes pre-gathered data
- ``build_workload_gatherer_prompt(prefetched_pod_metrics=...)`` includes pre-gathered data
- Pre-fetched metrics are wrapped with ``wrap_untrusted_content()`` (Fix #4)
- Step 2/3 instructions are conditional on pre-fetched data availability (Fix #5)
- Backward compatibility: empty string omits the pre-fetch section
"""

from __future__ import annotations

from unittest.mock import patch

from vaig.core.prompt_defense import DELIMITER_DATA_END, DELIMITER_DATA_START
from vaig.tools.base import ToolResult

# ── Pre-fetch helper tests ───────────────────────────────────────────────


FAKE_POD_METRICS = (
    "NAME                                              CONTAINER                     CPU(cores)          MEMORY\n"
    "payment-svc-abc123                                 payment                       0.056               128Mi"
)
FAKE_NODE_METRICS = (
    "NAME                                              CPU(cores)          CPU%      MEMORY           MEMORY%\n"
    "gke-cluster-pool-abc123                            0.250               12%       2048Mi           45%"
)


class TestPreFetchMetrics:
    """Tests for ServiceHealthSkill._pre_fetch_metrics()."""

    def test_returns_dict_with_pods_and_nodes_keys(self) -> None:
        """Result must contain 'pods' and 'nodes' keys."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=FAKE_POD_METRICS),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type] — mocked
                namespace="default",
                is_autopilot=False,
            )
        assert "pods" in result
        assert "nodes" in result

    def test_successful_pod_metrics(self) -> None:
        """When kubectl_top succeeds for pods, output is captured."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=FAKE_POD_METRICS),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="production",
                is_autopilot=False,
            )
        assert result["pods"] == FAKE_POD_METRICS

    def test_successful_node_metrics(self) -> None:
        """When kubectl_top succeeds for nodes, output is captured."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        def _mock_kubectl_top(resource_type, *, gke_config, **kwargs):  # noqa: ARG001
            if resource_type == "nodes":
                return ToolResult(output=FAKE_NODE_METRICS)
            return ToolResult(output=FAKE_POD_METRICS)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=_mock_kubectl_top,
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=False,
            )
        assert result["nodes"] == FAKE_NODE_METRICS

    def test_autopilot_skips_node_kubectl_top(self) -> None:
        """On Autopilot, node metrics are an informational message, NOT a kubectl_top call."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        call_count = {"nodes": 0}

        def _mock_kubectl_top(resource_type, *, gke_config, **kwargs):  # noqa: ARG001
            if resource_type == "nodes":
                call_count["nodes"] += 1
            return ToolResult(output=FAKE_POD_METRICS)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=_mock_kubectl_top,
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,
            )
        assert call_count["nodes"] == 0, "kubectl_top(nodes) must NOT be called on Autopilot"
        assert "Autopilot" in result["nodes"]

    def test_pod_metrics_error_captured_not_raised(self) -> None:
        """kubectl_top error for pods is captured as a string, never raised."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output="Metrics API not available", error=True),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,
            )
        assert "failed" in result["pods"].lower() or "unavailable" in result["pods"].lower()

    def test_exception_captured_not_raised(self) -> None:
        """Any exception during kubectl_top is captured, never propagated."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=RuntimeError("boom"),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=False,
            )
        # Both keys must have informational error strings, not empty
        assert "unavailable" in result["pods"].lower()
        assert "unavailable" in result["nodes"].lower()

    def test_node_metrics_error_captured(self) -> None:
        """kubectl_top error for nodes is captured as a string."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        def _mock_kubectl_top(resource_type, *, gke_config, **kwargs):  # noqa: ARG001
            if resource_type == "nodes":
                return ToolResult(output="Access denied to metrics API", error=True)
            return ToolResult(output=FAKE_POD_METRICS)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=_mock_kubectl_top,
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=False,
            )
        assert "failed" in result["nodes"].lower()
        assert result["pods"] == FAKE_POD_METRICS  # pods still succeed

    def test_empty_namespace_defaults_to_default(self) -> None:
        """When namespace is empty, 'default' is used for kubectl_top."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        captured_ns = {}

        def _mock_kubectl_top(resource_type, *, gke_config, namespace="default", **kwargs):  # noqa: ARG001
            if resource_type in ("pods", "pod"):
                captured_ns["ns"] = namespace
            return ToolResult(output=FAKE_POD_METRICS)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=_mock_kubectl_top,
        ):
            ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="",
                is_autopilot=False,
            )
        assert captured_ns["ns"] == "default"


# ── Sentinel detection tests (Fix #1) ───────────────────────────────────


class TestSentinelDetection:
    """Verify _pre_fetch_metrics detects non-metric outputs lacking CPU/MEMORY headers."""

    def test_sentinel_pod_output_detected(self) -> None:
        """Output without CPU/MEMORY headers is treated as unavailable."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        sentinel = "No metrics data available. Is metrics-server installed?"
        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=sentinel),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,  # skip node call
            )
        assert "unavailable" in result["pods"].lower()
        assert sentinel.strip() in result["pods"]

    def test_sentinel_node_output_detected(self) -> None:
        """Node output without CPU/MEMORY headers is treated as unavailable."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        sentinel = "error: Metrics API not available"

        def _mock_kubectl_top(resource_type, *, gke_config, **kwargs):  # noqa: ARG001
            if resource_type == "nodes":
                return ToolResult(output=sentinel)
            return ToolResult(output=FAKE_POD_METRICS)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=_mock_kubectl_top,
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=False,
            )
        assert "unavailable" in result["nodes"].lower()
        assert result["pods"] == FAKE_POD_METRICS  # pods still succeed

    def test_valid_output_not_flagged_as_sentinel(self) -> None:
        """Output with CPU and MEMORY headers passes validation."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=FAKE_POD_METRICS),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,  # skip node call
            )
        # Valid output should be passed through unchanged
        assert result["pods"] == FAKE_POD_METRICS
        assert "unavailable" not in result["pods"].lower()

    def test_informational_message_with_error_false(self) -> None:
        """Informational messages like 'metrics not available' with error=False are caught."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        msg = "Metrics not yet available for this cluster"
        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=msg),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,
            )
        assert "unavailable" in result["pods"].lower()


# ── Truncation tests (Fix #3) ───────────────────────────────────────────


class TestOutputTruncation:
    """Verify _pre_fetch_metrics truncates large outputs."""

    def test_output_within_limit_not_truncated(self) -> None:
        """Output with fewer lines than the limit is not truncated."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=FAKE_POD_METRICS),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,
            )
        assert "truncated" not in result["pods"].lower()

    def test_output_exceeding_limit_is_truncated(self) -> None:
        """Output with more than _MAX_KUBECTL_TOP_LINES lines is truncated."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        header = "NAME                  CPU(cores)     MEMORY\n"
        lines = [f"pod-{i:04d}              0.001          10Mi" for i in range(200)]
        large_output = header + "\n".join(lines)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=large_output),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,
            )
        assert "truncated" in result["pods"].lower()
        assert "201 total lines" in result["pods"]
        # First 100 lines should be present (header + 99 data lines)
        output_lines = result["pods"].splitlines()
        # 100 content lines + 1 truncation notice
        assert len(output_lines) == 101

    def test_truncation_preserves_header(self) -> None:
        """The header line must be preserved in truncated output."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        header = "NAME                  CPU(cores)     MEMORY\n"
        lines = [f"pod-{i:04d}              0.001          10Mi" for i in range(150)]
        large_output = header + "\n".join(lines)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            return_value=ToolResult(output=large_output),
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=True,
            )
        # Header (first line) should be present
        assert "CPU(cores)" in result["pods"].splitlines()[0]

    def test_node_output_also_truncated(self) -> None:
        """Node metrics are also truncated when exceeding the limit."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        header = "NAME                  CPU(cores)     CPU%     MEMORY      MEMORY%\n"
        lines = [f"node-{i:04d}              0.250       12%       2048Mi      45%" for i in range(150)]
        large_output = header + "\n".join(lines)

        def _mock_kubectl_top(resource_type, *, gke_config, **kwargs):  # noqa: ARG001
            if resource_type == "nodes":
                return ToolResult(output=large_output)
            return ToolResult(output=FAKE_POD_METRICS)

        with patch(
            "vaig.tools.gke.kubectl.kubectl_top",
            side_effect=_mock_kubectl_top,
        ):
            result = ServiceHealthSkill._pre_fetch_metrics(
                gke_config=None,  # type: ignore[arg-type]
                namespace="default",
                is_autopilot=False,
            )
        assert "truncated" in result["nodes"].lower()


# ── Pre-execute parallel and _prefetched_metrics tests (Fix #2) ─────────


class TestPreExecuteParallelPrefetch:
    """Verify pre_execute_parallel populates _prefetched_metrics (Fix #2)."""

    def test_init_sets_empty_prefetched_metrics(self) -> None:
        """ServiceHealthSkill.__init__ must set _prefetched_metrics to empty dict."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        assert skill._prefetched_metrics == {"pods": "", "nodes": ""}

    def test_pre_execute_parallel_populates_prefetched_metrics(self) -> None:
        """pre_execute_parallel must call _pre_fetch_metrics and store results."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        expected = {"pods": FAKE_POD_METRICS, "nodes": FAKE_NODE_METRICS}

        with (
            patch.object(
                ServiceHealthSkill, "_pre_fetch_metrics", return_value=expected,
            ),
            patch("vaig.skills.service_health.skill.ensure_client_initialized"),
            patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        ):
            skill.pre_execute_parallel("check health")

        assert skill._prefetched_metrics == expected

    def test_pre_execute_parallel_error_keeps_empty_defaults(self) -> None:
        """If _pre_fetch_metrics raises, _prefetched_metrics stays empty."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()

        with (
            patch.object(
                ServiceHealthSkill, "_pre_fetch_metrics",
                side_effect=RuntimeError("boom"),
            ),
            patch("vaig.skills.service_health.skill.ensure_client_initialized"),
            patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        ):
            # Must not raise
            skill.pre_execute_parallel("check health")

        # Stays at initial empty defaults
        assert skill._prefetched_metrics == {"pods": "", "nodes": ""}

    def test_get_parallel_agents_config_does_not_call_pre_fetch(self) -> None:
        """get_parallel_agents_config must NOT call _pre_fetch_metrics directly."""
        from vaig.skills.service_health.skill import ServiceHealthSkill

        skill = ServiceHealthSkill()
        # Pre-populate metrics as if pre_execute_parallel had been called
        skill._prefetched_metrics = {"pods": FAKE_POD_METRICS, "nodes": FAKE_NODE_METRICS}

        with patch.object(
            ServiceHealthSkill, "_pre_fetch_metrics",
        ) as mock_pre_fetch:
            skill.get_parallel_agents_config()

        mock_pre_fetch.assert_not_called()


# ── Node gatherer prompt injection tests ─────────────────────────────────


class TestNodeGathererPromptPrefetch:
    """Verify build_node_gatherer_prompt() handles prefetched_node_metrics."""

    def test_no_prefetch_section_when_empty(self) -> None:
        """When prefetched_node_metrics is empty, the pre-fetch section is absent."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(is_autopilot=False, prefetched_node_metrics="")
        assert "PRE-GATHERED METRICS DATA" not in prompt

    def test_prefetch_section_present_when_provided(self) -> None:
        """When prefetched_node_metrics has data, the pre-fetch section is injected."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(
            is_autopilot=False,
            prefetched_node_metrics=FAKE_NODE_METRICS,
        )
        assert "PRE-GATHERED METRICS DATA" in prompt
        assert FAKE_NODE_METRICS in prompt

    def test_prefetch_section_says_do_not_call_kubectl_top(self) -> None:
        """The pre-fetch section must instruct the agent NOT to call kubectl_top again."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(
            is_autopilot=False,
            prefetched_node_metrics=FAKE_NODE_METRICS,
        )
        assert "Do NOT call" in prompt and "kubectl_top" in prompt

    def test_autopilot_ignores_prefetch(self) -> None:
        """Autopilot prompt does not inject pre-fetch section (nodes aren't investigated)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(
            is_autopilot=True,
            prefetched_node_metrics=FAKE_NODE_METRICS,
        )
        # Autopilot prompt already says "Do NOT call kubectl_top" — the pre-gathered
        # section should NOT appear since the Autopilot prompt code path returns early.
        assert "PRE-GATHERED METRICS DATA" not in prompt

    def test_backward_compat_default_is_empty(self) -> None:
        """Default value of prefetched_node_metrics is empty — backward compatible."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt_no_arg = build_node_gatherer_prompt(is_autopilot=False)
        prompt_explicit = build_node_gatherer_prompt(is_autopilot=False, prefetched_node_metrics="")
        assert prompt_no_arg == prompt_explicit

    def test_prefetch_wrapped_with_untrusted_delimiters(self) -> None:
        """Pre-fetched data must be wrapped with DELIMITER_DATA_START/END markers (Fix #4)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(
            is_autopilot=False,
            prefetched_node_metrics=FAKE_NODE_METRICS,
        )
        assert DELIMITER_DATA_START in prompt
        assert DELIMITER_DATA_END in prompt

    def test_step3_conditional_with_prefetch(self) -> None:
        """Step 3 should NOT say 'kubectl_top' is a tool to call when data is pre-fetched (Fix #5)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(
            is_autopilot=False,
            prefetched_node_metrics=FAKE_NODE_METRICS,
        )
        # Step 3 should say the data was pre-gathered, not instruct to call kubectl_top
        assert "pre-gathered" in prompt.lower()

    def test_step3_normal_without_prefetch(self) -> None:
        """Step 3 should instruct to call kubectl_top when no pre-fetched data (Fix #5)."""
        from vaig.skills.service_health.prompts import build_node_gatherer_prompt

        prompt = build_node_gatherer_prompt(
            is_autopilot=False,
            prefetched_node_metrics="",
        )
        # Step 3 should contain the kubectl_top tool call instruction
        assert 'kubectl_top(resource_type="nodes")' in prompt


# ── Workload gatherer prompt injection tests ─────────────────────────────


class TestWorkloadGathererPromptPrefetch:
    """Verify build_workload_gatherer_prompt() handles prefetched_pod_metrics."""

    def test_no_prefetch_section_when_empty(self) -> None:
        """When prefetched_pod_metrics is empty, the pre-fetch section is absent."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(namespace="default", prefetched_pod_metrics="")
        assert "PRE-GATHERED METRICS DATA" not in prompt

    def test_prefetch_section_present_when_provided(self) -> None:
        """When prefetched_pod_metrics has data, the pre-fetch section is injected."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="production",
            prefetched_pod_metrics=FAKE_POD_METRICS,
        )
        assert "PRE-GATHERED METRICS DATA" in prompt
        assert FAKE_POD_METRICS in prompt

    def test_prefetch_section_says_do_not_call_kubectl_top(self) -> None:
        """The pre-fetch section must instruct the agent NOT to call kubectl_top again."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="default",
            prefetched_pod_metrics=FAKE_POD_METRICS,
        )
        assert "Do NOT call" in prompt and "kubectl_top" in prompt

    def test_prefetch_with_argo_rollouts(self) -> None:
        """Pre-fetch section works correctly when argo_rollouts_enabled=True."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="default",
            argo_rollouts_enabled=True,
            prefetched_pod_metrics=FAKE_POD_METRICS,
        )
        assert "PRE-GATHERED METRICS DATA" in prompt
        assert "Step 4d" in prompt  # Argo Rollouts section still present
        assert FAKE_POD_METRICS in prompt

    def test_backward_compat_default_is_empty(self) -> None:
        """Default value of prefetched_pod_metrics is empty — backward compatible."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt_no_arg = build_workload_gatherer_prompt(namespace="default")
        prompt_explicit = build_workload_gatherer_prompt(namespace="default", prefetched_pod_metrics="")
        assert prompt_no_arg == prompt_explicit

    def test_prefetch_mentions_get_pod_metrics_still_needed(self) -> None:
        """Pre-fetch section should note that get_pod_metrics is still needed for trends."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="default",
            prefetched_pod_metrics=FAKE_POD_METRICS,
        )
        assert "get_pod_metrics" in prompt

    def test_prefetch_wrapped_with_untrusted_delimiters(self) -> None:
        """Pre-fetched data must be wrapped with DELIMITER_DATA_START/END markers (Fix #4)."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="default",
            prefetched_pod_metrics=FAKE_POD_METRICS,
        )
        assert DELIMITER_DATA_START in prompt
        assert DELIMITER_DATA_END in prompt

    def test_step2_conditional_with_prefetch(self) -> None:
        """Step 2 should NOT say kubectl_top is MANDATORY when data is pre-fetched (Fix #5)."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="default",
            prefetched_pod_metrics=FAKE_POD_METRICS,
        )
        # Should NOT contain the original kubectl_top MANDATORY instruction for step 2
        assert "MANDATORY — the reporter needs real CPU and memory" not in prompt
        # Should have the conditional replacement
        assert "pre-gathered" in prompt.lower()

    def test_step2_normal_without_prefetch(self) -> None:
        """Step 2 should say kubectl_top is MANDATORY when no pre-fetched data (Fix #5)."""
        from vaig.skills.service_health.prompts import build_workload_gatherer_prompt

        prompt = build_workload_gatherer_prompt(
            namespace="default",
            prefetched_pod_metrics="",
        )
        assert "MANDATORY" in prompt

"""Prompt content tests for sub-gatherer builders — SPEC-DD-01, SPEC-SH-14, SPEC-SH-11, SPEC-LT-01, SPEC-LT-03."""

from __future__ import annotations


class TestDatadogGathererPromptDD01:
    """SPEC-DD-01 T3.6 — build_datadog_gatherer_prompt() contains error-span tool + Envoy."""

    def _get_prompt(self) -> str:
        from vaig.skills.service_health.prompts._sub_gatherers import build_datadog_gatherer_prompt

        return build_datadog_gatherer_prompt(datadog_api_enabled=True)

    def test_contains_query_datadog_error_spans(self) -> None:
        assert "query_datadog_error_spans" in self._get_prompt()

    def test_contains_envoy_admin_localhost_15000(self) -> None:
        assert "localhost:15000" in self._get_prompt()

    def test_contains_error_rate_threshold_rule(self) -> None:
        prompt = self._get_prompt()
        assert "error_rate" in prompt or "error rate" in prompt


class TestDatadogGathererPromptSH14:
    """SPEC-SH-14 T2 — build_datadog_gatherer_prompt() contains tool error recovery."""

    def _get_prompt(self) -> str:
        from vaig.skills.service_health.prompts._sub_gatherers import build_datadog_gatherer_prompt

        return build_datadog_gatherer_prompt(datadog_api_enabled=True)

    def test_contains_diagnose_datadog_metrics(self) -> None:
        assert "diagnose_datadog_metrics" in self._get_prompt()

    def test_contains_unknown_metric_recovery(self) -> None:
        assert "Unknown metric template" in self._get_prompt() or "unknown metric" in self._get_prompt().lower()

    def test_contains_no_service_catalog_skip_rule(self) -> None:
        assert "No service catalog found" in self._get_prompt() or "service catalog" in self._get_prompt().lower()

    def test_contains_valid_metric_keys_cpu(self) -> None:
        assert "cpu" in self._get_prompt()

    def test_contains_valid_metric_keys_error_rate(self) -> None:
        assert "error_rate" in self._get_prompt()

    def test_contains_valid_metric_keys_apdex(self) -> None:
        assert "apdex" in self._get_prompt()

    def test_contains_max_one_retry_instruction(self) -> None:
        prompt = self._get_prompt()
        assert "MAX ONE RETRY" in prompt or "one retry" in prompt.lower() or "RETRY" in prompt


class TestLoggingGathererPromptSH11:
    """SPEC-SH-11 T2 — build_logging_gatherer_prompt() contains expanded log queries."""

    def _get_prompt(self) -> str:
        from vaig.skills.service_health.prompts._sub_gatherers import build_logging_gatherer_prompt

        return build_logging_gatherer_prompt(namespace="test-ns")

    def test_contains_istio_proxy_container(self) -> None:
        assert 'container_name="istio-proxy"' in self._get_prompt()

    def test_contains_upstream_connect_error(self) -> None:
        assert "upstream connect error" in self._get_prompt()

    def test_contains_discovery_container(self) -> None:
        assert 'container_name="discovery"' in self._get_prompt()

    def test_contains_istio_init_cni_node(self) -> None:
        assert "istio-init" in self._get_prompt() and "istio-cni-node" in self._get_prompt()

    def test_contains_retry_on_zero_policy(self) -> None:
        assert "3 attempts" in self._get_prompt()

    def test_contains_istiod_caveat(self) -> None:
        prompt = self._get_prompt()
        assert "certificate rotation" in prompt or "xDS" in prompt


class TestWorkloadGathererPromptHPAAdapterHealth:
    """hpa-adapter-health spec — build_workload_gatherer_prompt() contains Step 6e adapter pod health."""

    def _get_prompt(self) -> str:
        from vaig.skills.service_health.prompts._sub_gatherers import build_workload_gatherer_prompt

        return build_workload_gatherer_prompt(namespace="test-ns")

    def test_contains_get_pods(self) -> None:
        assert "get_pods" in self._get_prompt()

    def test_contains_adapter_name(self) -> None:
        prompt = self._get_prompt()
        assert "prometheus-adapter" in prompt or "datadog-cluster-agent" in prompt or "custom-metrics-apiserver" in prompt

    def test_contains_adapter_health_reference(self) -> None:
        prompt = self._get_prompt()
        assert "adapter" in prompt.lower()


class TestLowThroughputHeuristicDD:
    """SPEC-LT-01 — build_datadog_gatherer_prompt() contains low-throughput heuristic block."""

    def _get_prompt(self) -> str:
        from vaig.skills.service_health.prompts._sub_gatherers import build_datadog_gatherer_prompt

        return build_datadog_gatherer_prompt(datadog_api_enabled=True)

    def test_contains_low_throughput_threshold(self) -> None:
        assert "0.1 req" in self._get_prompt()

    def test_contains_low_throughput_label(self) -> None:
        assert "low-throughput" in self._get_prompt().lower() or "low throughput" in self._get_prompt().lower()

    def test_low_throughput_triggers_span_investigation(self) -> None:
        assert "query_datadog_error_spans" in self._get_prompt()


class TestLowThroughputAnalyzerEscalation:
    """SPEC-LT-03 — HEALTH_ANALYZER_PROMPT contains low-throughput severity escalation."""

    def _get_prompt(self) -> str:
        from vaig.skills.service_health.prompts._analyzer import HEALTH_ANALYZER_PROMPT

        return HEALTH_ANALYZER_PROMPT

    def test_analyzer_contains_low_throughput_escalation(self) -> None:
        prompt = self._get_prompt()
        assert "low-throughput" in prompt.lower() or "low throughput" in prompt.lower()

    def test_analyzer_escalation_assigns_high_severity(self) -> None:
        prompt = self._get_prompt()
        # Find the low-throughput section and verify HIGH severity is present
        low_tp_idx = prompt.lower().find("low-throughput severity escalation")
        assert low_tp_idx != -1, "Low-Throughput Severity Escalation section not found"
        section = prompt[low_tp_idx : low_tp_idx + 600]
        assert "HIGH" in section

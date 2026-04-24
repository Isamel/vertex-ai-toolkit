"""Tests for Envoy admin API instructions in sub-gatherer prompts."""

from __future__ import annotations

from vaig.skills.service_health.prompts._sub_gatherers import build_workload_gatherer_prompt


class TestWorkloadGathererEnvoyInstructions:
    def test_workload_gatherer_prompt_includes_envoy_instructions(self) -> None:
        prompt = build_workload_gatherer_prompt(namespace="default")
        assert "localhost:15000" in prompt
        assert "clusters" in prompt

    def test_workload_gatherer_prompt_envoy_sidecar_detection(self) -> None:
        prompt = build_workload_gatherer_prompt(namespace="default")
        assert "istio-proxy" in prompt or "envoy" in prompt.lower()

    def test_workload_gatherer_prompt_envoy_admin_key(self) -> None:
        prompt = build_workload_gatherer_prompt(namespace="default")
        assert "envoy_admin" in prompt

    def test_workload_gatherer_prompt_envoy_stats_endpoint(self) -> None:
        prompt = build_workload_gatherer_prompt(namespace="default")
        assert "localhost:15000/stats" in prompt

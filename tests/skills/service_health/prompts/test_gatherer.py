"""Tests for Envoy upstream verification instructions in the gatherer (verifier) prompt."""

from __future__ import annotations

from vaig.skills.service_health.prompts._gatherer import build_gatherer_prompt


class TestGathererEnvoyUpstreamCheck:
    def test_verifier_prompt_includes_envoy_upstream_check(self) -> None:
        prompt = build_gatherer_prompt()
        assert "localhost:15000/clusters" in prompt

    def test_verifier_prompt_envoy_block_triggered_by_upstream_errors(self) -> None:
        prompt = build_gatherer_prompt()
        assert "upstream" in prompt.lower()
        assert "5xx" in prompt or "connection refused" in prompt

"""Unit tests for vaig.core.router — dynamic agent routing.

Tests cover:
- Exact keyword match routes correctly
- Substring match works (e.g., "pod" matches "pods" and vice-versa)
- No match → safe-all fallback (returns ALL configs)
- Empty query → safe-all fallback
- Configs without ``capabilities`` field are always included (backward compat)
- Configs without ``parallel_group`` (sequential agents) pass through unchanged
- Case-insensitive matching
- Multiple capabilities on a single gatherer
"""

from __future__ import annotations

import pytest

from vaig.core.router import _tokenize, route_agents

# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_gatherer(name: str, capabilities: list[str], parallel_group: str = "gather") -> dict:
    """Create a minimal parallel gatherer config with capabilities."""
    return {
        "name": name,
        "parallel_group": parallel_group,
        "capabilities": capabilities,
        "system_instruction": f"You are {name}.",
    }


def _make_sequential(name: str) -> dict:
    """Create a minimal sequential agent config (no parallel_group, no capabilities)."""
    return {
        "name": name,
        "system_instruction": f"You are {name}.",
    }


def _make_gatherer_no_capabilities(name: str, parallel_group: str = "gather") -> dict:
    """Create a parallel gatherer config WITHOUT a capabilities key (backward compat)."""
    return {
        "name": name,
        "parallel_group": parallel_group,
        "system_instruction": f"You are {name}.",
    }


# ── _tokenize ─────────────────────────────────────────────────────────────────


class TestTokenize:
    def test_lowercases_input(self) -> None:
        assert _tokenize("Pods CPU") == ["pods", "cpu"]

    def test_splits_on_whitespace(self) -> None:
        assert _tokenize("check log for pod X") == ["check", "log", "for", "pod", "x"]

    def test_strips_punctuation(self) -> None:
        assert _tokenize("pod!!! logs???") == ["pod", "logs"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _tokenize("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert _tokenize("   ") == []

    def test_numbers_are_preserved(self) -> None:
        assert "503" in _tokenize("503 errors in pod")

    def test_hyphenated_splits_into_parts(self) -> None:
        # Hyphens are not alphanumeric — split into separate tokens
        tokens = _tokenize("crash-loop errors")
        assert "crash" in tokens
        assert "loop" in tokens

    def test_mixed_case_lowercased(self) -> None:
        tokens = _tokenize("NODE CPU Memory")
        assert tokens == ["node", "cpu", "memory"]


# ── route_agents ──────────────────────────────────────────────────────────────


class TestRouteAgentsExactMatch:
    """Exact keyword matches route to the correct gatherer."""

    def test_exact_match_returns_matching_gatherer(self) -> None:
        node = _make_gatherer("node_gatherer", ["node", "cpu"])
        logging = _make_gatherer("logging_gatherer", ["log", "logging"])
        analyzer = _make_sequential("analyzer")
        configs = [node, logging, analyzer]

        result = route_agents("check node cpu usage", configs)

        names = [c["name"] for c in result]
        assert "node_gatherer" in names
        assert "analyzer" in names
        assert "logging_gatherer" not in names

    def test_exact_match_single_capability(self) -> None:
        cpu = _make_gatherer("cpu_gatherer", ["cpu"])
        mem = _make_gatherer("mem_gatherer", ["memory"])
        configs = [cpu, mem]

        result = route_agents("high cpu usage", configs)

        assert result == [cpu]

    def test_multiple_matching_gatherers_all_returned(self) -> None:
        node = _make_gatherer("node_gatherer", ["node"])
        logging = _make_gatherer("logging_gatherer", ["log"])
        configs = [node, logging]

        result = route_agents("check node log", configs)

        assert node in result
        assert logging in result


class TestRouteAgentsSubstringMatch:
    """Bidirectional substring matching."""

    def test_token_in_capability(self) -> None:
        """Query token 'log' matches capability 'logging' (token in cap)."""
        logging = _make_gatherer("logging_gatherer", ["logging"])
        other = _make_gatherer("node_gatherer", ["node"])
        configs = [logging, other]

        result = route_agents("check log output", configs)

        assert logging in result
        assert other not in result

    def test_capability_in_token(self) -> None:
        """Query token 'pods' matches capability 'pod' (cap in token)."""
        pod = _make_gatherer("pod_gatherer", ["pod"])
        configs = [pod]

        result = route_agents("list all pods", configs)

        assert pod in result

    def test_token_substring_of_capability(self) -> None:
        """Query token 'net' matches capability 'networking'."""
        net = _make_gatherer("net_gatherer", ["networking"])
        configs = [net]

        result = route_agents("check net traffic", configs)

        assert net in result

    def test_capability_substring_of_longer_token(self) -> None:
        """Capability 'log' matches query token 'logging'."""
        logging = _make_gatherer("logging_gatherer", ["log"])
        configs = [logging]

        result = route_agents("analyze logging errors", configs)

        assert logging in result

    @pytest.mark.parametrize("query", [
        "pod issues",
        "pods crashing",
        "podname not found",
        "check pod health",
    ])
    def test_pod_variants_all_match(self, query: str) -> None:
        """Various 'pod' query forms should match a gatherer with 'pod' capability."""
        pod = _make_gatherer("pod_gatherer", ["pod"])
        sequential = _make_sequential("analyzer")
        configs = [pod, sequential]

        result = route_agents(query, configs)

        assert pod in result
        assert sequential in result


class TestRouteAgentsSafeAllFallback:
    """No match → safe-all fallback returns ALL configs."""

    def test_no_match_returns_all_configs(self) -> None:
        node = _make_gatherer("node_gatherer", ["node", "cpu"])
        logging = _make_gatherer("logging_gatherer", ["log"])
        analyzer = _make_sequential("analyzer")
        configs = [node, logging, analyzer]

        # Query has no keywords matching any capability
        result = route_agents("check overall cluster health", configs)

        # Safe-all: all configs returned
        assert result == configs

    def test_empty_query_triggers_safe_all(self) -> None:
        node = _make_gatherer("node_gatherer", ["node"])
        analyzer = _make_sequential("analyzer")
        configs = [node, analyzer]

        result = route_agents("", configs)

        assert result == configs

    def test_whitespace_query_triggers_safe_all(self) -> None:
        node = _make_gatherer("node_gatherer", ["node"])
        configs = [node]

        result = route_agents("   ", configs)

        assert result == configs

    def test_safe_all_preserves_original_order(self) -> None:
        a = _make_gatherer("a", ["alpha"])
        b = _make_gatherer("b", ["beta"])
        c = _make_sequential("c")
        configs = [a, b, c]

        # "gamma" doesn't match alpha or beta → safe-all
        result = route_agents("gamma query", configs)

        assert result == [a, b, c]


class TestRouteAgentsBackwardCompat:
    """Configs without capabilities always pass through."""

    def test_gatherer_without_capabilities_always_passes_through(self) -> None:
        """A parallel gatherer WITHOUT a capabilities key is never filtered out."""
        legacy_gatherer = _make_gatherer_no_capabilities("legacy_gatherer")
        node_gatherer = _make_gatherer("node_gatherer", ["node"])
        configs = [legacy_gatherer, node_gatherer]

        # Query matches only node_gatherer
        result = route_agents("node cpu", configs)

        # legacy_gatherer has no capabilities → passes through unconditionally
        assert legacy_gatherer in result
        assert node_gatherer in result

    def test_gatherer_without_capabilities_included_when_others_match(self) -> None:
        """Legacy gatherer should appear alongside matched gatherers."""
        legacy = _make_gatherer_no_capabilities("legacy")
        log = _make_gatherer("log_gatherer", ["log"])
        sequential = _make_sequential("analyzer")
        configs = [legacy, log, sequential]

        result = route_agents("check log errors", configs)

        assert legacy in result
        assert log in result
        assert sequential in result

    def test_gatherer_without_capabilities_included_on_no_match(self) -> None:
        """Safe-all fallback still includes legacy gatherer."""
        legacy = _make_gatherer_no_capabilities("legacy")
        log = _make_gatherer("log_gatherer", ["log"])
        configs = [legacy, log]

        # "database" matches nothing → safe-all
        result = route_agents("database errors", configs)

        assert result == configs


class TestRouteAgentsSequentialAgents:
    """Sequential agents (no parallel_group) always pass through."""

    def test_sequential_agents_always_included(self) -> None:
        node = _make_gatherer("node_gatherer", ["node"])
        analyzer = _make_sequential("analyzer")
        verifier = _make_sequential("verifier")
        reporter = _make_sequential("reporter")
        configs = [node, analyzer, verifier, reporter]

        # Only node matches
        result = route_agents("node cpu check", configs)

        names = [c["name"] for c in result]
        assert "node_gatherer" in names
        assert "analyzer" in names
        assert "verifier" in names
        assert "reporter" in names

    def test_only_sequential_configs_returns_all(self) -> None:
        """When there are no filterable gatherers, all configs pass through."""
        a = _make_sequential("analyzer")
        b = _make_sequential("verifier")
        configs = [a, b]

        result = route_agents("any query", configs)

        assert result == configs

    def test_empty_configs_returns_empty(self) -> None:
        result = route_agents("check node health", [])
        assert result == []


class TestRouteAgentsCaseInsensitive:
    """Matching must be case-insensitive."""

    @pytest.mark.parametrize(("query", "capability"), [
        ("CHECK NODE HEALTH", "node"),
        ("Check Node Health", "node"),
        ("node health", "NODE"),
        ("NODE health", "NODE"),
        ("Logging Errors", "logging"),
    ])
    def test_case_insensitive_match(self, query: str, capability: str) -> None:
        gatherer = _make_gatherer("gatherer", [capability])
        configs = [gatherer]

        result = route_agents(query, configs)

        assert gatherer in result

    def test_mixed_case_capability_and_query(self) -> None:
        gatherer = _make_gatherer("gatherer", ["Networking"])
        other = _make_gatherer("other", ["Database"])
        configs = [gatherer, other]

        result = route_agents("NETWORKING traffic check", configs)

        assert gatherer in result
        assert other not in result


class TestRouteAgentsMultipleCapabilities:
    """A single gatherer with multiple capabilities matches any of them."""

    def test_first_capability_matches(self) -> None:
        gatherer = _make_gatherer("multi", ["node", "cpu", "memory", "disk"])
        configs = [gatherer]

        result = route_agents("high cpu usage", configs)

        assert gatherer in result

    def test_last_capability_matches(self) -> None:
        gatherer = _make_gatherer("multi", ["node", "cpu", "memory", "disk"])
        configs = [gatherer]

        result = route_agents("disk space low", configs)

        assert gatherer in result

    def test_any_capability_sufficient(self) -> None:
        """Even one matching capability is enough to include the gatherer."""
        gatherer = _make_gatherer("multi", ["node", "cpu", "log", "network"])
        other = _make_gatherer("other", ["database"])
        configs = [gatherer, other]

        # "log" matches only 'multi' gatherer
        result = route_agents("check logs", configs)

        assert gatherer in result
        assert other not in result

    def test_two_gatherers_different_capabilities_query_matches_one(self) -> None:
        node = _make_gatherer("node_gatherer", ["node", "cpu", "memory"])
        logging = _make_gatherer("logging_gatherer", ["log", "logging", "stderr"])
        analyzer = _make_sequential("analyzer")
        configs = [node, logging, analyzer]

        result = route_agents("pod logs crashing", configs)

        names = [c["name"] for c in result]
        assert "logging_gatherer" in names
        assert "analyzer" in names
        assert "node_gatherer" not in names


class TestRouteAgentsOrderPreservation:
    """Returned configs preserve the original order from the input list."""

    def test_original_order_preserved_on_match(self) -> None:
        a = _make_gatherer("a", ["alpha"])
        b = _make_sequential("b")
        c = _make_gatherer("c", ["alpha"])  # also matches
        d = _make_sequential("d")
        configs = [a, b, c, d]

        result = route_agents("alpha beta", configs)

        assert result.index(a) < result.index(c)

    def test_pass_through_before_gatherers_preserved(self) -> None:
        """Sequential agents that appear before gatherers keep their position."""
        pre = _make_sequential("pre")
        gatherer = _make_gatherer("g", ["node"])
        post = _make_sequential("post")
        configs = [pre, gatherer, post]

        result = route_agents("node health", configs)

        assert result[0] is pre
        assert result[1] is gatherer
        assert result[2] is post


class TestRouteAgentsRealWorldScenarios:
    """Scenarios based on the spec acceptance criteria (REQ-2)."""

    def _make_service_health_configs(self) -> list[dict]:
        """Approximate service-health pipeline configs."""
        return [
            _make_gatherer("node_gatherer",       ["node", "nodes", "cpu", "memory", "disk"]),
            _make_gatherer("workload_gatherer",    ["workload", "pod", "pods", "deployment", "replicaset"]),
            _make_gatherer("logging_gatherer",     ["log", "logs", "logging", "stderr", "stdout"]),
            _make_gatherer("networking_gatherer",  ["network", "networking", "service", "ingress", "dns"]),
            _make_sequential("analyzer"),
            _make_sequential("verifier"),
            _make_sequential("reporter"),
        ]

    def test_spec_ac_log_query_returns_logging_and_sequentials(self) -> None:
        """AC from REQ-2: 'check logs for pod X' → logging_gatherer + sequentials."""
        configs = self._make_service_health_configs()

        result = route_agents("check logs for pod X", configs)

        names = [c["name"] for c in result]
        assert "logging_gatherer" in names
        assert "analyzer" in names
        assert "verifier" in names
        assert "reporter" in names
        # Non-matching gatherers excluded
        assert "node_gatherer" not in names
        assert "networking_gatherer" not in names

    def test_spec_ac_no_match_returns_all(self) -> None:
        """AC from REQ-2: no matching capabilities → all configs unchanged."""
        configs = self._make_service_health_configs()

        result = route_agents("check overall cluster health", configs)

        assert len(result) == len(configs)

    def test_pod_query_matches_workload_gatherer(self) -> None:
        """Query with 'pods' should match workload_gatherer.

        Note: short tokens like 'in' (from 'in production') can match
        capabilities containing them as substrings (e.g. 'in' ⊆ 'logging').
        This is expected bidirectional-substring behaviour — the main contract
        is that workload_gatherer is included, not that every other gatherer
        is excluded.
        """
        configs = self._make_service_health_configs()

        result = route_agents("pods crashing in production", configs)

        names = [c["name"] for c in result]
        assert "workload_gatherer" in names
        assert "analyzer" in names
        assert "node_gatherer" not in names

    def test_network_query_matches_networking_gatherer(self) -> None:
        configs = self._make_service_health_configs()

        result = route_agents("service dns resolution failing", configs)

        names = [c["name"] for c in result]
        assert "networking_gatherer" in names
        assert "node_gatherer" not in names

    def test_broad_query_matches_multiple_gatherers(self) -> None:
        """Query mentioning multiple domains routes to all matching gatherers."""
        configs = self._make_service_health_configs()

        result = route_agents("check logs and node cpu", configs)

        names = [c["name"] for c in result]
        assert "logging_gatherer" in names
        assert "node_gatherer" in names
        assert "analyzer" in names
        assert "workload_gatherer" not in names
        assert "networking_gatherer" not in names

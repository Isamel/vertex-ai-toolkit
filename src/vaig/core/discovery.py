"""Discovery query builder — shared by CLI discover and fleet runner.

Extracted from ``vaig.cli.commands.discover`` so that
:mod:`vaig.core.fleet` can import it without a core → CLI layer inversion.
"""

from __future__ import annotations

from vaig.core.prompt_defense import _sanitize_namespace
from vaig.skills.discovery.prompts import SYSTEM_NAMESPACES_CSV

# ── Default auto-generated queries ────────────────────────────

_QUERY_SINGLE_NS = (
    "Scan namespace '{namespace}' and discover all workloads. "
    "Enumerate Deployments, StatefulSets, DaemonSets, and Services. "
    "Classify each workload as Healthy, Degraded, or Failing. "
    "Investigate any non-healthy workloads by checking pods, logs, events, "
    "and resource usage. Produce a comprehensive cluster health report."
)

_QUERY_ALL_NS = (
    "Scan ALL non-system namespaces and discover all workloads. "
    "Skip these system namespaces: {system_ns}. "
    "Enumerate Deployments, StatefulSets, DaemonSets, and Services in each namespace. "
    "Classify each workload as Healthy, Degraded, or Failing. "
    "Investigate any non-healthy workloads by checking pods, logs, events, "
    "and resource usage. Produce a comprehensive cluster health report."
)

_SKIP_HEALTHY_SUFFIX = (
    " Focus the report on 🟡 Degraded and 🔴 Failing workloads only. "
    "Do NOT include detailed output for healthy workloads — just a count."
)


def build_discover_query(
    *,
    namespace: str | None = None,
    all_namespaces: bool = False,
    skip_healthy: bool = False,
) -> str:
    """Build the auto-generated investigation query for the discovery pipeline.

    Args:
        namespace: Target namespace (used when *all_namespaces* is False).
        all_namespaces: When True, scan all non-system namespaces.
        skip_healthy: When True, append instructions to omit healthy workloads.

    Returns:
        A natural-language query string for the agent pipeline.

    Raises:
        ValueError: If the namespace name is invalid.
    """
    if all_namespaces:
        query = _QUERY_ALL_NS.format(system_ns=SYSTEM_NAMESPACES_CSV)
    else:
        ns = namespace or "default"
        safe_ns = _sanitize_namespace(ns)
        if not safe_ns:
            raise ValueError(
                f"Invalid namespace name: {ns!r}. "
                "Namespace must contain only lowercase alphanumeric characters or hyphens, "
                "start and end with an alphanumeric character, and be at most 63 characters."
            )
        query = _QUERY_SINGLE_NS.format(namespace=safe_ns)

    if skip_healthy:
        query += _SKIP_HEALTHY_SUFFIX

    return query

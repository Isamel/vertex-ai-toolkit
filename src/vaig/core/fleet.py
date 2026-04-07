"""Fleet scanning — multi-cluster discovery with cross-cluster correlation.

Provides :class:`FleetRunner` to iterate (sequentially or in parallel) over
a list of :class:`~vaig.core.config.FleetCluster` entries, call
:func:`~vaig.core.headless.execute_skill_headless` per cluster, and produce
a :class:`FleetReport` with optional cross-cluster correlations.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vaig.agents.orchestrator import OrchestratorResult
    from vaig.core.config import FleetCluster, FleetConfig, Settings
    from vaig.skills.service_health.schema import Finding

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────


@dataclass
class ClusterResult:
    """Result of scanning a single cluster."""

    cluster_name: str
    display_name: str
    status: Literal["success", "error", "skipped"]
    result: OrchestratorResult | None = None
    error: str | None = None
    duration_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class FleetCorrelation:
    """A cross-cluster pattern detected by the deterministic correlator."""

    pattern: str
    category: str
    affected_clusters: list[str] = field(default_factory=list)
    count: int = 0


@dataclass
class FleetReport:
    """Aggregated fleet scan results."""

    clusters: list[ClusterResult] = field(default_factory=list)
    correlations: list[FleetCorrelation] = field(default_factory=list)
    total_duration_s: float = 0.0
    total_cost_usd: float = 0.0
    budget_exceeded: bool = False


# ── Cross-cluster correlation ─────────────────────────────────


def _normalize_symptom(text: str) -> str:
    """Normalize a symptom string for grouping (lowercase, strip whitespace)."""
    return text.strip().lower()


def _extract_findings(result: ClusterResult) -> list[Finding]:
    """Extract :class:`Finding` instances from a successful cluster result."""
    if result.status != "success" or result.result is None:
        return []
    report = result.result.structured_report
    if report is None:
        return []
    # HealthReport has a `findings` attribute
    findings = getattr(report, "findings", None)
    if findings is None:
        return []
    return list(findings)


def correlate(cluster_results: list[ClusterResult]) -> list[FleetCorrelation]:
    """Group findings by (category, normalized symptom) across clusters.

    A :class:`FleetCorrelation` is emitted when ≥2 clusters share the same
    ``(category, normalized_title)`` pair.  Deterministic, no AI.

    Args:
        cluster_results: List of per-cluster scan results.

    Returns:
        List of cross-cluster correlations (may be empty).
    """
    # group_key → {cluster_names set, total count}
    groups: dict[tuple[str, str], dict[str, set[str] | int]] = {}

    for cr in cluster_results:
        findings = _extract_findings(cr)
        for finding in findings:
            category = finding.category or "unknown"
            symptom = _normalize_symptom(finding.title)
            key = (category, symptom)

            if key not in groups:
                groups[key] = {"clusters": set(), "count": 0}
            entry = groups[key]
            clusters_set: set[str] = entry["clusters"]  # type: ignore[assignment]
            clusters_set.add(cr.display_name)
            entry["count"] = int(entry["count"]) + 1  # type: ignore[arg-type]

    correlations: list[FleetCorrelation] = []
    for (category, symptom), info in groups.items():
        affected: set[str] = info["clusters"]  # type: ignore[assignment]
        if len(affected) >= 2:
            correlations.append(
                FleetCorrelation(
                    pattern=symptom,
                    category=category,
                    affected_clusters=sorted(affected),
                    count=int(info["count"]),  # type: ignore[arg-type]
                )
            )

    # Sort by count descending for display
    correlations.sort(key=lambda c: c.count, reverse=True)
    return correlations


# ── Fleet Runner ──────────────────────────────────────────────


class FleetRunner:
    """One-shot multi-cluster scanner.

    Reuses :func:`~vaig.core.headless.execute_skill_headless` per cluster
    and collects results into a :class:`FleetReport`.
    """

    def _scan_one(self, settings: Settings, cluster: FleetCluster) -> ClusterResult:
        """Scan a single cluster, catching all exceptions.

        Builds a :class:`~vaig.core.config.GKEConfig` from the
        :class:`FleetCluster` (inheriting ``project_id``/``location`` from
        ``settings.gcp`` when empty), resolves the discovery skill, builds
        the query, and calls ``execute_skill_headless``.

        Args:
            settings: Application settings.
            cluster: Fleet cluster to scan.

        Returns:
            :class:`ClusterResult` — ``status="success"`` or ``status="error"``.
        """
        start = time.monotonic()
        try:
            from vaig.cli.commands.discover import _build_discover_query
            from vaig.core.gke import build_gke_config
            from vaig.core.headless import execute_skill_headless
            from vaig.skills.discovery.skill import DiscoverySkill

            # Build GKE config — inherit project/location from settings when empty
            gke_config = build_gke_config(
                settings,
                cluster=cluster.cluster_name,
                namespace=cluster.namespace or None,
                project_id=cluster.project_id or None,
                location=cluster.location or None,
            )

            # Override kubeconfig / context / impersonate if set on the cluster
            if cluster.kubeconfig_path:
                gke_config = gke_config.model_copy(
                    update={"kubeconfig_path": cluster.kubeconfig_path}
                )
            if cluster.context:
                gke_config = gke_config.model_copy(update={"context": cluster.context})
            if cluster.impersonate_sa:
                gke_config = gke_config.model_copy(
                    update={"impersonate_sa": cluster.impersonate_sa}
                )

            # Build query
            query = _build_discover_query(
                namespace=cluster.namespace or None,
                all_namespaces=cluster.all_namespaces,
                skip_healthy=cluster.skip_healthy,
            )

            # Resolve discovery skill
            skill = DiscoverySkill()

            # Execute
            result = execute_skill_headless(settings, skill, query, gke_config)

            duration = time.monotonic() - start
            return ClusterResult(
                cluster_name=cluster.cluster_name,
                display_name=cluster.name,
                status="success",
                result=result,
                duration_s=duration,
                cost_usd=result.run_cost_usd,
            )

        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start
            logger.warning(
                "Fleet scan failed for cluster %s: %s",
                cluster.name,
                exc,
                exc_info=True,
            )
            return ClusterResult(
                cluster_name=cluster.cluster_name,
                display_name=cluster.name,
                status="error",
                error=str(exc),
                duration_s=duration,
            )

    def run(
        self,
        settings: Settings,
        fleet_config: FleetConfig,
    ) -> FleetReport:
        """Scan all clusters sequentially.

        Continues to the next cluster on failure (no fail-fast).
        When ``fleet_config.daily_budget_usd > 0``, tracks cumulative cost
        and skips remaining clusters once the budget is exceeded.

        Args:
            settings: Application settings.
            fleet_config: Fleet configuration with cluster list.

        Returns:
            :class:`FleetReport` with per-cluster results and correlations.
        """
        start = time.monotonic()
        results: list[ClusterResult] = []
        cumulative_cost = 0.0
        budget = fleet_config.daily_budget_usd
        budget_exceeded = False

        for cluster in fleet_config.clusters:
            # Budget check before scanning
            if budget > 0 and cumulative_cost >= budget:
                budget_exceeded = True
                results.append(
                    ClusterResult(
                        cluster_name=cluster.cluster_name,
                        display_name=cluster.name,
                        status="skipped",
                        error="Budget exceeded",
                    )
                )
                logger.info(
                    "Skipping cluster %s — budget exceeded ($%.2f / $%.2f)",
                    cluster.name,
                    cumulative_cost,
                    budget,
                )
                continue

            cr = self._scan_one(settings, cluster)
            cumulative_cost += cr.cost_usd
            results.append(cr)

        # Post-scan correlation
        correlations = correlate(results)

        total_duration = time.monotonic() - start
        return FleetReport(
            clusters=results,
            correlations=correlations,
            total_duration_s=total_duration,
            total_cost_usd=cumulative_cost,
            budget_exceeded=budget_exceeded,
        )

    def run_parallel(
        self,
        settings: Settings,
        fleet_config: FleetConfig,
    ) -> FleetReport:
        """Scan all clusters concurrently via :class:`ThreadPoolExecutor`.

        Each thread creates its own credentials (no shared google-auth state).
        Budget tracking is approximate in parallel mode — all clusters are
        submitted upfront but cost is tallied after completion.

        Args:
            settings: Application settings.
            fleet_config: Fleet configuration with cluster list.

        Returns:
            :class:`FleetReport` with per-cluster results and correlations.
        """
        start = time.monotonic()
        max_workers = fleet_config.max_workers or 4

        results: list[ClusterResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cluster = {
                executor.submit(self._scan_one, settings, cluster): cluster
                for cluster in fleet_config.clusters
            }

            for future in as_completed(future_to_cluster):
                cr = future.result()
                results.append(cr)

        # Sort results back to config order
        cluster_order = {c.cluster_name: i for i, c in enumerate(fleet_config.clusters)}
        results.sort(key=lambda r: cluster_order.get(r.cluster_name, 999))

        total_cost = sum(r.cost_usd for r in results)

        # Budget check — retroactive in parallel mode
        budget = fleet_config.daily_budget_usd
        budget_exceeded = budget > 0 and total_cost > budget

        # Post-scan correlation
        correlations = correlate(results)

        total_duration = time.monotonic() - start
        return FleetReport(
            clusters=results,
            correlations=correlations,
            total_duration_s=total_duration,
            total_cost_usd=total_cost,
            budget_exceeded=budget_exceeded,
        )

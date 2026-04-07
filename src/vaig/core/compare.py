"""Cross-cluster deployment comparison — snapshot, diff, and runner.

Provides :class:`CompareRunner` to collect :class:`DeploymentSnapshot`
from multiple clusters in parallel and produce a :class:`CompareReport`
with severity-annotated :class:`FieldDiff` entries.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from vaig.core.config import FleetCluster, GKEConfig, Settings

logger = logging.getLogger(__name__)

# ── Data Models ───────────────────────────────────────────────


class DeploymentSnapshot(BaseModel):
    """Point-in-time state of a Deployment on a single cluster (REQ-CMP-01)."""

    cluster_name: str
    namespace: str
    deployment_name: str

    # Core
    replicas_desired: int = 0
    replicas_ready: int = 0
    image_tag: str = ""
    rollout_generation: int = 0

    # HPA (optional — None when no HPA)
    hpa_min: int | None = None
    hpa_max: int | None = None

    # Metrics (optional — None when metrics unavailable)
    cpu_usage_cores: float | None = None
    memory_usage_gib: float | None = None
    error_rate_pct: float | None = None

    # Meta
    collected_at: datetime = datetime.min


@dataclass
class FieldDiff:
    """A single divergent field across clusters (REQ-CMP-04)."""

    field: str
    values: dict[str, Any]  # cluster_name → value
    severity: Literal["critical", "warning", "info"]


@dataclass
class CompareMetadata:
    """Metadata for a comparison run."""

    timestamp: str = ""
    clusters_requested: list[str] = field(default_factory=list)
    namespace: str = ""
    deployment: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=UTC).isoformat()


@dataclass
class CompareReport:
    """Aggregated comparison result (REQ-CMP-03)."""

    snapshots: dict[str, DeploymentSnapshot] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    diffs: list[FieldDiff] = field(default_factory=list)
    metadata: CompareMetadata = field(default_factory=CompareMetadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "snapshots": {
                name: snap.model_dump(mode="json") for name, snap in self.snapshots.items()
            },
            "errors": self.errors,
            "diffs": [
                {"field": d.field, "values": d.values, "severity": d.severity}
                for d in self.diffs
            ],
            "metadata": {
                "timestamp": self.metadata.timestamp,
                "clusters_requested": self.metadata.clusters_requested,
                "namespace": self.metadata.namespace,
                "deployment": self.metadata.deployment,
            },
        }


# ── Snapshot Collection (REQ-CMP-02) ─────────────────────────


def collect_deployment_snapshot(
    gke_config: GKEConfig,
    namespace: str,
    deployment: str,
) -> DeploymentSnapshot:
    """Query K8s APIs to build a DeploymentSnapshot for one cluster.

    Uses :func:`~vaig.tools.gke._clients._create_k8s_clients` directly.
    Gracefully degrades when HPA or metrics are unavailable (REQ-CMP-09).

    Args:
        gke_config: GKE cluster configuration.
        namespace: Kubernetes namespace.
        deployment: Deployment name.

    Returns:
        Assembled :class:`DeploymentSnapshot`.

    Raises:
        RuntimeError: If K8s client creation fails or deployment not found.
    """
    from vaig.tools.base import ToolResult
    from vaig.tools.gke._clients import _create_k8s_clients

    result = _create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        msg = f"K8s client error for {gke_config.cluster_name}: {result.output}"
        raise RuntimeError(msg)

    core_v1, apps_v1, _custom, _api_client = result

    # ── Read Deployment ──────────────────────────────────────
    try:
        dep = apps_v1.read_namespaced_deployment(
            name=deployment,
            namespace=namespace,
            _request_timeout=gke_config.request_timeout,
        )
    except Exception as exc:
        msg = f"Failed to read deployment {deployment} in {namespace} on {gke_config.cluster_name}: {exc}"
        raise RuntimeError(msg) from exc

    spec = dep.spec or type("", (), {"replicas": 0, "template": None})()
    status = dep.status or type("", (), {"ready_replicas": 0, "observed_generation": 0})()

    replicas_desired = spec.replicas or 0
    replicas_ready = status.ready_replicas or 0
    observed_gen = status.observed_generation or 0

    # Extract image tag from the first container
    image_tag = ""
    template = getattr(spec, "template", None)
    if template and template.spec and template.spec.containers:
        full_image = template.spec.containers[0].image or ""
        image_tag = full_image.split(":")[-1] if ":" in full_image else full_image

    # ── HPA (optional) ───────────────────────────────────────
    hpa_min: int | None = None
    hpa_max: int | None = None
    try:
        from kubernetes.client import AutoscalingV1Api

        autoscaling = AutoscalingV1Api(_api_client)
        hpa = autoscaling.read_namespaced_horizontal_pod_autoscaler(
            name=deployment,
            namespace=namespace,
            _request_timeout=gke_config.request_timeout,
        )
        hpa_min = hpa.spec.min_replicas if hpa.spec else None
        hpa_max = hpa.spec.max_replicas if hpa.spec else None
    except Exception:  # noqa: BLE001
        logger.debug("No HPA found for %s/%s on %s", namespace, deployment, gke_config.cluster_name)

    return DeploymentSnapshot(
        cluster_name=gke_config.cluster_name,
        namespace=namespace,
        deployment_name=deployment,
        replicas_desired=replicas_desired,
        replicas_ready=replicas_ready,
        image_tag=image_tag,
        rollout_generation=observed_gen,
        hpa_min=hpa_min,
        hpa_max=hpa_max,
        cpu_usage_cores=None,  # metrics enrichment deferred to v2
        memory_usage_gib=None,
        error_rate_pct=None,
        collected_at=datetime.now(tz=UTC),
    )


# ── Diff Engine (REQ-CMP-04) ─────────────────────────────────

_NA_SENTINEL = "N/A"

# Fields eligible for comparison and their severity classifiers.
_COMPARABLE_FIELDS: list[str] = [
    "image_tag",
    "replicas_desired",
    "replicas_ready",
    "hpa_min",
    "hpa_max",
    "cpu_usage_cores",
    "memory_usage_gib",
    "error_rate_pct",
    "rollout_generation",
]


def _delta_pct(values: dict[str, Any]) -> float:
    """Compute the max-spread percentage of numeric values.

    Returns 0.0 if fewer than two numeric values exist.
    """
    nums = [v for v in values.values() if isinstance(v, (int, float))]
    if len(nums) < 2:
        return 0.0
    max_val = max(abs(n) for n in nums)
    if max_val == 0:
        return 0.0
    spread = max(nums) - min(nums)
    return (spread / max_val) * 100.0


_SEVERITY_MAP: dict[str, Callable[[dict[str, Any]], Literal["critical", "warning", "info"]]] = {
    "image_tag": lambda _: "critical",
    "replicas_desired": lambda vals: "critical" if _delta_pct(vals) > 50 else "warning",
    "replicas_ready": lambda _: "warning",
    "hpa_min": lambda _: "warning",
    "hpa_max": lambda _: "warning",
    "cpu_usage_cores": lambda _: "info",
    "memory_usage_gib": lambda _: "info",
    "error_rate_pct": lambda _: "info",
    "rollout_generation": lambda _: "info",
}


def diff_snapshots(snapshots: dict[str, DeploymentSnapshot]) -> list[FieldDiff]:
    """Compare field values across snapshots and return divergent diffs (REQ-CMP-04).

    None values are replaced with the ``"N/A"`` sentinel for comparison
    so that missing HPA/metrics are surfaced as divergences (REQ-CMP-09).

    Args:
        snapshots: Mapping of cluster_name → DeploymentSnapshot.

    Returns:
        List of :class:`FieldDiff` for fields where values differ.
    """
    if len(snapshots) < 2:
        return []

    diffs: list[FieldDiff] = []

    for field_name in _COMPARABLE_FIELDS:
        values: dict[str, Any] = {}
        for cluster_name, snap in snapshots.items():
            raw = getattr(snap, field_name, None)
            values[cluster_name] = raw if raw is not None else _NA_SENTINEL

        # Check if all values are the same
        unique = {str(v) for v in values.values()}
        if len(unique) <= 1:
            continue

        def _default_severity(_: dict[str, Any]) -> Literal["critical", "warning", "info"]:
            return "info"

        severity_fn = _SEVERITY_MAP.get(field_name, _default_severity)
        severity: Literal["critical", "warning", "info"] = severity_fn(values)

        diffs.append(FieldDiff(field=field_name, values=values, severity=severity))

    return diffs


# ── CompareRunner (REQ-CMP-03) ────────────────────────────────


class CompareRunner:
    """Parallel snapshot collector + diff producer.

    Follows the :class:`~vaig.core.fleet.FleetRunner` pattern: accept
    a list of :class:`FleetCluster`, build per-cluster :class:`GKEConfig`,
    collect in parallel, then diff.
    """

    def __init__(
        self,
        clusters: list[FleetCluster],
        namespace: str,
        deployment: str,
        max_workers: int = 4,
        settings: Settings | None = None,
    ) -> None:
        self._clusters = clusters
        self._namespace = namespace
        self._deployment = deployment
        self._max_workers = max_workers
        self._settings = settings

    def _collect_one(self, cluster: FleetCluster) -> tuple[str, DeploymentSnapshot | None, str | None]:
        """Collect snapshot for a single cluster, catching errors (REQ-CMP-08)."""
        try:
            from vaig.core.gke import build_gke_config

            if self._settings is None:
                from vaig.core.config import get_settings
                settings = get_settings()
            else:
                settings = self._settings

            gke_config = build_gke_config(
                settings,
                cluster=cluster.cluster_name,
                namespace=cluster.namespace or None,
                project_id=cluster.project_id or None,
                location=cluster.location or None,
            )

            # Apply kubeconfig / context overrides
            updates: dict[str, Any] = {}
            if cluster.kubeconfig_path:
                updates["kubeconfig_path"] = cluster.kubeconfig_path
            if cluster.context:
                updates["context"] = cluster.context
            if cluster.impersonate_sa:
                updates["impersonate_sa"] = cluster.impersonate_sa
            if updates:
                gke_config = gke_config.model_copy(update=updates)

            snapshot = collect_deployment_snapshot(gke_config, self._namespace, self._deployment)
            return (cluster.name, snapshot, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Compare: failed to collect from %s: %s",
                cluster.name,
                exc,
                exc_info=True,
            )
            return (cluster.name, None, str(exc))

    def run_parallel(self) -> CompareReport:
        """Collect snapshots in parallel and produce a CompareReport.

        Per-cluster failures are captured in ``errors`` without
        aborting the comparison (REQ-CMP-08).
        """
        start = time.monotonic()
        snapshots: dict[str, DeploymentSnapshot] = {}
        errors: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_cluster = {
                executor.submit(self._collect_one, cluster): cluster
                for cluster in self._clusters
            }

            for future in as_completed(future_to_cluster):
                display_name, snapshot, error = future.result()
                if snapshot is not None:
                    snapshots[display_name] = snapshot
                elif error is not None:
                    errors[display_name] = error

        # Sort snapshots to config order
        cluster_order = {c.name: i for i, c in enumerate(self._clusters)}
        snapshots = dict(sorted(snapshots.items(), key=lambda kv: cluster_order.get(kv[0], 999)))

        # Diff only successful snapshots
        diffs = diff_snapshots(snapshots)

        duration = time.monotonic() - start
        metadata = CompareMetadata(
            clusters_requested=[c.name for c in self._clusters],
            namespace=self._namespace,
            deployment=self._deployment,
        )

        logger.info(
            "Compare completed: %d snapshots, %d errors, %d diffs in %.1fs",
            len(snapshots),
            len(errors),
            len(diffs),
            duration,
        )

        return CompareReport(
            snapshots=snapshots,
            errors=errors,
            diffs=diffs,
            metadata=metadata,
        )

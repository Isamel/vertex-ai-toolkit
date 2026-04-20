"""Shared report-metadata injection for CLI and web paths.

Extracts runtime context (cost metrics, tool usage, GKE cost estimation,
anomaly trends, header fields) into a ``HealthReport.metadata`` object so
that the HTML renderer can populate every section of the SPA report.

This module is intentionally dependency-light: all schema imports are
deferred so that it can be imported cheaply from both the CLI and the
web server.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime
from typing import Any

__all__ = ["inject_report_metadata", "format_models_used"]

logger = logging.getLogger(__name__)


def _extract_target_from_question(question: str) -> str:
    """Extract a Kubernetes resource target from a question string.

    Looks for patterns like ``pod/name``, ``deployment/name``, or just
    returns the first 60 chars of the question as a fallback.
    """
    import re  # noqa: PLC0415

    match = re.search(
        r"\b(pod|deployment|statefulset|daemonset|service|node)/[\w.-]+",
        question,
        re.IGNORECASE,
    )
    if match:
        return match.group(0)
    return question[:60] if question else ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def format_models_used(models_used: list[str]) -> str:
    """Format a list of model IDs for display.

    Returns a compact human-readable string:
    - Single unique model → returned as-is (e.g. ``"gemini-2.5-flash"``).
    - All agents use the same model → ``"gemini-2.5-flash ×7"``.
    - Multiple distinct models → comma-separated list.
    - Empty list → empty string.
    """
    if not models_used:
        return ""
    counts = Counter(models_used)
    if len(counts) == 1:
        model, n = next(iter(counts.items()))
        return f"{model} ×{n}" if n > 1 else model
    return ", ".join(sorted(counts))


# ── Main injection function ──────────────────────────────────────────────────


def inject_report_metadata(
    report: Any,
    *,
    gke_config: Any = None,
    model_id: str = "",
    orch_result: Any = None,
    tool_logger: Any = None,
    cost_namespaces: list[str] | None = None,
) -> None:
    """Fill metadata fields in *report* from runtime context.

    System-authoritative fields (``model_used``, ``cluster_name``,
    ``project_id``, ``generated_at``) are ALWAYS overwritten when the runtime
    has an authoritative value — the LLM may hallucinate these fields so we
    never trust whatever it wrote.  Cost and tool-usage fields are only set
    when not already populated (they are never known to the LLM).

    Args:
        report: A ``HealthReport`` instance (typed as ``Any`` to avoid a hard
            import; we access ``report.metadata`` defensively).
        gke_config: Optional :class:`~vaig.core.config.GKEConfig`.  When
            provided, its ``cluster_name`` and ``project_id`` fields are used
            to overwrite the corresponding metadata slots unconditionally.
        model_id: The model identifier used for the run.  Always overwrites
            ``metadata.model_used`` when a value is available.
        orch_result: Optional :class:`~vaig.agents.orchestrator.OrchestratorResult`.
            When provided, ``run_cost_usd`` and ``total_usage`` are extracted to
            populate ``metadata.cost_metrics``.
        tool_logger: Optional :class:`ToolCallLogger`.  When provided, its
            ``tool_name_counts`` and ``tool_count`` are used to populate
            ``metadata.tool_usage``.
        cost_namespaces: Optional explicit namespace list for cost estimation.
            ``None`` means "use the default_namespace from gke_config (if set)".
            Pass an empty list to analyse all non-system namespaces.
    """
    metadata = getattr(report, "metadata", None)
    if metadata is None:
        return

    def _is_empty(value: Any) -> bool:
        """Return True when *value* is falsy or the sentinel 'N/A'."""
        if not value:
            return True
        if isinstance(value, str) and value.strip().upper() == "N/A":
            return True
        return False

    if gke_config is not None:
        for attr in ["cluster_name", "project_id"]:
            value = getattr(gke_config, attr, None)
            # Overwrite unconditionally — even empty/None clears hallucinated values
            setattr(metadata, attr, value if value is not None else "")

    # ALWAYS overwrite model_used — the LLM may hallucinate this field.
    if orch_result is not None:
        actual_models = getattr(orch_result, "models_used", [])
        effective_model = format_models_used(actual_models) or model_id
    else:
        effective_model = model_id
    if effective_model:
        metadata.model_used = effective_model
    else:
        metadata.model_used = ""

    # ── Cost metrics ──────────────────────────────────────────
    if orch_result is not None and getattr(metadata, "cost_metrics", None) is None:
        from vaig.skills.service_health.schema import CostMetrics  # noqa: PLC0415

        run_cost = getattr(orch_result, "run_cost_usd", None)
        total_usage = getattr(orch_result, "total_usage", None)
        total_tokens: int | None = None
        if isinstance(total_usage, dict):
            total_tokens = total_usage.get("total_tokens") or (
                total_usage.get("prompt_tokens", 0) + total_usage.get("completion_tokens", 0)
            ) or None
        cost_str = f"${run_cost:.6f}" if (run_cost is not None and run_cost > 0) else None

        if run_cost is not None or total_tokens is not None:
            metadata.cost_metrics = CostMetrics(
                run_cost_usd=run_cost,
                total_tokens=total_tokens,
                estimated_cost=cost_str,
            )

    # ── Tool usage ────────────────────────────────────────────
    if tool_logger is not None and getattr(metadata, "tool_usage", None) is None:
        from vaig.skills.service_health.schema import ToolUsageSummary  # noqa: PLC0415

        pipeline_counts = dict(getattr(tool_logger, "pipeline_tool_name_counts", {}))
        live_counts = dict(getattr(tool_logger, "tool_name_counts", {}))
        tool_counts = (pipeline_counts or live_counts) or None
        tool_calls = getattr(tool_logger, "tool_count", None)
        if tool_calls == 0:
            tool_calls = None

        if tool_counts is not None or tool_calls is not None:
            metadata.tool_usage = ToolUsageSummary(
                tool_counts=tool_counts,
                tool_calls=tool_calls,
            )

    # ── Resolve effective namespaces (shared by cost estimation & trend analysis) ──
    if gke_config is not None:
        if cost_namespaces is None:
            # Not using --all-namespaces or an explicit list, so fall back to
            # the default namespace from config.
            default_ns = getattr(gke_config, "default_namespace", None)
            effective_namespaces: list[str] | None = [default_ns] if default_ns else None
        else:
            # An empty list from --all-namespaces means "all namespaces",
            # which is represented by ``None``.
            # A non-empty list is an explicit filter.
            effective_namespaces = cost_namespaces or None
    else:
        effective_namespaces = None

    # ── GKE workload cost estimation ──────────────────────────
    if gke_config is not None and getattr(metadata, "gke_cost", None) is None:
        try:
            from vaig.tools.gke.cost_estimation import fetch_workload_costs  # noqa: PLC0415

            metadata.gke_cost = fetch_workload_costs(gke_config, namespaces=effective_namespaces)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as _gke_cost_exc:  # noqa: BLE001
            logger.debug("GKE cost estimation failed: %s", _gke_cost_exc)
            try:
                from vaig.skills.service_health.schema import GKECostReport  # noqa: PLC0415

                metadata.gke_cost = GKECostReport(
                    supported=False,
                    cluster_type="unknown",
                    unsupported_reason=str(_gke_cost_exc),
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:  # noqa: BLE001
                pass  # schema import failed — leave gke_cost unset

    # ── Anomaly trend detection ───────────────────────────────
    trend_cfg = getattr(gke_config, "trends", None) if gke_config is not None else None
    if trend_cfg is not None and getattr(trend_cfg, "enabled", False):
        try:
            from vaig.tools.gke.trend_analysis import fetch_anomaly_trends  # noqa: PLC0415

            metadata.trends = fetch_anomaly_trends(gke_config, namespaces=effective_namespaces)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as _trend_exc:  # noqa: BLE001
            logger.debug("Trend analysis failed: %s", _trend_exc)

    # ── Generated-at timestamp — ALWAYS overwrite with actual time ────────────
    metadata.generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ── Skill version ─────────────────────────────────────────
    if _is_empty(getattr(metadata, "skill_version", None)):
        from vaig import __version__  # noqa: PLC0415

        metadata.skill_version = f"vaig {__version__}"

    # ── AUDIT-07: Pipeline version (git short SHA or package version) ─────────
    if _is_empty(getattr(metadata, "pipeline_version", None)) or getattr(metadata, "pipeline_version", None) == "unknown":
        try:
            import subprocess  # noqa: PLC0415

            result = subprocess.run(  # noqa: S603
                ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                metadata.pipeline_version = result.stdout.strip()
            else:
                from vaig import __version__ as _vaig_version  # noqa: PLC0415

                metadata.pipeline_version = _vaig_version
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001
            try:
                from vaig import __version__ as _vaig_version  # noqa: PLC0415

                metadata.pipeline_version = _vaig_version
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:  # noqa: BLE001
                metadata.pipeline_version = "unknown"

    # ── AUDIT-07: model_versions map (agent-name → resolved model-id) ─────────
    if orch_result is not None and not getattr(metadata, "model_versions", None):
        models_by_agent: dict[str, str] = getattr(orch_result, "models_by_agent", {})
        if isinstance(models_by_agent, dict) and models_by_agent:
            metadata.model_versions = dict(models_by_agent)
        elif model_id:
            metadata.model_versions = {"health_analyzer": model_id}
    elif not getattr(metadata, "model_versions", None) and model_id:
        metadata.model_versions = {"health_analyzer": model_id}

    # ── AUDIT-11: Autonomous mode visibility ──────────────────────────────────
    if orch_result is not None:
        agent_results: list[Any] = getattr(orch_result, "agent_results", []) or []
        investig_result: Any = None
        for ar in agent_results:
            if getattr(ar, "agent_name", "") == "health_investigator":
                investig_result = ar
                break
        if investig_result is not None:
            investig_meta: dict[str, Any] = getattr(investig_result, "metadata", {}) or {}
            metadata.autonomous_enabled = True
            metadata.autonomous_steps_executed = investig_meta.get("steps_completed")
            metadata.autonomous_replan_iterations = investig_meta.get("replan_count")

            # Populate investigation_evidence from the EvidenceLedger (final_state)
            final_state = getattr(orch_result, "final_state", None)
            ledger = getattr(final_state, "evidence_ledger", None) if final_state is not None else None
            if ledger is not None and hasattr(report, "investigation_evidence"):
                try:
                    from vaig.skills.service_health.schema import (  # noqa: PLC0415
                        InvestigationEvidenceSnapshot,
                    )

                    snapshots = []
                    for entry in ledger.entries:
                        supports = getattr(entry, "supports", ())
                        contradicts = getattr(entry, "contradicts", ())
                        if supports:
                            verdict: str = "CONFIRMED"
                        elif contradicts:
                            verdict = "CONTRADICTED"
                        else:
                            verdict = "INCONCLUSIVE"
                        step_id = getattr(entry, "id", "")
                        question = getattr(entry, "question", "")
                        answer = getattr(entry, "answer_summary", "")
                        tool = getattr(entry, "tool_name", "")
                        # Extract target from question heuristically (first k8s resource pattern)
                        target = _extract_target_from_question(question)
                        snapshots.append(
                            InvestigationEvidenceSnapshot(
                                step_id=step_id,
                                target=target,
                                tool_name=tool,
                                hypothesis=question[:320],
                                verdict=verdict,  # type: ignore[arg-type]
                                answer_preview=answer[:320],
                                iteration=0,
                            )
                        )
                    report.investigation_evidence = snapshots
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as _inv_exc:  # noqa: BLE001
                    logger.debug("Failed to build investigation_evidence: %s", _inv_exc)

    # ── Cluster overview: inject Namespace row when missing ───
    if gke_config is not None:
        ns = getattr(gke_config, "default_namespace", None)
        if isinstance(ns, str) and ns and hasattr(report, "cluster_overview") and report.cluster_overview is not None:
            from vaig.skills.service_health.schema import ClusterMetric  # noqa: PLC0415

            already_has_ns = any(
                isinstance(row.metric, str) and row.metric.strip().lower() == "namespace"
                for row in report.cluster_overview
            )
            if not already_has_ns:
                report.cluster_overview.insert(
                    0, ClusterMetric(metric="Namespace", value=ns)
                )

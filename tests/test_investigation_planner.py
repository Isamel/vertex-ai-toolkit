"""Tests for create_investigation_plan() (SPEC-SH-01).

Covers:
- Plan created from HealthReport with known root causes
- Root causes ordered before dependents
- Budget allocated across steps
- Empty findings → empty plan
- Budget cap per step is respected
"""

from __future__ import annotations

from vaig.core.config import GlobalBudgetConfig, SelfCorrectionConfig
from vaig.core.global_budget import GlobalBudgetManager
from vaig.core.hypothesis_library import HypothesisLibrary
from vaig.core.investigation import create_investigation_plan
from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    Finding,
    HealthReport,
    InvestigationPlan,
    OverallStatus,
    ReportMetadata,
    Severity,
    StepStatus,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_budget(max_cost_usd: float = 1.0) -> GlobalBudgetManager:
    cfg = GlobalBudgetConfig(max_cost_usd=max_cost_usd)
    return GlobalBudgetManager(cfg)


def _make_executive_summary() -> ExecutiveSummary:
    return ExecutiveSummary(
        overall_status=OverallStatus.DEGRADED,
        scope="Cluster-wide",
        summary_text="Test report",
        services_checked=3,
        issues_found=2,
    )


def _make_finding(
    finding_id: str,
    title: str,
    caused_by: list[str] | None = None,
    affected_resources: list[str] | None = None,
) -> Finding:
    return Finding(
        id=finding_id,
        title=title,
        severity=Severity.HIGH,
        caused_by=caused_by or [],
        affected_resources=affected_resources or [f"pod/{finding_id}-pod"],
    )


def _make_report(findings: list[Finding]) -> HealthReport:
    return HealthReport(
        executive_summary=_make_executive_summary(),
        findings=findings,
        metadata=ReportMetadata(generated_at="2026-04-18T00:00:00Z"),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCreateInvestigationPlan:
    __test__ = True

    def test_plan_created_from_report_with_three_root_causes(self) -> None:
        """Scenario: Plan created from HealthReport with known root causes."""
        findings = [
            _make_finding("f1", "OOMKilled pod", affected_resources=["pod/svc-oom"]),
            _make_finding("f2", "High latency detected", affected_resources=["pod/svc-lat"]),
            _make_finding("f3", "DNS resolution failing", affected_resources=["pod/svc-dns"]),
        ]
        report = _make_report(findings)
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        assert isinstance(plan, InvestigationPlan)
        assert len(plan.steps) == 3
        for step in plan.steps:
            assert step.tool_hint, "tool_hint should be non-empty"
            assert step.hypothesis, "hypothesis should be non-empty"

    def test_root_causes_ordered_before_dependents(self) -> None:
        """Scenario: Root causes ordered before dependents (A is root, B depends on A, C depends on B)."""
        f_root = _make_finding("a", "Root OOMKilled", caused_by=[])
        f_dep1 = _make_finding("b", "High latency caused by OOM", caused_by=["a"])
        f_dep2 = _make_finding("c", "DNS failure caused by latency", caused_by=["b"])

        report = _make_report([f_dep2, f_dep1, f_root])  # deliberately out-of-order
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        # Root cause (a) should come first (lowest priority number = 1)
        root_step = next(s for s in plan.steps if s.target == "pod/a-pod")
        dep1_step = next(s for s in plan.steps if s.target == "pod/b-pod")
        dep2_step = next(s for s in plan.steps if s.target == "pod/c-pod")

        assert root_step.priority <= dep1_step.priority
        assert dep1_step.priority <= dep2_step.priority

    def test_budget_allocated_across_steps(self) -> None:
        """Scenario: Budget allocated across steps, total <= remaining budget."""
        findings = [_make_finding(f"f{i}", f"Finding {i}") for i in range(4)]
        report = _make_report(findings)
        library = HypothesisLibrary.default()
        budget = _make_budget(max_cost_usd=1.0)

        plan = create_investigation_plan(report, library, budget)

        total = sum(s.budget_usd for s in plan.steps)
        assert total <= 1.0, f"Total budget {total} exceeds remaining budget 1.0"
        assert total > 0.0, "Budget should be > 0 when max_cost_usd is set"

    def test_empty_findings_returns_empty_plan(self) -> None:
        report = _make_report([])
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        assert len(plan.steps) == 0
        assert plan.total_budget_allocated == 0.0

    def test_budget_cap_per_step_is_respected(self) -> None:
        """Per-step budget is capped at SelfCorrectionConfig.max_budget_per_step_usd."""
        findings = [_make_finding(f"f{i}", f"Finding {i}") for i in range(2)]
        report = _make_report(findings)
        library = HypothesisLibrary.default()
        budget = _make_budget(max_cost_usd=100.0)  # huge budget

        cfg = SelfCorrectionConfig(max_budget_per_step_usd=0.05)
        plan = create_investigation_plan(report, library, budget, cfg)

        for step in plan.steps:
            assert step.budget_usd <= 0.05 + 1e-9, (
                f"Step budget {step.budget_usd} exceeds cap 0.05"
            )

    def test_all_steps_start_as_pending(self) -> None:
        findings = [_make_finding("f1", "OOMKilled"), _make_finding("f2", "Crash loop")]
        report = _make_report(findings)
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        assert all(s.status == StepStatus.pending for s in plan.steps)

    def test_plan_has_unique_step_ids(self) -> None:
        findings = [_make_finding(f"f{i}", f"Finding {i}") for i in range(5)]
        report = _make_report(findings)
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        ids = [s.step_id for s in plan.steps]
        assert len(ids) == len(set(ids)), "Step IDs should be unique"

    def test_created_from_uses_report_metadata(self) -> None:
        report = _make_report([])
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        assert plan.created_from == "2026-04-18T00:00:00Z"

    def test_oom_symptom_gets_kubectl_describe_hint(self) -> None:
        """OOM finding should be matched to kubectl_describe via hypothesis library."""
        finding = _make_finding("oom1", "OOMKilled pod in production")
        report = _make_report([finding])
        library = HypothesisLibrary.default()
        budget = _make_budget(1.0)

        plan = create_investigation_plan(report, library, budget)

        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.tool_hint == "kubectl_describe"

    def test_investigation_plan_is_importable_from_vaig_core(self) -> None:
        """SPEC requirement: InvestigationPlan importable from vaig.core.investigation."""
        from vaig.core.investigation import create_investigation_plan as _fn  # noqa: PLC0415

        assert callable(_fn)

    def test_default_budget_when_unlimited(self) -> None:
        """When max_cost_usd == 0 (unlimited), plan should still allocate steps."""
        findings = [_make_finding("f1", "OOM"), _make_finding("f2", "Crash")]
        report = _make_report(findings)
        library = HypothesisLibrary.default()
        budget = _make_budget(max_cost_usd=0.0)  # unlimited

        plan = create_investigation_plan(report, library, budget)

        assert len(plan.steps) == 2
        assert all(s.budget_usd > 0 for s in plan.steps)

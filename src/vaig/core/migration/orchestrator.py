"""MigrationOrchestrator: 5-phase migration orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vaig.core.migration.adapters.base import SourceAdapterRegistry
from vaig.core.migration.config import MigrationConfig
from vaig.core.migration.domain import DomainModel, DomainNode
from vaig.core.migration.gates.base import QualityGate
from vaig.core.migration.gates.sdd_gate import MigrationSpec, SddGate
from vaig.core.migration.gates.tdd_gate import TddGate
from vaig.core.migration.gates.test_pass import TestPassGate
from vaig.core.migration.jail import ReadOnlyFilesystemJail

__all__ = ["MigrationOrchestrator", "MigrationResult", "RetrievalContext"]


@dataclass
class MigrationPlan:
    target_files: list[dict[str, Any]] = field(default_factory=list)
    spec_per_file: dict[str, str] = field(default_factory=dict)
    dependency_graph: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class MigrationResult:
    success: bool
    phase_reached: str
    domain_model: DomainModel | None = None
    migration_plan: MigrationPlan | None = None
    files_done: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RetrievalContext:
    spec: str = ""
    domain_slice: list[DomainNode] = field(default_factory=list)
    source_snippets: list[str] = field(default_factory=list)
    example_snippets: list[str] = field(default_factory=list)
    test_suite: str = ""
    prior_iteration_feedback: str = ""

    def estimated_tokens(self) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        total = len(self.spec) + len(self.test_suite) + len(self.prior_iteration_feedback)
        total += sum(len(s) for s in self.source_snippets)
        total += sum(len(s) for s in self.example_snippets)
        total += sum(len(str(n)) for n in self.domain_slice)
        return total // 4

    def to_prompt_xml(self) -> str:
        """Assemble into the XML prompt block format from the spec."""
        parts = []
        if self.spec:
            parts.append(f"<spec>\n{self.spec}\n</spec>")
        if self.test_suite:
            parts.append(f"<test_suite>\n{self.test_suite}\n</test_suite>")
        if self.prior_iteration_feedback:
            parts.append(
                f"<prior_iteration_feedback>\n{self.prior_iteration_feedback}\n</prior_iteration_feedback>"
            )
        if self.domain_slice:
            slice_text = "\n".join(
                f"- {n.step_type}: {n.step_name}" for n in self.domain_slice
            )
            parts.append(f"<domain_slice>\n{slice_text}\n</domain_slice>")
        if self.source_snippets:
            parts.append(
                "<source_snippets>\n"
                + "\n---\n".join(self.source_snippets)
                + "\n</source_snippets>"
            )
        if self.example_snippets:
            parts.append(
                "<example_snippets>\n"
                + "\n---\n".join(self.example_snippets)
                + "\n</example_snippets>"
            )
        return "\n\n".join(parts)


class MigrationOrchestrator:
    """5-phase migration orchestrator.

    Phase 1 · Ingest (deterministic — implemented)
    Phase 2 · Architect (LLM — stub, Sprint 4)
    Phase 3 · TDD Scaffold (LLM — stub, Sprint 4)
    Phase 4 · Implement per-file iterative (LLM — stub, Sprint 4)
    Phase 5 · Integrate (LLM — stub, Sprint 4)
    """

    def __init__(
        self,
        migration_config: MigrationConfig,
        sdd_specs: dict[str, MigrationSpec] | None = None,
    ) -> None:
        self._config = migration_config
        self.sdd_specs: dict[str, MigrationSpec] = sdd_specs or {}
        self._jails = [ReadOnlyFilesystemJail(d) for d in migration_config.from_dirs]
        if migration_config.examples_dirs:
            self._example_jails = [
                ReadOnlyFilesystemJail(d) for d in migration_config.examples_dirs
            ]
        else:
            self._example_jails = []
        self.gates: list[QualityGate] = [SddGate(), TddGate(), TestPassGate()]

    def run(self, task: str) -> MigrationResult:
        """Run all 5 phases. Phases 2-5 are stubs (raise NotImplementedError)."""
        # Phase 1 · Ingest
        domain_model = self._phase_ingest()

        # Phase 2 · Architect
        migration_plan = self._phase_architect(task, domain_model)

        # Phase 3 · TDD Scaffold
        self._phase_tdd_scaffold(migration_plan)

        # Phase 4 · Implement
        files_done, files_failed = self._phase_implement(migration_plan)

        # Phase 5 · Integrate
        self._phase_integrate()

        return MigrationResult(
            success=len(files_failed) == 0,
            phase_reached="integrate",
            domain_model=domain_model,
            migration_plan=migration_plan,
            files_done=files_done,
            files_failed=files_failed,
        )

    def ingest_only(self) -> DomainModel:
        """Run only Phase 1 (deterministic, no LLM). Used by --show-domain."""
        return self._phase_ingest()

    def build_retrieval_context(
        self,
        spec: str = "",
        domain_slice: list[DomainNode] | None = None,
        source_snippets: list[str] | None = None,
        example_snippets: list[str] | None = None,
        test_suite: str = "",
        prior_iteration_feedback: str = "",
    ) -> RetrievalContext:
        """Assemble a RetrievalContext for a per-file executor call (no LLM)."""
        return RetrievalContext(
            spec=spec,
            domain_slice=domain_slice or [],
            source_snippets=source_snippets or [],
            example_snippets=example_snippets or [],
            test_suite=test_suite,
            prior_iteration_feedback=prior_iteration_feedback,
        )

    def _phase_ingest(self) -> DomainModel:
        """Phase 1: Parse source files into a DomainModel (no LLM)."""
        source_kind = self._config.source_kind or "generic"
        adapter = SourceAdapterRegistry.get(source_kind)
        return adapter.parse(self._config.from_dirs)

    def _phase_architect(self, task: str, domain_model: DomainModel) -> MigrationPlan:
        raise NotImplementedError(
            "Phase 2 (Architect) requires LLM integration — implemented in Sprint 4."
        )

    def _phase_tdd_scaffold(self, plan: MigrationPlan) -> None:
        raise NotImplementedError(
            "Phase 3 (TDD Scaffold) requires LLM integration — implemented in Sprint 4."
        )

    def _phase_implement(self, plan: MigrationPlan) -> tuple[list[str], list[str]]:
        raise NotImplementedError(
            "Phase 4 (Implement) requires LLM integration — implemented in Sprint 4."
        )

    def _phase_integrate(self) -> None:
        raise NotImplementedError(
            "Phase 5 (Integrate) requires LLM integration — implemented in Sprint 4."
        )

    def check_write_allowed(self, absolute_path: str | Path) -> None:
        """Raise ReadOnlySourceError if absolute_path is inside any from_dir jail."""
        for jail in self._jails:
            jail.check_write_blocked(absolute_path)
        for jail in self._example_jails:
            jail.check_write_blocked(absolute_path)

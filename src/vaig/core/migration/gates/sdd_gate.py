"""SDD Gate: validates generated code against a MigrationSpec."""
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vaig.core.migration.domain import Chunk
from vaig.core.migration.gates.base import GateResult, QualityGate

__all__ = ["MigrationSpec", "SddGate"]


class MigrationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: Path
    target_path: Path
    required_transformations: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    required_patterns: list[str] = Field(default_factory=list)
    notes: str = ""


class SddGate(QualityGate):
    """Validates generated code against a MigrationSpec."""

    def check(
        self,
        chunk: Chunk,
        generated_code: str,
        spec: MigrationSpec | None = None,
    ) -> GateResult:
        if spec is None:
            return GateResult(passed=True, notes="no spec defined, skipping")

        violations: list[str] = []

        for pattern in spec.forbidden_patterns:
            if re.search(pattern, generated_code):
                violations.append(f"Forbidden pattern found: {pattern!r}")

        for pattern in spec.required_patterns:
            if not re.search(pattern, generated_code):
                violations.append(f"Required pattern missing: {pattern!r}")

        return GateResult(passed=len(violations) == 0, violations=violations)

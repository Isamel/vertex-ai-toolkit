"""Base types for quality gates."""
from dataclasses import dataclass, field

__all__ = ["QualityGate", "GateResult"]


@dataclass
class GateResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    notes: str = ""


class QualityGate:
    """Abstract base for quality gates — subclasses define their own check() signature."""

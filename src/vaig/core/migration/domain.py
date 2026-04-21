"""Domain model: structured representation of parsed source code."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["DomainNode", "DomainModel", "SemanticKind", "Chunk", "SourceReference"]

SemanticKind = Literal["SOURCE", "TRANSFORM", "SINK", "CONTROL", "UNKNOWN"]


@dataclass(frozen=True)
class SourceReference:
    """Maps a DomainNode back to its origin file and line."""
    file: str
    line: int
    end_line: int | None = None


@dataclass(frozen=True)
class DomainNode:
    step_name: str
    step_type: str
    semantic_kind: SemanticKind
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    config: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    source_file: str = ""
    source_line: int = 0


@dataclass
class Chunk:
    """A retrieval chunk representing a portion of the domain model."""
    node: DomainNode
    text: str          # human-readable representation for embedding
    tags: list[str] = field(default_factory=list)


@dataclass
class DomainModel:
    """Structured representation of a parsed source tree."""
    source_kind: str
    nodes: list[DomainNode] = field(default_factory=list)
    hops: list[tuple[str, str]] = field(default_factory=list)  # (from_step, to_step)
    parameters: dict[str, str] = field(default_factory=dict)
    connections: list[dict[str, Any]] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def hop_count(self) -> int:
        return len(self.hops)

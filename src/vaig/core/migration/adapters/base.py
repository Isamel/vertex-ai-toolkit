"""SourceAdapter interface, registry, and supporting types."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from vaig.core.migration.domain import Chunk, DomainModel, DomainNode, SourceReference

__all__ = [
    "SourceAdapter",
    "SourceAdapterRegistry",
    "UnknownSourceError",
]


class UnknownSourceError(ValueError):
    """Raised when no adapter can handle the given source files."""


class SourceAdapter:
    """Base class (also usable as Protocol) for source adapters."""

    kind: ClassVar[str] = "base"
    file_globs: ClassVar[tuple[str, ...]] = ()

    def detect(self, paths: Sequence[Path]) -> float:
        """Return confidence in [0, 1] that the files are of this kind."""
        return 0.0

    def parse(self, paths: Sequence[Path]) -> DomainModel:
        """Extract a structured DomainModel from the source tree."""
        raise NotImplementedError

    def chunk(self, domain: DomainModel) -> Iterable[Chunk]:
        """Emit retrieval chunks."""
        for node in domain.nodes:
            yield Chunk(
                node=node,
                text=f"{node.step_type}: {node.step_name}",
                tags=[node.semantic_kind.lower(), node.step_type.lower()],
            )

    def trace(self, target_node: DomainNode) -> list[SourceReference]:
        """Map a DomainNode back to its origin file:line range."""
        if target_node.source_file and target_node.source_line:
            return [SourceReference(file=target_node.source_file, line=target_node.source_line)]
        return []


class SourceAdapterRegistry:
    """Registry of SourceAdapter implementations."""

    _adapters: ClassVar[dict[str, type[SourceAdapter]]] = {}

    @classmethod
    def register(cls, adapter: type[SourceAdapter]) -> None:
        cls._adapters[adapter.kind] = adapter

    @classmethod
    def get(cls, kind: str) -> SourceAdapter:
        if kind not in cls._adapters:
            raise UnknownSourceError(
                f"No adapter for source kind '{kind}'. "
                f"Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[kind]()

    @classmethod
    def auto_detect(cls, paths: Sequence[Path]) -> SourceAdapter:
        scores = {k: cls._adapters[k]().detect(paths) for k in cls._adapters}
        if not scores:
            raise UnknownSourceError("No adapters registered.")
        best_kind = max(scores, key=lambda k: (scores[k], k))  # score desc, kind asc for ties
        best_score = scores[best_kind]
        if best_score < 0.5:
            raise UnknownSourceError(
                f"Could not identify source kind (best: {best_kind!r} at {best_score:.2f}). "
                f"Use --source-kind to override. Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[best_kind]()

    @classmethod
    def available_kinds(cls) -> list[str]:
        return sorted(cls._adapters)

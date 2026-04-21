"""DDD overlay: BoundedContext and AggregateRoot models for domain-driven migration hints."""
from pydantic import BaseModel, Field

from vaig.core.migration.domain import Chunk

__all__ = ["AggregateRoot", "BoundedContext", "DddOverlay"]

# SemanticKind → typical DDD context role (heuristic)
_KIND_CONTEXT_MAP: dict[str, str] = {
    "SOURCE": "ingestion",
    "SINK": "persistence",
    "TRANSFORM": "domain",
    "CONTROL": "orchestration",
    "UNKNOWN": "",
}


class AggregateRoot(BaseModel):
    name: str
    entities: list[str]
    value_objects: list[str] = Field(default_factory=list)
    domain_events: list[str] = Field(default_factory=list)


class BoundedContext(BaseModel):
    name: str
    aggregates: list[AggregateRoot]
    ubiquitous_language: dict[str, str] = Field(default_factory=dict)  # term → definition


class DddOverlay(BaseModel):
    contexts: list[BoundedContext]

    def find_context(self, name: str) -> BoundedContext | None:
        for ctx in self.contexts:
            if ctx.name == name:
                return ctx
        return None

    def all_aggregates(self) -> list[AggregateRoot]:
        result: list[AggregateRoot] = []
        for ctx in self.contexts:
            result.extend(ctx.aggregates)
        return result

    def enrich_chunk(self, chunk: Chunk) -> dict[str, str]:
        """Return context hints for the chunk based on its SemanticKind and name.

        Returns a dict with keys: 'context', 'aggregate', 'language_hints'.
        Returns empty dict if no match found.
        """
        kind_role = _KIND_CONTEXT_MAP.get(chunk.node.semantic_kind, "")
        if not kind_role:
            return {}

        # Try to find a context whose name matches the kind role
        matched_ctx: BoundedContext | None = None
        for ctx in self.contexts:
            if kind_role in ctx.name.lower():
                matched_ctx = ctx
                break

        # Fallback: try to match by chunk step_name against aggregate entity names
        if matched_ctx is None:
            step_lower = chunk.node.step_name.lower()
            for ctx in self.contexts:
                for agg in ctx.aggregates:
                    if agg.name.lower() in step_lower or step_lower in agg.name.lower():
                        matched_ctx = ctx
                        break
                if matched_ctx is not None:
                    break

        if matched_ctx is None:
            return {}

        # Find best matching aggregate
        matched_agg_name = ""
        step_lower = chunk.node.step_name.lower()
        for agg in matched_ctx.aggregates:
            if agg.name.lower() in step_lower or step_lower in agg.name.lower():
                matched_agg_name = agg.name
                break
        if not matched_agg_name and matched_ctx.aggregates:
            matched_agg_name = matched_ctx.aggregates[0].name

        language_hints = ", ".join(
            f"{term}={defn}"
            for term, defn in matched_ctx.ubiquitous_language.items()
        )

        return {
            "context": matched_ctx.name,
            "aggregate": matched_agg_name,
            "language_hints": language_hints,
        }

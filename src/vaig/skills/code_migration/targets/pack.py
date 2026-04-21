"""TargetPack: Pydantic model for migration target knowledge packs."""
from pydantic import BaseModel

__all__ = ["CodeExample", "TargetPack"]


class CodeExample(BaseModel):
    description: str
    source: str  # source code snippet
    target: str  # expected migrated code snippet


class TargetPack(BaseModel):
    name: str
    version: str
    description: str
    imports: list[str]          # required imports for the target
    patterns: dict[str, str]    # transformation patterns: source_pattern → target_template
    forbidden_apis: list[str]   # APIs that must not appear in migrated code
    required_boilerplate: str   # boilerplate that must appear (e.g. GlueContext setup)
    examples: list[CodeExample]

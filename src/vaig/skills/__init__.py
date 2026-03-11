"""Skills package — pluggable skill system for specialized AI tasks."""

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase, SkillResult
from vaig.skills.registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "SkillMetadata",
    "SkillPhase",
    "SkillRegistry",
    "SkillResult",
]

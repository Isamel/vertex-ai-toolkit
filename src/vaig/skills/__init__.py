"""Skills package — pluggable skill system for specialized AI tasks."""

from vaig.skills.base import BaseSkill, CompositeSkill, SkillMetadata, SkillPhase, SkillResult
from vaig.skills.loader import load_from_directories, load_from_entry_points, load_from_packages
from vaig.skills.registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "CompositeSkill",
    "SkillMetadata",
    "SkillPhase",
    "SkillRegistry",
    "SkillResult",
    "load_from_directories",
    "load_from_entry_points",
    "load_from_packages",
]

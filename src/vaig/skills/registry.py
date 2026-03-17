"""Skill registry — discovery, loading, and management of skills."""

from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from vaig.skills.base import BaseSkill, SkillMetadata

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# Common stop words to exclude from query matching
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "do", "does", "did",
    "what", "why", "how", "when", "where", "who", "which",
    "my", "your", "our", "their", "its", "this", "that",
    "i", "me", "we", "you", "he", "she", "it", "they",
    "can", "could", "will", "would", "should", "shall",
    "have", "has", "had", "may", "might", "must",
})


def _tokenize_query(query: str) -> list[str]:
    """Tokenize and normalize a query string, removing stop words."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _score_skill(query_words: list[str], meta: SkillMetadata) -> float:
    """Score a skill against query words."""
    score = 0.0
    tags_lower = {t.lower() for t in meta.tags}
    name_parts = set(meta.name.lower().split("-"))
    description_lower = meta.description.lower()

    for word in query_words:
        # Exact tag match (highest weight)
        if word in tags_lower:
            score += 2.0
        # Name match
        if word in name_parts:
            score += 1.5
        # Description substring match
        if word in description_lower:
            score += 0.5

    # Normalize by query length to avoid bias toward long queries
    return score / len(query_words)

# Directory containing built-in skill packages (each has a ``skill.py`` module).
_SKILLS_DIR = Path(__file__).parent

# Skill class name convention: <Name>Skill (e.g., RCASkill, AnomalySkill)
_SKILL_CLASS_SUFFIXES = ("Skill",)


def _discover_builtin_skills() -> dict[str, str]:
    """Auto-discover built-in skills by scanning subdirectories of the skills package.

    Each subdirectory that contains a ``skill.py`` file is treated as a skill
    package.  The module path is derived from the directory name, following
    the pattern ``vaig.skills.<dir_name>.skill``.

    Returns a mapping of ``skill_name → module_path`` where ``skill_name``
    comes from the directory name with underscores replaced by hyphens
    (matching the convention used in ``SkillMetadata.name``).
    """
    skills: dict[str, str] = {}
    for entry in sorted(_SKILLS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "skill.py"
        if not skill_file.exists():
            continue
        # Derive the skill name from the directory (e.g. log_analysis → log-analysis)
        skill_name = entry.name.replace("_", "-")
        module_path = f"vaig.skills.{entry.name}.skill"
        skills[skill_name] = module_path
    return skills


class SkillRegistry:
    """Registry for discovering and managing skills.

    Handles both built-in skills and custom skills loaded from a directory.
    Skills are loaded lazily on first access.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._skills: dict[str, BaseSkill] = {}
        self._metadata_cache: dict[str, SkillMetadata] = {}
        self._loaded = False

    def _load_builtin_skills(self) -> None:
        """Load built-in skills that are enabled in config.

        Skills are auto-discovered by scanning subdirectories of the
        ``vaig.skills`` package for ``skill.py`` modules, eliminating
        the need to maintain a hardcoded registry dict.
        """
        enabled = self._settings.skills.enabled
        discovered = _discover_builtin_skills()

        for name, module_path in discovered.items():
            if name not in enabled:
                logger.debug("Skipping disabled built-in skill: %s", name)
                continue

            try:
                skill = _import_skill(module_path)
                self._register(skill)
                logger.info("Loaded built-in skill: %s", name)
            except Exception:
                logger.warning("Failed to load built-in skill: %s", name, exc_info=True)

    def _load_custom_skills(self) -> None:
        """Load custom skills from the configured directory."""
        custom_dir = self._settings.skills.custom_dir
        if not custom_dir:
            return

        custom_path = Path(custom_dir).expanduser().resolve()
        if not custom_path.is_dir():
            logger.warning("Custom skills directory not found: %s", custom_path)
            return

        for skill_dir in sorted(custom_path.iterdir()):
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "skill.py"
            if not skill_file.exists():
                continue

            try:
                skill = _import_skill_from_path(skill_file)
                self._register(skill)
                logger.info("Loaded custom skill: %s from %s", skill.get_metadata().name, skill_file)
            except Exception:
                logger.warning("Failed to load custom skill from: %s", skill_file, exc_info=True)

    def _register(self, skill: BaseSkill) -> None:
        """Register a skill instance."""
        meta = skill.get_metadata()
        self._skills[meta.name] = skill
        self._metadata_cache[meta.name] = meta

        # Telemetry: emit skill_use event via EventBus
        try:
            from vaig.core.event_bus import EventBus
            from vaig.core.events import SkillUsed

            EventBus.get().emit(SkillUsed(skill_name=meta.name))
        except Exception:  # noqa: BLE001
            pass

    def _ensure_loaded(self) -> None:
        """Ensure skills are loaded (lazy initialization)."""
        if self._loaded:
            return

        self._load_builtin_skills()
        self._load_custom_skills()
        self._loaded = True

    def get(self, name: str) -> BaseSkill | None:
        """Get a skill by name. Returns None if not found."""
        self._ensure_loaded()
        return self._skills.get(name)

    def get_metadata(self, name: str) -> SkillMetadata | None:
        """Get skill metadata without loading the full skill."""
        self._ensure_loaded()
        return self._metadata_cache.get(name)

    def list_skills(self) -> list[SkillMetadata]:
        """List all registered skills' metadata."""
        self._ensure_loaded()
        return list(self._metadata_cache.values())

    def list_names(self) -> list[str]:
        """List all registered skill names."""
        self._ensure_loaded()
        return list(self._skills.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a skill is registered."""
        self._ensure_loaded()
        return name in self._skills

    def suggest_skill(self, query: str, *, top_n: int = 3) -> list[tuple[str, float]]:
        """Suggest skills based on query keywords matching skill tags and descriptions.

        Uses a simple TF-based scoring:
        - Each query word is checked against skill tags (exact match = 2.0 points)
        - Each query word is checked against skill name (exact match = 1.5 points)
        - Each query word is checked against skill description (substring match = 0.5 points)
        - Scores are normalized by the number of query words

        Returns list of (skill_name, score) tuples, sorted by score descending.
        Only returns skills with score > 0.
        """
        self._ensure_loaded()
        query_words = _tokenize_query(query)
        if not query_words:
            return []

        scores: list[tuple[str, float]] = []
        for name, meta in self._metadata_cache.items():
            score = _score_skill(query_words, meta)
            if score > 0:
                scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    @property
    def count(self) -> int:
        """Number of registered skills."""
        self._ensure_loaded()
        return len(self._skills)


def _import_skill(module_path: str) -> BaseSkill:
    """Import a skill from a dotted module path.

    Expects the module to contain exactly one class that extends BaseSkill.
    """
    module = importlib.import_module(module_path)
    skill_class = _find_skill_class(module)
    return skill_class()


def _import_skill_from_path(file_path: Path) -> BaseSkill:
    """Import a skill from a file path (for custom skills).

    Uses importlib.util to load from an arbitrary file.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"vaig.skills.custom.{file_path.parent.name}",
        file_path,
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot load skill from: {file_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    skill_class = _find_skill_class(module)
    return skill_class()


def _find_skill_class(module: object) -> type[BaseSkill]:
    """Find the first BaseSkill subclass in a module."""
    import inspect

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseSkill) and obj is not BaseSkill:
            return obj

    msg = f"No BaseSkill subclass found in module: {module}"
    raise ImportError(msg)

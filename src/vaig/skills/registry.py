"""Skill registry — discovery, loading, and management of skills."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from vaig.skills.base import BaseSkill, SkillMetadata

if TYPE_CHECKING:
    from vaig.core.config import Settings

logger = logging.getLogger(__name__)

# Built-in skills mapped by name → module path
_BUILTIN_SKILLS: dict[str, str] = {
    "rca": "vaig.skills.rca.skill",
    "anomaly": "vaig.skills.anomaly.skill",
    "migration": "vaig.skills.migration.skill",
    "log-analysis": "vaig.skills.log_analysis.skill",
    "error-triage": "vaig.skills.error_triage.skill",
    "config-audit": "vaig.skills.config_audit.skill",
    "slo-review": "vaig.skills.slo_review.skill",
    "postmortem": "vaig.skills.postmortem.skill",
    "code-review": "vaig.skills.code_review.skill",
    "iac-review": "vaig.skills.iac_review.skill",
    "cost-analysis": "vaig.skills.cost_analysis.skill",
    "capacity-planning": "vaig.skills.capacity_planning.skill",
    "test-generation": "vaig.skills.test_generation.skill",
    "compliance-check": "vaig.skills.compliance_check.skill",
    "api-design": "vaig.skills.api_design.skill",
    "runbook-generator": "vaig.skills.runbook_generator.skill",
    "dependency-audit": "vaig.skills.dependency_audit.skill",
    "db-review": "vaig.skills.db_review.skill",
    "pipeline-review": "vaig.skills.pipeline_review.skill",
    "perf-analysis": "vaig.skills.perf_analysis.skill",
    "threat-model": "vaig.skills.threat_model.skill",
    "change-risk": "vaig.skills.change_risk.skill",
    "alert-tuning": "vaig.skills.alert_tuning.skill",
    "resilience-review": "vaig.skills.resilience_review.skill",
    "incident-comms": "vaig.skills.incident_comms.skill",
    "toil-analysis": "vaig.skills.toil_analysis.skill",
    "network-review": "vaig.skills.network_review.skill",
    "adr-generator": "vaig.skills.adr_generator.skill",
}

# Skill class name convention: <Name>Skill (e.g., RCASkill, AnomalySkill)
_SKILL_CLASS_SUFFIXES = ("Skill",)


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
        """Load built-in skills that are enabled in config."""
        enabled = self._settings.skills.enabled

        for name, module_path in _BUILTIN_SKILLS.items():
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
            return obj  # type: ignore[return-value]

    msg = f"No BaseSkill subclass found in module: {module}"
    raise ImportError(msg)

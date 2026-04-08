"""Skill loader — discover and load skills from directories, entry points, and packages.

This module provides three public functions for loading external
``BaseSkill`` subclasses:

1. :func:`load_from_directories` — scan filesystem directories for
   subdirectories containing ``skill.py`` with a ``BaseSkill`` subclass.
2. :func:`load_from_entry_points` — discover pip-installed packages
   that register the ``vaig.skills`` entry-point group.
3. :func:`load_from_packages` — like entry points, but filtered to
   specific distribution package names.

Errors in individual skills are logged and skipped — one broken skill
never blocks agent startup.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

from vaig.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _normalize_dist_name(name: str) -> str:
    """Normalize a distribution name per PEP 503.

    Collapses runs of hyphens, underscores, and dots into a single
    hyphen and lowercases the result, so that ``Vaig_Security_Skills``,
    ``vaig-security-skills``, and ``vaig.security.skills`` all compare
    as equal.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_skill_class(module: ModuleType) -> type[BaseSkill]:
    """Find the first ``BaseSkill`` subclass in a module.

    Args:
        module: The Python module to inspect.

    Returns:
        The first ``BaseSkill`` subclass found.

    Raises:
        ImportError: If no ``BaseSkill`` subclass is found.
    """
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseSkill) and obj is not BaseSkill:
            return obj

    msg = f"No BaseSkill subclass found in module: {module}"
    raise ImportError(msg)


def _import_skill_from_path(file_path: Path) -> BaseSkill:
    """Import a skill from a file path (for external/custom skills).

    Uses ``importlib.util`` to load from an arbitrary filesystem path.

    Args:
        file_path: Absolute path to a ``skill.py`` file.

    Returns:
        An instantiated ``BaseSkill`` subclass.

    Raises:
        ImportError: If the module cannot be loaded or has no ``BaseSkill`` subclass.
    """
    # Replace hyphens with underscores — directory names like "my-custom-skills"
    # would produce invalid Python module identifiers otherwise.
    module_name = file_path.parent.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(
        f"vaig.skills.external.{module_name}",
        file_path,
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot load skill from: {file_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    skill_class = _find_skill_class(module)
    return skill_class()


def load_from_directories(dirs: list[str]) -> list[BaseSkill]:
    """Scan directories for subdirectories containing ``skill.py``.

    Each subdirectory with a ``skill.py`` is imported via
    :func:`_import_skill_from_path`.  Directories that don't exist are
    logged and skipped.  Import errors are logged and skipped.

    Args:
        dirs: List of directory paths to scan.  Supports ``~`` expansion.

    Returns:
        A list of instantiated ``BaseSkill`` objects.  Never raises.
    """
    skills: list[BaseSkill] = []

    for dir_str in dirs:
        directory = Path(dir_str).expanduser().resolve()

        if not directory.is_dir():
            logger.warning("External skills directory not found: %s", directory)
            continue

        for skill_dir in sorted(directory.iterdir()):
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "skill.py"
            if not skill_file.exists():
                logger.debug("No skill.py in %s — skipping", skill_dir)
                continue

            try:
                skill = _import_skill_from_path(skill_file)
                skills.append(skill)
                logger.info(
                    "Loaded external skill: %s from %s",
                    skill.get_metadata().name,
                    skill_file,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to load external skill from: %s",
                    skill_file,
                    exc_info=True,
                )

    return skills


def load_from_entry_points(group: str = "vaig.skills") -> list[BaseSkill]:
    """Discover pip-installed skills via ``importlib.metadata.entry_points``.

    Each entry point in the given group must reference a ``BaseSkill``
    subclass.  The class is loaded via ``ep.load()`` and instantiated.

    Args:
        group: Entry-point group name to scan.  Defaults to ``"vaig.skills"``.

    Returns:
        A list of instantiated ``BaseSkill`` objects.  Never raises.
    """
    skills: list[BaseSkill] = []

    try:
        entry_points = importlib.metadata.entry_points(group=group)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to query entry points for group '%s'",
            group,
            exc_info=True,
        )
        return skills

    for ep in entry_points:
        try:
            skill_class = ep.load()
            if not (isinstance(skill_class, type) and issubclass(skill_class, BaseSkill)):
                logger.warning(
                    "Entry point '%s' does not reference a BaseSkill subclass — skipping",
                    ep.name,
                )
                continue

            skill = skill_class()
            skills.append(skill)
            logger.info(
                "Loaded entry-point skill: %s (from %s)",
                skill.get_metadata().name,
                ep.value,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load entry-point skill '%s' — skipping",
                ep.name,
                exc_info=True,
            )

    return skills


def load_from_packages(names: list[str]) -> list[BaseSkill]:
    """Load entry-point skills filtered to specific distribution package names.

    When ``names`` is empty, no skills are loaded (use
    :func:`load_from_entry_points` for unfiltered discovery).

    Args:
        names: List of pip distribution package names to accept.

    Returns:
        A list of instantiated ``BaseSkill`` objects.  Never raises.
    """
    if not names:
        return []

    skills: list[BaseSkill] = []
    names_set = {_normalize_dist_name(n) for n in names}

    try:
        entry_points = importlib.metadata.entry_points(group="vaig.skills")
    except Exception:  # noqa: BLE001
        logger.warning(
            "Failed to query entry points for package filtering",
            exc_info=True,
        )
        return skills

    for ep in entry_points:
        try:
            dist_name = ep.dist.name if ep.dist else None
        except Exception:  # noqa: BLE001
            dist_name = None

        if dist_name is None or _normalize_dist_name(dist_name) not in names_set:
            logger.debug(
                "Skipping entry point '%s' — dist '%s' not in packages filter",
                ep.name,
                dist_name,
            )
            continue

        try:
            skill_class = ep.load()
            if not (isinstance(skill_class, type) and issubclass(skill_class, BaseSkill)):
                logger.warning(
                    "Entry point '%s' (package '%s') does not reference a "
                    "BaseSkill subclass — skipping",
                    ep.name,
                    dist_name,
                )
                continue

            skill = skill_class()
            skills.append(skill)
            logger.info(
                "Loaded package skill: %s (from package %s)",
                skill.get_metadata().name,
                dist_name,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to load package skill '%s' from '%s' — skipping",
                ep.name,
                dist_name,
                exc_info=True,
            )

    return skills

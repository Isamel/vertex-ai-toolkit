"""Code Migration Skill — language-to-language code migration with 5-phase state machine."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.code_migration.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION
from vaig.tools.file_tools import create_file_tools

logger = logging.getLogger(__name__)

# Path to the bundled idiom maps directory
_IDIOMS_DIR = Path(__file__).parent / "idioms"


class MigrationPhase(StrEnum):
    """5-phase state machine for language-to-language code migration.

    The phases are ordered — each must complete before the next begins:
    INVENTORY → SEMANTIC_MAP → SPEC → IMPLEMENT → VERIFY

    Inherits from ``StrEnum`` — instances compare equal to their string value
    and serialise as plain strings in JSON (no custom encoder required).
    """

    INVENTORY = "inventory"
    SEMANTIC_MAP = "semantic_map"
    SPEC = "spec"
    IMPLEMENT = "implement"
    VERIFY = "verify"

    def next_phase(self) -> MigrationPhase | None:
        """Return the next migration phase, or None if this is the last phase."""
        order = list(MigrationPhase)
        idx = order.index(self)
        if idx + 1 < len(order):
            return order[idx + 1]
        return None

    @classmethod
    def from_skill_phase(cls, phase: SkillPhase) -> MigrationPhase:
        """Map a SkillPhase to the corresponding MigrationPhase for prompt selection."""
        _mapping: dict[SkillPhase, MigrationPhase] = {
            SkillPhase.ANALYZE: cls.INVENTORY,
            SkillPhase.PLAN: cls.SEMANTIC_MAP,
            SkillPhase.EXECUTE: cls.IMPLEMENT,
            SkillPhase.VALIDATE: cls.VERIFY,
            SkillPhase.REPORT: cls.VERIFY,
        }
        return _mapping.get(phase, cls.INVENTORY)


def _load_idiom_map(source_lang: str, target_lang: str) -> dict[str, Any] | None:
    """Load an idiom map YAML file for the given language pair.

    Looks for ``{source_lang}_to_{target_lang}.yaml`` in the bundled idioms directory.

    Args:
        source_lang: Source programming language (e.g. ``"python"``).
        target_lang: Target programming language (e.g. ``"go"``).

    Returns:
        Parsed YAML dict on success, or ``None`` if no map is found.
    """
    filename = f"{source_lang.lower()}_to_{target_lang.lower()}.yaml"
    path = _IDIOMS_DIR / filename

    if not path.exists():
        logger.debug("No idiom map found for %s→%s at %s", source_lang, target_lang, path)
        return None

    try:
        with path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        logger.debug(
            "Loaded idiom map %s: %d idioms, %d deps",
            filename,
            len(data.get("idioms", [])),
            len(data.get("dependencies", {})),
        )
        return data
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to load idiom map %s: %s", filename, exc)
        return None


def _format_idiom_map(idiom_data: dict[str, Any]) -> str:
    """Render a loaded idiom map as a Markdown string for prompt injection.

    Args:
        idiom_data: Parsed YAML dict from :func:`_load_idiom_map`.

    Returns:
        Formatted Markdown string describing idioms and dependency substitutions.
    """
    lines: list[str] = []
    src = idiom_data.get("source_lang", "source")
    tgt = idiom_data.get("target_lang", "target")
    lines.append(f"## Idiom Map: {src.title()} → {tgt.title()}\n")

    idioms: list[dict[str, str]] = idiom_data.get("idioms", [])
    if idioms:
        lines.append("### Idiom Transformations\n")
        lines.append("| Source Pattern | Target Pattern | Notes |")
        lines.append("|----------------|----------------|-------|")
        for entry in idioms:
            sp = entry.get("source_pattern", "").replace("|", "\\|")
            tp = entry.get("target_pattern", "").replace("|", "\\|")
            notes = entry.get("notes", "").replace("|", "\\|")
            lines.append(f"| {sp} | {tp} | {notes} |")
        lines.append("")

    deps: dict[str, str] = idiom_data.get("dependencies", {})
    if deps:
        lines.append("### Dependency Substitutions\n")
        lines.append("| Source Package | Target Package |")
        lines.append("|----------------|----------------|")
        for src_pkg, tgt_pkg in deps.items():
            lines.append(f"| {src_pkg} | {tgt_pkg} |")
        lines.append("")

    return "\n".join(lines)


class _MigrationTracker:
    """Per-file migration progress tracker.

    Tracks the migration status of each file as it moves through the phases.
    This is a lightweight in-memory record — for persistence, the skill's
    agents write the log to disk via ``write_file``.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def record(
        self,
        filename: str,
        phase: MigrationPhase,
        status: str,
        notes: str = "",
    ) -> None:
        """Record or update the migration status for a file.

        Args:
            filename: The source or target filename being tracked.
            phase: The :class:`MigrationPhase` at which this status was recorded.
            status: Human-readable status string (e.g. ``"complete"``, ``"skipped"``).
            notes: Optional notes about the migration result.
        """
        self._records[filename] = {
            "filename": filename,
            "phase": phase.value,
            "status": status,
            "notes": notes,
        }

    def get_all(self) -> list[dict[str, Any]]:
        """Return all tracked records as a list of dicts."""
        return list(self._records.values())

    def as_markdown(self) -> str:
        """Render the migration log as a Markdown table.

        Returns:
            A Markdown-formatted table string, or a placeholder message if empty.
        """
        if not self._records:
            return "*No files tracked yet.*"

        lines = ["| File | Phase | Status | Notes |", "|------|-------|--------|-------|"]
        for rec in self._records.values():
            fn = rec["filename"]
            ph = rec["phase"]
            st = rec["status"]
            nt = rec.get("notes", "")
            lines.append(f"| {fn} | {ph} | {st} | {nt} |")
        return "\n".join(lines)


class CodeMigrationSkill(BaseSkill):
    """Code Migration skill — migrate a codebase from one language to another.

    Implements a 5-phase state machine aligned to :class:`MigrationPhase`:

    1. **INVENTORY** — catalogue source files and identify constructs
    2. **SEMANTIC_MAP** — map source idioms to target idioms using YAML maps
    3. **SPEC** — write per-file migration specifications
    4. **IMPLEMENT** — produce complete migrated code with no placeholders
    5. **VERIFY** — run completeness checks and emit final migration report

    The skill exposes :func:`~vaig.tools.file_tools.create_file_tools` (including
    ``verify_completeness``) so agents can read source files and scan migrated
    output for placeholder patterns.

    Supports dynamic idiom map loading: if a ``{src}_to_{tgt}.yaml`` file is
    present in the ``idioms/`` directory, it is injected into the SEMANTIC_MAP
    phase prompt.

    Args:
        source_lang: Source programming language (e.g. ``"python"``).
        target_lang: Target programming language (e.g. ``"go"``).
        workspace: Workspace root path for file tool sandboxing.
            Defaults to the current working directory.
    """

    def __init__(
        self,
        source_lang: str = "python",
        target_lang: str = "go",
        workspace: Path | None = None,
    ) -> None:
        self._source_lang = source_lang.lower()
        self._target_lang = target_lang.lower()
        self._workspace = workspace or Path.cwd()
        self._tracker = _MigrationTracker()

        # Load idiom map eagerly; None if no map exists for the pair
        self._idiom_data = _load_idiom_map(self._source_lang, self._target_lang)

    # ── BaseSkill interface ───────────────────────────────────────────────

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="code-migration",
            display_name="Code Migration",
            description=(
                "Migrate a codebase from one programming language to another with semantic "
                "fidelity, idiomatic target code, and dependency mapping"
            ),
            version="1.0.0",
            tags=[
                "migration",
                "refactoring",
                "polyglot",
                "code-transformation",
                "language-migration",
            ],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.VALIDATE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
            requires_live_tools=True,
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        """Build a phase-specific prompt, injecting the idiom map for PLAN phase.

        For :attr:`~vaig.skills.base.SkillPhase.PLAN` (SEMANTIC_MAP), the loaded
        idiom map (if any) is appended to the context before formatting.

        Args:
            phase: The skill phase being executed.
            context: Loaded context (source files, inventory, etc.).
            user_input: The user's specific migration request.

        Returns:
            Formatted prompt string for the given phase.
        """
        # Inject idiom map into context during PLAN (SEMANTIC_MAP) phase
        enriched_context = context
        if phase == SkillPhase.PLAN and self._idiom_data is not None:
            idiom_markdown = _format_idiom_map(self._idiom_data)
            enriched_context = f"{context}\n\n{idiom_markdown}"

        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=enriched_context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return a 3-agent configuration for the migration pipeline.

        Agents:
        - **migration_analyst**: Handles INVENTORY and SEMANTIC_MAP phases
        - **migration_engineer**: Handles SPEC and IMPLEMENT phases, has file tools
        - **migration_lead**: Handles VERIFY phase and synthesises the final report

        All agents have access to file tools for reading source and scanning output.
        """
        lang_pair = f"{self._source_lang.title()} → {self._target_lang.title()}"

        return [
            {
                "name": "migration_analyst",
                "role": f"Migration Analyst ({lang_pair})",
                "system_instruction": (
                    f"You are a migration analyst specialising in {lang_pair} migrations. "
                    "Your job covers the INVENTORY and SEMANTIC_MAP phases: "
                    "catalogue all source files, classify their constructs, identify external "
                    "dependencies, assess migration complexity, and map every source idiom to "
                    f"its {self._target_lang.title()} equivalent. "
                    "Produce structured tables and complexity ratings. "
                    "Be precise — your inventory drives the entire migration pipeline."
                ),
                "model": "gemini-2.5-flash",
                "requires_tools": True,
                "tool_categories": ["coding"],
            },
            {
                "name": "migration_engineer",
                "role": f"Migration Engineer ({lang_pair})",
                "system_instruction": (
                    f"You are a senior engineer executing {lang_pair} code migrations. "
                    "Your job covers the SPEC and IMPLEMENT phases: "
                    "write per-file migration specs, then produce COMPLETE, idiomatic "
                    f"{self._target_lang.title()} code for each file. "
                    "NEVER emit TODO, FIXME, pass, ..., or NotImplementedError. "
                    "If you cannot complete a section, STOP and report what is missing. "
                    "Use the provided file tools to read source files and verify output."
                ),
                "model": "gemini-2.5-pro",
                "requires_tools": True,
                "tool_categories": ["coding"],
            },
            {
                "name": "migration_lead",
                "role": f"Migration Lead ({lang_pair})",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "gemini-2.5-pro",
                "requires_tools": True,
                "tool_categories": ["coding"],
            },
        ]

    # ── Public helpers ────────────────────────────────────────────────────

    def get_file_tools(self) -> list[Any]:
        """Return file tool definitions bound to this skill's workspace.

        The tools include: ``read_file``, ``write_file``, ``edit_file``,
        ``list_files``, ``search_files``, and ``verify_completeness``.

        Returns:
            List of :class:`~vaig.tools.base.ToolDef` instances.
        """
        return create_file_tools(self._workspace)

    def record_file_migration(
        self,
        filename: str,
        phase: MigrationPhase,
        status: str,
        notes: str = "",
    ) -> None:
        """Record or update the migration status for a specific file.

        Args:
            filename: The source or target filename being tracked.
            phase: The :class:`MigrationPhase` at which this status applies.
            status: Human-readable status (e.g. ``"complete"``, ``"skipped"``).
            notes: Optional notes about this file's migration.
        """
        self._tracker.record(filename, phase, status, notes)

    def get_migration_log(self) -> list[dict[str, Any]]:
        """Return all per-file migration records as a list of dicts.

        Returns:
            List of records, each with keys: ``filename``, ``phase``,
            ``status``, ``notes``.
        """
        return self._tracker.get_all()

    def get_migration_log_markdown(self) -> str:
        """Return the per-file migration log rendered as a Markdown table.

        Returns:
            Markdown table string, or ``*No files tracked yet.*`` if empty.
        """
        return self._tracker.as_markdown()

    def get_language_pair(self) -> tuple[str, str]:
        """Return the (source_lang, target_lang) tuple for this skill instance.

        Returns:
            A ``(source_lang, target_lang)`` tuple of lowercase strings.
        """
        return (self._source_lang, self._target_lang)

    def get_idiom_map(self) -> dict[str, Any] | None:
        """Return the loaded idiom map for the current language pair, or None.

        Returns:
            Parsed YAML dict if a map was found for the language pair, else ``None``.
        """
        return self._idiom_data

"""Code Migration Skill — language-to-language code migration with 6-phase state machine."""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from vaig.core.project import ensure_project_dir as _ensure_project_dir
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.code_migration.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION
from vaig.tools.file_tools import create_file_tools

if TYPE_CHECKING:
    from vaig.core.client import GeminiClient
    from vaig.core.config import IdiomConfig

logger = logging.getLogger(__name__)


class MigrationPhase(StrEnum):
    """6-phase state machine for language-to-language code migration.

    The phases are ordered — each must complete before the next begins:
    INVENTORY → SEMANTIC_MAP → SPEC → IMPLEMENT → VERIFY → REPORT

    Inherits from ``StrEnum`` — instances compare equal to their string value
    and serialise as plain strings in JSON (no custom encoder required).
    """

    INVENTORY = "inventory"
    SEMANTIC_MAP = "semantic_map"
    SPEC = "spec"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REPORT = "report"

    def next_phase(self) -> MigrationPhase | None:
        """Return the next migration phase, or None if this is the last phase."""
        order = list(MigrationPhase)
        idx = order.index(self)
        if idx + 1 < len(order):
            return order[idx + 1]
        return None

    @classmethod
    def from_skill_phase(cls, phase: SkillPhase) -> MigrationPhase:
        """Map a SkillPhase to the corresponding MigrationPhase for prompt selection.

        Mapping:
        - ANALYZE  → INVENTORY    (catalogue source files)
        - PLAN     → SEMANTIC_MAP (map idioms and dependencies)
        - EXECUTE  → SPEC         (write per-file migration specs)
        - VALIDATE → IMPLEMENT    (produce migrated code)
        - REPORT   → REPORT       (final summary report)

        Falls back to INVENTORY for any unknown phase.
        """
        _mapping: dict[SkillPhase, MigrationPhase] = {
            SkillPhase.ANALYZE: cls.INVENTORY,
            SkillPhase.PLAN: cls.SEMANTIC_MAP,
            SkillPhase.EXECUTE: cls.SPEC,
            SkillPhase.VALIDATE: cls.IMPLEMENT,
            SkillPhase.REPORT: cls.REPORT,
        }
        return _mapping.get(phase, cls.INVENTORY)


def _parse_idiom_yaml(content: str, filename: str) -> dict[str, Any] | None:
    """Parse and validate YAML content for an idiom map.

    Args:
        content: Raw YAML string.
        filename: Source filename used for log messages.

    Returns:
        Parsed dict on success, or ``None`` if parsing/validation fails.
    """
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse idiom map %s: %s", filename, exc)
        return None

    if raw is None:
        logger.warning("Idiom map %s is empty", filename)
        return None
    if not isinstance(raw, dict):
        logger.warning(
            "Idiom map %s has unexpected top-level type %s (expected dict)",
            filename,
            type(raw).__name__,
        )
        return None

    data: dict[str, Any] = raw
    logger.debug(
        "Loaded idiom map %s: %d idioms, %d deps",
        filename,
        len(data.get("idioms", [])),
        len(data.get("dependencies", {})),
    )
    return data


def _load_idiom_map(
    source_lang: str,
    target_lang: str,
    *,
    idiom_config: IdiomConfig | None = None,
    client: GeminiClient | None = None,
) -> dict[str, Any] | None:
    """Load an idiom map for the given language pair using a 3-tier fallback.

    **Tier 1 — Bundled**: Looks for ``{source_lang}_to_{target_lang}.yaml`` in
    the bundled ``vaig.skills.code_migration.idioms`` package, using
    :mod:`importlib.resources` so the file is accessible whether the package is
    installed as a wheel or run from source.

    **Tier 2 — User cache**: If the bundled map is not found, looks in the
    ``idiom_config.cache_dir`` directory (default: ``~/.vaig/idioms/``) for a
    previously generated map.

    **Tier 3 — LLM generate**: If neither bundled nor cached maps exist, and
    ``idiom_config.auto_generate`` is ``True`` and a ``client`` is provided,
    generates a new map using the LLM and caches it for future use.

    If all tiers fail, returns ``None`` (graceful degradation).

    Args:
        source_lang: Source programming language (e.g. ``"python"``).
        target_lang: Target programming language (e.g. ``"go"``).
        idiom_config: Optional :class:`~vaig.core.config.IdiomConfig` that
            controls caching and LLM generation.  When ``None``, tiers 2 and 3
            are skipped.
        client: Optional :class:`~vaig.core.client.GeminiClient` used for
            tier-3 generation.  Ignored when ``idiom_config.auto_generate`` is
            ``False`` or ``idiom_config`` is ``None``.

    Returns:
        Parsed YAML dict on success, or ``None`` if no map is found or could be
        generated.
    """
    source_lang = source_lang.lower()
    target_lang = target_lang.lower()
    filename = f"{source_lang}_to_{target_lang}.yaml"

    # ── Tier 1: bundled map ───────────────────────────────────────────────
    try:
        idioms_pkg = importlib.resources.files("vaig.skills.code_migration.idioms")
        resource = idioms_pkg.joinpath(filename)
        content = resource.read_text(encoding="utf-8")
        logger.debug("_load_idiom_map: bundled map found for %s→%s", source_lang, target_lang)
        return _parse_idiom_yaml(content, filename)
    except (FileNotFoundError, ModuleNotFoundError):
        logger.debug("_load_idiom_map: no bundled map for %s→%s", source_lang, target_lang)
    except OSError as exc:
        logger.warning("_load_idiom_map: failed to read bundled map %s: %s", filename, exc)

    if idiom_config is None:
        return None

    # ── Tier 2: user cache ────────────────────────────────────────────────
    cache_dir = Path(idiom_config.cache_dir).expanduser()
    cache_path = cache_dir / filename
    if cache_path.exists():
        try:
            cached_content = cache_path.read_text(encoding="utf-8")
            logger.debug("_load_idiom_map: cache hit for %s→%s at %s", source_lang, target_lang, cache_path)
            return _parse_idiom_yaml(cached_content, filename)
        except OSError as exc:
            logger.warning("_load_idiom_map: failed to read cache %s: %s", cache_path, exc)

    # ── Tier 3: LLM generation ────────────────────────────────────────────
    if not idiom_config.auto_generate or client is None:
        logger.debug(
            "_load_idiom_map: auto_generate=%s, client=%s — skipping LLM for %s→%s",
            idiom_config.auto_generate,
            "provided" if client is not None else "None",
            source_lang,
            target_lang,
        )
        return None

    logger.info("_load_idiom_map: generating idiom map via LLM for %s→%s", source_lang, target_lang)
    try:
        from vaig.skills.code_migration.idiom_generator import IdiomGenerator

        generator = IdiomGenerator(client, cache_dir=cache_dir)
        yaml_content = generator.generate(source_lang, target_lang)
        return _parse_idiom_yaml(yaml_content, filename)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_load_idiom_map: LLM generation failed for %s→%s: %s",
            source_lang,
            target_lang,
            exc,
        )
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

    def save_state(self, path: Path) -> None:
        """Persist tracker records to a JSON file using an atomic write.

        Writes to ``{path}.tmp`` first and then renames to ``path`` so the
        original file (if any) is never corrupted on a mid-write crash.

        Args:
            path: Destination file path (e.g. ``.vaig/migration-state.json``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"records": dict(self._records)}
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up tmp file on failure; do not leave partial files
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_state(self, path: Path) -> None:
        """Load tracker records from a JSON file.

        Silently does nothing if the file does not exist.

        Args:
            path: Source file path (e.g. ``.vaig/migration-state.json``).
        """
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load migration state from %s: %s", path, exc)
            return

        if not isinstance(raw, dict):
            logger.warning("Failed to load migration state from %s: invalid top-level JSON shape", path)
            self._records = {}
            return

        records_raw = raw.get("records", {})
        normalized_records: dict[str, dict[str, Any]] = {}
        if isinstance(records_raw, dict):
            for filename, record in records_raw.items():
                if isinstance(filename, str) and isinstance(record, dict):
                    normalized_records[filename] = record

        self._records = normalized_records


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
        resume: bool = False,
        idiom_config: IdiomConfig | None = None,
        client: GeminiClient | None = None,
    ) -> None:
        self._source_lang = source_lang.lower()
        self._target_lang = target_lang.lower()
        self._workspace = workspace or Path.cwd()
        self._tracker = _MigrationTracker()
        self._state_path = _ensure_project_dir() / "migration-state.json"

        # Resume from previous run if requested
        if resume:
            self._tracker.load_state(self._state_path)

        # Load idiom map eagerly via 3-tier fallback; None if no map exists
        self._idiom_data = _load_idiom_map(
            self._source_lang,
            self._target_lang,
            idiom_config=idiom_config,
            client=client,
        )

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
                "system_instruction": (
                    f"You are a senior engineering lead overseeing the final stages of a "
                    f"{lang_pair} code migration project. "
                    "Your responsibilities cover the VERIFY and REPORT phases: "
                    "run completeness checks on all migrated files (scan for TODO, FIXME, "
                    "stub bodies, placeholder patterns), confirm full semantic fidelity with "
                    "the original source, and synthesise a structured final migration report. "
                    "The report must include an executive summary, per-file status table, "
                    "idiom transformations applied, dependency substitutions, design decisions "
                    "made, and recommended next steps. "
                    "Be direct and precise — your verdict determines whether the migration "
                    "is accepted or sent back for rework. "
                    "Use the provided file tools to scan migrated output before reporting."
                ),
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
        try:
            self._tracker.save_state(self._state_path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to persist migration state to %s after recording %s (%s/%s): %s",
                self._state_path, filename, phase, status, exc,
            )

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

"""Greenfield Skill — prompts for 6-stage structured project generation.

Stage order:
  REQUIREMENTS → ARCHITECTURE_DECISION → PROJECT_SPEC → SCAFFOLD → IMPLEMENT → VERIFY
"""

from __future__ import annotations

from vaig.core.prompt_defense import (
    ANTI_HALLUCINATION_RULES,
    ANTI_INJECTION_RULE,
    COT_INSTRUCTION,
    DELIMITER_DATA_END,
    DELIMITER_DATA_START,
)

# ── System instruction ────────────────────────────────────────

SYSTEM_INSTRUCTION = f"""{ANTI_INJECTION_RULE}

You are a Principal Software Architect with 20+ years of experience creating \
production-grade projects from scratch.  Your role is to guide a project through \
six structured stages — from raw requirements all the way to a verified, \
runnable codebase.

## Core Principles

1. **Complete before confident** — Never emit placeholder code, TODO stubs, or ellipsis \
bodies.  Every function, class, and module must be production-ready.
2. **Explicit decisions** — Document every architectural decision with an ADR \
(Architecture Decision Record) before writing code.
3. **Minimal surface area** — Generate only what the requirements demand. Avoid \
over-engineering; add complexity only when the requirement explicitly calls for it.
4. **Testability by design** — Structure every module so it can be unit-tested without \
external dependencies.  Dependency injection over global state.
5. **Security by default** — Input validation, secure defaults, no hardcoded secrets.

## Chain-of-Thought Requirement
{COT_INSTRUCTION}

## Anti-Hallucination Rules
{ANTI_HALLUCINATION_RULES}
"""

# ── Stage prompts ─────────────────────────────────────────────

_REQUIREMENTS_PROMPT = f"""## Stage 1 — Requirements Analysis

{ANTI_INJECTION_RULE}

Analyse the user input and extract structured, unambiguous requirements.

### User Input:
{{user_input}}

### Your Task:

Produce a **Requirements Document** with:

1. **Project Overview** — One paragraph describing what is being built and why.
2. **Functional Requirements** — Numbered list of concrete capabilities (FR-1, FR-2, …).
   Each FR must be testable: "The system SHALL …"
3. **Non-Functional Requirements** — Performance, security, scalability, observability
   targets with measurable acceptance criteria.
4. **Constraints** — Technology stack, language, runtime, external integrations.
5. **Out of Scope** — Explicit list of things NOT being built in this iteration.
6. **Open Questions** — Ambiguities that require clarification before proceeding.

Be precise.  "Fast" is not a requirement.  "API response p99 < 200 ms under 100 RPS" is.
"""

_ARCHITECTURE_DECISION_PROMPT = f"""## Stage 2 — Architecture Decision

{ANTI_INJECTION_RULE}

Design the high-level architecture and document every significant decision.

### Requirements Document:
{DELIMITER_DATA_START}
{{context}}
{DELIMITER_DATA_END}

### Additional context:
{{user_input}}

### Your Task:

Produce an **Architecture Decision Record (ADR) set** containing:

1. **System Context** — Component diagram (ASCII or Mermaid) showing the system and its
   external actors/dependencies.
2. **ADR-001: Technology Stack** — Options considered, tradeoffs, decision, consequences.
3. **ADR-002: Module Structure** — Package layout, layer boundaries, dependency direction.
4. **ADR-003: Data Model** — Core entities, relationships, storage strategy.
5. **ADR-004: Error Strategy** — How errors propagate, logging conventions, retry policy.
6. **ADR-005: Testing Strategy** — Unit / integration / E2E boundaries, fixtures approach.
7. **Risk Register** — Top 3 risks with mitigation plans.

Each ADR must follow the format:
- **Status**: Proposed / Accepted / Superseded
- **Context**: Why this decision is needed
- **Options**: At least 2 alternatives with pros/cons
- **Decision**: What was chosen and why
- **Consequences**: What becomes easier / harder
"""

_PROJECT_SPEC_PROMPT = f"""## Stage 3 — Project Specification

{ANTI_INJECTION_RULE}

Translate the ADRs into a precise, file-level implementation specification.

### Architecture Decisions:
{DELIMITER_DATA_START}
{{context}}
{DELIMITER_DATA_END}

### Additional context:
{{user_input}}

### Your Task:

Produce a **Project Spec** containing:

1. **File Tree** — Complete directory structure with every file listed (no globs).
2. **Module Specs** — For each module:
   - Purpose (one sentence)
   - Public API: every exported function/class with full signature and docstring
   - Dependencies (imports required)
   - Key implementation notes
3. **Configuration Schema** — Every env var, config key, and its type/default/description.
4. **CLI Interface** — Commands, flags, arguments with examples (if CLI is in scope).
5. **Test Plan** — For each module: test file path, test function names, scenarios covered.
6. **Implementation Order** — Topologically sorted list (dependencies first).

This spec is the single source of truth the Scaffold stage will follow.
"""

_SCAFFOLD_PROMPT = f"""## Stage 4 — Scaffold

{ANTI_INJECTION_RULE}

Create the complete project skeleton: all files, configs, and boilerplate.

### Project Specification:
{DELIMITER_DATA_START}
{{context}}
{DELIMITER_DATA_END}

### Additional context:
{{user_input}}

### Your Task:

Write every file listed in the Project Spec that is NOT source logic:

1. **Project config** — `pyproject.toml` / `package.json` / `go.mod` (language-appropriate).
2. **Tooling config** — linter, formatter, type checker, test runner configurations.
3. **CI pipeline** — `.github/workflows/ci.yml` or equivalent.
4. **Docker** — `Dockerfile` and `docker-compose.yml` if in scope.
5. **Module stubs** — Empty (but valid) source files with correct package declarations,
   module docstrings, and import sections.  NO implementation yet — leave function bodies
   as a single `raise NotImplementedError` ONLY in stub phase.
6. **Test stubs** — Empty test files with correct imports and one `test_placeholder` function
   that asserts `True` (to verify the test runner finds the file).
7. **README.md** — Project overview, setup, usage, architecture summary.

Use the output directory from config.  Write each file individually using write_file.
Confirm every file was written by running `list_files` at the end.
"""

_IMPLEMENT_PROMPT = f"""## Stage 5 — Implement

{ANTI_INJECTION_RULE}

Replace every stub with complete, production-ready implementation.

### Project Specification:
{DELIMITER_DATA_START}
{{context}}
{DELIMITER_DATA_END}

### Additional context:
{{user_input}}

### Implementation Rules (NON-NEGOTIABLE):

1. **Zero placeholders** — NEVER leave `pass`, `...`, `TODO`, `FIXME`, or
   `raise NotImplementedError` in the final code.  Every function must have
   a complete, working body.
2. **All imports resolved** — Verify every import before writing.
3. **Type hints required** — Annotate all signatures.  Use modern typing idioms.
4. **Match project style** — Follow the conventions established in Stage 3.
5. **Docstrings on all public items** — Module, class, and function level.
6. **Error handling complete** — Handle edge cases; never swallow exceptions silently.

### Workflow:

1. Read the Project Spec (from context above).
2. For each module in implementation order, read its stub, then write the complete file.
3. Re-read each written file to confirm correctness.
4. Run `verify_completeness` on all written files.
5. Run linting and type checking if available.
"""

_VERIFY_PROMPT = f"""## Stage 6 — Verify

{ANTI_INJECTION_RULE}

Validate the complete project against every requirement in the spec.

### Project Specification:
{DELIMITER_DATA_START}
{{context}}
{DELIMITER_DATA_END}

### Additional context:
{{user_input}}

### Verification Checklist:

For every file in the Project Spec:

1. **Placeholder scan** — Run `verify_completeness`.  Zero tolerance for stubs.
2. **Syntax** — Run the language-appropriate syntax/type checker.
3. **Tests** — Run the test suite.  All tests must pass.
4. **Interface match** — Confirm every function signature matches the spec.
5. **Requirements coverage** — Map each FR to the code that satisfies it.

### Output Format:

Produce a **Final Verification Report** with:

- **Overall Result**: PASS ✅ or FAIL ❌
- **Requirements Coverage Table**: | FR | Status | File:Function |
- **Per-file Table**: | File | Placeholders | Syntax | Tests |
- **Test Results**: Pass / fail / error count
- **Failures**: Each failure with file, line, and description
- **Risks**: Any remaining concerns or recommended follow-up

The project is only DONE when this report shows PASS ✅ with zero failures.
"""

# ── Public map ────────────────────────────────────────────────

STAGE_PROMPTS: dict[str, str] = {
    "requirements": _REQUIREMENTS_PROMPT,
    "architecture_decision": _ARCHITECTURE_DECISION_PROMPT,
    "project_spec": _PROJECT_SPEC_PROMPT,
    "scaffold": _SCAFFOLD_PROMPT,
    "implement": _IMPLEMENT_PROMPT,
    "verify": _VERIFY_PROMPT,
}
"""Mapping from stage name to its prompt template.

Each template has ``{context}`` and ``{user_input}`` placeholders.
The ``requirements`` stage only uses ``{user_input}``.
"""

# Alias for compatibility with the standard skill prompt-defense test suite,
# which iterates over ``PHASE_PROMPTS``.  Greenfield uses stage terminology
# internally but exposes the same interface.
PHASE_PROMPTS = STAGE_PROMPTS

STAGE_ORDER: list[str] = [
    "requirements",
    "architecture_decision",
    "project_spec",
    "scaffold",
    "implement",
    "verify",
]
"""Ordered list of Greenfield stage names."""

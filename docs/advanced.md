# Advanced Usage

## Composite Skills

Combine multiple skills for comprehensive analysis by merging their system instructions, agent pipelines, tags, and phases:

```python
from vaig.skills.base import CompositeSkill
from vaig.skills.registry import SkillRegistry

registry = SkillRegistry(settings)
code_review = registry.get("code-review")
threat_model = registry.get("threat-model")
compliance = registry.get("compliance-check")

# Merge into a single composite skill
full_audit = CompositeSkill(
    name="full-security-audit",
    skills=[code_review, threat_model, compliance],
)
```

The `CompositeSkill` merges:
- **System instructions** — Concatenated from all skills
- **Agent pipelines** — All agents from all skills, run sequentially
- **Tags** — Union of all tags
- **Phases** — Union of all supported phases

## Chunked Processing

For files that exceed the model's context window (e.g., a 20 MB log file), VAIG uses the `ChunkedProcessor` with a Map-Reduce approach.

### How It Works

1. **Split**: The input is divided into overlapping chunks based on `chunking.chars_per_token` and the model's context limit
2. **Map**: Each chunk is processed independently by a specialist agent
3. **Overlap**: Adjacent chunks share `chunk_overlap_ratio` (default: 10%) of content to avoid missing patterns at boundaries
4. **Reduce**: All chunk results are merged into a final analysis by a synthesizer agent
5. **Delay**: `inter_chunk_delay` (default: 2s) pauses between API calls to respect rate limits

### Configuration

```yaml
chunking:
  chunk_overlap_ratio: 0.1          # 10% overlap
  token_safety_margin: 0.1          # 10% reserved for prompt/instructions
  chars_per_token: 2.0              # Character-to-token ratio
  inter_chunk_delay: 2.0            # Seconds between chunk API calls
```

### When It Activates

Chunked processing activates automatically when:
- A file exceeds the model's context window
- The total input (question + context + system instruction) exceeds the token limit

You do not need to do anything special — VAIG detects the situation and switches to chunked mode.

## Workspace Mode

The coding agent operates within a sandboxed workspace:

```bash
# Set workspace root
vaig ask "Refactor this module" -f src/auth.py --code -w /path/to/project

# In REPL
vaig chat -w /path/to/project
/code
```

### Workspace Security

- All file operations are sandboxed to the workspace root
- Paths outside the workspace are rejected
- `blocked_paths` config prevents access to sensitive directories
- Shell commands are restricted to the `allowed_commands` allowlist
- `confirm_actions` (default: `true`) requires user approval for write operations

### Adding Allowed Commands

```yaml
coding:
  allowed_commands:
    - python
    - pip
    - git
    - make
    - pytest
    - docker
    - kubectl                        # Add kubectl for k8s manifests
    - terraform                      # Add terraform for IaC
```

## Custom Skills

### Creating a Skill

```bash
vaig skills create my-skill -d "My custom analysis" -t "custom,review"
```

This generates:

```
~/.vaig/skills/my-skill/
├── __init__.py
├── skill.py          # Skill class with metadata and agent config
└── prompts.py        # Phase-specific prompt templates
```

### Skill Structure

```python
# skill.py
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase


class MySkill(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="my-skill",
            display_name="My Custom Skill",
            description="What this skill does",
            version="1.0.0",
            tags=["custom", "review"],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-pro",
        )

    def get_system_instruction(self) -> str:
        return "You are a specialist in..."

    def get_phase_prompt(self, phase, context, user_input):
        templates = {
            "analyze": "Analyze: {context}\n\nUser input: {user_input}",
            "report": "Generate report for: {context}\n\nFindings: {user_input}",
        }
        return templates.get(phase.value, templates["analyze"]).format(
            context=context, user_input=user_input
        )

    def get_agents_config(self) -> list[dict]:
        return [
            {
                "name": "analyzer",
                "role": "Analyzer",
                "system_instruction": "You analyze...",
                "model": "gemini-2.5-flash",
            },
            {
                "name": "reporter",
                "role": "Reporter",
                "system_instruction": "You generate reports...",
            },
        ]
```

### Adding Structured Output to a Custom Skill

You can use Gemini's `response_schema` to force a reporter agent to return validated JSON instead of free-form text. This is useful when reports need a consistent, machine-parseable structure.

**Step 1: Define a Pydantic v2 model**

```python
# my_skill/schema.py
from enum import StrEnum
from pydantic import BaseModel, Field


class Severity(StrEnum):          # Use StrEnum, not Enum
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Finding(BaseModel):
    title: str
    severity: Severity
    description: str = ""


class MyReport(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# Report\n\n{self.summary}\n"]
        for f in self.findings:
            lines.append(f"- **[{f.severity}]** {f.title}: {f.description}")
        return "\n".join(lines)
```

**Step 2: Set `response_schema` on the reporter agent**

```python
# In get_agents_config()
{
    "name": "reporter",
    "role": "Report Generator",
    "requires_tools": False,
    "system_instruction": "Generate a structured report...",
    "response_schema": MyReport,
    "response_mime_type": "application/json",
}
```

**Step 3: Override `post_process_report()`**

```python
from pydantic import ValidationError
from my_skill.schema import MyReport


class MySkill(BaseSkill):
    def post_process_report(self, content: str) -> str:
        try:
            report = MyReport.model_validate_json(content)
            return report.to_markdown()
        except (ValueError, ValidationError):
            return content  # Graceful fallback
```

> **Note:** Enum fields in the schema **must** use `StrEnum` (not `Enum`). Gemini's `response_schema` requires string-valued enum variants; plain `Enum` subclasses with non-string values will be rejected. Validate the raw JSON with `model_validate_json()`, not `model_validate()` — the model returns a JSON string, not a dict.

### Skill Discovery

Custom skills are auto-discovered at startup from:
1. `~/.vaig/skills/` (default)
2. The `skills.custom_dir` config path
3. Any directory containing a `skill.py` with a `BaseSkill` subclass

```yaml
skills:
  custom_dir: ./my-project-skills
  enabled:
    - my-skill                       # Add your custom skill name
```

## Pipeline Mode

Pipeline mode replaces the single `CodingAgent` loop with a structured 3-agent pipeline: **Planner → Implementer → Verifier**. Use it for complex coding tasks that benefit from an explicit planning phase and automated completeness verification.

### When to Use

| Situation | Recommendation |
|-----------|---------------|
| Small edits, one-file changes | Default single-agent (`--code`) |
| Multi-file features, refactors, new modules | `--pipeline` |
| Tasks where correctness must be verified automatically | `--pipeline` |

### How It Works

```
Planner ──────────────────────────────────────────────────────
  1. Reads the codebase (read_file, list_files, search_files)
  2. Writes PLAN.md to the workspace root

Implementer ──────────────────────────────────────────────────
  3. Reads PLAN.md
  4. Writes ALL files — no placeholders, no TODOs

Verifier ─────────────────────────────────────────────────────
  5. Reads PLAN.md + written files
  6. Runs verify_completeness on each file
  7. Checks syntax; emits a structured PASS / FAIL report
  8. Short-circuits (halts) the pipeline on first failure
```

### Usage

```bash
# One-shot pipeline mode
vaig ask "Add a rate-limiting middleware to the FastAPI app" \
    --code --pipeline -w /path/to/project

# Enable pipeline mode by default in config (avoid passing --pipeline every time)
# vaig.yaml
coding:
  pipeline_mode: true
```

> **Warning:** Pipeline mode does **not** support interactive `confirm_actions`. If `confirm_actions: true` is set in config, a warning is logged and the pipeline proceeds without prompts. For interactive confirmation, use single-agent mode (omit `--pipeline`).

### Configuration

```yaml
coding:
  pipeline_mode: false               # Default; set true to always use pipeline
  confirm_actions: true              # Always require approval for writes
  workspace_root: /path/to/project    # Workspace root (or pass -w on CLI)
```

---

## GreenfieldSkill — Full Project Generation

`GreenfieldSkill` generates a complete project from a plain-language description. It runs 6 sequential stages, each building on the previous stage's output.

### Stages

| # | Stage | What It Produces |
|---|-------|-----------------|
| 1 | **Requirements** | Structured requirements extracted from your description |
| 2 | **Architecture Decision** | ADRs (Architecture Decision Records) for key technical choices |
| 3 | **Project Spec** | File-level implementation spec derived from the ADRs |
| 4 | **Scaffold** | Project skeleton — config files, CI pipeline, stub modules |
| 5 | **Implement** | Every stub replaced with production-ready code |
| 6 | **Verify** | Full project validated against the original requirements |

### Phase-to-Stage Mapping

The `GreenfieldSkill` maps standard `SkillPhase` values to internal stages:

| SkillPhase | Greenfield Stage(s) |
|------------|---------------------|
| `ANALYZE` | `requirements` |
| `PLAN` | `architecture_decision` → `project_spec` |
| `EXECUTE` | `scaffold` → `implement` |
| `VALIDATE` | `verify` |
| `REPORT` | `verify` (summary report) |

When a phase covers multiple stages, `get_phase_prompt()` returns the first stage. Use `get_stage_prompt(stage_name, ...)` to address a specific stage directly.

### Invoking Greenfield

```bash
# Generate a full project from a description
vaig ask "Build a REST API in Python with FastAPI, Postgres, and Docker" \
    --skill greenfield -w /path/to/output-dir
```

```python
from vaig.skills.greenfield import GreenfieldSkill
from vaig.skills.base import SkillPhase

skill = GreenfieldSkill()

# Execute stages in order, passing previous output as context
requirements = skill.get_stage_prompt("requirements", context="", user_input="Build a REST API in Python with FastAPI")
# ... run model, collect output ...

adr = skill.get_stage_prompt("architecture_decision", context=requirements_output, user_input="")
# ... run model, collect output ...

# Or use SkillPhase enum (returns first stage in that phase)
prompt = skill.get_phase_prompt(SkillPhase.PLAN, context=requirements_output, user_input="")
```

### Requirements

- `requires_live_tools: true` — the Greenfield agent needs file tools (write_file, list_files, etc.) to scaffold the project
- A writable workspace root must be set via `-w` or `coding.workspace` in config
- Recommended model: `gemini-2.5-pro` (set automatically by skill metadata)
- All 6 stages run sequentially; do not skip stages — each stage depends on the previous output

### Advanced: Accessing Individual Stages

```python
# List stage order
skill = GreenfieldSkill()
print(skill.stage_order)
# ['requirements', 'architecture_decision', 'project_spec', 'scaffold', 'implement', 'verify']

# Address a stage by name (raises ValueError for unknown stage names)
prompt = skill.get_stage_prompt("project_spec", context=adr_output, user_input="")
```

---

## Auto-Skill Detection

The `--auto-skill` flag uses keyword matching to automatically select the best skill:

```bash
vaig ask "Why are my pods crashing?" --auto-skill
# → Matches tags: "sre", "incident", "kubernetes" → selects error-triage or rca

vaig ask "Is this Terraform safe to apply?" -f main.tf --auto-skill
# → Matches tags: "iac", "terraform", "security" → selects iac-review
```

The `suggest_skill()` method in `SkillRegistry` scores each skill by matching query words against skill tags, names, and descriptions.

## Per-Request Cost Tracking

VAIG estimates the cost of every API call based on model pricing:

```
Model: gemini-2.5-pro
Tokens: 15,234 total (12,100 prompt + 3,134 completion)
Estimated cost: $0.0234
```

Cost is displayed after each response and included in exports. The pricing module (`vaig.core.pricing`) maintains per-model pricing tables (per 1M tokens).

## Multi-Agent Orchestration Strategies

Skills default to `sequential` orchestration, but you can configure different strategies:

### Sequential (default)

Agents run one after another. Each agent receives the previous agent's output as context.

```
Agent 1 → Agent 2 → Agent 3 → Final Response
```

### Fan-Out

All agents run in parallel (up to `agents.max_concurrent`), then results are merged.

```
         ┌→ Agent 1 ─┐
Input ───┼→ Agent 2 ──┼→ Merge → Final Response
         └→ Agent 3 ─┘
```

### Lead-Delegate

A lead agent decides which specialists to invoke based on the task.

```
Lead Agent ──→ decides ──→ Specialist A
                       ──→ Specialist B
                       ──→ Merge → Final Response
```

## Retry and Error Handling

VAIG implements exponential backoff for API errors:

```yaml
retry:
  max_retries: 3
  initial_delay: 1.0
  max_delay: 60.0
  backoff_multiplier: 2.0
  retryable_status_codes: [429, 500, 503]
```

Retry flow:
1. API call fails with a retryable status code
2. Wait `initial_delay` seconds
3. Retry with delay multiplied by `backoff_multiplier`
4. Continue until `max_retries` is reached or `max_delay` is hit
5. If the primary model fails, automatically falls back to `models.fallback`

## Logging and Debugging

```bash
# Verbose mode — sets log level to INFO
vaig ask "test" --verbose

# Debug mode — sets log level to DEBUG
vaig ask "test" --debug

# Via config
# vaig.yaml
logging:
  level: DEBUG
  show_path: true                    # Include file paths in log lines

# Via environment
export VAIG_LOGGING__LEVEL=DEBUG
```

Log output includes:
- API calls and response times
- Tool invocations and results
- Skill loading and agent pipeline setup
- Session create/load/save events
- Retry attempts and fallbacks

---

## Surgical File Editing with `patch_file`

The `patch_file` tool applies a **unified diff** to an existing file. It is the precision instrument of the file-editing toolkit — safer and more transparent than `write_file` for modifying files that already exist.

### When to Use Which Tool

| Tool | Best For |
|------|----------|
| `read_file` | Reading files before editing |
| `write_file` | Creating new files or completely replacing content |
| `edit_file` | Single exact-string substitution (Claude Code pattern) |
| `patch_file` | Multi-hunk surgical edits — add/remove/modify specific lines |

Use `patch_file` over `write_file` when you need to modify an existing file and want to express the change as a diff rather than rewriting the entire content. This is especially valuable for large files where transmitting the full new content is expensive, and for changes that need to be human-auditable.

### Unified Diff Format

The patch must follow the standard unified diff format with `@@ ... @@` hunk headers:

```
@@ -<old_start>[,<old_count>] +<new_start>[,<new_count>] @@
 context line
-removed line
+added line
 context line
```

- Lines starting with ` ` (space) are context — they must match the file exactly
- Lines starting with `-` are removed
- Lines starting with `+` are added
- The `@@` header counts are optional; VAIG re-derives them from the ops

**Example: a two-hunk patch**

```diff
@@ -3,7 +3,8 @@
 import logging
 import os
+import re
 from pathlib import Path
 
 logger = logging.getLogger(__name__)
-LOG_LEVEL = "INFO"
+LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

### Atomic Application

All hunks in a patch succeed together or **none are applied**. If any hunk's context lines do not match the current file, the entire patch is rejected and the file is left unchanged. The response is a JSON object:

```json
{"success": true, "path": "src/mymodule.py"}
```

```json
{
  "success": false,
  "error": "Hunk context mismatch",
  "conflicts": [
    {
      "hunk_start": 42,
      "expected": "    LOG_LEVEL = \"INFO\"\n",
      "found":    "    LOG_LEVEL = \"WARNING\"\n"
    }
  ]
}
```

### Backup Behavior

Set `patch.backup_enabled: true` in config (or pass `backup: "true"` in the tool call) to create a `.orig` backup before patching:

```yaml
# vaig.yaml
coding:
  patch:
    backup_enabled: true             # Creates src/auth.py.orig before patching
    max_hunk_size: 500               # Max lines per hunk (0 = unlimited)
```

When `backup_enabled` is true, VAIG writes `<filepath>.orig` beside the target file before applying changes. If writing the backup fails (e.g. disk full), a warning is logged but patching continues.

### Error Handling

| Error | Meaning | Fix |
|-------|---------|-----|
| `"File not found"` | Target file does not exist | Use `write_file` to create it first |
| `"Invalid unified diff: missing hunk header"` | Patch has no `@@` header | Add proper `@@ -L,N +L,N @@` headers |
| `"Hunk context mismatch"` | Context lines don't match file | Re-read the file and regenerate the patch |
| `"Write failed"` | Disk/permissions issue | Check file permissions and disk space |

### Full Example: Applying a Multi-Hunk Patch via CLI

```bash
vaig ask "Add request ID logging to the auth middleware" \
    --code -w /path/to/project
```

The agent will call `patch_file` with something like:

```
path: "src/middleware/auth.py"
patch: |
  @@ -1,5 +1,6 @@
   import logging
  +import uuid
   from fastapi import Request
   
   logger = logging.getLogger(__name__)
  @@ -18,6 +19,7 @@
   async def auth_middleware(request: Request, call_next):
  +    request_id = str(uuid.uuid4())
  +    logger.info("request_id=%s path=%s", request_id, request.url.path)
       token = request.headers.get("Authorization")
backup: "true"
```

---

## Git Integration (Automated Workflow)

`GitManager` wraps the `git` and `gh` CLIs to automate the full branch → commit → push → PR lifecycle during coding pipeline runs. Disabled by default — enable explicitly in config.

### Safety Model

- **Never commits to `main` or `master`** — `GitSafetyError` is raised if the current branch is a protected branch when `commit_all` or `push` is called
- **Dirty tree detection** — `check_clean()` returns `False` when uncommitted changes exist; the pipeline can abort rather than creating a noisy diff
- **No-op when disabled** — every method returns immediately when `coding.git.enabled: false`, so the same code path works in both modes

### Full Lifecycle

```
check_clean()          → Assert working tree is clean before starting
create_branch(name)    → git checkout -b vaig/<task-slug>
  ... pipeline writes files ...
commit_all(message)    → git add -A && git commit -m "<message>"
push(set_upstream=True)→ git push -u origin vaig/<task-slug>
create_pr(title, body) → gh pr create --title "..." --body "..." --base main
```

### Configuration

```yaml
# vaig.yaml
coding:
  git:
    enabled: true                    # Master switch — off by default
    auto_branch: true                # Create feature branch before writes
    auto_commit: true                # Commit after each pipeline phase
    auto_pr: true                    # Open PR via gh CLI when run completes
    pr_provider: "gh"                # Only "gh" (GitHub CLI) is supported
    branch_prefix: "vaig/"          # All auto branches are "vaig/<slug>"
    commit_signoff: false            # Append Signed-off-by trailer
```

### Branch Naming

Branch names are derived from the task description using `_sanitize_branch_name()`:

- Lowercased
- Non-alphanumeric characters replaced with hyphens
- Leading/trailing hyphens stripped
- Truncated to 60 characters
- Prefixed with `branch_prefix` (default `vaig/`)

```
"Add retry logic to GCS upload"  →  vaig/add-retry-logic-to-gcs-upload
"Fix auth bug #123"               →  vaig/fix-auth-bug-123
```

Override the prefix in config:

```yaml
coding:
  git:
    branch_prefix: "bot/"           # → bot/add-retry-logic-to-gcs-upload
```

### Enabling Auto-Commit for Coding Tasks

```yaml
# vaig.yaml — minimal git integration (branch + commit, no PR)
coding:
  git:
    enabled: true
    auto_branch: true
    auto_commit: true
    auto_pr: false
```

```bash
vaig ask "Refactor the database connection pool" \
    --code --pipeline -w /path/to/project
# → Creates vaig/refactor-the-database-connection-pool
# → Commits all changes with a conventional commit message
```

### Full Auto-PR Pipeline

```yaml
# vaig.yaml — full auto-PR pipeline
coding:
  git:
    enabled: true
    auto_branch: true
    auto_commit: true
    auto_pr: true
    pr_provider: "gh"
    commit_signoff: true
```

```bash
# Prerequisites: gh must be authenticated (gh auth login)
vaig ask "Add OpenTelemetry tracing to the FastAPI app" \
    --code --pipeline -w /path/to/project
# → Branch: vaig/add-opentelemetry-tracing-to-the-fastapi-app
# → Commits changes
# → Pushes branch to origin
# → Opens PR: "Add OpenTelemetry tracing to the FastAPI app" → main
# → Prints the PR URL
```

### Programmatic Usage

```python
from pathlib import Path
from vaig.core.config import GitConfig
from vaig.core.git_integration import GitManager

config = GitConfig(enabled=True, auto_branch=True, auto_commit=True, auto_pr=True)
manager = GitManager(config, workspace=Path("my_project"))

if not manager.check_clean():
    raise RuntimeError("Workspace has uncommitted changes — stash or commit first")

manager.create_branch("vaig/my-feature")
# ... implement feature, write files ...
manager.commit_all("feat(auth): add JWT refresh token rotation")
manager.push()
pr_url = manager.create_pr(
    title="feat(auth): add JWT refresh token rotation",
    body="Implements sliding-window refresh token rotation with Redis backing.",
)
print(pr_url)
```

---

## Workspace RAG (Semantic Code Search)

Workspace RAG builds a local vector index over your workspace source files using **ChromaDB**, enabling semantic search during coding sessions. The agent can ask "find code related to authentication" and get relevant file chunks back rather than relying on regex grep alone.

### Prerequisites

```bash
pip install chromadb
# or
pip install 'vertex-ai-toolkit[rag]'
```

### What Gets Indexed

`WorkspaceRAG` walks the workspace recursively and indexes files matching `extensions`. It **skips** these directories automatically:

| Skipped Directory | Reason |
|-------------------|--------|
| `.vaig/` | Internal VAIG data |
| `.venv/`, `venv/`, `.env/` | Virtual environments |
| `node_modules/` | JS dependencies |
| `.git/` | Git objects |
| `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/` | Build/cache artifacts |
| `.tox/`, `.nox/` | Test environments |
| `dist/`, `build/`, `.eggs/` | Distribution outputs |

Files larger than **1 MB** are also skipped (logged at DEBUG level).

### Chunking Strategy

Files are split into overlapping **line-based chunks**:

- **Chunk size**: 200 lines
- **Overlap**: 20 lines between adjacent chunks
- **ID format**: `relative/path/to/file.py::0`, `::1`, `::2`, …

The overlap prevents boundary effects where a relevant code block spans the edge of two chunks.

### Index Storage

The ChromaDB index is stored persistently at:

```
<workspace_root>/.vaig/workspace-index/
```

On subsequent runs, the existing index is reused. Stale detection works by comparing the `mtime` of indexed files against the `build_timestamp` recorded when the index was last built.

### Configuration

```yaml
# vaig.yaml
coding:
  workspace_rag:
    enabled: true                    # Master switch — requires chromadb
    reindex_on_run: true             # Rebuild index if files changed since last run
    max_chunks: 500                  # Cap on total chunks indexed (discovery order)
    extensions:                      # File extensions to include
      - .py
      - .ts
      - .go
      - .java
      - .md
```

### Enabling Workspace RAG for a Large Codebase

```yaml
# vaig.yaml — tuned for a large monorepo
coding:
  workspace_rag:
    enabled: true
    reindex_on_run: false            # Manual reindex (too slow for huge repos)
    max_chunks: 2000                 # Increase cap for large projects
    extensions:
      - .py
      - .ts
      - .tsx
      - .go
      - .java
      - .kt
      - .md
      - .yaml
```

```bash
# Initial index build (explicit, before a long coding session)
vaig ask "build the workspace index" --code -w /path/to/monorepo

# Subsequent runs reuse the cached index
vaig ask "Find all places we handle authentication errors" \
    --code -w /path/to/monorepo
```

### Programmatic Usage

```python
from pathlib import Path
from vaig.core.config import WorkspaceRAGConfig
from vaig.core.workspace_rag import WorkspaceRAG

config = WorkspaceRAGConfig(
    enabled=True,
    reindex_on_run=True,
    max_chunks=1000,
    extensions=[".py", ".ts", ".go"],
)
rag = WorkspaceRAG(workspace=Path("/path/to/project"), config=config)

# Build or refresh the index
chunk_count = rag.build_index()
print(f"Indexed {chunk_count} chunks")

# Search
results = rag.search("JWT token validation", k=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['file']}")
    print(r['chunk'][:200])
    print("---")
```

Search returns a list of `{"file": str, "chunk": str, "score": float}` dicts sorted from most to least relevant. Scores are derived from ChromaDB's L2 distance via `score = 1.0 / (1.0 + distance)` (higher is better).

---

## Idiom Maps for Code Migration

The `migration` skill uses **idiom maps** — YAML files that describe idiomatic transformations between language pairs — to guide the LLM when migrating code. Rather than asking the model to invent transformations from scratch, it consults a curated catalog of patterns.

### 3-Tier Fallback Chain

```
1. Bundled maps (shipped with VAIG)
        ↓  not found
2. Cached maps (~/.vaig/idioms/)
        ↓  not found
3. LLM-generated map (IdiomGenerator) — requires idiom.auto_generate: true
```

At tier 3, `IdiomGenerator` calls the Gemini API, validates the output is valid YAML, and writes it atomically to the cache directory. Subsequent calls for the same language pair hit tier 2.

### Bundled Idiom Maps

VAIG ships 7 idiom maps out of the box:

| File | Migration Path |
|------|---------------|
| `python2_to_python3.yaml` | Python 2 → Python 3 |
| `python_to_go.yaml` | Python → Go |
| `javascript_to_typescript.yaml` | JavaScript → TypeScript |
| `java_to_kotlin.yaml` | Java → Kotlin |
| `angular_to_react.yaml` | Angular → React |
| `express_to_fastapi.yaml` | Express.js → FastAPI |
| `pentaho_to_glue.yaml` | Pentaho ETL → AWS Glue |

### Idiom Map Schema

Each YAML file follows this schema:

```yaml
source_lang: java
target_lang: kotlin

idioms:
  - source_pattern: "Null check with if statement"
    target_pattern: "Elvis operator or safe call"
    description: "Kotlin provides null-safe operators to replace verbose null checks"
    example_before: |
      String name = person.getName();
      if (name != null) {
          System.out.println(name.length());
      }
    example_after: |
      val name = person.name
      println(name?.length)

dependencies:
  "org.junit:junit": "org.junit.jupiter:junit-jupiter"
  "com.google.guava:guava": "standard library (use Kotlin stdlib)"
```

### Creating Custom Idiom Maps

Place custom YAML files in `~/.vaig/idioms/` following the naming convention `<source>_to_<target>.yaml`. They are loaded at tier 2 (cache) and take precedence over LLM generation.

```bash
# Create a custom map for Ruby → Python
cat > ~/.vaig/idioms/ruby_to_python.yaml << 'EOF'
source_lang: ruby
target_lang: python

idioms:
  - source_pattern: "Symbol to proc"
    target_pattern: "Lambda or operator.methodcaller"
    description: "Ruby's & :method shorthand maps to Python's operator module"
    example_before: |
      names = users.map(&:name)
    example_after: |
      from operator import attrgetter
      names = list(map(attrgetter('name'), users))

dependencies:
  "rails": "django or fastapi"
  "activerecord": "sqlalchemy"
EOF
```

### Configuration

```yaml
# vaig.yaml
idiom:
  enabled: true                      # Master switch for idiom map expansion
  auto_generate: true                # Generate maps via LLM on cache miss
  cache_dir: "~/.vaig/idioms"        # Where generated maps are stored
```

### Example: Migrating Java to Kotlin with Custom Idioms

```bash
# Uses the bundled java_to_kotlin.yaml
vaig ask "Migrate this service to Kotlin" \
    -f src/main/java/com/example/UserService.java \
    --skill migration

# With LLM generation enabled for an unsupported pair (e.g. Ruby → Python)
# vaig.yaml: idiom.auto_generate: true
vaig ask "Migrate this Rails model to Django" \
    -f app/models/user.rb \
    --skill migration
```

```python
from vaig.skills.code_migration.idiom_generator import IdiomGenerator
from vaig.core.client import GeminiClient

client = GeminiClient(settings)
generator = IdiomGenerator(client, cache_dir="~/.vaig/idioms")

# Check if already cached
if generator.is_cached("java", "kotlin"):
    print("Using cached map")

# Generate (or load from cache)
yaml_content = generator.generate("java", "kotlin")
idiom_map = generator.parse_yaml(yaml_content)
print(f"Loaded {len(idiom_map.get('idioms', []))} idioms")
```

---

## Workspace Isolation (Jail Mode)

Workspace isolation (also called **jail mode**) protects the original workspace by executing the pipeline against a **temporary copy** instead of the real directory. If the pipeline fails or produces bad output, the original workspace is untouched.

### How It Works

1. The workspace is copied to a system temp directory (via `shutil.copytree`)
2. The coding pipeline runs entirely inside the temp copy
3. On **success** — the temp directory's output is copied back to the original workspace
4. On **failure** — the temp directory is discarded; the original workspace is unchanged

This is particularly valuable when using `--pipeline` on production codebases where an interrupted run could leave partially-written files.

### Configuration

```yaml
# vaig.yaml
coding:
  workspace_isolation: true          # Enable jail mode

  # Patterns excluded when copying workspace to the temp jail
  # Uses shutil.ignore_patterns semantics
  jail_ignore_patterns:
    - ".git"
    - "node_modules"
    - "__pycache__"
    - "*.pyc"
    - ".venv"
    - "dist"
    - "build"
```

### When to Use It

| Situation | Use jail mode? |
|-----------|---------------|
| Exploratory refactoring of production code | ✅ Yes |
| Greenfield generation into an empty directory | ❌ Not needed |
| Long-running pipeline on a large codebase | ✅ Yes |
| Single-file edits with `confirm_actions: true` | ❌ Overkill |

```bash
# Enable jail mode for a risky refactor
VAIG_CODING__WORKSPACE_ISOLATION=true \
vaig ask "Migrate all async routes from aiohttp to FastAPI" \
    --code --pipeline -w /path/to/project
```

---

## Fix-Forward Loop

The fix-forward loop enables the **Implementer → Verifier** cycle to retry automatically when the verifier detects failures (syntax errors, incomplete placeholders, or test failures). Without it, a single verifier failure halts the pipeline permanently.

### How `max_fix_iterations` Works

```
Implementer writes files
    ↓
Verifier checks files
    ↓ PASS → done
    ↓ FAIL and iterations < max_fix_iterations
        → Implementer receives failure report, attempts fixes
        → Verifier re-checks
        → Repeat up to max_fix_iterations times
    ↓ FAIL and iterations == max_fix_iterations → pipeline reports error
```

The default (`max_fix_iterations: 1`) means no retry — the pipeline fails on first verifier rejection, matching the original behaviour. Set to `3` or higher to enable self-correction.

### Test Command Integration

When `test_command` is set, the verifier runs it after static checks:

```yaml
coding:
  test_command: "pytest -x --tb=short"   # Run pytest; fail fast on first error
  test_timeout: 120                       # Kill test run after 120 seconds
```

When `test_command` is empty, VAIG auto-detects pytest by looking for `pyproject.toml` or `conftest.py` in the workspace root.

### Configuration

```yaml
# vaig.yaml — Python project with self-correction
coding:
  pipeline_mode: true
  max_fix_iterations: 3              # Up to 3 Implementer→Verifier retries
  test_command: "pytest -x --tb=short src/tests/"
  test_timeout: 180                  # 3-minute timeout for the test suite
```

```yaml
# vaig.yaml — Go project
coding:
  pipeline_mode: true
  max_fix_iterations: 2
  test_command: "go test ./..."
  test_timeout: 120
  allowed_commands:
    - go
    - gofmt
```

```bash
# Override max_fix_iterations at runtime
VAIG_CODING__MAX_FIX_ITERATIONS=3 \
vaig ask "Implement the OAuth2 PKCE flow" \
    --code --pipeline -w /path/to/project
```

### Combining Fix-Forward with Git

```yaml
# vaig.yaml — full self-correcting pipeline with git
coding:
  pipeline_mode: true
  max_fix_iterations: 3
  test_command: "pytest -x --tb=short"
  test_timeout: 120
  git:
    enabled: true
    auto_branch: true
    auto_commit: true
    auto_pr: true
```

With this configuration, VAIG will:
1. Create a feature branch
2. Implement the requested changes
3. Run pytest — retry up to 3 times if tests fail
4. Commit the final passing state
5. Push the branch and open a PR

---

[Back to index](README.md)

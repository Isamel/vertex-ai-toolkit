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

> **Note:** Enum fields in the schema **must** use `StrEnum` (not `Enum`). Standard `Enum` values serialize as integers, which Gemini's `response_schema` rejects. Validate the raw JSON with `model_validate_json()`, not `model_validate()` — the model returns a JSON string, not a dict.

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
# Verbose mode — sets log level to DEBUG
vaig ask "test" --verbose

# Specific log level
vaig ask "test" --log-level DEBUG

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

[Back to index](README.md)

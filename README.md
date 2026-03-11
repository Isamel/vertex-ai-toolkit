# VAIG — Vertex AI Gemini Toolkit

Multi-agent AI assistant powered by **Google Vertex AI Gemini** models. Interactive CLI with pluggable skills for incident analysis, anomaly detection, and code migration.

## Features

- **Interactive REPL** — chat with Gemini models in your terminal with slash commands
- **Multi-model support** — switch between Gemini 2.5 Pro, Flash, and more on the fly
- **Multimodal context** — attach code, PDFs, images, audio, and Pentaho ETL files
- **Session persistence** — save and resume conversations (SQLite-backed)
- **Pluggable skills** — specialized multi-agent workflows:
  - **RCA** — Root Cause Analysis with 5 Whys + Fishbone methodology
  - **Anomaly Detection** — detect unusual patterns in logs, metrics, and data
  - **Code Migration** — migrate between platforms (e.g., Pentaho KTR/KJB → AWS Glue PySpark)
- **Multi-agent orchestration** — skills spawn specialized agents with different roles and models
- **Configurable auth** — Application Default Credentials (ADC) for GKE, service account impersonation for local dev

## Requirements

- Python 3.11+
- A Google Cloud project with Vertex AI API enabled
- Authentication configured (see [Authentication](#authentication))

## Installation

```bash
# From source
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Start interactive chat
vaig chat

# Ask a single question
vaig ask "What are the best practices for Kubernetes pod security?"

# Ask with file context
vaig ask "Analyze this code for issues" -f main.py -f utils.py

# Use a skill
vaig chat --skill rca
vaig ask "Investigate why API latency spiked" -s rca -f error.log

# Use a specific model
vaig chat --model gemini-2.5-flash
```

## CLI Commands

### `vaig chat`

Start an interactive REPL session.

```bash
vaig chat [OPTIONS]

Options:
  -c, --config PATH    Path to config YAML
  -m, --model TEXT     Model to use (overrides config)
  -s, --skill TEXT     Activate a skill
  --session TEXT       Resume an existing session by ID
  -n, --name TEXT      Name for new session (default: "chat")
```

### `vaig ask`

Ask a single question and get a response.

```bash
vaig ask QUESTION [OPTIONS]

Options:
  -c, --config PATH    Path to config YAML
  -m, --model TEXT     Model to use
  -f, --file PATH      Files to include as context (repeatable)
  -s, --skill TEXT     Use a specific skill
  --no-stream          Disable streaming output
```

### `vaig sessions list`

List saved chat sessions.

### `vaig sessions delete SESSION_ID`

Delete a saved session.

### `vaig models list`

List available Gemini models.

### `vaig skills list`

List available skills.

### `vaig skills info SKILL_NAME`

Show detailed info about a skill, including its agents.

## REPL Slash Commands

Inside the interactive chat (`vaig chat`):

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `/add <path>`        | Add a file or directory as context           |
| `/model <name>`      | Switch to a different model                  |
| `/skill <name>`      | Activate a skill                             |
| `/phase <phase>`     | Set the skill phase (analyze/plan/execute)   |
| `/agents`            | Show active agents                           |
| `/sessions`          | List saved sessions                          |
| `/new [name]`        | Start a new session                          |
| `/load <id>`         | Load a previous session                      |
| `/clear`             | Clear current context files                  |
| `/context`           | Show loaded context files                    |
| `/help`              | Show all commands                            |
| `/quit`              | Exit the REPL                                |

## Configuration

VAIG uses layered configuration: **environment variables > YAML config > defaults**.

### Config file

Default location: `config/default.yaml` or specify with `--config`.

```yaml
gcp:
  project_id: "my-project"
  location: "us-central1"

auth:
  mode: "adc"  # "adc" | "impersonate"
  impersonate_sa: "my-sa@my-project.iam.gserviceaccount.com"

models:
  default: "gemini-2.5-pro"
  fallback: "gemini-2.5-flash"

generation:
  temperature: 0.7
  max_output_tokens: 8192
  top_p: 0.95

session:
  db_path: "~/.vaig/sessions.db"
  auto_save: true

skills:
  enabled: [rca, anomaly, migration]
  custom_dir: null  # Path to custom skills directory
```

### Environment variables

All settings can be overridden with `VAIG_` prefixed env vars:

```bash
export VAIG_GCP_PROJECT_ID="my-project"
export VAIG_GCP_LOCATION="us-central1"
export VAIG_AUTH_MODE="impersonate"
export VAIG_AUTH_IMPERSONATE_SA="my-sa@my-project.iam.gserviceaccount.com"
export VAIG_MODELS_DEFAULT="gemini-2.5-flash"
```

## Authentication

### In GKE (recommended)

Use **Workload Identity** — the pod's service account authenticates automatically via ADC:

```yaml
# config/default.yaml
auth:
  mode: "adc"
```

### Local development

Use **service account impersonation** — your user account impersonates a service account:

```bash
# Authenticate with gcloud
gcloud auth application-default login

# Configure impersonation
export VAIG_AUTH_MODE="impersonate"
export VAIG_AUTH_IMPERSONATE_SA="vaig-sa@my-project.iam.gserviceaccount.com"
```

Required IAM roles on the service account:
- `roles/aiplatform.user` — Vertex AI API access
- Your user needs `roles/iam.serviceAccountTokenCreator` on the SA

## Skills Architecture

Skills follow a **phase-based execution** model:

```
ANALYZE → PLAN → EXECUTE → VALIDATE → REPORT
```

Each skill defines:
- **System instructions** — injected as the AI system prompt
- **Phase prompts** — templates for each execution phase
- **Agent configuration** — specialized agents with different roles and models

### Built-in Skills

#### RCA (Root Cause Analysis)

Agents: `log_analyzer`, `metric_correlator`, `rca_lead`

```bash
vaig chat --skill rca
# Then: paste logs, describe the incident, attach files
```

#### Anomaly Detection

Agents: `pattern_analyzer`, `anomaly_detector`

```bash
vaig ask "Find anomalies in this data" -s anomaly -f metrics.csv
```

#### Code Migration

Agents: `code_analyzer`, `code_generator`, `migration_validator`

```bash
vaig ask "Migrate this Pentaho job to AWS Glue" -s migration -f transform.ktr
```

### Custom Skills

Create your own skills by placing them in the custom skills directory:

```
~/.vaig/skills/
└── my-skill/
    ├── __init__.py
    ├── prompts.py
    └── skill.py      # Must contain a BaseSkill subclass
```

```python
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase, SkillResult


class MySkill(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="my-skill",
            display_name="My Custom Skill",
            description="Does something cool",
            tags=["custom"],
        )

    def get_system_instruction(self) -> str:
        return "You are a specialized assistant for..."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"Context:\n{context}\n\nTask: {user_input}"
```

Then enable it in config:

```yaml
skills:
  custom_dir: "~/.vaig/skills"
```

## Project Structure

```
vertex-ai-toolkit/
├── pyproject.toml
├── config/
│   └── default.yaml
├── src/vaig/
│   ├── __init__.py
│   ├── __main__.py
│   ├── core/
│   │   ├── config.py       # Pydantic Settings (layered config)
│   │   ├── auth.py         # ADC + SA impersonation
│   │   └── client.py       # GeminiClient (streaming, multi-model)
│   ├── context/
│   │   ├── filters.py      # .gitignore patterns, binary detection
│   │   ├── loader.py       # File loaders (text, PDF, image, audio, ETL)
│   │   └── builder.py      # ContextBuilder + ContextBundle
│   ├── session/
│   │   ├── store.py        # SQLite persistence
│   │   └── manager.py      # SessionManager + ActiveSession
│   ├── skills/
│   │   ├── base.py         # BaseSkill ABC, SkillPhase, SkillResult
│   │   ├── registry.py     # Discovery, loading, lazy initialization
│   │   ├── rca/            # Root Cause Analysis skill
│   │   ├── anomaly/        # Anomaly Detection skill
│   │   └── migration/      # Code Migration skill
│   ├── agents/
│   │   ├── base.py         # AgentRole, AgentConfig, BaseAgent ABC
│   │   ├── specialist.py   # SpecialistAgent (wraps GeminiClient)
│   │   ├── orchestrator.py # Multi-agent coordination
│   │   └── registry.py     # Agent factory
│   └── cli/
│       ├── app.py          # Typer commands
│       └── repl.py         # Interactive REPL (prompt-toolkit)
└── tests/
```

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/vaig/

# Test
pytest
```

## License

MIT

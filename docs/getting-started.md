# Getting Started

## Installation

### From source (recommended during development)

```bash
# Clone the repository
git clone https://github.com/Isamel/vertex-ai-toolkit.git
cd vertex-ai-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with live infrastructure support (GKE, Cloud Logging, Cloud Monitoring)
pip install -e ".[live]"

# Install with development dependencies (includes live deps + pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Dependencies

**Core dependencies:**
- `google-genai` — Google Generative AI SDK
- `typer` — CLI framework
- `rich` — Terminal formatting and spinners
- `prompt_toolkit` — Interactive REPL input
- `pyyaml` — Configuration parsing

**Live infrastructure (`[live]` extra):**
- `kubernetes` — GKE cluster access
- `google-cloud-logging` — Cloud Logging queries
- `google-cloud-monitoring` — Cloud Monitoring metrics
- `google-cloud-container` — GKE cluster API

**Dev (`[dev]` extra — includes all `[live]` deps):**
- `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-timeout` — Testing
- `ruff` — Linting and formatting
- `mypy` — Type checking

## Authentication

VAIG uses Google Cloud Application Default Credentials (ADC):

```bash
# Standard authentication
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### Service Account Impersonation

For environments where you need to impersonate a service account:

```yaml
# vaig.yaml
auth:
  mode: impersonate
  impersonate_sa: my-sa@my-project.iam.gserviceaccount.com
```

## Configuration

VAIG looks for configuration files in this order:

1. `--config` flag (highest priority)
2. `config/default.yaml` in the current directory
3. `vaig.yaml` in the current directory
4. `~/.vaig/config.yaml` (user default)

Create a minimal config:

```yaml
# vaig.yaml
gcp:
  project_id: my-gcp-project
  location: us-central1

models:
  default: gemini-2.5-pro
  fallback: gemini-2.5-flash
```

> **Note:** All config values can also be set via environment variables with the `VAIG_` prefix. See [Configuration](configuration.md) for details.

## Quick Start

### Ask a single question

```bash
# Simple question
vaig ask "What are the SOLID principles?"

# With a file for context
vaig ask "Review this code for security issues" -f src/auth.py

# With a specific skill
vaig ask "Review this Terraform plan" -f main.tf -s iac-review

# Auto-detect the best skill
vaig ask "Find vulnerabilities in this code" -f app.py --auto-skill
```

### Start an interactive chat

```bash
# Default chat session
vaig chat

# Chat with a specific model
vaig chat -m gemini-2.5-flash

# Chat with a skill loaded
vaig chat -s code-review

# Resume last session
vaig chat -r
```

### Investigate live infrastructure

```bash
# Investigate a service in GKE
vaig live "Why is the payment-service returning 503 errors?" \
  --cluster my-cluster \
  --namespace production

# With a skill
vaig live "Check the health of all services in staging" \
  -s service-health \
  --cluster staging-cluster
```

### Manage sessions

```bash
# List recent sessions
vaig sessions list

# Search sessions
vaig sessions search "kubernetes deployment"

# Show session details with messages
vaig sessions show SESSION_ID -m

# Export a session
vaig export SESSION_ID -f md -o report.md
```

### Work with skills

```bash
# List all available skills
vaig skills list

# Get details about a skill
vaig skills info rca

# Create a custom skill scaffold
vaig skills create my-custom-skill -d "My custom analysis skill"
```

## Next Steps

- [CLI Reference](cli-reference.md) — All commands and options
- [REPL Guide](repl-guide.md) — Master the interactive chat
- [Skills Guide](skills-guide.md) — Explore all 29 built-in skills
- [Configuration](configuration.md) — Fine-tune every setting

---

[Back to index](README.md)

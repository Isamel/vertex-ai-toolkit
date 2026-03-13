# Configuration Guide

VAIG is configured via YAML files and environment variables. This guide covers all 14 configuration sections.

## Config File Locations

VAIG searches for configuration in this order (first found wins):

1. `--config` / `-c` flag on the command line
2. `config/default.yaml` in the current directory
3. `vaig.yaml` in the current directory
4. `~/.vaig/config.yaml` (user default)

## Environment Variable Overrides

Any config value can be overridden via environment variables using the `VAIG_` prefix with `__` as the nesting delimiter:

```bash
export VAIG_GCP__PROJECT_ID=my-project
export VAIG_GCP__LOCATION=us-central1
export VAIG_MODELS__DEFAULT=gemini-2.5-flash
export VAIG_LOGGING__LEVEL=DEBUG
export VAIG_SESSION__AUTO_SAVE=false
export VAIG_CODING__MAX_TOOL_ITERATIONS=50
```

**Priority**: Environment variables > YAML config > defaults

## Full Configuration Reference

### `gcp` — Google Cloud Platform

```yaml
gcp:
  project_id: my-gcp-project        # GCP project ID (required)
  location: us-central1             # Vertex AI region (default: us-central1)
  fallback_location: null            # Fallback region if primary fails
```

### `auth` — Authentication

```yaml
auth:
  mode: adc                          # Auth mode: "adc" or "impersonate"
  impersonate_sa: null               # Service account email (for impersonate mode)
```

| Mode | Description |
|------|-------------|
| `adc` | Application Default Credentials (default) — uses `gcloud auth application-default login` |
| `impersonate` | Impersonates a service account — requires `impersonate_sa` to be set |

### `generation` — Model Generation Parameters

```yaml
generation:
  temperature: 0.7                   # Creativity (0.0 = deterministic, 1.0 = creative)
  max_output_tokens: 16384           # Maximum tokens in response
  top_p: 0.95                        # Nucleus sampling threshold
  top_k: 40                          # Top-k sampling
```

### `models` — Model Selection

```yaml
models:
  default: gemini-2.5-pro            # Primary model
  fallback: gemini-2.5-flash         # Fallback model (used on errors/retries)
  available:                         # List of available models
    - gemini-2.5-pro
    - gemini-2.5-flash
    - gemini-2.0-flash
```

### `session` — Session Persistence

```yaml
session:
  db_path: ~/.vaig/sessions.db      # SQLite database path
  auto_save: true                    # Auto-save messages to DB
  max_history_messages: 100          # Max messages kept in memory per session
```

### `skills` — Skills Configuration

```yaml
skills:
  enabled:                           # List of enabled skill names
    - rca
    - postmortem
    - error-triage
    - log-analysis
    - anomaly
    - code-review
    - test-generation
    - api-design
    - db-review
    - migration
    - perf-analysis
    - threat-model
    - compliance-check
    - dependency-audit
    - iac-review
    - config-audit
    - pipeline-review
    - change-risk
    - slo-review
    - alert-tuning
    - capacity-planning
    - cost-analysis
    - network-review
    - resilience-review
    - runbook-generator
    - incident-comms
    - adr-generator
    - toil-analysis
    - service-health
  custom_dir: null                   # Path to custom skills directory
```

### `agents` — Agent Configuration

```yaml
agents:
  max_concurrent: 3                  # Max parallel agents (for fan-out strategy)
  orchestrator_model: null           # Override model for orchestrator (uses default)
  specialist_model: null             # Override model for specialists (uses default)
```

### `context` — File Context Handling

```yaml
context:
  max_file_size_mb: 50               # Max file size to load as context
  supported_extensions:              # File extensions that can be loaded
    - .py
    - .js
    - .ts
    - .go
    - .rs
    - .java
    - .yaml
    - .yml
    - .json
    - .toml
    - .md
    - .txt
    - .sh
    - .sql
    - .tf
    - .hcl
    - .dockerfile
    # ... and more
  ignore_patterns:                   # Patterns to skip when listing/searching
    - __pycache__
    - .git
    - node_modules
    - .venv
    - "*.pyc"
```

### `retry` — API Retry Configuration

```yaml
retry:
  max_retries: 3                     # Maximum retry attempts
  initial_delay: 1.0                 # Initial delay in seconds
  max_delay: 60.0                    # Maximum delay between retries
  backoff_multiplier: 2.0            # Exponential backoff multiplier
  retryable_status_codes:            # HTTP status codes that trigger retry
    - 429                            # Too Many Requests
    - 500                            # Internal Server Error
    - 503                            # Service Unavailable
```

### `logging` — Logging Configuration

```yaml
logging:
  level: WARNING                     # Log level: DEBUG, INFO, WARNING, ERROR
  show_path: false                   # Show file path in log output
```

Override via CLI: `--verbose` sets `DEBUG`, `--log-level` sets any level.

### `coding` — Coding Agent Configuration

```yaml
coding:
  workspace_root: "."                # Root directory for file operations
  max_tool_iterations: 25            # Max tool-use loop iterations
  confirm_actions: true              # Require user confirmation for write ops
  allowed_commands:                  # Shell commands the agent can run
    - ls
    - cat
    - grep
    - find
    - wc
    - head
    - tail
    - sort
    - uniq
    - diff
    - python
    - pip
    - node
    - npm
    - git
    - make
    - cargo
    - go
  blocked_paths:                     # Paths that cannot be read/written
    - /etc
    - /var
    - /usr
```

### `chunking` — Large File Processing

```yaml
chunking:
  chunk_overlap_ratio: 0.1           # 10% overlap between chunks
  token_safety_margin: 0.1           # Reserve 10% of context for prompt
  chars_per_token: 2.0               # Estimated characters per token
  inter_chunk_delay: 2.0             # Seconds between chunk API calls
```

### `gke` — Google Kubernetes Engine

```yaml
gke:
  cluster_name: null                 # GKE cluster name
  project_id: null                   # GCP project (falls back to gcp.project_id)
  default_namespace: default         # Default K8s namespace
  kubeconfig_path: null              # Custom kubeconfig path
  context: null                      # Kubernetes context to use
  log_limit: 100                     # Default log tail lines
  metrics_interval_minutes: 60       # Default metrics time window
  proxy_url: null                    # Proxy URL for GKE API
```

### `mcp` — Model Context Protocol

```yaml
mcp:
  enabled: false                     # Enable MCP integration
  servers:                           # List of MCP servers
    - name: my-server
      command: npx
      args: ["-y", "@my-org/mcp-server"]
      env:
        API_KEY: "${MY_API_KEY}"
      description: "My custom MCP server"
```

See [MCP Guide](mcp-guide.md) for detailed MCP configuration.

## Complete Example

```yaml
# vaig.yaml — Full configuration example

gcp:
  project_id: my-production-project
  location: us-central1

auth:
  mode: adc

generation:
  temperature: 0.3                   # Lower for more deterministic output
  max_output_tokens: 32768

models:
  default: gemini-2.5-pro
  fallback: gemini-2.5-flash

session:
  db_path: ~/.vaig/sessions.db
  auto_save: true
  max_history_messages: 50

skills:
  enabled:
    - rca
    - code-review
    - service-health
    - threat-model
  custom_dir: ./my-skills

agents:
  max_concurrent: 5

coding:
  workspace_root: "."
  max_tool_iterations: 30
  confirm_actions: true
  allowed_commands:
    - python
    - pip
    - git
    - make
    - pytest

gke:
  cluster_name: production-cluster
  default_namespace: api
  log_limit: 200
  metrics_interval_minutes: 30

logging:
  level: INFO

retry:
  max_retries: 5
  initial_delay: 2.0
```

---

[Back to index](README.md)

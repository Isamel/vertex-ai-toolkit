# CLI Reference

Complete reference for all `vaig` CLI commands and options.

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--verbose` | `-V` | Enable verbose output |
| `--log-level` | | Set log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Top-Level Commands

### `vaig web`

Start the VAIG web server. Launches a FastAPI web interface with the same toolkit features available in the CLI (ask, chat, live modes) plus a browser-based UI with SSE streaming and dark/light theme toggle.

Requires web extras: `pip install vertex-ai-toolkit[web]`

```bash
vaig web [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Bind address | `0.0.0.0` |
| `--port` | `-p` | Bind port (also via `PORT` env var) | `8080` |
| `--reload` | | Enable auto-reload for development | `false` |

**Examples:**

```bash
# Start with defaults (0.0.0.0:8080)
vaig web

# Custom port
vaig web --port 9090

# Development mode with auto-reload
vaig web --reload
```

### `vaig ask`

Send a single question to the AI and get a response. Supports file context, skills, and export.

```bash
vaig ask "Your question here" [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |
| `--model` | `-m` | Model to use | `gemini-2.5-pro` |
| `--file` | `-f` | File(s) to include as context (repeatable) | — |
| `--output` | `-o` | Output file path | stdout |
| `--format` | | Export format: `json`, `md`, `html` | — |
| `--skill` | `-s` | Skill to use | — |
| `--auto-skill` | | Auto-detect best skill for the query | `false` |
| `--no-stream` | | Disable streaming output | `false` |
| `--code` | | Enable coding agent mode (file read/write/edit) | `false` |
| `--pipeline` | | Enable 3-agent pipeline mode (Planner→Implementer→Verifier). Requires `--code`. Maps to `coding.pipeline_mode` config key | `false` |
| `--live` | | Enable live infrastructure tools | `false` |
| `--workspace` | `-w` | Workspace root for coding agent | `.` |
| `--cluster` | | GKE cluster name | Config value |
| `--namespace` | | Kubernetes namespace | Config value |
| `--project-id` | | GCP project ID override | Config value |

**Examples:**

```bash
# Simple question
vaig ask "Explain the CAP theorem"

# Code review with file context
vaig ask "Review this for performance issues" -f slow_query.py -s perf-analysis

# Multiple files
vaig ask "Compare these implementations" -f v1/handler.py -f v2/handler.py

# Export to markdown
vaig ask "Analyze this config" -f nginx.conf --format md -o report.md

# Auto-skill detection
vaig ask "Is this Terraform plan safe to apply?" -f plan.tf --auto-skill

# Coding agent mode
vaig ask "Refactor this function to use async/await" -f server.py --code

# Pipeline mode — 3-agent Planner→Implementer→Verifier (requires --code)
vaig ask "Implement a retry decorator with exponential backoff" --code --pipeline

# Live infrastructure query
vaig ask "What's the CPU usage of the API pods?" --live --cluster prod
```

### `vaig chat`

Start an interactive REPL session. See [REPL Guide](repl-guide.md) for slash commands.

```bash
vaig chat [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |
| `--model` | `-m` | Model to use | `gemini-2.5-pro` |
| `--skill` | `-s` | Skill to preload | — |
| `--session` | | Session ID to load | — |
| `--resume` | `-r` | Resume last session | `false` |
| `--name` | `-n` | Name for the new session | Auto-generated |
| `--workspace` | `-w` | Workspace root for coding agent | `.` |

**Examples:**

```bash
# Start a new chat
vaig chat

# Chat with a skill
vaig chat -s code-review

# Resume last session
vaig chat -r

# Load a specific session
vaig chat --session abc123-def456

# Name your session
vaig chat -n "kubernetes-debugging"

# Use a specific model
vaig chat -m gemini-2.5-flash
```

### `vaig live`

Interactive infrastructure investigation mode with live GKE and GCloud tools.

```bash
vaig live "Your investigation query" [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |
| `--model` | `-m` | Model to use | `gemini-2.5-pro` |
| `--output` | `-o` | Output file path | stdout |
| `--format` | | Export format: `json`, `md`, `html` | — |
| `--skill` | `-s` | Skill to use | — |
| `--auto-skill` | | Auto-detect best skill | `false` |
| `--cluster` | | GKE cluster name | Config value |
| `--namespace` | | Kubernetes namespace | Config value |
| `--project-id` | | GCP project ID override | Config value |

**Examples:**

```bash
# Investigate a service
vaig live "Why is payment-service slow?" --cluster prod --namespace default

# Health check with a skill
vaig live "Full health check of the staging environment" \
  -s service-health --cluster staging

# Root cause analysis
vaig live "Investigate the spike in 5xx errors at 14:30 UTC" \
  -s rca --cluster prod --namespace api

# Export investigation results
vaig live "Check all pod resource utilization" \
  --cluster prod -o capacity-report.md --format md
```

### `vaig discover`

Autonomously scan a Kubernetes cluster and discover workload health issues. Unlike `vaig live` which takes a specific question, `vaig discover` auto-generates its investigation query and runs a 4-agent pipeline: Inventory Scanner → Triage Classifier → Deep Investigator → Cluster Reporter.

```bash
vaig discover [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--namespace` | `-n` | Kubernetes namespace to scan | Config default or `default` |
| `--all-namespaces` | `-A` | Scan all non-system namespaces | `false` |
| `--skip-healthy` | | Omit healthy workloads from the report — focus on issues only | `false` |
| `--config` | `-c` | Path to config file | Auto-detected |
| `--model` | `-m` | Model to use | `gemini-2.5-pro` |
| `--output` | `-o` | Save report to a file | stdout |
| `--format` | | Export format: `json`, `md`, `html` | — |
| `--cluster` | | GKE cluster name | Config value |
| `--project` | `-p` | GCP project ID override | Config value |
| `--location` | | GCP location override | Config value |
| `--gke-project` | | GKE project ID (overrides `gke.project_id`; defaults to `--project` if unset) | Config value |
| `--gke-location` | | GKE cluster location (overrides `gke.location`) | Config value |
| `--summary` | | Show compact summary instead of full report | `false` |
| `--detailed` | | Show every tool call as it happens (verbose execution output) | `false` |
| `--no-bell` | | Suppress terminal bell after pipeline completes | `false` |
| `--open` | `-O` | Open HTML report in default browser (requires `--format html`) | `false` |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Examples:**

```bash
# Scan a specific namespace
vaig discover --namespace production

# Scan all non-system namespaces
vaig discover --all-namespaces

# Focus on issues, skip healthy workloads
vaig discover --namespace staging --skip-healthy

# Export as HTML and open in browser
vaig discover --namespace production --format html --open

# Export to markdown file
vaig discover -A -o report.md

# Target a specific cluster and project
vaig discover --namespace default --cluster prod-cluster --gke-project my-gke-project
```

### `vaig doctor`

Run diagnostic checks on your VAIG environment. Verifies GCP authentication, Vertex AI API access, Kubernetes connectivity, observability integrations, and optional dependencies.

The command runs 10 sequential checks:
1. **GCP Authentication** — validates Application Default Credentials
2. **Vertex AI API** — verifies model accessibility via a lightweight `count_tokens` call
3. **GKE Connectivity** — tests Kubernetes cluster connection and detects Autopilot/Standard mode
4. **Cloud Logging** — checks Cloud Logging API availability
5. **Cloud Monitoring** — checks Cloud Monitoring API availability
6. **Helm Integration** (optional) — verifies Helm is enabled and binary is available
7. **ArgoCD Integration** (optional) — checks ArgoCD configuration
8. **Datadog Integration** (optional) — validates Datadog API key configuration
9. **Optional Dependencies** — checks importability of kubernetes, google-cloud-logging, google-cloud-monitoring, datadog-api-client
10. **MCP Servers** (optional) — verifies MCP server configuration

```bash
vaig doctor [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |
| `--project` | `-p` | GCP project ID override | Config value |
| `--cluster` | | GKE cluster name override | Config value |
| `--location` | | GCP location override | Config value |
| `--gke-project` | | GKE project ID (overrides `gke.project_id`) | Config value |
| `--gke-location` | | GKE cluster location (overrides `gke.location`) | Config value |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Examples:**

```bash
# Run all checks with default config
vaig doctor

# Target a specific project
vaig doctor --project my-project

# Use a custom config file
vaig doctor --config ~/custom-config.yaml

# Override cluster for GKE connectivity check
vaig doctor --cluster prod-cluster --gke-project my-gke-project
```

Exit codes: `0` if all critical checks pass, `1` if any critical check (GCP Auth or Vertex AI API) fails.

### `vaig export`

Export a past session to a file.

```bash
vaig export SESSION_ID [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--format` | `-f` | Export format: `json`, `md`, `html` | `md` |
| `--output` | `-o` | Output file path | stdout |
| `--config` | `-c` | Path to config file | Auto-detected |

**Examples:**

```bash
# Export to markdown
vaig export abc123 -f md -o session-report.md

# Export to JSON
vaig export abc123 -f json -o session-data.json

# Export to HTML
vaig export abc123 -f html -o report.html

# Print to stdout
vaig export abc123
```

### `vaig feedback`

Submit quality feedback for a completed `vaig live` analysis run. Rate the analysis from 1 (poor) to 5 (excellent) and optionally include a text comment. Feedback is exported to the configured BigQuery `feedback` table for quality tracking.

Requires either `--run-id` or `--last` to identify which run the feedback belongs to. These flags are mutually exclusive.

```bash
vaig feedback --rating RATING [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--rating` | `-r` | Rating from 1 (poor) to 5 (excellent) | **Required** |
| `--run-id` | | Run ID to attach feedback to | — |
| `--last` | | Use the most recent run ID | `false` |
| `--comment` | `-m` | Free-text feedback comment | `""` |
| `--config` | `-c` | Path to config file | Auto-detected |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Examples:**

```bash
# Rate the last run
vaig feedback --rating 5 --last

# Rate with a comment
vaig feedback -r 4 -m "Great analysis" --last

# Rate a specific run by ID
vaig feedback -r 3 --run-id 20250601T120000Z
```

> **Note:** Export must be enabled in your configuration (`export.enabled=true`) for feedback to be saved.

### `vaig optimize`

Analyze tool call efficiency and suggest optimizations. Scans recent run history for per-tool statistics, redundant calls, and performance patterns, then prints actionable suggestions.

Use `--reports` to switch to **report quality analysis**, which computes hallucination rate, evidence depth, actionability, and other quality signals from past HealthReports.

```bash
vaig optimize [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--last` | `-n` | Number of recent runs to analyze | `50` |
| `--reports` | | Analyze report quality instead of tool calls | `false` |
| `--config` | `-c` | Path to config file | Auto-detected |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Default mode (tool call analysis)** displays:
- Summary — total runs, total calls, total duration, average calls per run
- Per-tool statistics — call count, failures, fail %, avg/max duration, cache hits, unique arg combos
- Redundant calls — same tool called multiple times with identical arguments in a single run
- Suggestions — actionable optimization recommendations

**Report mode** (`--reports`) displays:
- Quality signals — hallucination rate, evidence depth, actionability, completeness
- Each signal's value, threshold, and pass/fail status
- Prompt improvement suggestions

**Examples:**

```bash
# Analyze tool call efficiency (default — last 50 runs)
vaig optimize

# Analyze fewer runs
vaig optimize --last 20

# Analyze report quality
vaig optimize --reports

# Analyze report quality for recent runs
vaig optimize --reports --last 10
```

## Sub-Command Groups

### `vaig sessions`

Manage chat sessions.

#### `vaig sessions list`

```bash
vaig sessions list [--limit/-n LIMIT]
```

Lists recent sessions sorted by last update time. Default limit is 20.

```bash
vaig sessions list
vaig sessions list -n 50
```

#### `vaig sessions show`

```bash
vaig sessions show SESSION_ID [--messages/-m]
```

Show details of a specific session. Add `-m` to include message history.

```bash
vaig sessions show abc123
vaig sessions show abc123 -m
```

#### `vaig sessions search`

```bash
vaig sessions search QUERY
```

Search sessions by name or message content.

```bash
vaig sessions search "kubernetes"
vaig sessions search "deployment rollback"
```

#### `vaig sessions rename`

```bash
vaig sessions rename SESSION_ID NEW_NAME
```

Rename an existing session.

```bash
vaig sessions rename abc123 "prod-incident-2024-03-15"
```

#### `vaig sessions delete`

```bash
vaig sessions delete SESSION_ID [--force/-f]
```

Delete a session and all its messages. Use `--force` to skip confirmation.

```bash
vaig sessions delete abc123
vaig sessions delete abc123 -f
```

### `vaig models`

#### `vaig models list`

```bash
vaig models list
```

List all available Gemini models configured in your project.

### `vaig skills`

Manage analysis skills.

#### `vaig skills list`

```bash
vaig skills list
```

List all available skills with their names, descriptions, and tags.

#### `vaig skills info`

```bash
vaig skills info SKILL_NAME
```

Show detailed information about a skill including phases, tags, recommended model, and agent configuration.

```bash
vaig skills info rca
vaig skills info service-health
```

#### `vaig skills create`

```bash
vaig skills create SKILL_NAME [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--description` | `-d` | Skill description |
| `--tags` | `-t` | Comma-separated tags |
| `--output` | `-o` | Output directory | `~/.vaig/skills/` |

Scaffold a new custom skill with boilerplate code.

```bash
vaig skills create my-audit -d "Custom security audit" -t "security,audit"
vaig skills create etl-review -d "ETL pipeline review" -o ./skills/
```

### `vaig stats`

View and manage usage telemetry data. See [Telemetry Guide](telemetry-guide.md) for full details.

#### `vaig stats show`

```bash
vaig stats show [--since DATE] [--until DATE]
```

Display a usage summary with event counts, token totals, and cost estimates.

```bash
vaig stats show
vaig stats show --since 2024-01-01 --until 2024-12-31
```

#### `vaig stats export`

```bash
vaig stats export [OPTIONS]
```

Export raw telemetry events for external analysis.

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--format` | `-f` | Export format: `jsonl`, `csv` | `jsonl` |
| `--output` | `-o` | Output file path | stdout |
| `--type` | `-t` | Filter by event type | All types |
| `--since` | | Start date (ISO 8601) | — |
| `--until` | | End date (ISO 8601) | — |

```bash
vaig stats export
vaig stats export --format csv --output usage.csv
vaig stats export --type tool_call --since 2024-01-01
```

#### `vaig stats clear`

```bash
vaig stats clear --days N --confirm
```

Remove telemetry events older than N days. Requires `--confirm` flag.

```bash
vaig stats clear --days 30 --confirm
```

### `vaig mcp`

Manage MCP (Model Context Protocol) servers.

#### `vaig mcp list-servers`

```bash
vaig mcp list-servers
```

List all configured MCP servers and their status.

#### `vaig mcp discover`

```bash
vaig mcp discover
```

Discover tools available from all configured MCP servers.

#### `vaig mcp call`

```bash
vaig mcp call
```

Interactively call an MCP tool for testing.

### `vaig login`

Authenticate with the platform backend using the OAuth PKCE flow. Opens a browser window for Google OAuth consent and stores tokens locally at `~/.vaig/credentials.json`.

Requires `platform.enabled: true` in your configuration.

```bash
vaig login [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |
| `--force` | | Re-authenticate even if already logged in | `false` |

**Examples:**

```bash
# Login to platform
vaig login

# Force re-authentication
vaig login --force
```

### `vaig logout`

Log out from the platform and clear local credentials.

Requires `platform.enabled: true` in your configuration.

```bash
vaig logout [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |

**Examples:**

```bash
vaig logout
```

### `vaig whoami`

Show the currently authenticated platform user, including email, organization, role, and CLI ID.

Requires `platform.enabled: true` in your configuration.

```bash
vaig whoami [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |

**Examples:**

```bash
vaig whoami
```

### `vaig status`

Show platform registration status. Displays authentication state, organization, role, CLI ID, and whether a config policy is enforced.

Requires `platform.enabled: true` in your configuration.

```bash
vaig status [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |

**Examples:**

```bash
vaig status
```

## Environment Variables

All configuration can be overridden via environment variables using the `VAIG_` prefix with `__` as the nesting delimiter:

```bash
# Override GCP project
export VAIG_GCP__PROJECT_ID=my-project

# Override default model
export VAIG_MODELS__DEFAULT=gemini-2.5-flash

# Override log level
export VAIG_LOGGING__LEVEL=DEBUG

# Override GKE cluster
export VAIG_GKE__CLUSTER_NAME=my-cluster
```

---

[Back to index](README.md)

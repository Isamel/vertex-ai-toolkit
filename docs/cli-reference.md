# CLI Reference

Complete reference for all `vaig` CLI commands and options.

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--verbose` | `-V` | Enable verbose output |
| `--log-level` | | Set log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Top-Level Commands

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

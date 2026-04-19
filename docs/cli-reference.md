# CLI Reference

Complete reference for all `vaig` CLI commands and options.

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--verbose` | `-V` | Enable verbose logging (INFO level) |
| `--debug` | `-d` | Enable debug logging (DEBUG level). **Note:** `-d` is context-sensitive — it maps to `--dir` in `vaig ask` and `--deployment` in `vaig compare run`. Always check the command's option table. |
| `--log-level TEXT` | | Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Takes precedence over `--verbose`/`--debug`. |

---

## Top-Level Commands

### `vaig ask`

Send a single question to the AI and get a response. Supports file context, directory context, skills, code-migration, and export.

```bash
vaig ask "Your question here" [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |
| `--model` | `-m` | Model to use | `gemini-2.5-pro` |
| `--file` | `-f` | File(s) to include as context (repeatable) | — |
| `--dir` | `-d` | Directory(ies) as context — recursively included (repeatable) | — |
| `--examples` | `-e` | Reference/example files for code migration (repeatable) | — |
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
| `--project` | `-p` | GCP project ID (alias for `--project-id`) | Config value |
| `--location` | | GCP location override | Config value |
| `--gke-project` | | GKE project ID (overrides `gke.project_id`) | Config value |
| `--gke-location` | | GKE cluster location (overrides `gke.location`) | Config value |
| `--phases` | | Comma-separated skill phases to run (e.g. `plan,implement`) | All phases |
| `--resume` | | Resume a code-migration run from saved state | `false` |
| `--from-repo` | | Shallow-clone source repo as context: `owner/repo[@ref]` | — |
| `--to-repo` | | Target repo for code migration (stub) | — |
| `--push` / `--no-push` | | Push changes after migration (stub) | `false` |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | | Enable debug logging (DEBUG level) | `false` |

**Examples:**

```bash
# Simple question
vaig ask "Explain the CAP theorem"

# Code review with file context
vaig ask "Review this for performance issues" -f slow_query.py -s perf-analysis

# Multiple files
vaig ask "Compare these implementations" -f v1/handler.py -f v2/handler.py

# Directory context (recursive)
vaig ask "Find any SQL injection risks" --dir src/

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

# Code migration from a remote repo
vaig ask "Migrate this service from Flask to FastAPI" \
  --from-repo my-org/legacy-service --phases plan,implement

# Resume an interrupted migration
vaig ask "Continue the migration" --resume
```

---

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

---

### `vaig live`

Interactive infrastructure investigation mode with live GKE and GCloud tools. Runs a multi-agent pipeline against your cluster and produces a structured health report. After the report, use `--interactive` to open a drill-in REPL to ask follow-up questions with full cluster context.

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
| `--project` | `-p` | GCP project ID (alias for `--project-id`) | Config value |
| `--location` | | GCP location override | Config value |
| `--gke-project` | | GKE project ID (overrides `gke.project_id`) | Config value |
| `--gke-location` | | GKE cluster location (overrides `gke.location`) | Config value |
| `--all-namespaces` | | Scan all non-system namespaces | `false` |
| `--interactive` | `-i` | Open drill-in REPL after the report to ask follow-up questions | `false` |
| `--watch` | `-w` | Re-execute every N seconds (minimum 10) — live polling mode | — |
| `--dry-run` / `--dry` | | Show execution plan without running | `false` |
| `--summary` | | Show compact summary instead of full report | `false` |
| `--detailed` | | Show every tool call as it happens | `false` |
| `--no-bell` | | Suppress terminal bell after pipeline completes | `false` |
| `--open` | `-O` | Open HTML report in default browser (requires `--format html`) | `false` |
| `--repo` | | GitHub repo for correlation: `owner/repo` | — |
| `--repo-ref` | | Git ref for repo correlation (branch, tag, SHA) | — |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Examples:**

```bash
# Basic investigation
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

# Open HTML report in browser
vaig live "Cluster health summary" --format html --open

# Drill-in REPL after the report
vaig live "What pods are in CrashLoopBackOff?" --cluster prod --interactive

# Preview execution plan without running
vaig live "Investigate memory pressure" --cluster prod --dry-run

# Live polling every 60 seconds
vaig live "Are there any OOMKilled pods?" --cluster prod --watch 60

# Correlate findings with a GitHub repo
vaig live "Did recent deploys cause errors?" \
  --cluster prod --repo my-org/my-service --repo-ref main

# Compact summary
vaig live "Cluster overview" --cluster prod --summary

# Scan all namespaces
vaig live "Any issues across all namespaces?" --all-namespaces --cluster prod
```

> **Tip:** Combine `--interactive` with `--skill rca` to investigate root causes in depth after the initial report.

> **Note on short flags with multiple meanings:** Several short flags differ by command. `-w`: on `vaig live` maps to `--watch INTEGER`; on `vaig ask`/`vaig chat` maps to `--workspace PATH`. `-d`: globally maps to `--debug`, but maps to `--dir` in `vaig ask` and `--deployment` in `vaig compare run`. `-c`: globally maps to `--config`, but maps to `--cluster` in `vaig schedule add` and `vaig fleet` commands. Always check the command's option table to avoid confusion.

---

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

---

### `vaig check`

Terraform-compatible health check for a Kubernetes namespace. Returns a structured JSON result and exits with a machine-readable code — designed for use in CI/CD pipelines and infrastructure-as-code health gates.

**Exit codes:**
- `0` — healthy
- `1` — unhealthy (issues found)
- `2` — error or timeout

```bash
vaig check [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--namespace` | `-n` | Kubernetes namespace to check | Config default |
| `--cluster` | | GKE cluster name | Config value |
| `--project` | `-p` | GCP project ID override | Config value |
| `--location` | `-l` | GCP location override | Config value |
| `--timeout` | | Maximum seconds to wait for analysis | Config value |
| `--cached` | | Return cached result if available (skip live analysis) | `false` |
| `--cache-ttl` | | Cache TTL in seconds (how long a cached result is valid) | Config value |
| `--config` | `-c` | Path to config file | Auto-detected |
| `--model` | `-m` | Model to use | Config value |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Output (stdout, always JSON):**

```json
{
  "status": "healthy",
  "namespace": "production",
  "cluster": "prod-cluster",
  "findings": [],
  "run_id": "20250601T120000Z"
}
```

**Examples:**

```bash
# Basic health check (CI/CD gate)
vaig check --namespace production

# Use cached result for fast checks
vaig check --namespace production --cached

# Override cluster and project
vaig check -n staging --cluster staging-cluster -p my-project

# With a timeout
vaig check --namespace production --timeout 120

# Use in shell scripts
if vaig check --namespace production; then
  echo "Cluster is healthy"
else
  echo "Issues detected — check output"
fi
```

---

### `vaig remediate`

Execute recommended actions from the last health report. Reads recommendations from the most recent `vaig discover` / `vaig live` report and applies them with safety tier enforcement.

**Safety tiers:**
- `SAFE` (green) — auto-executable with `--approve`
- `REVIEW` (yellow) — requires `--execute` after showing the plan
- `BLOCKED` (red) — never executed; shows reason and alternatives

Requires `remediation.enabled: true` in your configuration.

```bash
vaig remediate [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--list` | | List all recommended actions from the last health report | `false` |
| `--finding` | `-f` | Finding ID (or index, or partial title) to remediate | — |
| `--approve` | | Auto-approve and execute SAFE tier commands | `false` |
| `--dry-run` | | Show what would happen without executing | `false` |
| `--execute` | | Approve and execute REVIEW tier commands after showing plan | `false` |
| `--config` | `-c` | Path to config file | Auto-detected |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |
| `--debug` | `-d` | Enable debug logging (DEBUG level) | `false` |

**Examples:**

```bash
# See all recommended actions from last report
vaig remediate --list

# Preview what a finding would do (safe, no changes)
vaig remediate --finding crashloop-payment-svc --dry-run

# Execute a SAFE action
vaig remediate --finding crashloop-payment-svc --approve

# Execute a REVIEW action after reviewing the plan
vaig remediate --finding high-memory-usage --execute

# Reference by index from --list output
vaig remediate --finding 3 --approve

# Reference by partial title
vaig remediate --finding "memory pressure" --dry-run
```

> **Note:** `vaig remediate` always reads from the _most recent_ local health report. Run `vaig discover` or `vaig live` first to generate a fresh report.

---

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

---

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

---

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

---

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

---

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

---

### `vaig webhook-server`

Start the VAIG webhook server for Datadog alerts. Launches a Uvicorn/FastAPI server that receives Datadog alert webhooks (`POST /webhook/datadog`), triggers `vaig` health analyses on affected services, and dispatches results to PagerDuty and Google Chat.

Requires web extras: `pip install vertex-ai-toolkit[web]`

```bash
vaig webhook-server [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Bind address | `0.0.0.0` |
| `--port` | `-p` | Bind port (also via `PORT` env var) | `8080` |
| `--hmac-secret` | | Datadog webhook HMAC secret for signature verification (also via `VAIG_WEBHOOK_SERVER__HMAC_SECRET`) | `""` |
| `--max-analyses` | | Maximum analyses per UTC day (cost protection) | `50` |
| `--dedup-cooldown` | | Seconds before re-analyzing the same alert | `300` |
| `--reload` | | Enable auto-reload for development | `false` |

**Examples:**

```bash
# Start with defaults
vaig webhook-server

# Custom port
vaig webhook-server --port 9090

# Enable HMAC signature verification
vaig webhook-server --hmac-secret my-secret

# Raise analysis budget and dedup cooldown
vaig webhook-server --max-analyses 100 --dedup-cooldown 600

# Development mode
vaig webhook-server --reload
```

---

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

---

### `vaig logout`

Log out from the platform and clear local credentials.

Requires `platform.enabled: true` in your configuration.

```bash
vaig logout [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |

```bash
vaig logout
```

---

### `vaig whoami`

Show the currently authenticated platform user, including email, organization, role, and CLI ID.

Requires `platform.enabled: true` in your configuration.

```bash
vaig whoami [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |

```bash
vaig whoami
```

---

### `vaig status`

Show platform registration status. Displays authentication state, organization, role, CLI ID, and whether a config policy is enforced.

Requires `platform.enabled: true` in your configuration.

```bash
vaig status [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to config file | Auto-detected |

```bash
vaig status
```

---

## Sub-Command Groups

### `vaig schedule`

Manage scheduled health checks. Schedule recurring `vaig discover` runs using an interval or a cron expression, and control the scheduler daemon.

#### `vaig schedule add`

Register a new scheduled scan.

```bash
vaig schedule add [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cluster` | `-c` | GKE cluster name (⚠️ conflicts with global `-c` for `--config`) | Config value |
| `--cron` | | Cron expression (e.g. `"0 */6 * * *"`) | — |
| `--all-namespaces` | `-A` | Scan all non-system namespaces | `false` |
| `--skip-healthy` / `--include-healthy` | | Omit healthy workloads from reports | `false` |

Either `--interval` or `--cron` is required.

```bash
# Every 30 minutes on the default namespace
vaig schedule add --interval 30 --namespace production

# Every 6 hours via cron
vaig schedule add --cron "0 */6 * * *" --namespace staging

# All namespaces, skip healthy
vaig schedule add --interval 60 --all-namespaces --skip-healthy
```

#### `vaig schedule list`

```bash
vaig schedule list
```

List all registered scheduled scans with their IDs, clusters, namespaces, schedule, and last-run time.

#### `vaig schedule remove`

```bash
vaig schedule remove SCHEDULE_ID
```

Remove a scheduled scan by ID.

#### `vaig schedule run-now`

```bash
vaig schedule run-now SCHEDULE_ID
```

Trigger an immediate run of a scheduled scan, regardless of its next scheduled time.

#### `vaig schedule start`

Start the scheduler daemon. Must be running for scheduled scans to fire.

```bash
vaig schedule start [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Bind address for scheduler API | `0.0.0.0` |
| `--port` | `-p` | Port for scheduler API | `8081` |

```bash
# Start with defaults
vaig schedule start

# Custom port
vaig schedule start --port 9000
```

#### `vaig schedule stop`

```bash
vaig schedule stop
```

Stop the running scheduler daemon.

#### `vaig schedule status`

```bash
vaig schedule status
```

Show scheduler daemon status and upcoming run times.

---

### `vaig compare`

Cross-cluster comparison tools.

#### `vaig compare run`

Compare deployment configurations across multiple clusters side-by-side.

```bash
vaig compare run [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--clusters` | | Comma-separated list of cluster names to compare | **Required** |
| `--namespace` | `-n` | Kubernetes namespace | Config default |
| `--deployment` | | Deployment name to compare | — |
| `--export` | | Output format for comparison report: `json`, `md`, `html` | — |
| `--verbose` | `-V` | Enable verbose output | `false` |

```bash
# Compare two clusters
vaig compare run --clusters prod-us,prod-eu --namespace default

# Compare a specific deployment
vaig compare run --clusters prod,staging -n payments -d payment-api

# Export the comparison
vaig compare run --clusters prod,staging --export json
```

---

### `vaig fleet`

Multi-cluster fleet management.

#### `vaig fleet discover`

Scan multiple clusters in parallel and produce a consolidated fleet health report.

```bash
vaig fleet discover [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--parallel` | | Run cluster scans in parallel | `false` |
| `--max-workers` | | Maximum parallel workers | Config value |
| `--budget` | | Maximum total cost budget (USD) | — |
| `--detailed` | | Show every tool call as it happens | `false` |
| `--export` | | Output format for fleet report: `json`, `md`, `html` | — |
| `--namespace` | `-n` | Namespace to scan on each cluster | Config default |
| `--all-namespaces` | `-A` | Scan all non-system namespaces on each cluster | `false` |
| `--skip-healthy` / `--no-skip-healthy` | | Omit healthy workloads from the report | `false` |
| `--verbose` | `-V` | Enable verbose logging (INFO level) | `false` |

```bash
# Scan all configured clusters sequentially
vaig fleet discover

# Parallel scan with a cost budget
vaig fleet discover --parallel --max-workers 4 --budget 5.00

# Focus on issues only
vaig fleet discover --parallel --skip-healthy

# Export consolidated report
vaig fleet discover --parallel --export json
```

---

### `vaig incident`

Incident management — export findings and review recent incidents.

#### `vaig incident export`

Export a finding to an external incident management system (Jira or PagerDuty).

```bash
vaig incident export [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--to` | | Target system: `jira` or `pagerduty` | **Required** |
| `--finding` | | Finding ID to export | **Required** |
| `--cluster` | | Cluster the finding belongs to | Config value |

```bash
# Export to Jira
vaig incident export --to jira --finding crashloop-payment-svc

# Export to PagerDuty
vaig incident export --to pagerduty --finding high-memory-usage --cluster prod
```

#### `vaig incident list`

List recent findings across all clusters.

```bash
vaig incident list [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--last` | | Number of recent findings to show | Config default |

```bash
# List recent findings
vaig incident list

# Show last 20
vaig incident list --last 20
```

---

### `vaig train`

Fine-tuning pipeline — prepare training data and submit Vertex AI tuning jobs.

#### `vaig train prepare`

Extract rated examples from BigQuery (via `vaig feedback`) and generate a training JSONL file for Vertex AI fine-tuning.

Requires `training.enabled: true` in your configuration.

```bash
vaig train prepare [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--min-rating` | | Minimum feedback rating to include (1–5) | Config value |
| `--output` | `-o` | Output JSONL file path | stdout |
| `--dry-run` | | Report statistics without writing a file | `false` |
| `--max-examples` | | Maximum number of examples to include | Config value |

```bash
# Prepare training data (all rated examples)
vaig train prepare -o training-data.jsonl

# Only include high-quality examples (rating ≥ 4)
vaig train prepare --min-rating 4 -o training-data.jsonl

# Preview statistics without writing
vaig train prepare --dry-run

# Limit to 1000 examples
vaig train prepare --max-examples 1000 -o training-data.jsonl
```

#### `vaig train submit`

Upload training JSONL to GCS and submit a Vertex AI supervised fine-tuning job.

Requires `training.enabled: true` in your configuration.

```bash
vaig train submit [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Path to JSONL training file | **Required** |
| `--dry-run` | | Validate and report without submitting | `false` |

```bash
# Submit a tuning job
vaig train submit --input training-data.jsonl

# Validate without submitting
vaig train submit --input training-data.jsonl --dry-run
```

---

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

---

### `vaig models`

#### `vaig models list`

```bash
vaig models list
```

List all available Gemini models configured in your project.

---

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

---

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

---

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

---

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

# Enable remediation
export VAIG_REMEDIATION__ENABLED=true

# Enable training pipeline
export VAIG_TRAINING__ENABLED=true

# Webhook HMAC secret
export VAIG_WEBHOOK_SERVER__HMAC_SECRET=my-hmac-secret
```

---

## Common Patterns

### Quick cluster health check

```bash
# Full scan — all namespaces, skip healthy, open HTML report
vaig discover --all-namespaces --skip-healthy --format html --open
```

### CI/CD health gate

```bash
# Exit 0 if healthy, 1 if issues found — safe for pipelines
vaig check --namespace production --cached
```

### Investigate → remediate workflow

```bash
# 1. Discover issues
vaig discover --namespace production --skip-healthy -o report.md

# 2. See recommended actions
vaig remediate --list

# 3. Preview a safe action
vaig remediate --finding 1 --dry-run

# 4. Execute it
vaig remediate --finding 1 --approve
```

### Live investigation with drill-in

```bash
# Run analysis then drop into interactive REPL
vaig live "Check for memory pressure" --cluster prod --interactive
```

### Scheduled recurring scans

```bash
# Register a nightly scan at 2am
vaig schedule add --cron "0 2 * * *" --namespace production --skip-healthy

# Start the scheduler daemon
vaig schedule start

# Check status
vaig schedule status
```

### Fleet-wide parallel scan

```bash
# Scan all clusters in parallel, export consolidated report
vaig fleet discover --parallel --max-workers 4 --skip-healthy --export fleet.md
```

### Fine-tuning pipeline

```bash
# 1. Collect feedback over time
vaig feedback -r 5 --last -m "Excellent RCA"

# 2. Prepare high-quality training examples
vaig train prepare --min-rating 4 -o training.jsonl

# 3. Submit tuning job
vaig train submit --input training.jsonl
```

### Watch mode — live polling

```bash
# Re-run analysis every 2 minutes, compact output
vaig live "Any OOMKilled pods?" --cluster prod --watch 120 --summary
```

---

[Back to index](README.md)

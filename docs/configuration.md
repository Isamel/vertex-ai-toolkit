# Configuration Guide

VAIG is configured via YAML files and environment variables. This guide covers all configuration sections.

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

### `language` — Output Language

```yaml
language: en                           # BCP-47 language code (default: "en")
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `language` | string | `"en"` | Preferred output language as a BCP-47 code (e.g. `"es"`, `"pt"`, `"ja"`). When set to a non-`"en"` value, ALL agent output is produced in this language regardless of the query language. When `"en"` (the default), language is auto-detected from the user query at runtime. Override via `VAIG_LANGUAGE`. |

**Examples:**

```yaml
# Produce all reports in Spanish
language: "es"

# Produce all reports in Japanese
language: "ja"
```

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
    - code-migration
    - greenfield
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

VAIG uses a **two-layer retry strategy** for Vertex AI API calls:

1. **SDK layer** — The `google-genai` SDK retries automatically via `HttpRetryOptions` using your configured backoff settings. This handles transient HTTP errors (429, 500, 502, 503, 504) at the transport level.
2. **Application layer** — Catches `google.genai.errors.ClientError` exceptions that escape the SDK and converts them to typed exceptions (`GeminiRateLimitError` for 429, `GeminiConnectionError` for 500/502/503/504, `GeminiClientError` for non-retryable errors). The application layer does **not** re-retry SDK-retryable codes to avoid retry multiplication.

```yaml
retry:
  max_retries: 3                     # Maximum retry attempts (SDK + app layer)
  initial_delay: 1.0                 # Initial delay in seconds (SDK backoff base)
  max_delay: 60.0                    # Maximum delay between retries
  backoff_multiplier: 2.0            # Exponential backoff multiplier
  retryable_status_codes:            # HTTP status codes that trigger retry
    - 429                            # Too Many Requests / RESOURCE_EXHAUSTED
    - 500                            # Internal Server Error
    - 503                            # Service Unavailable
```

> **Note**: The SDK's `HttpRetryOptions` also retries 502 and 504 automatically. The `retryable_status_codes` list is passed to `HttpRetryOptions` to configure which HTTP status codes trigger SDK-level retries.

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
  pipeline_mode: false               # When true, routes --code through CodingSkillOrchestrator (Planner→Implementer→Verifier). Also set by --pipeline CLI flag.
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
  exec_enabled: false                # Enable exec_command tool (DISABLED by default)
  crd_check_timeout: 5               # Timeout (seconds) for CRD existence probes
  argo_request_timeout: 10           # Timeout (seconds) for Argo Rollouts API calls
  argo_rollouts_enabled: null        # null=auto-detect, true=force-enable, false=disable
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `exec_enabled` | boolean | `false` | When `true`, enables the `exec_command` tool which allows executing diagnostic commands inside running containers. **Disabled by default for security.** Commands are still validated against a denylist (dangerous patterns) and an allowlist (read-only diagnostics) even when enabled. |
| `crd_check_timeout` | integer | `5` | Timeout in seconds for CRD existence probes (used before ArgoCD and Argo Rollouts tool invocations). Prevents ~84s hangs when the apiextensions endpoint is unreachable. |
| `argo_request_timeout` | integer | `10` | Timeout in seconds for all Argo Rollouts Kubernetes API calls (`list`, `get` on rollouts, analysis runs, etc.). Keeps Argo tools fast-fail when the Argo Rollouts cluster is unreachable. Override via `VAIG_GKE__ARGO_REQUEST_TIMEOUT`. |
| `argo_rollouts_enabled` | boolean \| null | `null` | `null` = auto-detect via CRD probe + annotation scan. `true` = force-enable without CRD check (use when Argo Rollouts is on a **separate cluster**). `false` = disable entirely. |

### `datadog` — Datadog API Integration

Datadog tools are auto-enabled when both `api_key` and `app_key` are set.

```yaml
datadog:
  enabled: false                     # true = enable; or just set api_key+app_key (auto-enables)
  api_key: ""                        # Datadog API key — prefer VAIG_DATADOG__API_KEY env var
  app_key: ""                        # Datadog app key — prefer VAIG_DATADOG__APP_KEY env var
  site: "datadoghq.com"              # Datadog site (e.g. datadoghq.eu for EU)
  timeout: 30                        # API request timeout (seconds)
  metric_mode: "k8s_agent"           # "k8s_agent" | "apm" — see below
  cluster_name_override: ""          # Override auto-detected cluster name tag
  default_lookback_hours: 4.0        # Default lookback for APM trace queries
  # ssl_verify: true                 # true | false | "/path/to/ca-bundle.crt"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metric_mode` | string | `"k8s_agent"` | `"k8s_agent"` — query `kubernetes.*` metrics (requires Datadog DaemonSet Agent with kubelet check). `"apm"` — query `trace.*` metrics (for APM-only setups without a DaemonSet Agent). Override via `VAIG_DATADOG__METRIC_MODE`. |
| `cluster_name_override` | string | `""` | Override the `cluster_name` tag value used in all Datadog metric queries. When empty (default), VAIG uses the GKE cluster name. Set this when the Datadog Agent tags the cluster differently (e.g. `"prod-us-east"` vs `"gke-prod-us"`). Override via `VAIG_DATADOG__CLUSTER_NAME_OVERRIDE`. |
| `default_lookback_hours` | float | `4.0` | Default lookback window (hours) for APM trace queries. Increase for low-traffic services where 1h may return no data. Override via `VAIG_DATADOG__DEFAULT_LOOKBACK_HOURS`. |
| `ssl_verify` | bool \| string | `true` | SSL certificate verification for Datadog API requests. `true` = standard verification. `false` = disable (not recommended; for debugging only). `"/path/to/ca.crt"` = path to a custom CA bundle file for corporate proxies with SSL inspection. Override via `VAIG_DATADOG__SSL_VERIFY`. |

**Corporate proxy setup** — if your environment uses SSL inspection (e.g. Zscaler, Palo Alto), set:

```yaml
datadog:
  ssl_verify: "/etc/ssl/certs/corporate-ca.crt"  # path to CA bundle
```

Or set the standard `REQUESTS_CA_BUNDLE` environment variable (respected by the `requests` library when `ssl_verify=true`).

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

language: en                             # Output language (BCP-47 code, e.g. "es" for Spanish)

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
  exec_enabled: false                # Set true to enable exec_command tool
  impersonate_sa: ""                 # SA for GKE/GCP observability APIs

helm:
  enabled: true                      # Enable Helm release introspection

argocd:
  enabled: true
  context: gke_mgmt-project_us-central1_mgmt-cluster  # Management cluster

budget:
  enabled: true
  max_cost_usd: 5.0
  warn_threshold: 0.8
  action: warn                       # "warn" or "stop"

safety:
  enabled: true
  settings:
    - category: HARM_CATEGORY_DANGEROUS_CONTENT
      threshold: BLOCK_MEDIUM_AND_ABOVE

telemetry:
  enabled: true
  buffer_size: 50

logging:
  level: INFO

retry:
  max_retries: 5
  initial_delay: 2.0
```

### `helm` — Helm Integration

```yaml
helm:
  enabled: true                      # Helm release introspection (enabled by default)
```

When enabled, VAIG registers 4 read-only Helm tools that introspect release data from Kubernetes Secrets. No Helm binary required. See [Tools Reference](tools-reference.md#helm-tools).

### `argocd` — Argo CD Integration

```yaml
argocd:
  enabled: false                     # Set true to enable ArgoCD tools
  server: ""                         # ArgoCD API URL (for remote ArgoCD)
  token: ""                          # Auth token (prefer VAIG_ARGOCD__TOKEN env var)
  insecure: false                    # Skip TLS verification
  context: ""                        # kubeconfig context for ArgoCD management cluster
  kubeconfig_path: ""                # kubeconfig path for ArgoCD cluster
  namespace: argocd                  # Namespace where ArgoCD is installed
```

Supports 4 deployment topologies — see [Architecture](architecture.md#argocd-connection-topologies):

| Topology | Config |
|----------|--------|
| Same cluster | `enabled: true` (nothing else needed) |
| Management cluster (same project) | `enabled: true` + `context: <mgmt-context>` |
| Management cluster (different project) | `enabled: true` + `context: <mgmt-context>` |
| ArgoCD API server (any topology) | `enabled: true` + `server: https://...` + `token: ...` |

### `budget` — Cost Budget Tracking

```yaml
budget:
  enabled: false                     # Set true to enable budget enforcement
  max_cost_usd: 5.0                  # Maximum spend per session (USD)
  warn_threshold: 0.8                # Warn at 80% of max
  action: warn                       # "warn" (proceed with warning) or "stop" (block requests)
```

Use `/cost` in the REPL to see current session cost. Budget is checked before every API call.

### `safety` — Gemini Safety Settings

```yaml
safety:
  enabled: true                      # Set false to skip sending safety settings
  settings: []                       # List of category → threshold mappings
  # - category: HARM_CATEGORY_HARASSMENT
  #   threshold: BLOCK_MEDIUM_AND_ABOVE
```

### `telemetry` — Local Usage Telemetry

```yaml
telemetry:
  enabled: true                      # Set false to disable (or VAIG_TELEMETRY_ENABLED=false)
  buffer_size: 50                    # Events buffered before flushing to SQLite
```

All telemetry data stays local (`~/.vaig/telemetry.db`). View with `vaig stats summary`.

### `plugins` — Python Plugin Tools

```yaml
plugins:
  enabled: false                     # Set true to load custom Python tool plugins
  directories: []                    # Paths to plugin module directories
  # - ./plugins
  # - ~/.vaig/plugins
```

---

[Back to index](README.md)

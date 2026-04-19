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
  fallback_location: us-central1    # Fallback region if primary fails (default: us-central1)
  available_projects: []            # Catalog of GCP projects (optional)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project_id` | string | `""` | GCP project ID. Required for Vertex AI calls. Falls back to Application Default Credentials project when empty. |
| `location` | string | `"us-central1"` | Vertex AI region for model API calls. |
| `fallback_location` | string | `"us-central1"` | Fallback Vertex AI region when the primary fails. |
| `available_projects` | list | `[]` | Optional catalog of GCP projects with `project_id`, `description`, and `role` fields. |

💡 **Example usage** — multi-project setup:

```yaml
gcp:
  project_id: prod-analytics-42
  location: us-central1
  available_projects:
    - project_id: prod-analytics-42
      description: Production analytics cluster
      role: gke
    - project_id: ml-platform-99
      description: Vertex AI workloads
      role: vertex-ai
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
  thinking:                          # Gemini thinking mode (nested)
    enabled: false                   # Enable thinking mode
    budget_tokens: null              # Token budget (null=model default, 0=disable, -1=auto)
    include_thoughts: true           # Include thought content in response
```

#### `generation.thinking` — Gemini Thinking Mode

Controls Gemini's extended thinking capability. When enabled on a supported model (e.g. `gemini-2.5-flash`, `gemini-2.5-pro`), a `thinking_config` parameter is included in the `GenerateContentConfig` sent to the API. Thinking mode is **opt-in** — disabled by default — so existing behaviour is unchanged unless explicitly enabled.

```yaml
generation:
  thinking:
    enabled: false                   # Enable thinking mode (default: false)
    budget_tokens: null              # Token budget for thinking (default: null)
    include_thoughts: true           # Include thought parts in response (default: true)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | When `true` and the model supports thinking, the thinking config is sent with every request. |
| `budget_tokens` | integer \| null | `null` | Token budget for the thinking phase. `null` = use model default. `0` = disable thinking. `-1` = automatic budget. Positive integer = fixed budget. |
| `include_thoughts` | boolean | `true` | Whether to include thought content alongside the output in the response. |

> **Note**: Thinking mode only activates on models that support it (`gemini-2.5-flash`, `gemini-2.5-pro` and their versioned variants). Enabling it on other models has no effect. Override via `VAIG_GENERATION__THINKING__ENABLED`, `VAIG_GENERATION__THINKING__BUDGET_TOKENS`, etc.

### `models` — Model Selection

```yaml
models:
  default: gemini-2.5-pro            # Primary model
  fallback: gemini-2.5-flash         # Fallback model (used on errors/retries)
  available:                         # List of available models with metadata
    - id: gemini-2.5-pro
      description: "Best for complex reasoning"
      max_output_tokens: 65536
      context_window: 1048576
    - id: gemini-2.5-flash
      description: "Fast and efficient"
      max_output_tokens: 65536
      context_window: 1048576
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default` | string | `"gemini-2.5-pro"` | Primary model for all requests. |
| `fallback` | string | `"gemini-2.5-flash"` | Fallback model used after repeated failures. |
| `available` | list | `[]` | Catalog of models with `id`, `description`, `max_output_tokens`, and `context_window` metadata. When an entry exists for the active model, its `context_window` overrides the global `context_window.context_window_size`. |

### `session` — Session Persistence

```yaml
session:
  db_path: ~/.vaig/sessions.db      # SQLite database path
  repl_history_path: ~/.vaig/repl_history  # REPL input history file
  auto_save: true                    # Auto-save messages to DB
  max_history_messages: 100          # Max messages kept in memory per session
  max_history_tokens: 28000          # Token budget for conversation history
  summarization_threshold: 0.8       # Fraction of max_history_tokens that triggers summarization
  summary_target_tokens: 4000        # Target tokens for the generated summary message
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_history_tokens` | integer | `28000` | Conservative token budget for in-memory history. When history approaches this limit, older messages are summarized. |
| `summarization_threshold` | float | `0.8` | Fraction of `max_history_tokens` at which summarization triggers (0.8 = 80%). |
| `summary_target_tokens` | integer | `4000` | Target size in tokens for the summary message that replaces older history. |

💡 **Example usage** — long-running investigation sessions:

```yaml
session:
  db_path: ~/.vaig/sessions.db
  auto_save: true
  max_history_messages: 200
  max_history_tokens: 50000
  summarization_threshold: 0.75
  summary_target_tokens: 6000
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
    - discovery
  custom_dir: null                   # Deprecated — use external_dirs instead
  external_dirs: []                  # Paths to custom skills directories
  packages: []                       # Installed skill packages
  auto_routing: true                 # Enable automatic skill routing
  auto_routing_threshold: 1.5        # Minimum score gap for confident auto-routing
```

> **Note**: `custom_dir` is deprecated. Use `external_dirs` instead. If both are set, `external_dirs` wins.

### `agents` — Agent Configuration

```yaml
agents:
  max_concurrent: 3                  # Max parallel agents (for fan-out strategy)
  orchestrator_model: gemini-2.5-pro   # Model for orchestrator agent
  specialist_model: gemini-2.5-flash   # Model for specialist agents
  max_iterations_retry: 15           # Max retry iterations per agent
  parallel_tool_calls: true          # Execute independent tool calls in parallel
  max_concurrent_tool_calls: 5       # Semaphore limit for parallel tool calls
  max_failures_before_fallback: 2    # Consecutive failures before switching to fallback model
  min_inter_call_delay: 0.0          # Seconds between LLM API calls (0 = disabled)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `parallel_tool_calls` | boolean | `true` | When Gemini returns multiple tool calls in a single response, executes them concurrently via `asyncio.gather`. The sync path always runs sequentially. |
| `max_concurrent_tool_calls` | integer | `5` | Semaphore limit for concurrent tool calls. Prevents API throttling when the model requests many calls at once. |
| `max_failures_before_fallback` | integer | `2` | Consecutive rate-limit or connection failures before the orchestrator switches the agent to `models.fallback`. Set to `0` to disable model fallback. |
| `min_inter_call_delay` | float | `0.0` | Seconds to sleep between LLM API calls. Set to `1.0` to limit to ~60 calls/min for quota-limited projects. |

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
  rate_limit_initial_delay: 8.0      # Initial delay for 429 rate-limit errors (longer wait)
  retryable_status_codes:            # HTTP status codes that trigger retry
    - 429                            # Too Many Requests / RESOURCE_EXHAUSTED
    - 500                            # Internal Server Error
    - 502                            # Bad Gateway
    - 503                            # Service Unavailable
    - 504                            # Gateway Timeout
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rate_limit_initial_delay` | float | `8.0` | Longer initial backoff (seconds) used when a 429 rate-limit error is detected. Applied instead of `initial_delay` so the client waits longer before retrying quota-exhausted requests. |

> **Note**: The SDK's `HttpRetryOptions` also retries 502 and 504 automatically. The `retryable_status_codes` list is passed to `HttpRetryOptions` to configure which HTTP status codes trigger SDK-level retries.

### `logging` — Logging Configuration

```yaml
logging:
  level: WARNING                     # Console log level: DEBUG, INFO, WARNING, ERROR
  show_path: false                   # Show file path in log output
  file_enabled: true                 # Enable file logging
  file_path: ~/.vaig/logs/vaig.log   # Log file path
  file_level: DEBUG                  # Log level for file output
  file_max_bytes: 5242880            # Max log file size in bytes (5 MB)
  file_backup_count: 3               # Number of rotated log files to keep
  tool_results: true                 # Save tool results to disk
  tool_results_dir: ~/.vaig          # Directory for tool result files
```

Override via CLI: `--verbose` sets INFO-level detail, `--debug` enables full DEBUG output.

### `coding` — Coding Agent Configuration

```yaml
coding:
  workspace_root: "."                # Root directory for file operations
  max_tool_iterations: 25            # Max tool-use loop iterations
  confirm_actions: true              # Require user confirmation for write ops
  pipeline_mode: false               # When true, routes --code through CodingSkillOrchestrator (Planner→Implementer→Verifier). Also set by --pipeline CLI flag.
  workspace_isolation: false         # Copy workspace to temp dir before pipeline execution
  jail_ignore_patterns:              # Patterns to exclude when copying workspace to jail
    - .git
    - node_modules
    - __pycache__
    - "*.pyc"
  max_fix_iterations: 1              # Max Implementer→Verifier fix-forward loop iterations
  test_timeout: 120                  # Timeout in seconds for test runner
  test_command: ""                   # Explicit test command (empty = auto-detect)
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
  denied_commands: []                # Additional regex patterns for blocked commands
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `workspace_isolation` | boolean | `false` | When `true`, copies workspace to a temp directory before pipeline execution. The original workspace is untouched until the pipeline completes successfully. |
| `max_fix_iterations` | integer | `1` | Maximum fix-forward loop iterations (Implementer→Verifier retries). Default `1` means no retry. Set to `3`+ to enable automatic self-correction. |
| `test_timeout` | integer | `120` | Timeout in seconds for test runner execution. |
| `test_command` | string | `""` | Explicit test command (e.g. `"pytest -x --tb=short"`). When empty, auto-detects pytest via `pyproject.toml` / `conftest.py` presence. |
| `denied_commands` | list | (built-in list) | Regex patterns for commands that should NEVER be executed. Extend this list in config to add project-specific denials. |

#### `coding.git` — Git Integration

Controls automatic git branch/commit/PR lifecycle for coding pipeline runs. **Disabled by default.**

```yaml
coding:
  git:
    enabled: false                   # Master flag — no git ops when false (default)
    auto_branch: true                # Create a feature branch before any file writes
    auto_commit: true                # Commit all changes after each pipeline phase
    auto_pr: false                   # Open a pull request via `gh` CLI after completion
    pr_provider: gh                  # PR provider — only "gh" (GitHub CLI) supported
    commit_signoff: false            # Add Signed-off-by trailer to commits
    branch_prefix: "vaig/"          # Prefix prepended to auto-generated branch names
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master flag. When `false` (default), no git commands are run and pipeline behavior is identical to pre-git operation. |
| `auto_branch` | boolean | `true` | When enabled, creates a feature branch (e.g. `vaig/fix-auth-bug`) before any file writes. |
| `auto_commit` | boolean | `true` | When enabled, commits all changes after each pipeline phase. |
| `auto_pr` | boolean | `false` | When enabled, opens a pull request via `gh` CLI after the run completes. |
| `commit_signoff` | boolean | `false` | Adds `Signed-off-by` trailer to commits (for DCO compliance). |
| `branch_prefix` | string | `"vaig/"` | Prefix prepended to auto-generated branch names. |

💡 **Example usage** — full git lifecycle for CI/CD integration:

```yaml
coding:
  git:
    enabled: true
    auto_branch: true
    auto_commit: true
    auto_pr: true
    branch_prefix: "ai/"
    commit_signoff: true
```

#### `coding.patch` — Patch Write Configuration

Controls behavior of the `patch_file` tool. All defaults are safe for general use.

```yaml
coding:
  patch:
    backup_enabled: false            # Save .orig backup before applying each patch
    max_hunk_size: 500               # Max lines allowed in a single hunk (0 = unlimited)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backup_enabled` | boolean | `false` | When `true`, saves `<path>.orig` alongside each patched file before applying the patch. |
| `max_hunk_size` | integer | `500` | Maximum number of lines allowed in a single patch hunk. Set to `0` for unlimited. |

💡 **Example usage** — safe patching with backups on large codebases:

```yaml
coding:
  patch:
    backup_enabled: true
    max_hunk_size: 200
```

#### `coding.workspace_rag` — Workspace RAG Index

Controls a local vector-search index over workspace files using ChromaDB. **Disabled by default.** Requires `chromadb` to be installed.

```yaml
coding:
  workspace_rag:
    enabled: false                   # Build and expose a local vector-search index
    reindex_on_run: false            # Rebuild index when stale before first search
    max_chunks: 500                  # Max chunks to index (excess discarded)
    extensions:                      # File extensions to include in the index
      - .py
      - .ts
      - .go
      - .java
      - .md
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | When `true`, builds a local ChromaDB vector-search index over workspace files and exposes it as a search tool. |
| `reindex_on_run` | boolean | `false` | When `true` and the index is stale, rebuild before the first search in the run. |
| `max_chunks` | integer | `500` | Maximum number of chunks to index. Files are processed in discovery order; excess chunks are discarded. |
| `extensions` | list | `[".py", ".ts", ".go", ".java", ".md"]` | File extensions to include in the workspace index. |

💡 **Example usage** — semantic code search in a Python monorepo:

```yaml
coding:
  workspace_rag:
    enabled: true
    reindex_on_run: true
    max_chunks: 2000
    extensions:
      - .py
      - .md
      - .yaml
```

#### `coding.idiom` — Idiom Map Configuration

Controls CM-07 idiom map expansion for language-aware code generation. When `auto_generate` is enabled, the LLM can generate new idiom maps that are cached locally.

```yaml
coding:
  idiom:
    enabled: false                   # Enable idiom map expansion (default: false)
    auto_generate: false             # Allow LLM to generate new idiom maps (default: false)
    cache_dir: ~/.vaig/idioms        # Directory for cached generated idiom maps
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | When `false` (default), only bundled static idiom maps are used. |
| `auto_generate` | boolean | `false` | When `true`, the LLM can generate new idiom maps on demand; generated maps are cached to `cache_dir`. |
| `cache_dir` | string | `"~/.vaig/idioms"` | Directory for cached LLM-generated idiom maps. |

💡 **Example usage** — enable idiom generation for a Go project:

```yaml
coding:
  idiom:
    enabled: true
    auto_generate: true
    cache_dir: ~/.vaig/idioms
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
  location: null                     # GKE cluster location
  default_namespace: default         # Default K8s namespace
  kubeconfig_path: null              # Custom kubeconfig path
  context: null                      # Kubernetes context to use
  log_limit: 100                     # Default log tail lines
  metrics_interval_minutes: 60       # Default metrics time window
  proxy_url: null                    # Proxy URL for GKE API
  impersonate_sa: null               # SA for GKE/GCP observability APIs (overrides auth.impersonate_sa)
  exec_enabled: false                # Enable exec_command tool (DISABLED by default)
  helm_enabled: true                 # Enable Helm tools
  argocd_enabled: null               # null=auto-detect, true=force-enable, false=disable
  argocd_server: null                # ArgoCD API server URL
  argocd_token: null                 # ArgoCD auth token
  argocd_context: null               # kubeconfig context for ArgoCD management cluster
  argocd_namespace: null             # Namespace where ArgoCD is installed
  argocd_verify_ssl: true            # Verify TLS for ArgoCD API
  argo_rollouts_enabled: null        # null=auto-detect, true=force-enable, false=disable
  request_timeout: 30                # Timeout (seconds) for all K8s API calls
  crd_check_timeout: 5               # Timeout (seconds) for CRD existence probes
  argo_request_timeout: 10           # Timeout (seconds) for Argo Rollouts API calls
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `exec_enabled` | boolean | `false` | When `true`, enables the `exec_command` tool which allows executing diagnostic commands inside running containers. **Disabled by default for security.** Commands are still validated against a denylist (dangerous patterns) and an allowlist (read-only diagnostics) even when enabled. |
| `impersonate_sa` | string | `""` | Service account to impersonate for GKE/GCP observability APIs (Cloud Logging, Cloud Monitoring, GKE cluster API). Overrides `auth.impersonate_sa` for GKE tools only, enabling dual-auth scenarios. |
| `request_timeout` | integer | `30` | Timeout in seconds for all Kubernetes API calls. Prevents indefinite hangs when the cluster is unreachable. |
| `crd_check_timeout` | integer | `5` | Timeout in seconds for CRD existence probes (used before ArgoCD and Argo Rollouts tool invocations). Prevents ~84s hangs when the apiextensions endpoint is unreachable. |
| `argo_request_timeout` | integer | `10` | Timeout in seconds for all Argo Rollouts Kubernetes API calls (`list`, `get` on rollouts, analysis runs, etc.). Keeps Argo tools fast-fail when the Argo Rollouts cluster is unreachable. Override via `VAIG_GKE__ARGO_REQUEST_TIMEOUT`. |
| `argo_rollouts_enabled` | boolean \| null | `null` | `null` = auto-detect via CRD probe + annotation scan. `true` = force-enable without CRD check (use when Argo Rollouts is on a **separate cluster**). `false` = disable entirely. |

#### `gke.trends` — Anomaly Trend Detection

Compares current GKE metrics against historical Cloud Monitoring baselines to detect slowly-degrading services. **Disabled by default.**

```yaml
gke:
  trends:
    enabled: false                   # Enable trend analysis (default: false)
    baseline_days:                   # Baseline window sizes in days (max 42)
      - 7
    memory_warning_pct: 10.0         # Memory % increase over baseline → warning
    memory_critical_pct: 25.0        # Memory % increase over baseline → critical
    cpu_warning_pct: 20.0            # CPU % increase over baseline → warning
    cpu_critical_pct: 50.0           # CPU % increase over baseline → critical
    restart_warning_count: 5         # Absolute restart delta over baseline → warning
    restart_critical_count: 15       # Absolute restart delta over baseline → critical
    memory_limit_gib: 4.0            # Assumed memory limit (GiB) for projection
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | When `true`, trend analysis is enabled and findings are included in health reports. |
| `baseline_days` | list[int] | `[7]` | Baseline window sizes in days. Maximum 42 (Cloud Monitoring retention limit). Deduplicated and sorted automatically. |
| `memory_warning_pct` | float | `10.0` | Memory percentage increase over baseline to trigger a warning. |
| `memory_critical_pct` | float | `25.0` | Memory percentage increase over baseline to trigger a critical alert. |
| `cpu_warning_pct` | float | `20.0` | CPU percentage increase over baseline to trigger a warning. |
| `cpu_critical_pct` | float | `50.0` | CPU percentage increase over baseline to trigger a critical alert. |
| `restart_warning_count` | integer | `5` | Absolute restart delta over baseline to trigger a warning. |
| `restart_critical_count` | integer | `15` | Absolute restart delta over baseline to trigger a critical alert. |
| `memory_limit_gib` | float | `4.0` | Assumed memory limit in GiB for days-to-threshold projection calculations. |

💡 **Example usage** — tighter thresholds for memory-sensitive services:

```yaml
gke:
  trends:
    enabled: true
    baseline_days:
      - 7
      - 14
    memory_warning_pct: 5.0
    memory_critical_pct: 15.0
    cpu_warning_pct: 15.0
    cpu_critical_pct: 30.0
    memory_limit_gib: 8.0
```

### `datadog` — Datadog API Integration

Datadog tools are auto-enabled when both `api_key` and `app_key` are set.

```yaml
datadog:
  enabled: false                     # true = enable; or just set api_key+app_key (auto-enables)
  api_key: ""                        # Datadog API key — prefer VAIG_DATADOG__API_KEY env var
  app_key: ""                        # Datadog app key — prefer VAIG_DATADOG__APP_KEY env var
  site: "datadoghq.com"              # Datadog site (e.g. datadoghq.eu for EU)
  timeout: 30                        # API request timeout (seconds)
  metric_mode: "auto"                # "k8s_agent" | "apm" | "both" | "auto"
  cluster_name_override: ""          # Override auto-detected cluster name tag
  default_lookback_hours: 4.0        # Default lookback for APM trace queries
  apm_operation: "auto"              # APM operation name ("auto" = probe for data)
  apm_operation_overrides: {}        # Per-service APM operation overrides
  apm_discovery_enabled: false       # Auto-resolve APM operation via Datadog /api/v1/search
  custom_metrics: {}                 # Custom metric name overrides
  # ssl_verify: true                 # true | false | "/path/to/ca-bundle.crt"
  labels:                            # Configurable Datadog tag/label names
    cluster_name: kube_cluster_name
    service: service
    env: env
    version: version
    namespace: kube_namespace
    deployment: kube_deployment
    pod_name: pod_name
    custom: {}
  detection:                         # Annotation/label prefixes for Datadog detection
    annotation_prefixes:
      - "ad.datadoghq.com/"
      - "admission.datadoghq.com/"
    label_prefix: "tags.datadoghq.com/"
    env_vars:
      - DD_AGENT_HOST
      - DD_TRACE_AGENT_URL
      - DD_SERVICE
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metric_mode` | string | `"auto"` | `"k8s_agent"` — query `kubernetes.*` metrics (requires Datadog DaemonSet Agent with kubelet check). `"apm"` — query `trace.*` metrics (APM-only setups). `"both"` — combined kubernetes.* and trace.* metrics. `"auto"` (default) — tries k8s_agent first, falls back to apm if queries return 0 data points. Override via `VAIG_DATADOG__METRIC_MODE`. |
| `cluster_name_override` | string | `""` | Override the `cluster_name` tag value used in all Datadog metric queries. When empty (default), VAIG uses the GKE cluster name. Set this when the Datadog Agent tags the cluster differently (e.g. `"prod-us-east"` vs `"gke-prod-us"`). Override via `VAIG_DATADOG__CLUSTER_NAME_OVERRIDE`. |
| `default_lookback_hours` | float | `4.0` | Default lookback window (hours) for APM trace queries. Increase for low-traffic services where 1h may return no data. Override via `VAIG_DATADOG__DEFAULT_LOOKBACK_HOURS`. |
| `apm_operation` | string | `"auto"` | APM operation name for trace.* metrics (e.g. `"servlet.request"`, `"grpc.server"`). When `"auto"` (default), the system probes common operation names to find one with data. |
| `apm_operation_overrides` | dict | `{}` | Per-service APM operation override. Key: sanitized service name. Value: operation name. Takes highest precedence — checked before `apm_operation`, cache, discovery, and probe order. |
| `apm_discovery_enabled` | boolean | `false` | When `true`, queries Datadog `/api/v1/search` to auto-resolve the APM operation name before falling back to probe order. Adds one HTTP call per cache miss. Safe to enable — falls back on any API error. |
| `ssl_verify` | bool \| string | `true` | SSL certificate verification for Datadog API requests. `true` = standard verification. `false` = disable (not recommended; for debugging only). `"/path/to/ca.crt"` = path to a custom CA bundle file for corporate proxies with SSL inspection. Override via `VAIG_DATADOG__SSL_VERIFY`. |

**Corporate proxy setup** — if your environment uses SSL inspection (e.g. Zscaler, Palo Alto), set:

```yaml
datadog:
  ssl_verify: "/etc/ssl/certs/corporate-ca.crt"  # path to CA bundle
```

Or set the standard `REQUESTS_CA_BUNDLE` environment variable (respected by the `requests` library when `ssl_verify=true`).

### `memory` — Pattern Memory

Controls recurrence detection for health findings. When enabled, `vaig live` persists finding fingerprints to a local JSONL store and annotates repeated findings with recurrence badges (`NEW` / `RECURRING` / `CHRONIC`). **Disabled by default.**

```yaml
memory:
  enabled: false                     # Enable pattern memory (default: false)
  store_path: ~/.vaig/memory         # Directory for JSONL pattern files
  recurrence_threshold: 2            # Min occurrences to mark finding RECURRING
  chronic_threshold: 5               # Min occurrences to mark finding CHRONIC
  max_age_days: 90                   # Ignore entries older than this many days

  # MEM-03: Fix outcome tracking
  outcome_tracking_enabled: false    # Track whether fixes actually resolved findings
  outcome_store_path: ~/.vaig/memory/outcomes  # JSONL store for FixOutcome records
  outcome_fuzzy_match: false         # Allow fuzzy fingerprint matching during correlation
  outcome_correlation_window_runs: 3 # Max subsequent runs to search before marking outcome unknown

  # MEM-04: Semantic memory RAG
  memory_rag_enabled: false          # Enable semantic memory via Vertex AI RAG corpus
  memory_rag_corpus_name: ""         # Vertex AI RAG corpus name for memory narratives
  memory_rag_max_narratives: 500     # Max narratives to ingest (oldest dropped first)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master flag. When `true`, finding fingerprints are persisted across runs. Enable with `VAIG_MEMORY__ENABLED=true`. |
| `store_path` | string | `"~/.vaig/memory"` | Directory where JSONL pattern files are stored. Each run produces a `{run_id}.jsonl` file. |
| `recurrence_threshold` | integer | `2` | Minimum occurrence count before a finding is marked `RECURRING`. Must be ≥ 2. |
| `chronic_threshold` | integer | `5` | Minimum occurrence count before a finding is marked `CHRONIC`. Must be ≥ 2. |
| `max_age_days` | integer | `90` | Entries older than this number of days are ignored during analysis. |
| `outcome_tracking_enabled` | boolean | `false` | When `true` (MEM-03), tracks whether remediation actions actually fixed findings by correlating future runs. |
| `outcome_fuzzy_match` | boolean | `false` | Allow fuzzy fingerprint matching when correlating outcomes across runs. |
| `outcome_correlation_window_runs` | integer | `3` | Max subsequent runs to search before marking a fix outcome as `unknown`. |
| `memory_rag_enabled` | boolean | `false` | When `true` (MEM-04), ingests finding narratives into a Vertex AI RAG corpus for semantic retrieval. |
| `memory_rag_corpus_name` | string | `""` | Vertex AI RAG corpus name for memory narratives (separate from the knowledge RAG corpus). |
| `memory_rag_max_narratives` | integer | `500` | Maximum narratives to ingest. When exceeded, oldest narratives are dropped first. |

💡 **Example usage** — enable full memory with outcome tracking:

```yaml
memory:
  enabled: true
  store_path: ~/.vaig/memory
  recurrence_threshold: 3
  chronic_threshold: 7
  max_age_days: 60
  outcome_tracking_enabled: true
  outcome_fuzzy_match: true
  outcome_correlation_window_runs: 5
```

### `investigation` — Autonomous Investigation Pipeline

Controls the autonomous investigation pipeline for service-health analysis. When `enabled` is `true`, `health_planner` and `health_investigator` agents are inserted into the service-health pipeline between the analyzer and verifier. **Disabled by default.**

```yaml
investigation:
  enabled: false                     # Enable investigation pipeline (default: false)
  autonomous_mode: false             # Run fully autonomously with budget enforcement
  budget_per_run_usd: 0.0            # Max USD per pipeline run (0.0 = no cap)
  max_iterations: 10                 # Hard cap on investigator loop iterations per run
  max_steps_per_plan: 15             # Max steps the planner can generate
  circle_threshold: 2                # Same (tool, args_hash) count before flagging a loop
  memory_correction: true            # Enable MEM-05 memory-aware pre-action hook
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master flag. Enable with `VAIG_INVESTIGATION__ENABLED=true`. |
| `autonomous_mode` | boolean | `false` | When `true`, the pipeline runs fully autonomously with budget enforcement and memory awareness (SH-09). **Requires `enabled: true`** — raises a `ValueError` at construction if `autonomous_mode=true` and `enabled=false`. |
| `budget_per_run_usd` | float | `0.0` | Maximum spend in USD per pipeline run when `autonomous_mode` is `true`. Set to `0.0` for no cap. |
| `max_iterations` | integer | `10` | Hard cap on investigator loop iterations per run. Must be ≥ 1. |
| `max_steps_per_plan` | integer | `15` | The planner will not generate more than this many steps per plan. Must be ≥ 1. |
| `circle_threshold` | integer | `2` | Same `(tool, args_hash)` count before flagging a circular investigation loop. Overrides `self_correction.max_repeated_calls` when set. |
| `memory_correction` | boolean | `true` | When `true` (MEM-05), enables the memory-aware pre-action hook inside `InvestigationAgent` to avoid repeating previously attempted actions. |

💡 **Example usage** — fully autonomous investigation with $1 budget:

```yaml
investigation:
  enabled: true
  autonomous_mode: true
  budget_per_run_usd: 1.00
  max_iterations: 20
  max_steps_per_plan: 25
  memory_correction: true
```

### `hypothesis` — Hypothesis Prompt Library

Controls the hypothesis prompt library (SPEC-X-03). When `enabled`, the `HypothesisLibrary` is instantiated and used by the investigation planner to seed step `tool_hints`. **Disabled by default.**

```yaml
hypothesis:
  enabled: false                     # Enable hypothesis library (default: false)
  custom_templates_path: null        # Path to YAML file with custom HypothesisTemplate overrides
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | When `true`, the hypothesis library seeds investigation steps with tool hints. Auto-enabled when `custom_templates_path` is set. |
| `custom_templates_path` | path \| null | `null` | Optional path to a YAML file with user-defined `HypothesisTemplate` overrides. When set, `enabled` is automatically set to `true`. |

💡 **Example usage** — custom templates for your org's runbooks:

```yaml
hypothesis:
  enabled: true
  custom_templates_path: ~/.vaig/hypothesis-templates.yaml
```

### `self_correction` — Self-Correction Controller

Controls loop-detection and stale-iteration thresholds for `InvestigationAgent` (SPEC-SH-06). **Disabled by default.** Auto-enables when any non-default value is set.

```yaml
self_correction:
  enabled: false                     # Enable self-correction controller (default: false)
  max_repeated_calls: 3              # Same (tool, args_hash) count before flagging a circle
  max_stale_iterations: 5            # Consecutive iterations without progress before FORCE_DIFFERENT
  contradiction_sensitivity: 0.8     # Sensitivity for contradiction detection (0=off, 1=strict)
  max_budget_per_step_usd: 0.10      # Max USD budget per investigation step
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master flag. Auto-enabled when any non-default value is configured. Enable with `VAIG_SELF_CORRECTION__ENABLED=true`. |
| `max_repeated_calls` | integer | `3` | Number of times the same `(tool, args_hash)` pair can appear before flagging a loop. Must be ≥ 1. |
| `max_stale_iterations` | integer | `5` | Consecutive iterations without a newly completed step before the controller triggers `FORCE_DIFFERENT` to break the loop. Must be ≥ 1. |
| `contradiction_sensitivity` | float | `0.8` | Sensitivity for contradiction detection. `0.0` = off, `1.0` = strict. Must be between 0.0 and 1.0. |
| `max_budget_per_step_usd` | float | `0.10` | Maximum USD budget allocated to a single investigation step. Set to `0.0` for no limit. |

💡 **Example usage** — tighter loop detection for long-running investigations:

```yaml
self_correction:
  enabled: true
  max_repeated_calls: 2
  max_stale_iterations: 3
  contradiction_sensitivity: 0.9
  max_budget_per_step_usd: 0.05
```

### `remediation` — Remediation Engine

Controls the runbook execution engine that classifies recommended commands into `SAFE`/`REVIEW`/`BLOCKED` tiers and enforces approval workflows before execution. **Disabled by default.**

```yaml
remediation:
  enabled: false                     # Master flag — entire engine disabled by default
  auto_approve_safe: false           # Execute SAFE-tier commands without user confirmation
  blocked_commands: []               # Regex patterns for commands ALWAYS blocked
  timeout: 30                        # Execution timeout in seconds per command
  dry_run: false                     # Show execution plan without side effects
  tier_overrides: {}                 # Map of command pattern → tier (safe/review/blocked)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master flag. When `false`, the remediation engine is completely disabled. Enable with `VAIG_REMEDIATION__ENABLED=true`. |
| `auto_approve_safe` | boolean | `false` | When `true`, `SAFE`-tier commands execute without user confirmation. |
| `blocked_commands` | list | `[]` | Regex patterns for commands that should always be `BLOCKED`. Checked against the full command string before tier assignment. |
| `timeout` | integer | `30` | Execution timeout in seconds per command. |
| `dry_run` | boolean | `false` | When `true`, all commands show the execution plan without side effects. |
| `tier_overrides` | dict | `{}` | Map of command regex pattern → tier name (`safe`/`review`/`blocked`). Promotes or demotes specific commands without code changes. |

💡 **Example usage** — allow automatic rollout restarts but require review for scale changes:

```yaml
remediation:
  enabled: true
  auto_approve_safe: true
  dry_run: false
  timeout: 60
  tier_overrides:
    "kubectl rollout restart": "safe"
    "kubectl scale": "review"
    "kubectl delete pod": "review"
```

### `review` — Report Review Gate

Controls the human review/approval gate that can block remediation execution until a report is approved. **Disabled by default.**

```yaml
review:
  enabled: false                     # Master flag — review gate disabled by default
  require_review_for_remediation: true  # Block remediation until report is approved
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master flag. When `true`, the review gate is active. Enable with `VAIG_REVIEW__ENABLED=true`. |
| `require_review_for_remediation` | boolean | `true` | When `true`, remediation is blocked until the report review is approved. |

💡 **Example usage** — require review before any automated remediation:

```yaml
review:
  enabled: true
  require_review_for_remediation: true

remediation:
  enabled: true
  auto_approve_safe: false
```

### `mcp` — Model Context Protocol

```yaml
mcp:
  enabled: false                     # Enable MCP integration
  auto_register: false               # Auto-register MCP tools into agent tool lists
  servers:                           # List of MCP servers
    - name: my-server
      command: npx
      args: ["-y", "@my-org/mcp-server"]
      env:
        API_KEY: "${MY_API_KEY}"
      description: "My custom MCP server"
```

See [MCP Guide](mcp-guide.md) for detailed MCP configuration.

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
  max_cost_per_run: 0.0              # Max cost per single orchestrator pipeline run (0.0 = no cap)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_cost_per_run` | float | `0.0` | Maximum USD cost allowed for a single orchestrator pipeline run. When exceeded, the pipeline halts with `OrchestratorResult.budget_exceeded=True`. Set to `0.0` to disable. |

Use `/cost` in the REPL to see current session cost. Budget is checked before every API call.

### `global_budget` — Global Run Budget

Controls pipeline-level hard limits independent of USD cost tracking. All values default to `0` (unlimited). **Does not require `budget.enabled`.** This directly enforces token, tool-call, wall-time, and cost caps at the pipeline level.

```yaml
global_budget:
  max_tokens: 0                      # Max input+output tokens per run (0 = unlimited)
  max_tool_calls: 0                  # Max tool calls per run (0 = unlimited)
  max_wall_seconds: 0.0              # Max wall-clock seconds per run (0 = unlimited)
  max_cost_usd: 0.0                  # Max USD cost per run (0 = unlimited)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_tokens` | integer | `0` | Maximum total input+output tokens across all agents in one run. `0` = unlimited. |
| `max_tool_calls` | integer | `0` | Maximum total tool calls across all agents in one run. `0` = unlimited. |
| `max_wall_seconds` | float | `0.0` | Maximum wall-clock seconds allowed for one run. `0.0` = unlimited. |
| `max_cost_usd` | float | `0.0` | Maximum USD cost allowed for one run. `0.0` = unlimited. |

💡 **Example usage** — hard cap for CI pipelines:

```yaml
global_budget:
  max_tokens: 500000
  max_tool_calls: 100
  max_wall_seconds: 300.0
  max_cost_usd: 2.00
```

### `circuit_breaker` — Tool Circuit Breaker

Controls when a tool's circuit breaker opens after consecutive failures and how long before it probes again.

```yaml
circuit_breaker:
  failure_threshold: 3               # Consecutive failures before breaker opens
  recovery_timeout: 30.0             # Seconds in OPEN state before allowing a probe
  window_size: 10                    # Number of recent calls to consider (reserved)
```

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

### `effectiveness` — Tool Effectiveness Learning

Controls automatic tool scoring based on historical call data. When enabled, tools are auto-skipped, deprioritized, or boosted based on failure rate and duration thresholds. **Disabled by default.**

```yaml
effectiveness:
  enabled: false                     # Enable tool effectiveness learning (default: false)
  skip_threshold: 0.8                # Failure rate above which tool is SKIP tier
  deprioritize_threshold: 0.5        # Failure rate above which tool is DEPRIORITIZE tier
  boost_threshold: 0.1               # Failure rate below which tool is BOOST tier
  slow_tool_threshold_s: 10.0        # Avg duration (seconds) → DEPRIORITIZE tier
  min_calls_for_scoring: 3           # Min historical calls before a tool can be scored
  lookback_days: 7                   # Days of history to consider when computing scores
  cache_ttl_seconds: 300             # Seconds to cache computed scores
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `skip_threshold` | float | `0.8` | Failure rate above which a tool is assigned SKIP tier and not executed. |
| `deprioritize_threshold` | float | `0.5` | Failure rate above which a tool is assigned DEPRIORITIZE tier (warning logged). |
| `boost_threshold` | float | `0.1` | Failure rate below which a reliable tool is assigned BOOST tier. |
| `slow_tool_threshold_s` | float | `10.0` | Average duration in seconds above which a tool is assigned DEPRIORITIZE tier. |
| `min_calls_for_scoring` | integer | `3` | Minimum historical calls required before a tool can be scored. |

### `audit` — Audit Logging

Sends audit events to BigQuery and Cloud Logging. **Disabled by default.**

```yaml
audit:
  enabled: false                     # Enable audit logging (default: false)
  bigquery_dataset: vaig_audit       # BigQuery dataset name
  bigquery_table: audit_events       # BigQuery table name
  cloud_logging_log_name: vaig-audit # Cloud Logging log name
  buffer_size: 20                    # Events buffered before flushing
  flush_interval_seconds: 30         # Flush interval in seconds
```

### `rate_limit` — Rate Limiting (Platform Mode)

Controls quota policy enforcement loaded from GCS. Used in platform/multi-tenant deployments. **Disabled by default.**

```yaml
rate_limit:
  enabled: false                     # Enable rate limiting (default: false)
  policy_gcs_bucket: ""              # GCS bucket containing the quota policy YAML
  policy_gcs_path: vaig/quota-policy.yaml  # Object path within the bucket
  cache_ttl_seconds: 300             # TTL for cached policy (seconds)
```

### `webhook_server` — Webhook Server

Receives Datadog alert webhooks and triggers automated health analysis. Auto-enables when `hmac_secret` is provided. **Disabled by default.**

```yaml
webhook_server:
  enabled: false                     # Auto-enables when hmac_secret is set
  host: "0.0.0.0"                    # Bind host
  port: 8080                         # Bind port
  hmac_secret: ""                    # HMAC secret for webhook validation — prefer env var
  max_analyses_per_day: 50           # Daily analysis cap
  dedup_cooldown_seconds: 300        # Cooldown between identical alerts (seconds)
  analysis_timeout_seconds: 600      # Timeout for a single analysis run (seconds)
```

💡 **Example usage** — receive Datadog webhooks and route to Google Chat:

```yaml
webhook_server:
  hmac_secret: "${VAIG_DD_WEBHOOK_SECRET}"  # auto-enables
  port: 8080
  max_analyses_per_day: 100

google_chat:
  webhook_url: "${VAIG_GCHAT_WEBHOOK_URL}"  # auto-enables
  notify_on:
    - critical
    - high
```

### `jira` — Jira Integration

Exports health report findings as Jira issues. Auto-enables when `base_url` is set. **Disabled by default.**

```yaml
jira:
  enabled: false                     # Auto-enables when base_url is set
  base_url: ""                       # Jira Cloud base URL (e.g. https://myorg.atlassian.net)
  email: ""                          # Jira account email
  api_token: ""                      # Jira API token — prefer VAIG_JIRA__API_TOKEN env var
  project_key: ""                    # Jira project key (e.g. OPS)
  issue_type: Bug                    # Issue type for created issues
  severity_field_mapping:            # Map VAIG severity → Jira priority
    CRITICAL: Highest
    HIGH: High
    MEDIUM: Medium
    LOW: Low
    INFO: Lowest
```

### `pagerduty` — PagerDuty Integration

Triggers PagerDuty incidents from health findings. Auto-enables when `routing_key` is provided. **Disabled by default.**

```yaml
pagerduty:
  enabled: false                     # Auto-enables when routing_key is set
  routing_key: ""                    # Events API v2 routing key — prefer VAIG_PAGERDUTY__ROUTING_KEY
  api_token: ""                      # REST API v2 token (for enrichment features)
  service_id: ""                     # PagerDuty service ID
  base_url: "https://api.pagerduty.com"
  auto_create_incident: true         # Auto-create incidents for critical findings
  severity_mapping:                  # Map VAIG severity → PagerDuty severity
    critical: critical
    high: error
    medium: warning
    info: info
  alert_service_ids: []              # Service IDs for alert correlation tool
  alert_fetch_limit: 25              # Max alerts to fetch for correlation
```

### `google_chat` — Google Chat Integration

Sends Card v2 notifications to a Google Chat webhook. Auto-enables when `webhook_url` is set. **Disabled by default.**

```yaml
google_chat:
  enabled: false                     # Auto-enables when webhook_url is set
  webhook_url: ""                    # Incoming webhook URL — prefer VAIG_GOOGLE_CHAT__WEBHOOK_URL
  notify_on:                         # Severities that trigger notifications
    - critical
    - high
```

### `slack` — Slack Integration

Sends Block Kit notifications to a Slack webhook. Auto-enables when `webhook_url` is set. **Disabled by default.**

```yaml
slack:
  enabled: false                     # Auto-enables when webhook_url is set
  webhook_url: ""                    # Incoming webhook URL — prefer VAIG_SLACK__WEBHOOK_URL
  notify_on:                         # Severities that trigger notifications
    - critical
    - high
  bot_token: ""                      # Bot token for reading channel history (alert correlation)
```

### `opsgenie` — OpsGenie Integration

Alert correlation via OpsGenie v2 API. Auto-enables when `api_key` is set. **Disabled by default.**

```yaml
opsgenie:
  enabled: false                     # Auto-enables when api_key is set
  api_key: ""                        # OpsGenie API key — prefer VAIG_OPSGENIE__API_KEY
  base_url: "https://api.opsgenie.com"  # Use https://api.eu.opsgenie.com for EU
  team_ids: []                       # Filter alerts by team IDs
  alert_fetch_limit: 25              # Max alerts to fetch
```

### `email` — Email Notifications

SMTP email notifications for health findings. Auto-enables when `smtp_host`, `from_address`, and `recipients` are all set. **Disabled by default.**

```yaml
email:
  enabled: false                     # Auto-enables when smtp_host+from_address+recipients set
  smtp_host: ""                      # SMTP server hostname
  smtp_port: 587                     # SMTP port (default: 587 for STARTTLS)
  username: ""                       # SMTP auth username
  password: ""                       # SMTP auth password — prefer env var
  from_address: ""                   # Sender email address
  recipients: []                     # List of recipient email addresses
  use_tls: true                      # Enable STARTTLS
  timeout: 30                        # SMTP connection timeout (seconds)
  notify_on:                         # Severities that trigger email
    - critical
    - high
```

### `schedule` — Scheduled Health Scans

Automatically scans GKE clusters on a schedule. Auto-enables when `targets` is non-empty. **Disabled by default.**

```yaml
schedule:
  enabled: false                     # Auto-enables when targets are configured
  default_interval_minutes: 30       # Scan interval in minutes
  cron_expression: null              # Cron expression (overrides interval when set)
  targets: []                        # GKE clusters to scan
  alert_severity_threshold: HIGH     # Minimum severity to trigger alerts
  daily_max_analyses: 48             # Max analyses per day across all targets
  per_schedule_max_analyses: null    # Max analyses per individual schedule (null = no cap)
  max_concurrent_scans: 1            # Max concurrent scans
  store_results: true                # Persist scan results
  misfire_grace_time: 900            # Seconds grace for missed schedules (15 min)
  db_path: ~/.vaig/scheduler.db      # Scheduler state database
```

💡 **Example usage** — scan two clusters every hour:

```yaml
schedule:
  default_interval_minutes: 60
  alert_severity_threshold: HIGH
  targets:
    - cluster_name: prod-us-central1
      namespace: api
      skip_healthy: true
    - cluster_name: prod-eu-west1
      namespace: api
      skip_healthy: true
```

### `fleet` — Multi-Cluster Fleet Scanning

Scans multiple GKE clusters in a single command. Auto-enables when `clusters` is non-empty. **Disabled by default.**

```yaml
fleet:
  enabled: false                     # Auto-enables when clusters are configured
  parallel: false                    # Scan clusters in parallel
  max_workers: 4                     # Max parallel workers when parallel=true
  daily_budget_usd: 0.0              # Daily cost cap for fleet scans (0 = unlimited)
  clusters: []                       # List of clusters to scan
```

Each cluster in `clusters` supports:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Display name for the cluster. |
| `cluster_name` | string | required | GKE cluster name. |
| `project_id` | string | `""` | GCP project; falls back to `gcp.project_id`. |
| `location` | string | `""` | GKE location; falls back to `gcp.location`. |
| `namespace` | string | `""` | Target namespace (empty = `default_namespace`). |
| `all_namespaces` | boolean | `false` | Scan all namespaces. |
| `skip_healthy` | boolean | `true` | Skip clusters with no findings. |
| `kubeconfig_path` | string | `""` | Custom kubeconfig path. |
| `context` | string | `""` | Kubernetes context. |
| `impersonate_sa` | string | `""` | Service account to impersonate for this cluster. |

💡 **Example usage** — parallel fleet scan across three clusters:

```yaml
fleet:
  parallel: true
  max_workers: 3
  daily_budget_usd: 10.0
  clusters:
    - name: prod-us
      cluster_name: gke-prod-us-central1
      project_id: prod-project-123
      location: us-central1
      namespace: api
    - name: prod-eu
      cluster_name: gke-prod-eu-west1
      project_id: prod-project-123
      location: europe-west1
      namespace: api
    - name: staging
      cluster_name: gke-staging
      project_id: staging-project-456
      location: us-central1
      all_namespaces: true
      skip_healthy: false
```

### `github` — GitHub Integration

Read-only GitHub API integration for repo browsing tools. Auto-enables when `token` is set. **Disabled by default.**

```yaml
github:
  enabled: false                     # Auto-enables when token is set
  token: ""                          # GitHub Personal Access Token — prefer VAIG_GITHUB__TOKEN
  api_base: "https://api.github.com" # GitHub API base URL (or GitHub Enterprise URL)
  default_ref: main                  # Default branch ref
  allowed_repos: []                  # Allowlist of "owner/repo" strings (empty = all allowed)
  rate_limit_rpm: 60                 # Max requests per minute to GitHub API
  max_file_size_kb: 2048             # Max file size (KB) for repo_read_file
  cache_ttl_seconds: 300             # TTL for cached GitHub API responses (seconds)
```

### `knowledge` — External Knowledge Tools

Controls web search, document fetch, and RAG knowledge tools. Auto-enables when `web_search.api_key` is set. **Disabled by default.**

```yaml
knowledge:
  enabled: false                     # Auto-enables when web_search.api_key is set
  web_search:
    provider: tavily                 # Search provider (only "tavily" supported)
    api_key: ""                      # Tavily API key — prefer VAIG_KNOWLEDGE__WEB_SEARCH__API_KEY
    max_results: 5                   # Max search results (1–20)
    allowed_domains:                 # Domains allowed for search results
      - kubernetes.io
      - cloud.google.com
      - docs.datadoghq.com
      - argoproj.io
      - stackoverflow.com
      - github.com
  doc_fetch:
    max_bytes: 500000                # Max bytes per fetched document (1KB–5MB)
    timeout_seconds: 10              # HTTP fetch timeout (1–60 seconds)
    per_run_cap: 10                  # Max doc fetches per run (1–50)
  rag:
    enabled: true                    # Enable RAG knowledge base search
    top_k: 5                         # Number of RAG results to return (1–20)
```

### `training` — Fine-Tuning Pipeline

Controls extraction of rated examples from BigQuery and submission of fine-tuning jobs to Vertex AI. **Disabled by default.** Requires `[rag]` optional dependency group.

```yaml
training:
  enabled: false                     # Enable fine-tuning pipeline (default: false)
  base_model: gemini-2.5-flash       # Model to fine-tune
  min_examples: 50                   # Minimum examples required to start tuning
  max_examples: 10000                # Maximum examples to extract
  min_rating: 4                      # Minimum user rating for example inclusion (1–5)
  output_dir: training_data          # Local directory for JSONL output
  epochs: 3                          # Training epochs (1–10)
  learning_rate_multiplier: 1.0      # Learning rate multiplier (must be > 0)
  gcs_staging_prefix: training_data/ # GCS object prefix for staging data
```

### `platform` — Platform Mode

When `enabled`, the CLI operates in platform mode with centralized auth, config enforcement, and admin management. **Disabled by default.**

```yaml
platform:
  enabled: false                     # Enable platform mode (default: false)
  backend_url: ""                    # Backend API URL (required when enabled)
  org_id: ""                         # Organization ID for multi-tenant isolation
```

> **Note**: `backend_url` is required when `platform.enabled = true`. When `platform.org_id` is set, it is automatically copied to `export.org_id` for per-org data isolation in GCS.

### `ollama` — Ollama-Compatible Proxy

When enabled, registers Ollama-compatible API endpoints (`/api/generate`, `/api/chat`, `/api/tags`) in the FastAPI application, allowing Ollama clients (VS Code Continue, Cody, CLI tools) to use VAIG's Vertex AI backend. **Disabled by default.**

```yaml
ollama:
  enabled: false                     # Enable Ollama-compatible proxy endpoints
```

### `auto_activation` — Auto-Activation Policy

Controls whether capabilities can be auto-activated based on context. **Disabled by default.**

```yaml
auto_activation:
  enabled: false                     # Enable auto-activation (default: false)
  default_mode: auto_triggered       # Default mode for new capabilities
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_mode` | string | `"auto_triggered"` | Default `ActivationMode` for capabilities that don't specify one. Options: `auto_always`, `auto_triggered`, `auto_on_input`, `opt_in`. |

### `plugins` — Python Plugin Tools

```yaml
plugins:
  enabled: false                     # Set true to load custom Python tool plugins
  directories: []                    # Paths to plugin module directories
  # - ./plugins
  # - ~/.vaig/plugins
```

### `context_window` — Context Window Monitoring

Controls thresholds for warning and error states when the model's prompt token usage approaches the context window limit. This helps detect and diagnose situations where large prompts or long conversation histories are consuming most of the available context.

```yaml
context_window:
  warn_threshold_pct: 80.0           # Warn when usage exceeds this % (default: 80.0)
  error_threshold_pct: 95.0          # Error when usage exceeds this % (default: 95.0)
  context_window_size: 1048576       # Default context window in tokens (default: 1048576)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `warn_threshold_pct` | float | `80.0` | Percentage of context window usage that triggers a WARNING log. Must be between 0.0 and 100.0. |
| `error_threshold_pct` | float | `95.0` | Percentage of context window usage that triggers an ERROR log. Must be between 0.0 and 100.0. |
| `context_window_size` | integer | `1048576` | Default context window size in tokens. This is overridden per model when `models.available` entries specify a `context_window` value. |

> **Note**: `warn_threshold_pct` must be ≤ `error_threshold_pct`. Override via `VAIG_CONTEXT_WINDOW__WARN_THRESHOLD_PCT`, etc.

### `cache` — Response Caching

Controls the in-memory LRU cache for non-streaming, non-tool-use Gemini API responses. **Disabled by default** because LLM responses often depend on conversation context — enable only when appropriate (e.g. repeated stateless queries).

```yaml
cache:
  enabled: false                     # Enable response caching (default: false)
  max_size: 128                      # Maximum cached entries (default: 128)
  ttl_seconds: 300                   # Time-to-live per entry in seconds (default: 300)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable the in-memory response cache. Only caches non-streaming, non-tool-use responses. |
| `max_size` | integer | `128` | Maximum number of entries in the LRU cache. When full, the least-recently-used entry is evicted. |
| `ttl_seconds` | integer | `300` | Time-to-live in seconds for each cached entry. Entries older than this are discarded. |

### `export` — RAG Pipeline Export

Controls export of VAIG telemetry, tool calls, and health reports to GCP (BigQuery and GCS). **Disabled by default** — enable explicitly with `export.enabled = true` in config or `VAIG_EXPORT__ENABLED=true`.

```yaml
export:
  enabled: false                     # Enable RAG data export (default: false)
  gcp_project_id: ""                 # GCP project for BigQuery/GCS
  bigquery_dataset: vaig_analytics   # BigQuery dataset name (default: vaig_analytics)
  gcs_bucket: ""                     # GCS bucket for exported data
  gcs_prefix: "rag_data/"           # GCS object prefix (default: "rag_data/")
  auto_export_reports: false         # Auto-export health reports (default: false)
  auto_export_telemetry: false       # Auto-export telemetry data (default: false)
  vertex_rag_corpus_id: ""           # Vertex AI RAG corpus ID
  rag_enabled: false                 # Enable RAG corpus integration (default: false)
  rag_chunk_size: 1024               # Chunk size for RAG documents (default: 1024)
  rag_chunk_overlap: 200             # Overlap between chunks (default: 200)
  org_id: ""                         # Org ID for per-org GCS isolation (auto-set from platform.org_id)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable export pipeline. |
| `gcp_project_id` | string | `""` | GCP project ID for BigQuery and GCS operations. Also accepts the legacy alias `bigquery_project`. |
| `bigquery_dataset` | string | `"vaig_analytics"` | BigQuery dataset name where telemetry and report tables are created. |
| `gcs_bucket` | string | `""` | GCS bucket name for exported data (e.g. `"my-vaig-exports"`). |
| `gcs_prefix` | string | `"rag_data/"` | Object key prefix inside the GCS bucket. Automatically normalized to end with `/`. |
| `auto_export_reports` | boolean | `false` | When `true`, health reports are automatically exported after generation. |
| `auto_export_telemetry` | boolean | `false` | When `true`, telemetry events are exported on flush. |
| `vertex_rag_corpus_id` | string | `""` | Vertex AI RAG corpus resource ID for ingestion. Also accepts the legacy alias `rag_corpus_name`. |
| `rag_enabled` | boolean | `false` | Enable RAG corpus integration (chunking + ingestion to Vertex AI RAG). |
| `rag_chunk_size` | integer | `1024` | Chunk size in tokens for RAG document splitting. Must be a positive integer. |
| `rag_chunk_overlap` | integer | `200` | Token overlap between adjacent chunks. Must be non-negative and less than `rag_chunk_size`. |
| `org_id` | string | `""` | Organization ID for per-org GCS path isolation (`{gcs_prefix}{org_id}/`). Auto-set from `platform.org_id`. |

> **Note**: Requires the `[rag]` optional dependency group: `pip install 'vertex-ai-toolkit[rag]'`. Override individual fields via `VAIG_EXPORT__GCP_PROJECT_ID`, `VAIG_EXPORT__RAG_ENABLED`, etc.

---

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
  thinking:
    enabled: true                    # Enable Gemini thinking mode
    budget_tokens: null              # Use model default budget
    include_thoughts: true

models:
  default: gemini-2.5-pro
  fallback: gemini-2.5-flash

session:
  db_path: ~/.vaig/sessions.db
  auto_save: true
  max_history_messages: 50
  max_history_tokens: 28000
  summarization_threshold: 0.8

skills:
  enabled:
    - rca
    - code-review
    - service-health
    - threat-model
  external_dirs:
    - ./my-skills

agents:
  max_concurrent: 5
  parallel_tool_calls: true
  min_inter_call_delay: 0.5          # Gentle RPM throttle for quota-limited projects

coding:
  workspace_root: "."
  max_tool_iterations: 30
  confirm_actions: true
  pipeline_mode: false
  max_fix_iterations: 3              # Enable fix-forward loop
  allowed_commands:
    - python
    - pip
    - git
    - make
    - pytest
  git:
    enabled: true
    auto_branch: true
    auto_commit: true
    branch_prefix: "vaig/"
  workspace_rag:
    enabled: true
    max_chunks: 1000
    extensions:
      - .py
      - .md

context_window:
  warn_threshold_pct: 80.0
  error_threshold_pct: 95.0

cache:
  enabled: false                     # Enable for repeated stateless queries
  max_size: 128
  ttl_seconds: 300

gke:
  cluster_name: production-cluster
  default_namespace: api
  log_limit: 200
  metrics_interval_minutes: 30
  exec_enabled: false                # Set true to enable exec_command tool
  trends:
    enabled: true
    baseline_days: [7, 14]
    memory_critical_pct: 20.0

helm:
  enabled: true                      # Enable Helm release introspection

argocd:
  enabled: true
  context: gke_mgmt-project_us-central1_mgmt-cluster  # Management cluster

memory:
  enabled: true
  recurrence_threshold: 3
  chronic_threshold: 7
  outcome_tracking_enabled: true

investigation:
  enabled: true
  autonomous_mode: false             # Set true for fully autonomous runs
  max_iterations: 15
  memory_correction: true

hypothesis:
  enabled: true

self_correction:
  enabled: true
  max_repeated_calls: 2
  max_stale_iterations: 4

remediation:
  enabled: true
  dry_run: false
  auto_approve_safe: false
  tier_overrides:
    "kubectl rollout restart": "safe"

review:
  enabled: true
  require_review_for_remediation: true

budget:
  enabled: true
  max_cost_usd: 5.0
  warn_threshold: 0.8
  action: warn

safety:
  enabled: true
  settings:
    - category: HARM_CATEGORY_DANGEROUS_CONTENT
      threshold: BLOCK_MEDIUM_AND_ABOVE

telemetry:
  enabled: true
  buffer_size: 50

export:
  enabled: false
  gcp_project_id: my-analytics-project
  bigquery_dataset: vaig_analytics
  gcs_bucket: my-vaig-exports
  gcs_prefix: rag_data/
  rag_enabled: false

logging:
  level: INFO

retry:
  max_retries: 5
  initial_delay: 2.0
  rate_limit_initial_delay: 10.0
```

---

[Back to index](README.md)

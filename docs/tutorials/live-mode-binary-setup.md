# Live Mode — Binary Setup & Usage Tutorial

> Run autonomous GKE/GCP infrastructure investigations from a single binary — no Python, no pip.

---

## 1. Prerequisites

### GCP / Vertex AI
- A GCP project with the **Vertex AI API** enabled
- `gcloud` CLI installed and authenticated:
  ```sh
  gcloud auth application-default login
  gcloud config set project YOUR_PROJECT_ID
  ```

### Kubernetes
- `kubectl` installed and configured with cluster access
- A valid kubeconfig (`~/.kube/config` by default)
- Sufficient RBAC — vaig uses mostly **read-only** tools (get, list, describe, logs), but also registers
  write tools (`kubectl_scale`, `kubectl_restart`, `kubectl_label`, `kubectl_annotate`).
  Use cluster RBAC to restrict write access if needed.

### Required RBAC (minimal)
```yaml
rules:
  - apiGroups: ["", "apps", "autoscaling", "batch"]
    resources:
      - pods
      - deployments
      - replicasets
      - statefulsets
      - daemonsets
      - services
      - endpoints
      - events
      - namespaces
      - nodes
      - configmaps
      - horizontalpodautoscalers
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
```

---

## 2. Download & Install

Download from: **https://github.com/Isamel/vertex-ai-toolkit/releases/latest**

> **Supported platforms:** Linux amd64 (`vaig`) and Windows amd64 (`vaig.exe`).
> macOS users can install via `pip install vertex-ai-toolkit` (no binary is shipped for macOS).

### Linux

```sh
# Replace vX.Y.Z with the latest version from the releases page
curl -Lo vaig https://github.com/Isamel/vertex-ai-toolkit/releases/download/vX.Y.Z/vaig

# Make executable
chmod +x vaig

# Move to PATH
sudo mv vaig /usr/local/bin/

# Verify
vaig --version
vaig --help
```

### Windows

```powershell
# Replace vX.Y.Z with the latest version from the releases page
# (e.g. C:\tools\ — add that directory to System PATH if not already)
Invoke-WebRequest -Uri "https://github.com/Isamel/vertex-ai-toolkit/releases/download/vX.Y.Z/vaig.exe" -OutFile "C:\tools\vaig.exe"

# Verify
vaig.exe --version
vaig.exe --help
```

Expected output:
```
vaig, version 0.9.0
```

---

## 3. Configuration

Config is loaded in this priority order (highest wins):
1. Environment variables (`VAIG_*`)
2. Explicit path (`--config path/to/config.yaml`)
3. `./vaig.yaml` in the current working directory
4. `~/.vaig/config.yaml` (user home config — **recommended location**)
5. `config/default.yaml` (project defaults — shipped with the binary)
6. Built-in defaults

### Minimal config for live mode

Create `~/.vaig/config.yaml`:

```yaml
gcp:
  project_id: "my-gcp-project"   # Your GCP project ID
  location: "us-central1"        # Vertex AI region

gke:
  cluster_name: "my-cluster"     # GKE cluster name
  default_namespace: "production" # Default namespace for queries
```

That's it. vaig uses your existing `gcloud` ADC credentials and `kubectl` current context.

---

### Full annotated config

```yaml
# ~/.vaig/config.yaml

# ── GCP / Vertex AI ────────────────────────────────────────────
gcp:
  project_id: "my-gcp-project"   # Override: VAIG_GCP__PROJECT_ID or --project
  location: "us-central1"        # Override: VAIG_GCP__LOCATION

# ── AI Models ─────────────────────────────────────────────────
models:
  default: "gemini-2.5-pro"      # Main model for orchestration
  fallback: "gemini-2.5-flash"   # Used when default hits rate limits

# ── GKE Cluster ───────────────────────────────────────────────
gke:
  cluster_name: "my-cluster"         # Override: --cluster
  project_id: ""                     # GKE project (defaults to gcp.project_id)
  location: "us-central1"            # GKE region/zone
  default_namespace: "production"    # Override: --namespace
  kubeconfig_path: ""                # Empty = use ~/.kube/config
  context: ""                        # Empty = use current-context
  log_limit: 100                     # Max log lines per pod
  metrics_interval_minutes: 60       # Cloud Monitoring lookback window
  proxy_url: ""                      # For private clusters behind a proxy
  impersonate_sa: ""                 # SA email to impersonate for GKE/GCP APIs
  exec_enabled: false                # Allow exec_command in containers (off by default)

# ── Helm ──────────────────────────────────────────────────────
helm:
  enabled: true                      # Discover Helm-managed releases

# ── ArgoCD ────────────────────────────────────────────────────
# ArgoCD is DISABLED by default — requires explicit opt-in via config.
# Set enabled: true to enable, or null to auto-detect via CRD probe + annotation scan.
argocd:
  # enabled: false  # false = disabled (default), true = force-enable, null = auto-detect
  namespace: "argocd"

# ── Datadog (optional) ────────────────────────────────────────
datadog:
  enabled: false
  api_key: ""   # Or: VAIG_DATADOG__API_KEY
  app_key: ""   # Or: VAIG_DATADOG__APP_KEY
  site: "datadoghq.com"

# ── Agents ────────────────────────────────────────────────────
agents:
  max_concurrent: 3
  orchestrator_model: "gemini-2.5-pro"
  specialist_model: "gemini-2.5-flash"
  max_iterations_retry: 15
```

---

### GKE-specific setups

**Separate project for GKE vs Vertex AI:**
```yaml
gcp:
  project_id: "my-vertex-project"   # Vertex AI billing project

gke:
  project_id: "my-gke-project"      # GKE cluster lives here (different project)
  cluster_name: "prod-cluster"
  location: "us-east1"
```

**Use a non-default kubeconfig or context:**
```yaml
gke:
  kubeconfig_path: "/home/user/.kube/prod-config"
  context: "gke_my-project_us-central1_prod-cluster"
```

**Argo Rollouts on a separate cluster:**

If your Argo Rollouts control plane is on a different cluster than the workloads vaig connects to, the auto-detection CRD probe will always fail (the apiextensions endpoint is unreachable), causing ~5s timeouts per tool call. Force-enable it:

```yaml
gke:
  argo_rollouts_enabled: true   # Skip CRD check — always enable Rollouts tools
```

Or via environment variable:
```sh
export VAIG_GKE__ARGO_ROLLOUTS_ENABLED=true
```

---

### Datadog APM integration

**Basic setup (standard cloud):**
```yaml
datadog:
  api_key: "dd_api_xxxxxxxx"
  app_key: "dd_app_xxxxxxxx"
  site: "datadoghq.com"
```

**Behind a corporate proxy with SSL inspection:**
```yaml
datadog:
  api_key: "dd_api_xxxxxxxx"
  app_key: "dd_app_xxxxxxxx"
  site: "datadoghq.com"
  ssl_verify: "/etc/ssl/certs/corporate-ca.crt"  # Path to CA bundle
  # ssl_verify: false  # Last resort — disables verification entirely
```

**Using environment variables (recommended for secrets):**
```sh
export VAIG_DATADOG__API_KEY="dd_api_xxxxxxxx"
export VAIG_DATADOG__APP_KEY="dd_app_xxxxxxxx"
```
When both keys are set, Datadog is auto-enabled — no need to set `enabled: true`.

**APM-only setup (no DaemonSet Agent):**
```yaml
datadog:
  api_key: "..."
  app_key: "..."
  metric_mode: "apm"          # Use trace.* metrics instead of kubernetes.*
  default_lookback_hours: 8.0 # Increase for low-traffic services
```

---

### Environment variables reference

Any config value can be overridden via env vars using the `VAIG_` prefix with `__` as the nesting delimiter:

| Variable | Config equivalent | Example |
|---|---|---|
| `VAIG_GCP__PROJECT_ID` | `gcp.project_id` | `my-project` |
| `VAIG_GCP__LOCATION` | `gcp.location` | `us-central1` |
| `VAIG_GKE__CLUSTER_NAME` | `gke.cluster_name` | `prod-cluster` |
| `VAIG_GKE__DEFAULT_NAMESPACE` | `gke.default_namespace` | `production` |
| `VAIG_MODELS__DEFAULT` | `models.default` | `gemini-2.5-flash` |
| `VAIG_DATADOG__API_KEY` | `datadog.api_key` | `dd_api_...` |
| `VAIG_DATADOG__APP_KEY` | `datadog.app_key` | `dd_app_...` |
| `VAIG_LOGGING__LEVEL` | `logging.level` | `DEBUG` |

> **Note:** `GOOGLE_CLOUD_PROJECT` is respected by `gcloud` ADC but vaig uses `VAIG_GCP__PROJECT_ID` for its own config. Set both for full compatibility.

---

## 4. Basic Usage

### Health check

```sh
vaig live "check overall health of the payment-service deployment"
```

### Target a specific namespace

```sh
vaig live --namespace production "are there any CrashLoopBackOff pods?"
```

Windows:
```powershell
vaig.exe live --namespace production "are there any CrashLoopBackOff pods?"
```

### Watch mode — re-run every N seconds

```sh
# Refresh every 60 seconds (minimum is 10s)
vaig live --watch 60 "check pod restart counts in production"
# Press Ctrl+C to stop
```

### Use a specific skill

```sh
# Available skills: rca, anomaly, service-health, log-analysis, error-triage, slo-review, postmortem, etc.
vaig live --skill service-health "full health check for checkout-service"
vaig live --skill rca "root cause analysis for payment-service 503s"
vaig live --skill log-analysis "analyze error patterns in the last hour"
```

### Auto-detect the best skill

```sh
# vaig reads the question and routes to the most relevant skill automatically
vaig live --auto-skill "why is the checkout service returning 503s?"
```

### Dry run — see what would happen without executing

```sh
vaig live --dry-run "investigate high latency in order-service"
```

Expected output:
```
┌─ Dry Run — vaig live "investigate high latency in order-service" ─┐
│ Configuration                                                      │
│  Cluster    prod-cluster                                          │
│  Namespace  production                                            │
│  Project    my-gcp-project                                        │
│  Model      gemini-2.5-pro                                        │
│                                                                   │
│ Available tools (24): kubectl_get, kubectl_describe, kubectl_logs,  │
│   get_events, gcloud_logging_query, ...                              │
│                                                                   │
│ Estimated cost: depends on tool usage (typically $0.02-0.10/run)  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5. Advanced Usage

### Export output

**Save to Markdown:**
```sh
vaig live --format md --output report.md "full health check for my-service"
```

**Save to JSON:**
```sh
vaig live --format json --output report.json "check HPA status"
```

**Generate HTML report and open in browser:**
```sh
vaig live --format html --open "service health investigation for payment-service"
```

**Generate HTML report to a specific path:**
```sh
vaig live --format html --output /tmp/report.html "check production health"
```

> When using `--format html` without `--output`, vaig saves `vaig-report-<timestamp>.html` in the current directory.

### Override the model

```sh
# Use Flash for faster (cheaper) investigations
vaig live --model gemini-2.5-flash "quick check — any pods restarting?"

# Use Pro for deep analysis
vaig live --model gemini-2.5-pro --skill rca "root cause for payment latency spike"
```

### Override cluster/project at runtime

```sh
# Switch cluster without changing config
vaig live --cluster staging-cluster --namespace staging "check staging environment"

# Different GCP project for Vertex AI, separate GKE project
vaig live --project my-vertex-project --gke-project my-gke-project "check prod cluster"

# Different region
vaig live --location europe-west4 "check EU cluster health"
```

### Custom config file

```sh
# Use a project-specific config
vaig live --config ./ops/vaig-prod.yaml "check production"

# Or use vaig.yaml in the current directory (auto-loaded)
# Just create ./vaig.yaml and vaig will pick it up
```

### Multiple services — iterate via watch

```sh
# Monitor a service continuously during an incident
vaig live --watch 30 --namespace production --skill service-health \
  "monitor checkout-service health and alert on any degradation"
```

### Argo Rollouts investigation

```sh
# Requires Argo Rollouts CRDs present or argo_rollouts_enabled: true in config
vaig live --namespace production "show rollout status for payment-service"
vaig live --skill rca "why did the canary rollout for order-service pause?"
```

### Enable verbose output

```sh
# INFO level — see agent steps
vaig live --verbose "check service health"

# DEBUG level — see full tool args, paths, and tracebacks
vaig live --debug "investigate pod crashes"
```

### Suppress terminal bell

```sh
# Useful in scripts or noisy terminals
vaig live --no-bell "check cluster health"
```

### Analyze all namespaces (cost estimation)

```sh
# Include all non-system namespaces in GKE cost estimation
vaig live --all-namespaces "estimate cluster resource costs"
```

---

## 6. Troubleshooting

### `No GCP project specified and could not auto-detect from environment`

vaig can't determine your GCP project. Set it via:
```sh
# Option 1: config
echo 'gcp:\n  project_id: "my-project"' > ~/.vaig/config.yaml

# Option 2: env var
export VAIG_GCP__PROJECT_ID=my-project

# Option 3: CLI flag
vaig live --project my-project "..."
```

---

### `Error: No infrastructure tools available!`

The GKE tools could not be loaded. This usually means a kubeconfig issue:

1. Verify `kubectl` works: `kubectl get pods -n production`
2. Verify the cluster context: `kubectl config current-context`
3. If using a non-default kubeconfig, set `gke.kubeconfig_path` in config

---

### `RESOURCE_EXHAUSTED` / rate limit errors

You're hitting Vertex AI quota limits:
```sh
# Switch to Flash (higher quota) for the current run
vaig live --model gemini-2.5-flash "..."

# Or set Flash as default in config
# models.default: "gemini-2.5-flash"
```
vaig automatically falls back to `models.fallback` after repeated failures — you can also configure `agents.max_failures_before_fallback`.

---

### SSL / proxy issues

**Kubernetes API behind a proxy:**
```yaml
gke:
  proxy_url: "http://proxy.corp.example.com:8080"
```

**Datadog API behind SSL inspection proxy:**
```yaml
datadog:
  ssl_verify: "/etc/ssl/certs/corporate-ca.crt"
```

**Vertex AI / GCP behind a proxy:**

Set standard proxy environment variables before running vaig:
```sh
export HTTPS_PROXY=http://proxy.corp.example.com:8080
export NO_PROXY=localhost,169.254.169.254  # GCP metadata server
vaig live "..."
```

---

### Timeout errors from GKE API

For private clusters or slow API servers, increase the request timeout:
```yaml
gke:
  request_timeout: 60  # Default is 30s
```

If Argo Rollouts CRD checks are causing ~5s delays per call (separate cluster scenario):
```yaml
gke:
  argo_rollouts_enabled: true   # Skip the CRD probe entirely — recommended fix for separate-cluster setups
  crd_check_timeout: 3          # Reduce CRD existence check timeout (default is 5s)
```

> **Note:** `crd_check_timeout: 5` is already the default. Use `argo_rollouts_enabled: true` to skip
> the CRD check completely when Argo Rollouts is on a different cluster.

---

### Permission errors (`PERMISSION_DENIED`)

Verify your ADC has the required roles:
```sh
# Check current identity
gcloud auth list
gcloud config get-value project

# Required roles for Vertex AI
# roles/aiplatform.user

# Required roles for Cloud Logging / Monitoring (GKE observability)
# roles/logging.viewer
# roles/monitoring.viewer

# Re-authenticate if needed
gcloud auth application-default login
```

For GKE RBAC issues, check that your K8s user/SA has the read-only RBAC rules listed in Section 1.

---

### Watch mode stops with exit code 1

In watch mode, agent errors don't kill the loop — they log and continue. If the loop exits immediately:

```sh
# Run once first to catch config errors
vaig live "..." 
# Then add --watch once it works
vaig live --watch 30 "..."
```

---

### Windows: binary not found after adding to PATH

Restart your terminal (or PowerShell session) after modifying `PATH`. Alternatively, use the full path:
```powershell
C:\tools\vaig.exe live "check pod health"
```

---

## Quick Reference

```sh
# Basic
vaig live "QUESTION"

# With namespace + cluster
vaig live --namespace production --cluster prod-gke "QUESTION"

# With skill
vaig live --skill service-health "QUESTION"

# Watch mode
vaig live --watch 60 "QUESTION"

# Export HTML and open
vaig live --format html --open "QUESTION"

# Dry run
vaig live --dry-run "QUESTION"

# Debug
vaig live --debug "QUESTION"
```

All flags:

| Flag | Short | Description | Default |
|---|---|---|---|
| `--config` | `-c` | Path to config YAML | Auto-detected |
| `--model` | `-m` | Model to use | `gemini-2.5-pro` |
| `--output` | `-o` | Save output to file | — |
| `--format` | | Export format: `json`, `md`, `html` | — |
| `--skill` | `-s` | SRE skill to apply | — |
| `--auto-skill` | | Auto-detect best skill | `false` |
| `--cluster` | | GKE cluster name | Config value |
| `--namespace` | | Kubernetes namespace | Config `gke.default_namespace` |
| `--project` / `--project-id` | `-p` | GCP project ID | Config `gcp.project_id` |
| `--location` | | GCP location | Config `gcp.location` |
| `--gke-project` | | GKE project ID | Defaults to `--project` |
| `--gke-location` | | GKE cluster location | Config `gke.location` |
| `--watch` | `-w` | Re-execute every N seconds (min 10) | — |
| `--dry-run` / `--dry` | | Show plan without running | `false` |
| `--verbose` | `-V` | INFO level logging | `false` |
| `--debug` | `-d` | DEBUG level logging | `false` |
| `--summary` | | Compact summary output | `false` |
| `--no-bell` | | Suppress terminal bell | `false` |
| `--open` | `-O` | Open HTML in browser (requires `--format html`) | `false` |
| `--all-namespaces` | | Include all namespaces in cost estimation | `false` |

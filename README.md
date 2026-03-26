# VAIG — Vertex AI Gemini Toolkit

Multi-agent AI assistant powered by **Google Vertex AI Gemini** models. Interactive CLI with pluggable skills for incident analysis, anomaly detection, code migration, GKE diagnostics, and more.

## Features

- **Interactive REPL** — chat with Gemini models in your terminal with slash commands
- **Multi-model support** — switch between Gemini 2.5 Pro, Flash, and more on the fly
- **Multimodal context** — attach code, PDFs, images, audio, and Pentaho ETL files
- **Session persistence** — save and resume conversations (SQLite-backed)
- **Pluggable skills** — specialized multi-agent workflows:
  - **RCA** — Root Cause Analysis with 5 Whys + Fishbone methodology
  - **Anomaly Detection** — detect unusual patterns in logs, metrics, and data
  - **ETL Migration** — migrate ETL pipelines between platforms (e.g., Pentaho KTR/KJB → AWS Glue PySpark)
  - **Code Migration** — migrate source code between programming languages (e.g., Python → Go) with YAML-driven idiom mappings
  - **Greenfield** — scaffold new projects from scratch with a 6-stage pipeline
  - **Service Health** — comprehensive GKE service diagnostics (4-agent pipeline with two-pass gathering)
  - Plus 25+ built-in skills for SRE, DevOps, and platform engineering
- **CodingSkillOrchestrator** — 3-agent orchestration (Planner → Implementer → Verifier) for complex coding tasks; activate with `--pipeline` on `vaig ask --code`
- **Multi-agent orchestration** — skills spawn specialized agents with different roles and models
- **Async fanout** — true parallel agent execution via ThreadPoolExecutor for multi-agent workflows
- **Cost tracking** — per-request token and cost tracking with live CLI display and export report summaries; GKE Autopilot workload cost estimation with per-container breakdown and namespace-level waste/efficiency metrics
- **Usage telemetry** — local-only event capture (tool calls, API usage, errors) with `vaig stats` for analytics and export
- **Token budget enforcement** — configurable spending limits per session with warn/stop actions
- **Plugin tool registration** — extend the toolkit with custom Python modules or MCP servers
- **Safety settings** — configurable harm category thresholds for Gemini API content filtering
- **Dual-auth** — separate GCP project authentication for Vertex AI vs GKE observability via SA impersonation
- **Runtime config switching** — change GCP project, location, or GKE cluster at runtime without restarting
- **GKE live diagnostics** — connect to GKE clusters for pod inspection, log analysis, and metric queries (23 base tools)
- **Helm introspection** — read-only Helm release status, history, and values via K8s secrets (4 tools, enabled by default)
- **ArgoCD integration** — read-only ArgoCD Application status, sync history, diff, and managed resources (5 tools, opt-in)
- **ASM/Istio mesh introspection** — service mesh overview, traffic config, security policies, and sidecar status
- **ToolCallStore** — per-tool-call result storage (JSONL) for post-run analysis, debugging, and auditing
- **Configurable auth** — Application Default Credentials (ADC) for GKE, service account impersonation for local dev
- **Cross-platform** — UTF-8 enforcement on all file I/O for Windows compatibility

## Requirements

- Python 3.11+
- A Google Cloud project with Vertex AI API enabled
- Authentication configured (see [Authentication](#authentication))

## Installation

```bash
# From source
pip install -e .

# With live infrastructure support (GKE, Cloud Logging, Cloud Monitoring)
pip install -e ".[live]"

# With dev dependencies (includes live deps + pytest, ruff, mypy)
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
  -p, --project TEXT   GCP project ID (overrides config)
  --location TEXT      GCP location (overrides config)
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
  -o, --output PATH    Save response to file (includes cost summary footer)
  -p, --project TEXT   GCP project ID (overrides config)
  --location TEXT      GCP location (overrides config)
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

### `vaig stats show`

Show usage telemetry summary (event counts, tokens, costs). Supports `--since` and `--until` date filters.

### `vaig stats export`

Export telemetry events as JSONL or CSV. Supports `--type`, `--since`, `--format`, and `--output` filters.

### `vaig stats clear --days N --confirm`

Delete telemetry events older than N days.

## REPL Slash Commands

Inside the interactive chat (`vaig chat`):

| Command                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `/add <path>`            | Add a file or directory as context                |
| `/model <name>`          | Switch to a different model                       |
| `/skill <name>`          | Activate a skill                                  |
| `/phase <phase>`         | Set the skill phase (analyze/plan/execute)        |
| `/agents`                | Show active agents                                |
| `/cost`                  | Show session cost summary and budget status       |
| `/project [id]`          | Show or switch the active GCP project             |
| `/location [name]`       | Show or switch the GCP location                   |
| `/cluster [name] [ctx]`  | Show or switch the GKE cluster                    |
| `/config`                | Show current configuration snapshot               |
| `/sessions`              | List saved sessions                               |
| `/new [name]`            | Start a new session                               |
| `/load <id>`             | Load a previous session                           |
| `/clear`                 | Clear current context files                       |
| `/context`               | Show loaded context files                         |
| `/help`                  | Show all commands                                 |
| `/quit`                  | Exit the REPL                                     |

## Runtime Config Switching

Change GCP project, location, or GKE cluster at runtime without restarting the CLI. The Gemini client is automatically reinitialized, and settings are validated with rollback on failure.

### REPL commands

Use slash commands inside `vaig chat` to switch config on the fly:

```
# Show current project (and list available_projects from config)
/project

# Switch to a different GCP project
/project my-other-project

# Show current location
/location

# Switch location
/location europe-west1

# Show current GKE cluster
/cluster

# Switch cluster (optional kubeconfig context as second arg)
/cluster staging-cluster gke_my-project_us-east1_staging-cluster

# Show full config snapshot (project, location, model, cluster, etc.)
/config
```

When `available_projects` is defined in your config YAML, `/project` without arguments lists them and marks the current one. Tab completion in the REPL covers all slash commands including `/project`, `/location`, `/cluster`, and `/config`.

### CLI flags

Override project and location from the command line on `ask`, `chat`, and `live` subcommands:

```bash
# Start a chat session targeting a specific project and location
vaig chat --project my-other-project --location europe-west1

# Single-shot question against a different project
vaig ask "List running pods" -p my-gke-project

# Live infrastructure investigation with project override
vaig live "Check pod health" --project infra-project --location us-east1
```

The `--project`/`-p` flag overrides both `gcp.project_id` and `gke.project_id`. The `--location` flag overrides `gcp.location`.

### Prompt prefix

The REPL prompt displays the active project so you always know which project you're targeting:

```
[my-project] [gemini-2.5-pro] > what pods are running?
```

### After switching

When you switch project or location, the Gemini client is reinitialized automatically. However, previously created tools and agents may still hold references to the old config. The CLI warns you after each switch:

```
Note: Tools and agents will use the new project on next creation.
Use /clear to reset agents now.
```

For GKE cluster switches, internal Kubernetes caches are cleared so the next tool invocation picks up the new cluster. Infrastructure tools will warn similarly:

```
Note: Infrastructure tools will use the new cluster on next invocation.
```

## Configuration

VAIG uses layered configuration: **environment variables > YAML config > defaults**.

### Config file

Default location: `config/default.yaml` or specify with `--config`.

```yaml
gcp:
  project_id: "my-project"
  location: "us-central1"
  available_projects:             # Optional: catalog of GCP projects you work with
    # - project_id: "my-vertex-project"
    #   description: "Vertex AI / Gemini billing"
    #   role: "vertex-ai"
    # - project_id: "my-gke-project"
    #   description: "GKE clusters and monitoring"
    #   role: "gke"

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

telemetry:
  enabled: true            # Disable with false; or env VAIG_TELEMETRY_ENABLED=false
  buffer_size: 50          # Events buffered before flush to SQLite

skills:
  enabled: [rca, anomaly, migration]
  custom_dir: null  # Path to custom skills directory

helm:
  enabled: true       # Helm release introspection (4 read-only tools)

argocd:
  enabled: false      # ArgoCD Application introspection (5 read-only tools)
  namespace: argocd   # Namespace where ArgoCD Applications live
  context: ""         # kubeconfig context for ArgoCD management cluster
```

### Environment variables

All settings can be overridden with `VAIG_` prefixed env vars:

```bash
export VAIG_GCP__PROJECT_ID="my-project"
export VAIG_GCP__LOCATION="us-central1"
export VAIG_AUTH__MODE="impersonate"
export VAIG_AUTH__IMPERSONATE_SA="my-sa@my-project.iam.gserviceaccount.com"
export VAIG_MODELS__DEFAULT="gemini-2.5-flash"
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
export VAIG_AUTH__MODE="impersonate"
export VAIG_AUTH__IMPERSONATE_SA="vaig-sa@my-project.iam.gserviceaccount.com"
```

Required IAM roles on the service account:
- `roles/aiplatform.user` — Vertex AI API access
- Your user needs `roles/iam.serviceAccountTokenCreator` on the SA

### Dual-Auth (Separate Projects for Vertex AI and GKE)

For environments where Vertex AI billing and GKE clusters live in different GCP projects, configure independent SA impersonation:

```yaml
# config/default.yaml
gcp:
  project_id: "vertex-ai-project"    # Used for Gemini API calls

auth:
  mode: "impersonate"
  impersonate_sa: "vertex-sa@vertex-ai-project.iam.gserviceaccount.com"

gke:
  project_id: "gke-infra-project"    # Used for GKE observability APIs
  cluster_name: "prod-cluster"
  impersonate_sa: "gke-sa@gke-infra-project.iam.gserviceaccount.com"
```

When `gke.impersonate_sa` is set, GKE tools (Cloud Logging, Cloud Monitoring, GKE cluster API) use that SA instead of `auth.impersonate_sa`. This enables true dual-auth: one identity for Vertex AI, another for GKE observability.

Required IAM roles for the GKE SA:
- `roles/logging.viewer` — Cloud Logging read access
- `roles/monitoring.viewer` — Cloud Monitoring read access
- `roles/container.viewer` — GKE cluster read access
- Your ADC identity needs `roles/iam.serviceAccountTokenCreator` on both SAs

## Cost Tracking and Budget

VAIG tracks token usage and estimated costs for every API request.

### Live feedback

After each agent execution in the CLI, a cost summary line is displayed:

```
Tokens: 1,234 in / 567 out | Cost: $0.0042
```

### Export reports

When saving output with `-o`, a `## Cost & Usage Summary` section is appended:

```bash
vaig ask "Analyze this service" -s service-health -f app.log -o report.md
```

### Budget enforcement

Set spending limits per session:

```yaml
budget:
  enabled: true
  max_cost_usd: 5.0          # Maximum spend per session
  warn_threshold: 0.8        # Warn at 80% of budget
  action: "warn"             # "warn" or "stop" — stop blocks further requests
```

Use `/cost` in the REPL to check current session spend at any time.

## Safety Settings

Configure content filtering thresholds for Gemini API responses:

```yaml
safety:
  enabled: true
  settings:
    - category: "HARM_CATEGORY_HARASSMENT"
      threshold: "BLOCK_MEDIUM_AND_ABOVE"
    - category: "HARM_CATEGORY_HATE_SPEECH"
      threshold: "BLOCK_MEDIUM_AND_ABOVE"
    - category: "HARM_CATEGORY_SEXUALLY_EXPLICIT"
      threshold: "BLOCK_MEDIUM_AND_ABOVE"
    - category: "HARM_CATEGORY_DANGEROUS_CONTENT"
      threshold: "BLOCK_MEDIUM_AND_ABOVE"
```

Available thresholds: `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE`, `OFF`.

> **Note:** `BLOCK_NONE`/`OFF` may be rejected by some Vertex AI projects. `HARM_CATEGORY_CIVIC_INTEGRITY` is not supported in all regions.

Set `safety.enabled: false` to skip sending safety settings entirely (uses Gemini server defaults).

## Plugin Tool Registration

Extend the toolkit with external tools via MCP servers or Python plugin modules.

### MCP servers

```yaml
mcp:
  enabled: true
  auto_register: true       # Auto-register MCP tools into agent pipelines
  servers:
    - name: "filesystem"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      description: "Filesystem access via MCP"
    - name: "github"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
      description: "GitHub API via MCP"
```

### Python plugins

```yaml
plugins:
  enabled: true
  directories:
    - "./plugins"
    - "~/.vaig/plugins"
```

Place Python modules in the configured directories. Each module can export tool functions that are auto-discovered and registered into the agent pipeline.

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

#### ETL Migration

Agents: `code_analyzer`, `code_generator`, `migration_validator`

```bash
vaig ask "Migrate this Pentaho job to AWS Glue" -s migration -f transform.ktr
```

#### Code Migration (Language-to-Language)

Agents: `analyzer`, `transformer`, `validator`

Migrates source code between programming languages using YAML-driven idiom and dependency mappings. For example, Python → Go. Use `--code` for filesystem access during transformation.

```bash
vaig ask "Migrate this Python module to Go" -s code-migration -f service.py --code
```

> **Note:** For ETL pipeline migration (Pentaho → AWS Glue), use the `migration` skill instead.

#### Greenfield Project Scaffolding

Agents: 6-stage pipeline (requirements → architecture_decision → project_spec → scaffold → implement → verify)

```bash
vaig ask "Create a new Go REST API service" -s greenfield --code
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

## Anthos Service Mesh / Istio Introspection

VAIG includes 4 read-only diagnostic tools for GKE clusters running Anthos Service Mesh (ASM) or open-source Istio. All tools are safe to run in production — they only read cluster state, never mutate it.

### Prerequisites

- A GKE cluster with ASM or Istio installed
- RBAC permissions to read Istio CRDs (`networking.istio.io`, `security.istio.io`) and pods/namespaces
- When ASM/Istio is **not** installed, tools return `"No service mesh detected"` gracefully — no errors

### Tools

#### `get_mesh_overview`

Shows mesh presence, version, control plane health, and per-namespace sidecar injection status.

| Parameter       | Type   | Default | Description                          |
| --------------- | ------ | ------- | ------------------------------------ |
| `namespace`     | `str`  | `""`    | Filter to a single namespace         |
| `force_refresh` | `bool` | `false` | Bypass 30s TTL cache                 |

What the agent can diagnose:
- Is a service mesh installed? Managed ASM or open-source Istio?
- What Istio version is running? Is istiod healthy (replica count)?
- Which namespaces have sidecar injection enabled vs disabled?
- Are any namespaces using a specific revision label (e.g., `asm-managed-rapid`)?

#### `get_mesh_config`

Lists VirtualServices, DestinationRules, and Gateways with traffic routing details.

| Parameter       | Type   | Default | Description                          |
| --------------- | ------ | ------- | ------------------------------------ |
| `namespace`     | `str`  | `""`    | Filter to a single namespace         |
| `force_refresh` | `bool` | `false` | Bypass 30s TTL cache                 |

What the agent can diagnose:
- Traffic splitting: which VirtualService routes what percentage to which subset?
- Gateway configuration: ports, TLS mode, host bindings
- DestinationRule policies: load balancer settings, connection pools, outlier detection
- Missing or misconfigured routing rules causing 503s or unexpected traffic behavior

#### `get_mesh_security`

Shows PeerAuthentication (mTLS), AuthorizationPolicy (RBAC), and RequestAuthentication (JWT) resources.

| Parameter       | Type   | Default | Description                          |
| --------------- | ------ | ------- | ------------------------------------ |
| `namespace`     | `str`  | `""`    | Filter to a single namespace         |
| `force_refresh` | `bool` | `false` | Bypass 30s TTL cache                 |

What the agent can diagnose:
- mTLS enforcement: which namespaces use STRICT vs PERMISSIVE mode?
- Port-level mTLS overrides that might create security gaps
- Authorization policies: who can call what (principals, namespaces, methods, paths)?
- JWT validation rules: expected issuers and audiences
- Conflicting allow/deny policies that block legitimate traffic

#### `get_sidecar_status`

Checks per-pod sidecar injection status and identifies anomalies.

| Parameter       | Type   | Default | Description                          |
| --------------- | ------ | ------- | ------------------------------------ |
| `namespace`     | `str`  | `""`    | Filter to a single namespace         |
| `force_refresh` | `bool` | `false` | Bypass 30s TTL cache                 |

What the agent can diagnose:
- Which pods have the `istio-proxy` sidecar and which don't?
- **MISSING** anomaly: pod in an injection-enabled namespace without a sidecar (likely needs a restart)
- **UNEXPECTED** anomaly: pod with a sidecar in a non-injected namespace
- Sidecar version mismatches across pods
- Coverage statistics: what percentage of pods are in the mesh?

### Use Cases

- **mTLS debugging** — "Why can't service A talk to service B?" → check PeerAuthentication mode, look for STRICT mTLS blocking a non-mesh client
- **Traffic routing analysis** — "Why is canary getting 0% traffic?" → inspect VirtualService weight distribution and DestinationRule subsets
- **Sidecar injection issues** — "New deployment isn't in the mesh" → verify namespace labels and check for MISSING anomalies in sidecar status
- **Mesh health check** — "Is our mesh healthy?" → get overview for version, istiod status, and injection coverage across namespaces
- **Security audit** — "What authorization policies are in place?" → list all AuthorizationPolicies with their allow/deny rules

### Cross-Cutting Behavior

- **Auto-detects CRD API versions**: tries v1 → v1beta1 → v1alpha3 for each resource type
- **Fail-open per resource type**: if VirtualServices return 403 but Gateways work, you still get Gateway results with a warning
- **30s TTL caching**: repeated calls within 30 seconds return cached results (use `force_refresh` to bypass)
- **Truncation**: large clusters are capped at 50 resources per type to keep responses manageable
- **Works with both** managed ASM and open-source Istio installations

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
│   │   ├── config.py          # Pydantic Settings (layered config)
│   │   ├── config_switcher.py # Runtime config switching (project, location, cluster)
│   │   ├── auth.py            # ADC + SA impersonation + dual-auth
│   │   ├── client.py          # GeminiClient (streaming, multi-model)
│   │   ├── cost_tracker.py    # Per-request cost tracking (SQLite)
│   │   ├── prompt_defense.py  # wrap_untrusted_content() for prompt injection defense
│   │   └── tool_call_store.py # ToolCallStore — per-tool-call JSONL recording
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
│   │   ├── migration/      # ETL Migration skill (Pentaho → AWS Glue)
│   │   ├── code_migration/ # Code Migration skill (language-to-language, e.g. Python → Go)
│   │   │   └── idioms/     # YAML idiom + dependency mappings
│   │   ├── greenfield/     # Greenfield project scaffolding (6-stage pipeline)
│   │   └── ...             # 25+ additional built-in skills
│   ├── agents/
│   │   ├── base.py            # AgentRole, AgentConfig, BaseAgent ABC
│   │   ├── specialist.py      # SpecialistAgent (wraps GeminiClient)
│   │   ├── orchestrator.py    # Multi-agent coordination + async fanout
│   │   ├── coding_pipeline.py # CodingSkillOrchestrator (Planner→Implementer→Verifier)
│   │   └── registry.py        # Agent factory
│   ├── tools/
│   │   ├── base.py           # ToolResult, ToolCallRecord
│   │   ├── file_tools.py     # File tools incl. verify_completeness
│   │   ├── gke_tools.py      # GKE tool wrappers (legacy)
│   │   └── gke/
│   │       ├── _cache.py       # TTL cache for discovery/mesh resources
│   │       ├── _clients.py     # K8s client factory + auth
│   │       ├── _formatters.py  # Output formatters for GKE resources
│   │       ├── _registry.py    # Tool registration (23 base + 4 Helm + 5 ArgoCD)
│   │       ├── _resources.py   # Core resource readers
│   │       ├── argocd.py       # ArgoCD Application introspection (5 tools)
│   │       ├── diagnostics.py  # Pod/workload diagnostics
│   │       ├── discovery.py    # Cluster discovery tools
│   │       ├── helm.py         # Helm release introspection (4 tools)
│   │       ├── kubectl.py      # kubectl-style operations + get_labels
│   │       ├── mesh.py         # ASM/Istio mesh introspection tools
│   │       ├── mutations.py    # Write operations (scale, restart, etc.)
│   │       └── security.py     # RBAC check + exec_command
│   └── cli/
│       ├── app.py          # Typer commands
│       └── repl.py         # Interactive REPL (prompt-toolkit)
├── tests/                  # 60+ test files
└── .github/workflows/
    ├── ci.yml              # Test + Lint + Type check on PR/push
    └── build.yml           # PyInstaller standalone binary builds
```

## Development

```bash
# Install with dev deps (includes runtime + live infrastructure deps)
pip install -e ".[dev]"

# Lint
ruff check src/ tests/

# Format
ruff format src/

# Type check
mypy src/vaig/

# Test
pytest

# Test with timeout (CI default)
python -m pytest tests/ -x -q --tb=short --timeout=120
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **CI** (`.github/workflows/ci.yml`) — Runs on PR and push to `main`: pytest, ruff lint, mypy type check
- **Build** (`.github/workflows/build.yml`) — PyInstaller standalone binary builds for Linux and Windows

## License

MIT

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-03-27

### Added
- Datadog `metric_mode` config option — choose `"k8s_agent"` (default, kubernetes.* metrics) or `"apm"` (trace.* metrics for APM-only setups without DaemonSet Agent) (#124)
- Datadog `cluster_name_override` config option — override the auto-detected cluster name tag used in Datadog metric queries (e.g. when Datadog agent tags the cluster differently from the GKE name) (#124)
- Datadog `default_lookback_hours` config option — configurable lookback window for APM trace queries (default: 4 hours); increase for low-traffic services (#124)
- Datadog `ssl_verify` config option — set to `false` or a CA bundle path to support corporate proxy environments with SSL inspection (#121)
- GKE `crd_check_timeout` config option — short timeout (default: 5s) for CRD existence probes to prevent ~84s hangs when the apiextensions endpoint is unreachable (#125)
- GKE `argo_rollouts_enabled` config option (under `gke:`) — set to `true` to skip CRD check and force-enable Argo Rollouts tools when Argo is deployed on a separate cluster (#125)
- Unknown tool fuzzy matching — when the LLM hallucinates a tool name, agents now return helpful suggestions with a scored list of close matches and the full tool registry (#127)
- `CodingSkillOrchestrator` — 3-agent coding pipeline (Planner → Implementer → Verifier) for complex coding tasks — activate with `--pipeline` flag on `vaig ask --code` (#111)
- `GreenfieldSkill` with 6-stage pipeline for new project scaffolding from scratch: Requirements → Architecture Decision → Project Spec → Scaffold → Implement → Verify (`skills/greenfield/`) (#111)
- `CodeMigrationSkill` with 6-phase state machine (Inventory → Semantic Map → Spec → Implement → Verify → Report) for language-to-language code migration, e.g., Python → Go (`skills/code_migration/`) — distinct from the ETL `migration` skill (Pentaho → AWS Glue) (#110)
- Python-to-Go idiom mappings: 12 idiom translations + 17 dependency mappings (YAML-driven, `skills/code_migration/idioms/python_to_go.yaml`) (#110)
- GKE Autopilot workload cost estimation with per-container breakdown (#106)
- Namespace-level cost summaries with waste/efficiency metrics (#108)
- Usage metrics integration via Cloud Monitoring API for cost estimation (#108)
- 32 GCP regions in Autopilot pricing table (#107)
- `verify_completeness` file tool scanning for incomplete placeholder patterns (TODO, FIXME, HACK, XXX, bare `pass`, ellipsis, NotImplementedError) in coding agent workflows (#109)
- Chain-of-thought enforcement in coding agent system prompts for more reliable output (#109)
- XML context boundaries (`<DATA>` delimiters) via `wrap_untrusted_content()` for prompt injection defense (#109)
- SPEC phase (Phase 0) in coding workflow for specification-first development (#109)

### Changed
- Rich console now detects TTY on Windows and non-ANSI terminals at startup — falls back to plain text logging to prevent WinError 1 and garbled ANSI escape codes on non-ANSI handles (#126)
- urllib3 retries disabled globally for Kubernetes API calls — previously 3 retries × 30s caused ~84s total hangs; now fails fast at the `crd_check_timeout` boundary (#125)
- Split monolithic `prompts.py` into 9-file `prompts/` package in `service_health` skill for improved maintainability (#105)
- Enhanced prompt defense with `wrap_untrusted_content()` integration in CodingAgent (#109)
- Anti-hallucination rules now enforced in coding agent system prompts (#109)

### Fixed
- Service status regression — CPU/memory metrics and replica counts showed `N` / `0/0` for Argo Rollout-managed services after Rollout stubs were introduced (#119)
- Explicit timeout added to all Kubernetes API calls in `argo_rollouts.py` — prevents indefinite hangs on `_request_timeout=None` when the cluster is unreachable (#120)
- SSL verification errors for Datadog API on corporate proxy — improved error messages with actionable remediation steps now surface on all Datadog API functions (#123)
- `vaig ask --code --file` now shows output when the LLM uses file tools (edit_file/write_file) without returning a text response — previously the result was silently dropped (#122)

## [0.8.0] - 2026-03-18

### Added
- `--summary` flag for compact 3-5 line report output via `HealthReport.to_summary()`
- `--no-bell` flag to suppress terminal bell notification on pipeline completion
- Per-tool call counter and `[cached]` indicator in `ToolCallLogger` with breakdown summary
- Live agent progress indicator with Rich Status spinner showing active agent name
- Rich Panel for executive summary with status-aware colors and emoji
- Rich Table for recommended actions with urgency-based color coding

## [0.7.0] - 2026-03-17

### Added
- Improved report formatting — timeline events grouped by service/resource for better readability, evidence blocks use language-specific code fences (`yaml`, `json`, etc.), finding evidence rendered as individual sub-bullets instead of semicolon-joined single lines (#24)

## [0.6.0] - 2026-03-17

### Added
- Tool call deduplication cache with SHA-256 keying, LRU eviction, and TTL — shared across orchestrator passes; 7 write/volatile tools marked non-cacheable (#20)
- Defense-in-depth prompt security hardening and structured output documentation (#16)

### Fixed
- `gcloud_monitoring_query` NoneType crash — defensive None guards in `_format_time_series()` and improved tool description (#19)
- Cache wiring in `InfraAgent`, `CodingAgent`, and `live.py` — `tool_result_cache` parameter was not being forwarded to agent execute methods (#21)
- Cache TTL changed from 60s to 0 (no expiration) during pipeline runs — cache is already scoped to pipeline lifetime, TTL expiry caused mid-run misses (#22)
- Data loss in multi-agent pipeline when report sections returned empty results (#18)
- `Content` objects in `estimate_history_tokens` causing `AttributeError` (#17)
- `cast()` protocol gap in `repl.py` removed in favor of proper Protocol typing (#15)

### Changed
- Tech debt cleanup — event bus subscription model, Protocol type annotations, container migration for remaining direct instantiation, mypy strict fixes (#14)

## [0.1.0] - 2026-03-16

Initial public release of the Vertex AI Toolkit (`vaig`) — a multi-agent AI
toolkit powered by Vertex AI Gemini for GKE diagnostics, code analysis, and
SRE workflows.

### Added

#### Core Platform
- Multi-agent system with specialist and orchestrator agents (`InfraAgent`, `CodingAgent`, `ToolAwareAgent`)
- Pluggable skill framework with RCA, anomaly detection, and migration skills
- Interactive REPL with prompt history persistence across sessions
- Typer CLI with `ask`, `chat`, `live`, and `code` commands
- `--workspace/-w` flag, `--output` flag, `--dry-run`, and `--watch` modes
- SQLite-backed session persistence and session manager
- File filtering, context loading, and context building pipeline
- Configurable structured logging with visual feedback

#### AI & Model
- Google GenAI SDK integration (migrated from vertexai SDK)
- Automatic location fallback for SSL/VPN errors
- gcloud CLI auth fallback and SA impersonation for dual-auth
- Configurable safety settings for Gemini API
- Gemini 3.1 series support as default models
- Opt-in response cache for repeated queries
- Per-request cost tracking with token budget management
- Reduced LLM non-determinism via low temperature and frequency penalty

#### Agents & Tools
- Coding agent with filesystem tools and function calling
- GKE live infrastructure tools — kubectl, gcloud wrappers
- 6 GKE diagnostic tools for SRE troubleshooting
- 3 GKE auto-discovery tools (workloads, service mesh, network topology)
- Anthos Service Mesh / Istio introspection tools
- Helm release introspection tools
- ArgoCD integration tools with multi-topology support
- `kubectl_get_labels` tool for label and annotation inspection
- `get_rollout_history` tool for deployment rollout tracking
- Shell tools with command denylist for security hardening
- Plugin tool registration with MCP auto-discovery and Python module plugins
- Parallel tool execution via `ThreadPoolExecutor`

#### Skills (26 total)
- **Wave 1** — log-analysis, error-triage, postmortem, config-audit, slo-review
- **Wave 2** — code-review, iac-review, cost-analysis, capacity-planning, test-generation
- **Wave 3** — compliance-check, api-design, runbook-generator
- **Waves 4–6** — dependency-audit, db-review, pipeline-review, perf-analysis, threat-model, change-risk, alert-tuning, resilience-review, incident-comms, toil-analysis, network-review, adr-generator
- **service-health** — Two-pass diagnostic pipeline with anti-hallucination rules, investigation checklist enforcement, causal reasoning, and remediation framework
- Automatic skill routing via `suggest_skill()`

#### Async Architecture
- Full async foundation — `async_utils.py` and `GeminiClient` async API
- Async telemetry with aiosqlite dependency
- Async session layer — `SessionStore` and `SessionManager`
- Async agent and tool layer — `ToolLoopMixin`, `Orchestrator`, `BaseAgent`
- Async CLI and REPL entry points with `asyncio.run` wrappers

#### Observability
- Usage telemetry and analytics system
- Per-tool-call result storage with JSONL backend
- Cost and token usage display in CLI output and export reports
- Severity coloring in live diagnostic output
- Live tool execution log with namespace autocompletion

#### Configuration
- YAML-based configuration with `config/default.yaml`
- Runtime config switching for project, location, and cluster
- Available projects list in GCP config
- Expanded language detection (9 languages)
- GKE Autopilot detection via google-cloud-container API

#### CI/CD
- PyInstaller build workflow producing Linux and Windows standalone binaries
- CI workflow with ruff lint, mypy type checking, and pytest (2902 tests)
- CODEOWNERS file for code review enforcement
- Branch protection rules on `main`
- GitHub Release automation on `v*` tags

#### Documentation
- Comprehensive README with usage, architecture, and configuration
- Architecture diagrams (agents, skills, tools)
- CLI reference, REPL guide, export guide
- Tools reference, skills guide, sessions guide
- MCP guide, telemetry guide, advanced usage
- Getting started guide and configuration reference

### Fixed
- Gatherer validation failure and quality regression from Autopilot changes
- Gemini repetition bug mitigation with frequency penalty and deduplication
- Empty prompt handling on iteration 2+ of tool-calling loop
- Noisy third-party warnings suppressed at package init
- `Part.from_function_call` replaced with `Part.from_dict` for compatibility
- Thinking spinner for streaming chat mode
- `ResponseValidationError` handling and SDK warning suppression
- Frozen terminal fix in code mode spinner
- Chunk budget calculation and rate limiting for large files
- VPN/proxy malformed response detection with location fallback
- `_strip_empty_strings` list recursion and defensive `.get()` guards
- gcloud CLI auth fallback and config env var shadowing
- Debug flag on subcommands, `--location` param, node reads on Autopilot
- Tool parameter names in prompts and GKE Autopilot awareness
- Gatherer `max_iterations` increase for mandatory Cloud Logging
- Service-health pipeline reliability and deterministic output
- Report quality constraints in service-health skill prompts
- `NoneType.strip()` crash from kubernetes library TTY detection
- Markdown tables and code fences preserved in severity coloring
- UTF-8 encoding enforced on all file I/O for Windows compatibility
- Anti-hallucination regression prevention in Helm/ArgoCD gatherer steps
- ANSI escape codes stripped in CLI help tests for CI compatibility
- aiosqlite connections closed in async tests to prevent CI hang
- All 137 mypy type errors resolved; type checking now blocking in CI
- Runtime deps included in dev dependencies for CI test execution
- 846 ruff lint errors resolved with proper exclusion configuration

### Changed
- Migrated from `vertexai` SDK to `google-genai` SDK
- Split monolithic `app.py` into focused CLI modules
- Split monolithic `gke_tools.py` into `src/vaig/tools/gke/` package
- Unified `system_prompt` to `system_instruction` across agents
- Applied composition root pattern for telemetry settings
- Extracted shared test fixtures into `conftest.py`
- Applied 12 code quality improvements from audit

[0.9.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.9.0
[0.7.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.7.0
[0.6.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.6.0
[0.1.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.1.0

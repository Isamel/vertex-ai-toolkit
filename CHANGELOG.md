# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `/live` REPL command — run live mode analysis within an existing chat session ([#154](https://github.com/Isamel/vertex-ai-toolkit/pull/154))
- `vaig feedback` command — submit bug reports and feature requests from the CLI (writes to `~/.vaig/feedback/`) ([#155](https://github.com/Isamel/vertex-ai-toolkit/pull/155))
- `vaig optimize` command — analyze tool call patterns across runs and suggest efficiency improvements ([#156](https://github.com/Isamel/vertex-ai-toolkit/pull/156))
- `ToolCallOptimizer` engine for tool call pattern analysis (`src/vaig/core/optimizer.py`) ([#156](https://github.com/Isamel/vertex-ai-toolkit/pull/156))
- RAG integration with Vertex AI RAG Engine for enriching analyses with historical report context ([#157](https://github.com/Isamel/vertex-ai-toolkit/pull/157))
- `export` config section for RAG pipeline configuration and `vaig export` command with RAG corpus management ([#157](https://github.com/Isamel/vertex-ai-toolkit/pull/157))
- Shared `PipelineState` object passed across all agents in the pipeline, enabling cross-agent communication via read/write state patches ([#158](https://github.com/Isamel/vertex-ai-toolkit/pull/158))
- `PromptTuner` engine for analyzing past HealthReport quality — hallucination rate, actionability, evidence coverage, severity calibration, scope balance, and command coverage ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- `ReportStore` for local JSONL persistence of structured reports ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- `vaig optimize --reports` flag to display quality insights and prompt improvement suggestions ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- Automatic report persistence in `vaig live` sync and async paths ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- Multiline input support in REPL via backslash continuation ([#153](https://github.com/Isamel/vertex-ai-toolkit/pull/153))
- Priority-based deduplication for duplicate findings ([#153](https://github.com/Isamel/vertex-ai-toolkit/pull/153))

### Changed
- Anti-hallucination prompt rules consolidated ([#153](https://github.com/Isamel/vertex-ai-toolkit/pull/153))
- Updated roadmap status table — all 4 phases marked complete ([#161](https://github.com/Isamel/vertex-ai-toolkit/pull/161))

### Fixed
- `DATABASE_URL` credential redaction in debug logs ([#153](https://github.com/Isamel/vertex-ai-toolkit/pull/153))
- `_age()` crash (`AttributeError: 'str' object has no attribute 'tzinfo'`) when formatting CRD resources (Argo Rollouts, ExternalSecrets) that provide `creationTimestamp` as ISO-8601 string ([#159](https://github.com/Isamel/vertex-ai-toolkit/pull/159))

## [0.11.0] - 2026-03-28

### Added
- `vaig discover` command — autonomous cluster health scanning without requiring a question; auto-generates investigation queries and runs a 4-agent discovery pipeline ([#142](https://github.com/Isamel/vertex-ai-toolkit/pull/142))
- `vaig doctor` command — environment healthcheck with 10 diagnostic checks: GCP auth, Vertex AI API, GKE connectivity, Cloud Logging, Cloud Monitoring, Helm, ArgoCD, Datadog, optional deps, and MCP servers ([#143](https://github.com/Isamel/vertex-ai-toolkit/pull/143))
- Watch mode diff — `--watch` iterations now show new, resolved, and severity-changed findings between each pass ([#146](https://github.com/Isamel/vertex-ai-toolkit/pull/146))
- Watch session HTML export — pressing Ctrl+C during a `--watch` session exports a self-contained HTML report with diff timeline ([#147](https://github.com/Isamel/vertex-ai-toolkit/pull/147))
- ArgoCD management cluster support — 3 connection modes: API server (`server` + `token`), separate kubeconfig context, and same-cluster ([#149](https://github.com/Isamel/vertex-ai-toolkit/pull/149))
- Datadog `cluster_name_override` config field and fallback metric guidance in agent prompts ([#148](https://github.com/Isamel/vertex-ai-toolkit/pull/148))
- Language configuration override — set `language: "es"` (or any BCP-47 code) in config YAML to produce all agent output in that language ([#144](https://github.com/Isamel/vertex-ai-toolkit/pull/144))
- Autopilot detection timeout (10s) and pre-warm discovery clients for faster startup ([#145](https://github.com/Isamel/vertex-ai-toolkit/pull/145))

### Changed
- CLI visual overhaul with Rich components — real-time agent tree, status spinners, and cleaner output formatting ([#139](https://github.com/Isamel/vertex-ai-toolkit/pull/139))
- Cleaner CLI output with log-only noise suppression and real-time agent tree display ([#140](https://github.com/Isamel/vertex-ai-toolkit/pull/140))

### Fixed
- Restored per-tool breakdown in `--detailed` mode ([#141](https://github.com/Isamel/vertex-ai-toolkit/pull/141))
- Fixed demo CLI output and removed PR delivery slide from C-Suite presentation ([#138](https://github.com/Isamel/vertex-ai-toolkit/pull/138))
- Fixed language config override not applying and improved Helm/Argo discovery prompts ([#144](https://github.com/Isamel/vertex-ai-toolkit/pull/144))
- Added timeout to autopilot detection preventing discovery from hanging ([#145](https://github.com/Isamel/vertex-ai-toolkit/pull/145))
- Reused `_load_k8s_config` in doctor GKE connectivity check — fixed `ConnectTimeoutError` on unreachable clusters ([#150](https://github.com/Isamel/vertex-ai-toolkit/pull/150))

## [0.10.0] - 2026-03-27

### Changed
- Vertex AI API retry now uses a two-layer strategy — the SDK's built-in `HttpRetryOptions` handles retries with exponential backoff at the transport level, while the application layer catches `google.genai.errors.ClientError` and converts to typed exceptions without re-retrying (avoids retry multiplication) (#135)
- Argo Rollouts tools now reuse the shared Kubernetes client infrastructure (`_clients.py`) instead of creating standalone API clients — inherits proxy-url, kubeconfig, context, and auth-plugin settings automatically (#134)

### Fixed
- Vertex AI 429 `RESOURCE_EXHAUSTED` errors no longer crash with zero retries — SDK-level `HttpRetryOptions` now configured with backoff from `RetryConfig` (previously `http_options=None` meant `stop_after_attempt(1)`) (#135)
- Argo Rollouts API calls now use a dedicated configurable `argo_request_timeout` (default: 10s) instead of sharing the general `request_timeout` (default: 30s) — reduces hang duration when the Argo Rollouts cluster is unreachable (#134)

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

[Unreleased]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.10.0
[0.9.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.9.0
[0.7.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.7.0
[0.6.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.6.0
[0.1.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.1.0

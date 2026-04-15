# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.16.0] - 2026-04-15

**Phase F — Platform Integration & Extensibility (7 SPECs)**

### Added
- SPEC-1.2: Anomaly Trend Detection — temporal trend analysis for anomalies with spike/gradual/recurring classification and predictive scoring ([#222](https://github.com/Isamel/vertex-ai-toolkit/pull/222))
- SPEC-5.1: Skill Plugin System — dynamic skill loading with plugin registry, YAML-based skill definitions, and hot-reload support ([#218](https://github.com/Isamel/vertex-ai-toolkit/pull/218))
- SPEC-5.3: Skill Marketplace Admin Portal — web-based admin UI for managing skill marketplace with approval workflows and usage analytics ([#219](https://github.com/Isamel/vertex-ai-toolkit/pull/219))
- SPEC-6.2: Report Review/Approval Workflow — multi-stage review process with role-based approvals, comments, and audit trail ([#223](https://github.com/Isamel/vertex-ai-toolkit/pull/223))
- SPEC-7.1: VS Code Extension — native VS Code extension for running diagnostics, viewing reports, and managing skills from the editor ([#224](https://github.com/Isamel/vertex-ai-toolkit/pull/224))
- SPEC-7.2: GitHub Actions Health Check — reusable GitHub Action for automated health checks in CI/CD pipelines ([#220](https://github.com/Isamel/vertex-ai-toolkit/pull/220))
- SPEC-7.3: `vaig check` + Terraform — CLI `vaig check` command and Terraform provider for infrastructure-as-code health check integration ([#226](https://github.com/Isamel/vertex-ai-toolkit/pull/226))
- Monitoring diagnostic logging — debug-level diagnostic logging for workload usage metrics collection with missing pod detection and sorted truncation ([#240](https://github.com/Isamel/vertex-ai-toolkit/pull/240))

### Fixed
- Corrected cost and resource values in health reports ([#217](https://github.com/Isamel/vertex-ai-toolkit/pull/217))
- SSE chunking for live web mode — proper streaming of large payloads ([#221](https://github.com/Isamel/vertex-ai-toolkit/pull/221))
- Web report missing metadata sections — 5 sections (Cost & Usage, GKE Workload Cost Estimation, Anomaly Trends, Tool Usage, Header metadata) now properly rendered in web live mode ([#227](https://github.com/Isamel/vertex-ai-toolkit/pull/227))
- esbuild/vitest vulnerabilities in VS Code extension dependencies ([#228](https://github.com/Isamel/vertex-ai-toolkit/pull/228))
- 503 ServiceUnavailable handler in GCP API error handling with SA impersonation guidance ([#241](https://github.com/Isamel/vertex-ai-toolkit/pull/241))

### Dependencies
- Bumped GitHub Actions (actions/checkout, actions/setup-node, actions/setup-python, actions/upload-artifact) ([#229](https://github.com/Isamel/vertex-ai-toolkit/pull/229))
- Updated google-auth requirement ([#230](https://github.com/Isamel/vertex-ai-toolkit/pull/230))
- Updated humanize requirement ([#231](https://github.com/Isamel/vertex-ai-toolkit/pull/231))
- Updated pyyaml requirement ([#233](https://github.com/Isamel/vertex-ai-toolkit/pull/233))
- Updated google-cloud-bigquery requirement ([#234](https://github.com/Isamel/vertex-ai-toolkit/pull/234))
- Updated prompt-toolkit requirement ([#236](https://github.com/Isamel/vertex-ai-toolkit/pull/236))
- Updated apscheduler requirement ([#238](https://github.com/Isamel/vertex-ai-toolkit/pull/238))
- Updated uvicorn requirement ([#239](https://github.com/Isamel/vertex-ai-toolkit/pull/239))

## [0.13.0] - 2026-04-06

### Added

#### Phase A — Proactive Operations & Collaboration
- SPEC-1.1: Scheduled Health Scans — APScheduler-based background scanner with configurable cron, severity filtering, `HealthReport.diff()` comparison, and BigQuery storage ([#196](https://github.com/Isamel/vertex-ai-toolkit/pull/196))
- SPEC-5.2: Skill Templates & Scaffolding — preset-based skill scaffolding system for rapid skill creation ([#197](https://github.com/Isamel/vertex-ai-toolkit/pull/197))
- SPEC-6.3: Notification Hub — Slack webhook and SMTP email notification channels with multi-channel dispatcher and severity-based routing ([#198](https://github.com/Isamel/vertex-ai-toolkit/pull/198))

#### Phase B — Knowledge Compound Effect
- SPEC-4.1: Automated Fine-Tuning Pipeline — training data preparation from historical reports, SFT job management, model evaluation with comparison scoring ([#200](https://github.com/Isamel/vertex-ai-toolkit/pull/200))
- SPEC-4.3: Tool Effectiveness Learning — tool call scoring engine with latency/failure/redundancy tracking, tier classification (BOOST/NEUTRAL/PENALIZE), LRU caching, and optimizer feedback loop ([#201](https://github.com/Isamel/vertex-ai-toolkit/pull/201))
- SPEC-4.2: Per-Org Knowledge Specialization — org-scoped RAG corpus resolution (`vaig-{org_id}`), org-aware GCS export paths, config bridging via `PlatformConfig.org_id`, and retrieval fallback chain ([#202](https://github.com/Isamel/vertex-ai-toolkit/pull/202))

#### Integrations & Resilience
- Google Chat channel parity — migrated Google Chat to shared formatters for full feature parity with Slack across all notification types ([#199](https://github.com/Isamel/vertex-ai-toolkit/pull/199))
- Rate-limit resilience — 429-specific backoff with `rate_limit_initial_delay` (8s floor), RPM-aware inter-call throttling via `min_inter_call_delay`, deduplicated retry logic via `_compute_backoff_delay` helper, app-level retry for `genai_errors.APIError` code 429 ([#204](https://github.com/Isamel/vertex-ai-toolkit/pull/204))
- PagerDuty, Google Chat, and notification dispatcher integrations for multi-channel incident alerting ([#194](https://github.com/Isamel/vertex-ai-toolkit/pull/194))
- Datadog webhook server for automated incident analysis from Datadog alert payloads ([#195](https://github.com/Isamel/vertex-ai-toolkit/pull/195))

#### Web & Platform
- `vaig web` command — FastAPI/HTMX web server with ask, chat, and live modes ([#175](https://github.com/Isamel/vertex-ai-toolkit/pull/175))
- Web UI: ask mode with SSE streaming ([#176](https://github.com/Isamel/vertex-ai-toolkit/pull/176)), chat mode with multi-turn UI and Firestore sessions ([#177](https://github.com/Isamel/vertex-ai-toolkit/pull/177)), per-session config editor ([#178](https://github.com/Isamel/vertex-ai-toolkit/pull/178)), live mode diagnosis pipeline with real-time SSE progress ([#184](https://github.com/Isamel/vertex-ai-toolkit/pull/184))
- Web UI: error handling, responsive CSS, Cloud Run deployment support ([#179](https://github.com/Isamel/vertex-ai-toolkit/pull/179)), settings panel in chat mode ([#185](https://github.com/Isamel/vertex-ai-toolkit/pull/185))
- Dark/light theme toggle with OS preference detection via `prefers-color-scheme`, manual override, `localStorage` persistence, and FOUC prevention ([#188](https://github.com/Isamel/vertex-ai-toolkit/pull/188))
- Ollama-compatible proxy endpoint for VS Code Copilot integration ([#181](https://github.com/Isamel/vertex-ai-toolkit/pull/181))
- CLI auth + admin portal — `vaig login` (PKCE), `vaig logout`, `vaig whoami`, `vaig status`; JWT-based backend auth; Firestore user/org repository; FastAPI admin API for config enforcement ([#186](https://github.com/Isamel/vertex-ai-toolkit/pull/186))
- Platform config section (`platform.enabled`, `platform.backend_url`) for opt-in centralized auth ([#186](https://github.com/Isamel/vertex-ai-toolkit/pull/186))
- Enhanced dependency mapping with mandatory execution, Datadog API integration, and visual dependency graph ([#191](https://github.com/Isamel/vertex-ai-toolkit/pull/191))

#### Reports & Enrichment
- Report action quality improvement with rich examples and two-pass enrichment pipeline ([#182](https://github.com/Isamel/vertex-ai-toolkit/pull/182))
- Action recommendations enriched with expected output and interpretation guidance ([#180](https://github.com/Isamel/vertex-ai-toolkit/pull/180))

### Changed
- Project documentation updated for recent changes ([#189](https://github.com/Isamel/vertex-ai-toolkit/pull/189))

### Fixed
- TOCTOU race condition in web live mode pipeline semaphore — replaced double-acquire pattern with single-acquire to eliminate 429 errors from concurrent requests ([#187](https://github.com/Isamel/vertex-ai-toolkit/pull/187))
- Agent counter and progress bar rendering in web live mode ([#190](https://github.com/Isamel/vertex-ai-toolkit/pull/190))
- Async report post-processing wrapped in `asyncio.to_thread` for live mode compatibility ([#192](https://github.com/Isamel/vertex-ai-toolkit/pull/192))
- Datadog diagnostic feedback and auto-fallback for metric retrieval failures ([#193](https://github.com/Isamel/vertex-ai-toolkit/pull/193))
- urllib3 warning suppression, enrichment spinner, and finding context injection ([#183](https://github.com/Isamel/vertex-ai-toolkit/pull/183))
- Context window overflow protection and improved error handling ([#173](https://github.com/Isamel/vertex-ai-toolkit/pull/173))
- Datadog SDK Point object handling in metric pointlist extraction ([#169](https://github.com/Isamel/vertex-ai-toolkit/pull/169))
- Custom label exclusion from APM queries and context manager fix in Datadog tools ([#168](https://github.com/Isamel/vertex-ai-toolkit/pull/168))
- Datadog APM metrics rewritten to use Timeseries API with auto-detection ([#167](https://github.com/Isamel/vertex-ai-toolkit/pull/167))
- Label resolution from actual workload types instead of only deployments ([#166](https://github.com/Isamel/vertex-ai-toolkit/pull/166))
- Default `metric_mode` set to `both`, fuzzy template matching, and tag filter fallback ([#170](https://github.com/Isamel/vertex-ai-toolkit/pull/170))
- APM severity thresholds and standalone findings added to health analysis ([#171](https://github.com/Isamel/vertex-ai-toolkit/pull/171))
- Datadog service name pre-resolved from K8s workload labels ([#172](https://github.com/Isamel/vertex-ai-toolkit/pull/172))

### Security
- Pre-validation, delimiter defense, output redaction, and pre-commit hooks for prompt injection hardening ([#165](https://github.com/Isamel/vertex-ai-toolkit/pull/165))

## [0.12.0] - 2026-03-29

### Added
- `/live` REPL command — run live mode analysis within an existing chat session ([#154](https://github.com/Isamel/vertex-ai-toolkit/pull/154))
- `vaig feedback` command — collect rating/comment feedback and export it via the configured export pipeline (exports to BigQuery when `export.enabled=true`) ([#155](https://github.com/Isamel/vertex-ai-toolkit/pull/155))
- `vaig optimize` command — analyze tool call patterns across runs and suggest efficiency improvements ([#156](https://github.com/Isamel/vertex-ai-toolkit/pull/156))
- `ToolCallOptimizer` engine for tool call pattern analysis (`src/vaig/core/optimizer.py`) ([#156](https://github.com/Isamel/vertex-ai-toolkit/pull/156))
- RAG integration with Vertex AI RAG Engine for enriching analyses with historical report context ([#157](https://github.com/Isamel/vertex-ai-toolkit/pull/157))
- `export` config section for RAG pipeline configuration and `vaig export` command with RAG corpus management ([#157](https://github.com/Isamel/vertex-ai-toolkit/pull/157))
- Shared `PipelineState` object passed across all agents in the pipeline, enabling cross-agent communication via read/write state patches ([#158](https://github.com/Isamel/vertex-ai-toolkit/pull/158))
- `PromptTuner` engine for analyzing past HealthReport quality — hallucination rate, actionability, evidence coverage, severity calibration, scope balance, and command coverage ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- `ReportStore` for local JSONL persistence of structured reports ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- `vaig optimize --reports` flag to display quality insights and prompt improvement suggestions ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- Automatic report persistence in `vaig live` sync and async paths ([#160](https://github.com/Isamel/vertex-ai-toolkit/pull/160))
- Multiline input support in REPL using triple-quote (`"""`) delimiters ([#153](https://github.com/Isamel/vertex-ai-toolkit/pull/153))
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

[Unreleased]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.13.0...v0.16.0
[0.13.0]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Isamel/vertex-ai-toolkit/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.10.0
[0.9.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.9.0
[0.7.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.7.0
[0.6.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.6.0
[0.1.0]: https://github.com/Isamel/vertex-ai-toolkit/releases/tag/v0.1.0

# VAIG Documentation

**Vertex AI Toolkit** — An AI-powered CLI for SRE, code review, infrastructure analysis, and multi-agent workflows, built on Google Vertex AI (Gemini).

## Quick Navigation

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, authentication, and your first query |
| [CLI Reference](cli-reference.md) | Complete reference for all CLI commands and options |
| [REPL Guide](repl-guide.md) | Interactive chat mode with 17 slash commands |
| [Skills Guide](skills-guide.md) | All 31 built-in skills for SRE, security, code quality, and more |
| [Agents Guide](agents-guide.md) | Multi-agent architecture and orchestration strategies |
| [Tools Reference](tools-reference.md) | 34 infrastructure tools: GKE, GCloud, Helm, ArgoCD, Mesh, Labels, MCP |
| [Configuration](configuration.md) | All 14 config sections, YAML format, and environment variables |
| [Sessions Guide](sessions-guide.md) | Session persistence, search, and history management |
| [Export Guide](export-guide.md) | Export results to JSON, Markdown, or HTML |
| [MCP Guide](mcp-guide.md) | Model Context Protocol server integration |
| [Telemetry Guide](telemetry-guide.md) | Local-only usage telemetry, analytics, and event export |
| [Architecture](architecture.md) | Mermaid diagrams: system overview, pipeline flow, tool structure, ArgoCD topologies |
| [Advanced Usage](advanced.md) | Composite skills, chunked processing, custom skills, and workspace mode |
| [Web Deployment](web-deployment.md) | Deploy the web UI to Cloud Run with IAP authentication |

## Architecture Overview

See [Architecture](architecture.md) for detailed Mermaid diagrams.

```
┌──────────────────────────────────────────────────────────┐
│                       CLI Layer                          │
│  chat (REPL) │ ask │ live │ web │ login/logout │ stats   │
├──────────────────────────────────────────────────────────┤
│                      Web Layer                           │
│  FastAPI + HTMX + SSE │ ask/chat/live routes │ themes    │
├──────────────────────────────────────────────────────────┤
│                     Agent Layer                          │
│  Orchestrator │ CodingAgent │ CodingSkillOrchestrator │ InfraAgent │ Chunked │
├──────────────────────────────────────────────────────────┤
│                     Skills Layer                         │
│  31 built-in skills with phase-based multi-agent pipes   │
├──────────────────────────────────────────────────────────┤
│                     Tools Layer                          │
│  File │ Shell │ GKE │ GCloud │ Helm │ ArgoCD │ Mesh │MCP│
├──────────────────────────────────────────────────────────┤
│                    Core Layer                            │
│  GeminiClient │ Auth │ Cache │ CostTracker │ Telemetry   │
├──────────────────────────────────────────────────────────┤
│                  Platform Layer                          │
│  PlatformAuth (PKCE) │ JWT │ Firestore │ Config Policy   │
├──────────────────────────────────────────────────────────┤
│               Google Vertex AI (Gemini)                  │
│  gemini-2.5-pro │ gemini-2.5-flash │ gemini-3.x         │
└──────────────────────────────────────────────────────────┘
```

## Key Features

- **Web UI** — Browser-based interface with FastAPI + HTMX + SSE streaming, ask/chat/live modes, dark/light theme toggle with OS preference detection
- **Multi-agent orchestration** — Sequential, fan-out, and lead-delegate strategies
- **Async-native** — Full async stack from CLI to API calls, with sync backward compat
- **31 SRE/DevOps skills** — From root cause analysis to threat modeling, code migration, and greenfield generation
- **CodingSkillOrchestrator** — 3-agent Planner → Implementer → Verifier pipeline for complex coding tasks
- **Live infrastructure** — Query GKE clusters, read logs, check metrics in real-time
- **Helm & ArgoCD integration** — Read release status, sync state, drift detection (opt-in)
- **Istio/ASM mesh tools** — VirtualService, DestinationRule, sidecar introspection
- **Coding agent** — Read, write, edit files and run shell commands
- **Platform mode** — Centralized CLI fleet management with PKCE login, JWT auth, Firestore-backed config policy enforcement, and a FastAPI admin portal
- **Session persistence** — SQLite-backed history with search and resume
- **Response caching** — LRU + TTL cache for repeated queries (opt-in)
- **Chunked processing** — Map-Reduce for files that exceed model context limits
- **MCP integration** — Extend capabilities via Model Context Protocol servers
- **Plugin system** — Python module plugins and MCP auto-registration
- **Export** — Save analysis results as JSON, Markdown, or HTML reports
- **Per-session cost tracking** — Budget enforcement with warn/stop thresholds
- **ToolCallStore** — Per-tool-call JSONL recording for post-run analysis and auditing
- **Usage telemetry** — Local SQLite analytics with `vaig stats` commands
- **Auto-skill detection** — Automatically selects the best skill for your query
- **Structured output (JSON Schema)** — Skills can constrain Gemini output to a Pydantic schema for deterministic, type-safe reports
- **9-language detection** — Responds in the user's language (es, pt, fr, de, it, ja, zh, ko)

## Requirements

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- Authenticated via `gcloud auth application-default login`

---

*VAIG — MIT License*

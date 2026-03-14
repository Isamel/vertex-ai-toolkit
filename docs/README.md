# VAIG Documentation

**Vertex AI Toolkit** — An AI-powered CLI for SRE, code review, infrastructure analysis, and multi-agent workflows, built on Google Vertex AI (Gemini).

## Quick Navigation

| Guide | Description |
|-------|-------------|
| [Getting Started](getting-started.md) | Installation, authentication, and your first query |
| [CLI Reference](cli-reference.md) | Complete reference for all CLI commands and options |
| [REPL Guide](repl-guide.md) | Interactive chat mode with 17 slash commands |
| [Skills Guide](skills-guide.md) | All 29 built-in skills for SRE, security, code quality, and more |
| [Agents Guide](agents-guide.md) | Multi-agent architecture and orchestration strategies |
| [Tools Reference](tools-reference.md) | File, shell, GKE, GCloud, and MCP tool details |
| [Configuration](configuration.md) | All 14 config sections, YAML format, and environment variables |
| [Sessions Guide](sessions-guide.md) | Session persistence, search, and history management |
| [Export Guide](export-guide.md) | Export results to JSON, Markdown, or HTML |
| [MCP Guide](mcp-guide.md) | Model Context Protocol server integration |
| [Telemetry Guide](telemetry-guide.md) | Local-only usage telemetry, analytics, and event export |
| [Advanced Usage](advanced.md) | Composite skills, chunked processing, custom skills, and workspace mode |

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                    CLI Layer                     │
│  chat (REPL) │ ask (single-shot) │ live (infra) │
├─────────────────────────────────────────────────┤
│                 Agent Layer                      │
│  Orchestrator │ CodingAgent │ InfraAgent │ ...   │
├─────────────────────────────────────────────────┤
│                 Skills Layer                     │
│  29 built-in skills with phase-based workflows   │
├─────────────────────────────────────────────────┤
│                 Tools Layer                      │
│  File I/O │ Shell │ GKE │ GCloud │ MCP Bridge    │
├─────────────────────────────────────────────────┤
│              Google Vertex AI (Gemini)           │
│  gemini-2.5-pro │ gemini-2.5-flash │ ...         │
└─────────────────────────────────────────────────┘
```

## Key Features

- **Multi-agent orchestration** — Sequential, fan-out, and lead-delegate strategies
- **29 SRE/DevOps skills** — From root cause analysis to threat modeling
- **Live infrastructure** — Query GKE clusters, read logs, check metrics in real-time
- **Coding agent** — Read, write, edit files and run shell commands
- **Session persistence** — SQLite-backed history with search and resume
- **Chunked processing** — Map-Reduce for files that exceed model context limits
- **MCP integration** — Extend capabilities via Model Context Protocol servers
- **Export** — Save analysis results as JSON, Markdown, or HTML reports
- **Per-request cost tracking** — See estimated cost for every API call
- **Auto-skill detection** — Automatically selects the best skill for your query

## Requirements

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- Authenticated via `gcloud auth application-default login`

---

*VAIG v0.1.0 — MIT License*

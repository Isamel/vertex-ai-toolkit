# MCP Guide

VAIG supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) to extend its capabilities with external tool servers.

## What is MCP?

MCP is an open protocol that allows AI applications to connect to external tool servers. Each MCP server exposes a set of tools that VAIG can discover and use during agent execution — just like built-in tools.

## Configuration

Enable MCP and configure servers in your YAML config:

```yaml
mcp:
  enabled: true
  servers:
    - name: filesystem
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
      description: "Filesystem access server"

    - name: github
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
      description: "GitHub API access"

    - name: custom-server
      command: python
      args: ["-m", "my_mcp_server"]
      env:
        API_KEY: "${MY_API_KEY}"
      description: "Custom internal tools"
```

### Server Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique server identifier |
| `command` | string | Yes | Executable to run |
| `args` | list | No | Command arguments |
| `env` | object | No | Environment variables (supports `${VAR}` expansion) |
| `description` | string | No | Human-readable description |

## CLI Commands

### List Configured Servers

```bash
vaig mcp list-servers
```

Shows all configured MCP servers, their commands, and descriptions.

### Discover Tools

```bash
vaig mcp discover
```

Connects to all enabled MCP servers and lists the tools they expose. This is useful for understanding what capabilities are available.

Example output:

```
Server: filesystem
  - read_file: Read a file's contents
  - write_file: Write content to a file
  - list_directory: List directory contents

Server: github
  - search_repositories: Search GitHub repositories
  - get_file_contents: Get file contents from a repo
  - create_issue: Create a GitHub issue
```

### Call a Tool

```bash
vaig mcp call
```

Interactively select and call an MCP tool for testing. Useful for verifying that your MCP servers are working correctly.

## How MCP Tools Work with Agents

When MCP is enabled, the `MCPBridge` class:

1. **Discovers** tools from all configured servers at startup
2. **Creates** Gemini-compatible function declarations from MCP tool definitions
3. **Registers** them in the `ToolRegistry` alongside built-in tools
4. **Bridges** tool calls — when the AI model calls an MCP tool, the bridge routes the call to the correct server and returns the result

MCP tools are available to any agent that has tool access (`ToolAwareAgent`, `CodingAgent`, `InfraAgent`).

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  MCP Server  │     │  MCP Server  │     │  MCP Server  │
│ (filesystem) │     │   (github)   │     │   (custom)   │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────┬───────┘────────────────────┘
                    │
              ┌─────┴─────┐
              │ MCPBridge  │
              │            │
              │ discover() │
              │ call()     │
              └─────┬──────┘
                    │
              ┌─────┴──────┐
              │ToolRegistry │ ← Built-in tools + MCP tools
              └─────┬──────┘
                    │
              ┌─────┴──────┐
              │   Agents   │ ← Use all tools transparently
              └────────────┘
```

## Environment Variables in Config

MCP server configs support environment variable expansion using `${VAR}` syntax:

```yaml
mcp:
  servers:
    - name: my-server
      command: my-mcp-server
      env:
        API_KEY: "${MY_API_KEY}"        # Expands from environment
        DB_URL: "${DATABASE_URL}"
```

> **Note:** Make sure the referenced environment variables are set before starting VAIG.

## Examples

### Connecting to a Database MCP Server

```yaml
mcp:
  enabled: true
  servers:
    - name: postgres
      command: npx
      args: ["-y", "@modelcontextprotocol/server-postgres"]
      env:
        POSTGRES_URL: "${DATABASE_URL}"
      description: "PostgreSQL query access"
```

```bash
# Verify it works
vaig mcp discover

# Use it in an ask query
vaig ask "Analyze the schema of the users table" --live
```

### Multiple Servers

```yaml
mcp:
  enabled: true
  servers:
    - name: filesystem
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "."]

    - name: brave-search
      command: npx
      args: ["-y", "@modelcontextprotocol/server-brave-search"]
      env:
        BRAVE_API_KEY: "${BRAVE_API_KEY}"

    - name: slack
      command: npx
      args: ["-y", "@modelcontextprotocol/server-slack"]
      env:
        SLACK_TOKEN: "${SLACK_BOT_TOKEN}"
```

## Troubleshooting

**Server not starting:**
- Check that the command is installed (`npx`, `python`, etc.)
- Verify arguments are correct
- Check environment variables are set

**Tools not discovered:**
- Run `vaig mcp discover` to see errors
- Check server logs with `--verbose`
- Ensure `mcp.enabled` is `true` in config

**Tool calls failing:**
- Use `vaig mcp call` to test individual tools
- Check that required tool parameters are provided
- Verify API keys and credentials

---

[Back to index](README.md)

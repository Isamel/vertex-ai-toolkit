# Usage Telemetry & Analytics

VAIG includes a local-only telemetry system that captures operational events into a queryable SQLite store. All data stays on your machine — nothing is sent externally.

## What's Tracked

| Event Type | Fields Captured |
|------------|-----------------|
| `tool_call` | Tool name, duration (ms), arguments, error (if any) |
| `api_call` | Model name, tokens in/out, estimated cost, duration (ms) |
| `cli_command` | Command name, duration (ms) |
| `skill_activation` | Skill name |
| `session_start` | Session ID, timestamp |
| `session_end` | Session ID, timestamp, duration |
| `orchestrator_event` | Strategy, skill |
| `error` | Error type, message |

## Design Principles

- **Fire-and-forget** — telemetry never breaks normal operation; emit failures are silently dropped
- **Thread-safe** — concurrent emits from multiple agents are safe (lock-free list append)
- **Buffer-then-flush** — events are buffered in memory and flushed in batches for low overhead
- **Local-only** — no external data transmission, ever

## CLI Commands

### `vaig stats show`

Display a usage summary with event counts, token totals, and cost estimates.

```bash
# Show all-time stats
vaig stats show

# Filter by date range
vaig stats show --since 2024-01-01 --until 2024-12-31
```

Output example:

```
Usage Summary (2024-01-01 to 2024-12-31)
────────────────────────────────────────
Total events:         1,247
Tool calls:             832
API calls:              215
CLI commands:           142
Skill activations:       38
Errors:                  20

Tokens in:        1,204,500
Tokens out:         312,800
Estimated cost:      $14.27

Top tools:
  read_file             312
  shell_exec            198
  gke_get_pods          147

Top models:
  gemini-2.5-pro        142
  gemini-2.5-flash       73
```

### `vaig stats export`

Export raw telemetry events for external analysis.

```bash
# Export all events as JSONL to stdout
vaig stats export

# Export as CSV to a file
vaig stats export --format csv --output usage.csv

# Filter by event type
vaig stats export --type tool_call

# Filter by date range
vaig stats export --since 2024-01-01

# Combine filters
vaig stats export --type api_call --since 2024-06-01 --format csv -o api-usage.csv
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--format` | `-f` | Export format: `jsonl`, `csv` | `jsonl` |
| `--output` | `-o` | Output file path | stdout |
| `--type` | `-t` | Filter by event type | All types |
| `--since` | | Start date (ISO 8601) | — |
| `--until` | | End date (ISO 8601) | — |

### `vaig stats clear`

Remove old telemetry events from the database.

```bash
# Clear events older than 30 days (requires --confirm)
vaig stats clear --days 30 --confirm
```

| Option | Description | Default |
|--------|-------------|---------|
| `--days` | Delete events older than N days | Required |
| `--confirm` | Skip confirmation prompt | `false` |

> **Note:** This permanently deletes events from the SQLite database. Use `vaig stats export` first if you want to keep a backup.

## Configuration

### YAML Config

```yaml
telemetry:
  enabled: true          # Enable/disable telemetry collection
  buffer_size: 50        # Number of events to buffer before flushing to SQLite
```

### Environment Variables

```bash
# Disable telemetry entirely
export VAIG_TELEMETRY_ENABLED=false

# Adjust buffer size
export VAIG_TELEMETRY__BUFFER_SIZE=100
```

### Defaults

| Setting | Default | Description |
|---------|---------|-------------|
| `telemetry.enabled` | `true` | Telemetry is enabled by default |
| `telemetry.buffer_size` | `50` | Events buffered in memory before flush |

## Storage

Telemetry events are stored in the same SQLite database as sessions (`~/.vaig/sessions.db`). The telemetry table uses the following schema:

```
telemetry_events
├── id (auto-increment)
├── event_type (text)        — tool_call, api_call, cli_command, etc.
├── event_data (JSON)        — type-specific payload
├── session_id (text, nullable)
├── timestamp (ISO timestamp)
└── duration_ms (integer, nullable)
```

The database uses WAL mode for safe concurrent writes from multiple agents.

## Privacy

- **No network calls** — telemetry data never leaves your machine
- **No PII** — only operational metadata is captured (tool names, token counts, durations)
- **User-controlled** — disable with a single config flag or environment variable
- **Deletable** — use `vaig stats clear` or delete `~/.vaig/sessions.db` to remove all data

## Tips

- **Monitor costs over time** — `vaig stats show --since 2024-01-01` gives you a quick spend overview
- **Identify expensive patterns** — export API call events and sort by cost to find optimization targets
- **Debug tool failures** — filter `--type error` exports to review what went wrong and when
- **Automate reporting** — pipe `vaig stats export --format csv` into your own dashboards or spreadsheets
- **Keep the database lean** — schedule periodic `vaig stats clear --days 90 --confirm` to prune old data

---

[Back to index](README.md)

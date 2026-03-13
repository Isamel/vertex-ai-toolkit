# Session Management

VAIG persists chat sessions in a local SQLite database, allowing you to resume conversations, search history, and export past analyses.

## How Sessions Work

Every `vaig chat` interaction creates a session that stores:
- **Session metadata** — ID, name, model, skill, timestamps
- **Messages** — Full conversation history (role, content, model, token count)
- **Context files** — Files added during the session

Sessions are stored at `~/.vaig/sessions.db` by default (configurable via `session.db_path`).

## Database Schema

```
sessions
├── id (UUID)
├── name (text)
├── model (text)
├── skill (text, nullable)
├── created_at (ISO timestamp)
├── updated_at (ISO timestamp)
└── metadata (JSON)

messages
├── id (auto-increment)
├── session_id (FK → sessions)
├── role ("user" or "model")
├── content (text)
├── model (text, nullable)
├── token_count (integer)
└── created_at (ISO timestamp)

context_files
├── id (auto-increment)
├── session_id (FK → sessions)
├── file_path (text)
├── file_type (text)
├── size_bytes (integer)
└── added_at (ISO timestamp)
```

## Session Lifecycle

### Creating Sessions

Sessions are created automatically when you start a chat:

```bash
# Auto-named session
vaig chat

# Named session
vaig chat -n "incident-investigation"

# Session with a skill preloaded
vaig chat -s rca -n "payment-outage-rca"
```

In the REPL, start a new session mid-conversation:

```
/new debugging-auth-flow
```

### Auto-Save

When `session.auto_save` is `true` (default), every message is persisted to SQLite as it happens. If the REPL crashes, your history is safe.

### Resuming Sessions

```bash
# Resume the most recent session
vaig chat -r

# Load a specific session by ID
vaig chat --session abc123-def456
```

In the REPL:

```
/resume                          # Resume last session
/load abc123-def456              # Load by ID
```

### Listing Sessions

```bash
vaig sessions list               # Last 20 sessions
vaig sessions list -n 50         # Last 50 sessions
```

In the REPL:

```
/sessions
```

Output example:

```
ID          Name                     Model              Messages  Updated
────────────────────────────────────────────────────────────────────────────
a1b2c3d4    incident-investigation   gemini-2.5-pro     24        2024-03-13 14:30
e5f6g7h8    code-review-auth         gemini-2.5-flash   8         2024-03-12 09:15
i9j0k1l2    kubernetes-debugging     gemini-2.5-pro     42        2024-03-11 16:45
```

### Searching Sessions

Search by session name or message content:

```bash
vaig sessions search "kubernetes"
vaig sessions search "deployment rollback"
```

In the REPL:

```
/search kubernetes
```

### Viewing Session Details

```bash
# Session metadata only
vaig sessions show SESSION_ID

# Include message history
vaig sessions show SESSION_ID -m
```

### Renaming Sessions

```bash
vaig sessions rename SESSION_ID "new-name"
```

In the REPL:

```
/rename better-session-name
```

### Deleting Sessions

```bash
vaig sessions delete SESSION_ID         # With confirmation
vaig sessions delete SESSION_ID -f      # Force delete
```

> **Note:** Deleting a session removes all its messages and context files (cascade delete).

## History Management

### In-Memory Trimming

The `session.max_history_messages` config (default: 100) controls how many messages are kept in memory. When the limit is exceeded, older messages are trimmed from the in-memory history but remain in the database.

### Clearing History

In the REPL, `/clear` clears the in-memory history without deleting the database records:

```
/clear
```

This is useful when you want to "reset" the conversation context without losing the saved history.

## Configuration

```yaml
session:
  db_path: ~/.vaig/sessions.db      # Database file location
  auto_save: true                    # Persist messages automatically
  max_history_messages: 100          # Max messages in memory
```

### Environment Variables

```bash
export VAIG_SESSION__DB_PATH=~/custom/sessions.db
export VAIG_SESSION__AUTO_SAVE=false
export VAIG_SESSION__MAX_HISTORY_MESSAGES=50
```

## Tips

- **Name sessions descriptively** — Makes search much easier: `/new prod-incident-2024-03-13`
- **Use search to find past analyses** — `vaig sessions search "terraform"` finds all IaC-related sessions
- **Resume for ongoing work** — `vaig chat -r` picks up exactly where you left off
- **Export before sharing** — Use `vaig export SESSION_ID -f md` to create shareable reports
- **WAL mode** — The database uses SQLite WAL (Write-Ahead Logging) for safe concurrent access

---

[Back to index](README.md)

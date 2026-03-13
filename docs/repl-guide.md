# REPL Guide

The interactive REPL (Read-Eval-Print Loop) is started with `vaig chat`. It provides a rich conversational interface with slash commands for controlling the session.

## Starting the REPL

```bash
# Basic start
vaig chat

# With a skill preloaded
vaig chat -s code-review

# Resume last session
vaig chat -r

# With a specific model
vaig chat -m gemini-2.5-flash

# Named session
vaig chat -n "debugging-session"
```

## Slash Commands

All commands start with `/` and are available during any chat session.

### File & Context Commands

#### `/add <file_path>`

Add a file to the conversation context. The file contents are included in the next message to the AI.

```
/add src/main.py
/add config/nginx.conf
/add ../shared/utils.py
```

> **Note:** File size is limited by the `context.max_file_size_mb` config (default: 50 MB). Unsupported file types are filtered by `context.supported_extensions`.

#### `/context`

Show the current conversation context — loaded files, active skill, model, and token usage.

```
/context
```

#### `/code`

Toggle coding agent mode. When enabled, the AI can read, write, and edit files in your workspace, and run shell commands.

```
/code
```

> **Note:** In coding mode, the AI has access to file tools (`read_file`, `write_file`, `edit_file`, `list_files`, `search_files`) and the `run_command` shell tool. All operations are sandboxed to the workspace directory.

### Model & Skill Commands

#### `/model [model_name]`

Switch the active model. Without arguments, shows available models.

```
/model                          # List available models
/model gemini-2.5-flash         # Switch to flash
/model gemini-2.5-pro           # Switch to pro
```

#### `/skill [skill_name]`

Load or switch skills. Without arguments, lists available skills.

```
/skill                          # List all skills
/skill rca                      # Load root cause analysis
/skill code-review              # Load code review
/skill none                     # Unload current skill
```

#### `/phase [phase_name]`

Set the active skill phase. Skills support up to 5 phases: `analyze`, `plan`, `execute`, `validate`, `report`.

```
/phase analyze                  # Start with analysis
/phase plan                     # Switch to planning
/phase execute                  # Switch to execution
/phase report                   # Generate report
```

#### `/agents`

Show the agent configuration for the currently loaded skill, including agent names, roles, and orchestration strategy.

```
/agents
```

### Session Commands

#### `/new [name]`

Start a new session, optionally with a name. The current session is saved automatically.

```
/new
/new kubernetes-debugging
```

#### `/load <session_id>`

Load a previously saved session by its ID.

```
/load abc123-def456-789
```

#### `/resume`

Resume the most recently updated session.

```
/resume
```

#### `/rename <new_name>`

Rename the current session.

```
/rename prod-incident-march-2024
```

#### `/search <query>`

Search through all saved sessions by name or message content.

```
/search kubernetes
/search "deployment rollback"
```

#### `/sessions`

List recent sessions with their IDs, names, and message counts.

```
/sessions
```

### Utility Commands

#### `/clear`

Clear the current conversation history from memory (the database record is preserved).

```
/clear
```

#### `/help`

Show the help message with all available slash commands.

```
/help
```

#### `/quit` / `/exit`

Exit the REPL. The session is saved automatically if auto-save is enabled.

```
/quit
/exit
```

## Usage Patterns

### Code Review Workflow

```
$ vaig chat -s code-review

> /add src/api/handlers.py
> Review this handler for security issues and suggest improvements

> /add src/api/middleware.py
> Now check the middleware — are there any auth bypasses?

> /phase report
> Generate a final report of all findings
```

### Infrastructure Investigation

```
$ vaig chat -s rca -n "payment-outage"

> We're seeing 503 errors on the payment-service since 14:30 UTC.
> The service runs on GKE in the production namespace.

> /code
> Can you check the deployment manifest at k8s/payment-service.yaml?

> /phase report
> Write up the RCA with timeline and action items
```

### Multi-Skill Session

```
$ vaig chat

> /skill code-review
> /add src/auth.py
> Review this authentication module

> /skill threat-model
> Now do a threat model of the same auth flow

> /skill test-generation
> Generate tests for the issues you found
```

### Session Management

```
$ vaig chat -n "daily-review"

> /add src/changes/*.py
> Review today's changes

> /quit

# Later...
$ vaig chat -r          # Resume where you left off

> /sessions             # List all sessions
> /search "auth"        # Find auth-related sessions
> /load abc123          # Load a specific session
```

## Keyboard Shortcuts

The REPL uses `prompt_toolkit` and supports standard readline keybindings:

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current input |
| `Ctrl+D` | Exit the REPL |
| `Up/Down` | Navigate input history |
| `Tab` | Command completion |

## Tips

- **Multiline input**: The REPL accepts multiline input. Press Enter twice to send.
- **Token awareness**: Use `/context` to check how much context you're using before hitting model limits.
- **Skill phases**: Not all skills support all phases. Use `/agents` to see the skill's agent pipeline.
- **Coding mode**: Toggle `/code` only when you need file operations — it adds tool definitions to every request, using extra tokens.
- **Session names**: Name your sessions descriptively for easier search later.

---

[Back to index](README.md)

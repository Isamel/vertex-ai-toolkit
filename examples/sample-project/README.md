# Task Manager — Sample Project for VAIG Testing

A simple task management CLI app with intentional bugs and missing features.
Use this project as a workspace for testing the VAIG coding agent.

## Test Scenarios

### read_file
- "Read the README and summarize what this project does"
- "What does the TaskManager class do?"
- "Show me the config file"

### write_file
- "Create a new file called src/priority.py with a Priority enum (LOW, MEDIUM, HIGH, CRITICAL)"
- "Create a CHANGELOG.md file documenting version 0.1.0"
- "Add a .gitignore file for Python projects"

### edit_file
- "Fix the bug in src/models.py — the is_overdue method has a logic error"
- "Add a delete_task method to the TaskManager class in src/manager.py"
- "Fix the broken test in tests/test_manager.py"

### list_files
- "List all Python files in this project"
- "Show me the directory structure"
- "What files are in the config directory?"

### search_files
- "Find all TODO comments in the codebase"
- "Search for where TaskStatus is used"
- "Find all functions that raise exceptions"

### run_command
- "Run the tests and tell me which ones fail"
- "Run python src/models.py to check for syntax errors"
- "Count the lines of code in this project"

## Structure

```
sample-project/
  src/
    __init__.py
    models.py        # Task and TaskStatus — has intentional bugs
    manager.py       # TaskManager CRUD — missing delete method
    formatter.py     # Output formatting — has a TODO
    utils.py         # Date/time helpers
  tests/
    __init__.py
    test_models.py   # Working tests
    test_manager.py  # Has a broken test
  config/
    settings.yaml    # App config — missing fields
    logging.json     # Logging config
  docs/
    architecture.md  # Design notes
  run.sh             # Entry point script — has a bug
```

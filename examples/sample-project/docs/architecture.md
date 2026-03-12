# Architecture Notes

## Overview

The Task Manager follows a simple layered architecture:

```
CLI / Entry Point (run.sh)
        │
   TaskManager (manager.py)    ← CRUD operations, persistence
        │
     Task Model (models.py)    ← Data structures, validation
        │
   Utilities (utils.py)        ← Date parsing, formatting helpers
```

## Design Decisions

### 1. In-Memory Storage with JSON Persistence
Tasks are stored in a Python list for fast access and serialized to JSON
for persistence. This is simple but doesn't scale for large datasets.

**Future improvement**: Add SQLite backend option.

### 2. Dataclass-Based Models
Using `@dataclass` for the Task model provides automatic `__init__`,
`__repr__`, and comparison methods. Field defaults make creation ergonomic.

### 3. Status as Enum
`TaskStatus` uses Python's `Enum` to ensure only valid statuses are used.
This prevents typos and provides IDE autocompletion.

## Known Issues

1. **is_overdue bug**: The comparison in `models.py` is reversed — returns
   True when not overdue and False when overdue.

2. **Missing delete_task**: The `TaskManager` has no method to remove tasks.
   There's a TODO comment and a test expecting it, but the implementation
   is missing.

3. **No input validation**: The `update_task` method uses `setattr` without
   type checking. Passing wrong types will silently corrupt data.

4. **Thread safety**: No locking on the task list. Concurrent access from
   multiple threads would cause race conditions.

## File Descriptions

| File              | Purpose                              | Lines |
|-------------------|--------------------------------------|-------|
| `src/models.py`   | Task + TaskStatus data models        | ~95   |
| `src/manager.py`  | TaskManager CRUD + persistence       | ~155  |
| `src/formatter.py` | Display formatting utilities        | ~80   |
| `src/utils.py`    | Date parsing, string utilities       | ~90   |
| `config/settings.yaml` | App configuration              | ~18   |
| `config/logging.json`  | Python logging configuration    | ~35   |

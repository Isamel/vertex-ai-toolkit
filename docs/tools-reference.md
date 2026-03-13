# Tools Reference

VAIG provides tools that agents can call during their execution. Tools are organized by domain and registered via `ToolRegistry`.

## Tool Architecture

Each tool is defined as a `ToolDef`:

```python
@dataclass(frozen=True)
class ToolDef:
    name: str                    # Unique identifier
    description: str             # What the tool does
    parameters: list[ToolParam]  # Input parameters
    handler: Callable            # The function to call

@dataclass(frozen=True)
class ToolParam:
    name: str
    type: str                    # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None
```

Tools return a `ToolResult`:

```python
@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
```

## File Tools

Available in **coding agent** mode (`--code` or `/code`). All file operations are sandboxed to the workspace directory.

### `read_file`

Read the contents of a file.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative path from workspace root |
| `start_line` | integer | No | Start reading from this line (1-indexed) |
| `end_line` | integer | No | Stop reading at this line |

```
read_file(path="src/main.py")
read_file(path="src/main.py", start_line=10, end_line=50)
```

### `write_file`

Create or overwrite a file.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative path from workspace root |
| `content` | string | Yes | File contents to write |

```
write_file(path="src/new_module.py", content="def hello(): ...")
```

### `edit_file`

Apply a targeted edit to an existing file using search-and-replace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative path from workspace root |
| `old_text` | string | Yes | Exact text to find |
| `new_text` | string | Yes | Replacement text |

```
edit_file(path="src/main.py", old_text="print('debug')", new_text="logger.debug('debug')")
```

> **Note:** The `old_text` must match exactly (including whitespace). If not found, the edit fails with an error.

### `list_files`

List files in a directory, optionally filtered by pattern.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | No | Directory to list (default: workspace root) |
| `pattern` | string | No | Glob pattern filter (e.g., `*.py`) |
| `recursive` | boolean | No | Include subdirectories (default: `false`) |

```
list_files()
list_files(path="src", pattern="*.py", recursive=true)
```

### `search_files`

Search for text across files in the workspace.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search text or pattern |
| `path` | string | No | Directory to search (default: workspace root) |
| `pattern` | string | No | File glob filter (e.g., `*.py`) |

```
search_files(query="def authenticate", pattern="*.py")
search_files(query="TODO", path="src")
```

## Shell Tool

### `run_command`

Execute a shell command in the workspace directory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `command` | string | Yes | The command to run |

**Security constraints:**
- **Allowlist**: Only commands in `coding.allowed_commands` can be run (configurable)
- **Blocked patterns**: Certain argument patterns are blocked (e.g., `rm -rf /`)
- **Timeout**: 30-second execution limit
- **Output cap**: 100,000 character maximum output
- **Confirmation**: Requires user confirmation if `coding.confirm_actions` is `true`

Default allowed commands include: `ls`, `cat`, `grep`, `find`, `wc`, `head`, `tail`, `sort`, `uniq`, `diff`, `python`, `pip`, `node`, `npm`, `git`, `make`, `cargo`, `go`, etc.

```
run_command(command="python -m pytest tests/ -v")
run_command(command="git diff HEAD~1")
```

## GKE Tools

Available in **live** mode (`vaig live` or `--live`). Require the `[live]` extra and a configured GKE cluster.

### Read Operations

#### `kubectl_get`

List Kubernetes resources.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | Resource type (e.g., `pods`, `deployments`, `services`) |
| `namespace` | string | No | Namespace (default from config) |
| `label_selector` | string | No | Label filter (e.g., `app=nginx`) |
| `field_selector` | string | No | Field filter (e.g., `status.phase=Running`) |
| `all_namespaces` | boolean | No | List across all namespaces |

Supports 20+ resource types with aliases: `pods`/`po`/`pod`, `deployments`/`deploy`, `services`/`svc`, `configmaps`/`cm`, `secrets`, `ingresses`/`ing`, `statefulsets`/`sts`, `daemonsets`/`ds`, `jobs`, `cronjobs`/`cj`, `namespaces`/`ns`, `nodes`, `persistentvolumeclaims`/`pvc`, `persistentvolumes`/`pv`, `serviceaccounts`/`sa`, `networkpolicies`/`netpol`, `horizontalpodautoscalers`/`hpa`, `replicasets`/`rs`, `endpoints`/`ep`, `events`.

```
kubectl_get(resource_type="pods", namespace="production", label_selector="app=api")
kubectl_get(resource_type="deploy", all_namespaces=true)
```

#### `kubectl_describe`

Get detailed information about a specific resource.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | Resource type |
| `name` | string | Yes | Resource name |
| `namespace` | string | No | Namespace |

```
kubectl_describe(resource_type="pod", name="api-server-xyz", namespace="production")
```

#### `kubectl_logs`

Get container logs from a pod.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pod_name` | string | Yes | Pod name |
| `namespace` | string | No | Namespace |
| `container` | string | No | Container name (for multi-container pods) |
| `tail_lines` | integer | No | Number of lines from the end |
| `since_seconds` | integer | No | Logs from the last N seconds |
| `previous` | boolean | No | Get logs from previous container instance |

```
kubectl_logs(pod_name="api-server-xyz", namespace="production", tail_lines=100)
kubectl_logs(pod_name="api-server-xyz", container="sidecar", previous=true)
```

#### `kubectl_top`

Get resource utilization metrics.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | `pods` or `nodes` |
| `namespace` | string | No | Namespace (for pods) |
| `name` | string | No | Specific resource name |

```
kubectl_top(resource_type="pods", namespace="production")
kubectl_top(resource_type="nodes")
```

### Write Operations

> **Note:** Write operations require explicit confirmation. Use with caution in production.

#### `kubectl_scale`

Scale a deployment or statefulset.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | `deployment` or `statefulset` |
| `name` | string | Yes | Resource name |
| `replicas` | integer | Yes | Desired replica count |
| `namespace` | string | No | Namespace |

```
kubectl_scale(resource_type="deployment", name="api-server", replicas=5, namespace="production")
```

#### `kubectl_restart`

Trigger a rolling restart of a deployment.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | `deployment` |
| `name` | string | Yes | Resource name |
| `namespace` | string | No | Namespace |

```
kubectl_restart(resource_type="deployment", name="api-server", namespace="production")
```

#### `kubectl_label`

Add or update labels on a resource.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | Resource type |
| `name` | string | Yes | Resource name |
| `labels` | object | Yes | Key-value pairs to set |
| `namespace` | string | No | Namespace |

```
kubectl_label(resource_type="pod", name="api-xyz", labels={"team": "platform"})
```

#### `kubectl_annotate`

Add or update annotations on a resource.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | Resource type |
| `name` | string | Yes | Resource name |
| `annotations` | object | Yes | Key-value pairs to set |
| `namespace` | string | No | Namespace |

```
kubectl_annotate(resource_type="deployment", name="api", annotations={"deploy-note": "hotfix"})
```

## GCloud Tools

#### `gcloud_logging_query`

Query Cloud Logging for log entries.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filter_query` | string | Yes | Cloud Logging filter expression |
| `project_id` | string | No | GCP project (default from config) |
| `limit` | integer | No | Max entries to return (default from config) |
| `order_by` | string | No | Sort order: `timestamp desc` (default) |

```
gcloud_logging_query(
    filter_query='resource.type="k8s_container" AND severity>=ERROR AND resource.labels.namespace_name="production"',
    limit=50
)
```

#### `gcloud_monitoring_query`

Query Cloud Monitoring for time-series metrics.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `metric_type` | string | Yes | Metric type (e.g., `kubernetes.io/container/cpu/core_usage_time`) |
| `project_id` | string | No | GCP project |
| `filter_str` | string | No | Additional MQL filter |
| `interval_minutes` | integer | No | Time window in minutes (default from config: 60) |

```
gcloud_monitoring_query(
    metric_type="kubernetes.io/container/cpu/core_usage_time",
    filter_str='resource.labels.namespace_name="production"',
    interval_minutes=30
)
```

## MCP Tools

See [MCP Guide](mcp-guide.md) for details on MCP server integration.

The MCP bridge dynamically discovers and registers tools from configured MCP servers. These tools become available to agents just like built-in tools.

```bash
# Discover available MCP tools
vaig mcp discover

# Call an MCP tool interactively
vaig mcp call
```

---

[Back to index](README.md)

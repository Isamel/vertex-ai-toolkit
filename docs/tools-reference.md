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

### `verify_completeness`

Verify that one or more files exist, are non-empty, and optionally match a set of regex patterns. Useful for the Verifier agent in the CodingPipeline to confirm implementation completeness before approving a result.

Files larger than `_MAX_FILE_SIZE` (1 MB) are skipped with a warning rather than causing a hard failure.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paths` | array of strings | Yes | Relative paths from workspace root to verify |
| `patterns` | array of strings | No | Regex patterns that must match somewhere in the file content |

```
verify_completeness(paths=["src/auth.py", "tests/test_auth.py"])
verify_completeness(paths=["src/server.go"], patterns=["func main\\(", "func NewServer\\("])
```

Returns a `ToolResult` with `success=True` if all files exist, are non-empty, and all patterns match. On failure, `output` contains a detailed per-file report of what failed.

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

VAIG provides **23 base GKE tools** + **4 Helm tools** (enabled by default) + **5 ArgoCD tools** (opt-in) + **2 GCloud tools** — up to **34 infrastructure tools** total. Additionally, 5 file tools and 1 shell tool are available in coding agent mode.

### Read Operations

#### `kubectl_get`

List Kubernetes resources.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource` | string | Yes | Resource type (e.g., `pods`, `deployments`, `services`) |
| `name` | string | No | Specific resource name (omit to list all) |
| `namespace` | string | No | Namespace (default from config). Use `all` for all namespaces |
| `output_format` | string | No | Output format: `table` (default), `yaml`, `json`, `wide` |
| `label_selector` | string | No | Label filter (e.g., `app=nginx,tier=frontend`) |
| `field_selector` | string | No | Field filter (e.g., `status.phase=Running`) |

Supports 25 resource types with aliases: `pods`/`po`/`pod`, `deployments`/`deploy`, `services`/`svc`, `configmaps`/`cm`, `secrets`, `ingresses`/`ing`, `statefulsets`/`sts`, `daemonsets`/`ds`, `jobs`, `cronjobs`/`cj`, `namespaces`/`ns`, `nodes`, `persistentvolumeclaims`/`pvc`, `persistentvolumes`/`pv`, `serviceaccounts`/`sa`, `networkpolicies`/`netpol`, `horizontalpodautoscalers`/`hpa`, `replicasets`/`rs`, `endpoints`/`ep`, `poddisruptionbudgets`/`pdb`, `resourcequotas`/`quota`.

```
kubectl_get(resource="pods", namespace="production", label_selector="app=api")
kubectl_get(resource="deploy", namespace="all")
```

#### `kubectl_describe`

Get detailed information about a specific resource.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource` | string | Yes | Resource type |
| `name` | string | Yes | Resource name |
| `namespace` | string | No | Namespace |

```
kubectl_describe(resource="pod", name="api-server-xyz", namespace="production")
```

#### `kubectl_logs`

Get container logs from a pod. Automatically fetches previous container logs when the current container is in CrashLoopBackOff.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pod` | string | Yes | Pod name |
| `namespace` | string | No | Namespace |
| `container` | string | No | Container name (for multi-container pods) |
| `tail_lines` | integer | No | Number of recent log lines (default: 100) |
| `since` | string | No | Duration filter (e.g., `1h`, `30m`, `1h30m`) |

```
kubectl_logs(pod="api-server-xyz", namespace="production", tail_lines=100)
kubectl_logs(pod="api-server-xyz", container="sidecar", since="30m")
```

#### `kubectl_top`

Get resource utilization metrics. Requires metrics-server installed in the cluster.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | No | `pods` (default) or `nodes` |
| `name` | string | No | Specific resource name (omit for all) |
| `namespace` | string | No | Namespace for pod metrics. Use `all` for all namespaces |

```
kubectl_top(resource_type="pods", namespace="production")
kubectl_top(resource_type="nodes")
```

#### `get_events`

List Kubernetes events in a namespace, filtered by type and/or involved object. Events reveal WHY pods fail, WHY nodes have issues, and what the scheduler is doing. Critical for SRE triage.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string | No | Namespace (default from config) |
| `event_type` | string | No | Filter by type: `Warning`, `Normal`, or omit for all |
| `involved_object_name` | string | No | Filter by involved object name (e.g., a pod or node name) |
| `involved_object_kind` | string | No | Filter by involved object kind (e.g., `Pod`, `Node`, `Deployment`) |
| `limit` | integer | No | Max events to return (default: 50, max: 500) |

```
get_events(namespace="production", event_type="Warning")
get_events(namespace="default", involved_object_name="api-server-xyz", involved_object_kind="Pod")
get_events(namespace="production", limit=100)
```

#### `get_rollout_status`

Check the rollout status of a deployment — whether it is progressing, complete, stalled, or failed. Reports replica counts, conditions, and rollout strategy. Equivalent to `kubectl rollout status deployment/<name>`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Deployment name |
| `namespace` | string | No | Namespace |

```
get_rollout_status(name="api-server", namespace="production")
```

#### `get_rollout_history`

Show the revision history of a Kubernetes deployment. Lists all revisions by examining ReplicaSets owned by the deployment, similar to `kubectl rollout history deployment/<name>`. When a specific revision number is provided, shows detailed pod template information for that revision (containers, images, ports, env var names, volume mounts, resource requests/limits). Use BEFORE recommending a rollback to understand what changed in each revision.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Deployment name |
| `namespace` | string | No | Namespace |
| `revision` | integer | No | Specific revision number for detailed view. Omit to list all revisions |

```
get_rollout_history(name="api-server", namespace="production")
get_rollout_history(name="api-server", namespace="production", revision=3)
```

#### `get_node_conditions`

Show node health conditions, resource pressure, taints, and capacity. Without a node name, lists ALL nodes with status, roles, version, OS, kernel, container runtime, and CPU/memory capacity. With a node name, shows detailed conditions (MemoryPressure, DiskPressure, PIDPressure, NetworkUnavailable), taints, labels, and capacity vs allocatable.

Fills the gap where `kubectl_get nodes` only shows Ready/NotReady but hides pressure conditions that indicate imminent node failures.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | No | Specific node name for detailed view. Omit to list all nodes |

```
get_node_conditions()
get_node_conditions(name="gke-prod-pool-abc123")
```

#### `get_container_status`

Show detailed container-level status for ALL containers in a pod (init, regular, and ephemeral). For each container: name, image, state (Waiting/Running/Terminated with reason and exit code), ready flag, restart count, last termination state (crucial for CrashLoopBackOff debugging), resource requests/limits, volume mounts, and env var sources (names only — no secret values exposed).

Essential for debugging multi-container pods where `kubectl_get pods` only shows pod-level status.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Pod name to inspect |
| `namespace` | string | No | Namespace |

```
get_container_status(name="api-server-xyz", namespace="production")
```

#### `check_rbac`

Check whether a service account or the current user has permission to perform a specific action on a Kubernetes resource. Uses SubjectAccessReview (for service accounts) or SelfSubjectAccessReview (for current user). Use BEFORE operations that might fail with permission errors.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `verb` | string | Yes | Action to check: `get`, `list`, `watch`, `create`, `update`, `patch`, `delete` |
| `resource` | string | Yes | Resource type (aliases accepted: `po`, `svc`, `deploy`, etc.) |
| `namespace` | string | Yes | Namespace |
| `service_account` | string | No | Service account name. Omit to check current user |
| `resource_name` | string | No | Specific resource name to check access for |

```
check_rbac(verb="get", resource="pods", namespace="production")
check_rbac(verb="delete", resource="deployments", namespace="production", service_account="ci-bot")
check_rbac(verb="list", resource="secrets", namespace="kube-system", service_account="app-sa")
```

### Write Operations

> **Note:** Write operations require explicit confirmation. Use with caution in production.

#### `kubectl_scale`

Scale a deployment, statefulset, or replicaset. Replicas are clamped to the 0-50 range. Reports the previous and new replica count.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource` | string | Yes | `deployments`, `statefulsets`, or `replicasets` |
| `name` | string | Yes | Resource name |
| `replicas` | integer | Yes | Target replica count (0-50) |
| `namespace` | string | No | Namespace |

```
kubectl_scale(resource="deployment", name="api-server", replicas=5, namespace="production")
```

#### `kubectl_restart`

Trigger a rolling restart of a deployment, statefulset, or daemonset. Equivalent to `kubectl rollout restart`. Causes a zero-downtime rolling update.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource` | string | Yes | `deployments`, `statefulsets`, or `daemonsets` |
| `name` | string | Yes | Resource name |
| `namespace` | string | No | Namespace |

```
kubectl_restart(resource="deployment", name="api-server", namespace="production")
```

#### `kubectl_label`

Add or update labels on a resource. System labels (`kubernetes.io/`, `k8s.io/`) are protected and cannot be modified.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource` | string | Yes | Resource type |
| `name` | string | Yes | Resource name |
| `labels` | string | Yes | Labels to set: `key1=value1,key2=value2`. Use `key-` to remove |
| `namespace` | string | No | Namespace |

```
kubectl_label(resource="pod", name="api-xyz", labels="team=platform,env=prod")
```

#### `kubectl_annotate`

Add or update annotations on a resource. System annotations (`kubernetes.io/`, `k8s.io/`) are protected and cannot be modified.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource` | string | Yes | Resource type |
| `name` | string | Yes | Resource name |
| `annotations` | string | Yes | Annotations to set: `key1=value1,key2=value2`. Use `key-` to remove |
| `namespace` | string | No | Namespace |

```
kubectl_annotate(resource="deployment", name="api", annotations="deploy-note=hotfix")
```

### Exec Operations

> **Security:** Exec operations are **disabled by default** and require explicit opt-in.

#### `exec_command`

Execute a diagnostic command inside a running container.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pod_name` | string | Yes | Pod name |
| `namespace` | string | Yes | Namespace |
| `command` | string | Yes | Diagnostic command to execute (must match allowlist) |
| `container` | string | No | Container name (for multi-container pods) |
| `timeout` | integer | No | Execution timeout in seconds (default: 30, max: 300) |

**Security model (three layers):**

1. **Config gate:** `gke.exec_enabled` must be `true` (disabled by default)
2. **Denylist:** Commands matching dangerous patterns are always rejected (shell metacharacters `; & | \``, redirection `>`, destructive commands like `rm`, `kill`, `sudo`, `chmod`, etc.)
3. **Allowlist:** Command must start with an allowed prefix:
   `cat`, `head`, `tail`, `ls`, `env`, `printenv`, `whoami`, `id`, `hostname`, `date`, `ps`, `top -bn1`, `df`, `du`, `mount`, `ip`, `ifconfig`, `netstat`, `ss`, `nslookup`, `dig`, `ping`, `curl`, `wget`, `nc`, `java -version`, `python --version`, `node --version`, `cat /etc/resolv.conf`, `cat /etc/hosts`

Output is truncated to 10,000 characters.

```
exec_command(pod_name="api-server-xyz", namespace="production", command="cat /etc/resolv.conf")
exec_command(pod_name="api-server-xyz", namespace="production", command="ps aux", container="app")
exec_command(pod_name="debug-pod", namespace="default", command="curl -s http://localhost:8080/healthz", timeout=10)
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

## Labels & Annotations Tool

### `kubectl_get_labels`

Get labels and annotations for Kubernetes resources. Supports filtering by label selector (server-side) or annotation prefix (client-side). Essential for detecting management context (GitOps, Helm, Operator).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `resource_type` | string | Yes | Resource type (e.g. `deployments`, `pods`, `services`) |
| `namespace` | string | No | Namespace (default from config) |
| `name` | string | No | Specific resource name. Omit to list all. |
| `label_filter` | string | No | Kubernetes label selector (e.g. `app.kubernetes.io/managed-by=Helm`) |
| `annotation_filter` | string | No | Annotation key prefix to match (client-side, e.g. `argocd.argoproj.io`) |

```
# Get labels for a specific deployment
kubectl_get_labels(resource_type="deployments", name="my-app", namespace="production")

# Find all Helm-managed deployments
kubectl_get_labels(resource_type="deployments", namespace="production", label_filter="app.kubernetes.io/managed-by=Helm")

# Find all ArgoCD-managed resources
kubectl_get_labels(resource_type="deployments", namespace="production", annotation_filter="argocd.argoproj.io")
```

## Helm Tools

**Enabled by default** — disable with `helm.enabled: false` in config.

Read-only introspection of Helm releases. Reads Helm release data from Kubernetes Secrets (type `helm.sh/release.v1`). No Helm binary required.

### `helm_list_releases`

List all Helm releases with status, revision, and chart info.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string | No | Filter by namespace. Empty for all. |
| `force_refresh` | boolean | No | Bypass cache (default: false) |

### `helm_release_status`

Get detailed status of a specific Helm release.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `release_name` | string | Yes | Name of the Helm release |
| `namespace` | string | No | Namespace of the release |

### `helm_release_history`

Show revision history for a Helm release (when it was upgraded, rolled back, etc.).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `release_name` | string | Yes | Name of the Helm release |
| `namespace` | string | No | Namespace of the release |

### `helm_release_values`

Show user-supplied values for a Helm release (overrides only, not chart defaults).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `release_name` | string | Yes | Name of the Helm release |
| `namespace` | string | No | Namespace of the release |
| `all_values` | boolean | No | Include chart defaults (default: false — user overrides only) |

## ArgoCD Tools

**Disabled by default** — enable with `argocd.enabled: true` in config.

Read-only introspection of Argo CD Applications. Supports multiple connection topologies: same cluster, separate management cluster, or ArgoCD API server. See [Architecture](architecture.md) for topology diagrams.

### `argocd_list_applications`

List all Argo CD Applications with sync and health status.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | string | No | Filter by destination namespace |

### `argocd_app_status`

Get detailed status of an ArgoCD Application: source, destination, sync status, health, conditions, and managed resources.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `app_name` | string | Yes | Name of the ArgoCD Application |

### `argocd_app_history`

Show sync history for an ArgoCD Application (which commits were deployed and when).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `app_name` | string | Yes | Name of the ArgoCD Application |

### `argocd_app_diff`

Show resources that are OutOfSync between the desired state (Git) and the live cluster.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `app_name` | string | Yes | Name of the ArgoCD Application |

### `argocd_app_managed_resources`

List all Kubernetes resources managed by an ArgoCD Application with per-resource sync and health status.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `app_name` | string | Yes | Name of the ArgoCD Application |

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

"""Tools package — tool definitions and registry for agent tool use."""

from vaig.tools.agent_tool import agent_as_tool
from vaig.tools.base import ToolDef, ToolParam, ToolRegistry, ToolResult
from vaig.tools.file_tools import (
    create_file_tools,
    edit_file,
    list_files,
    read_file,
    search_files,
    write_file,
)
from vaig.tools.knowledge import create_knowledge_tools
from vaig.tools.plugin_loader import load_all_plugin_tools
from vaig.tools.shell_tools import create_shell_tools, run_command

# GKE tools — optional dependency (kubernetes package)
try:
    from vaig.tools.gke_tools import (
        create_gke_tools,
        kubectl_describe,
        kubectl_get,
        kubectl_logs,
        kubectl_top,
    )
except ImportError:
    pass

# GCP observability tools — optional dependency (google-cloud-logging/monitoring)
try:
    from vaig.tools.gcloud_tools import (
        create_gcloud_tools,
        gcloud_logging_query,
        gcloud_monitoring_query,
    )
except ImportError:
    pass

__all__ = [
    "ToolDef",
    "ToolParam",
    "ToolRegistry",
    "ToolResult",
    "agent_as_tool",
    # Always-available tools
    "create_file_tools",
    "create_knowledge_tools",
    "create_shell_tools",
    "edit_file",
    "list_files",
    "load_all_plugin_tools",
    "read_file",
    "run_command",
    "search_files",
    "write_file",
    # Optional: GKE tools (requires 'kubernetes')
    "create_gke_tools",
    "kubectl_describe",
    "kubectl_get",
    "kubectl_logs",
    "kubectl_top",
    # Optional: GCP observability tools (requires 'google-cloud-logging/monitoring')
    "create_gcloud_tools",
    "gcloud_logging_query",
    "gcloud_monitoring_query",
]

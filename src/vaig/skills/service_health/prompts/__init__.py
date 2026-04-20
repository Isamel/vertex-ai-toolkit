"""Public API for service_health prompts package.

This package replaces the monolithic prompts.py with focused modules.
All external imports remain backward-compatible.
"""

from vaig.core.prompt_defense import (
    ANTI_HALLUCINATION_RULES,
    ANTI_INJECTION_RULE,
    COT_INSTRUCTION,
    DELIMITER_DATA_END,
    DELIMITER_DATA_START,
    _sanitize_namespace,
    wrap_untrusted_content,
)

from ._analyzer import (
    _CHANGE_CORRELATION_PROMPT,
    _CONTRADICTION_RULES_PROMPT,
    _RECENT_CHANGES_PROMPT,
    AUTONOMOUS_OVERLAY as ANALYZER_AUTONOMOUS_OVERLAY,
    HEALTH_ANALYZER_PROMPT,
)
from ._gatherer import _GATHERER_PROMPT_TEMPLATE, HEALTH_GATHERER_PROMPT, build_gatherer_prompt
from ._investigator import AUTONOMOUS_OVERLAY as INVESTIGATOR_AUTONOMOUS_OVERLAY, HEALTH_INVESTIGATOR_PROMPT
from ._phases import PHASE_PROMPTS
from ._planner import HEALTH_PLANNER_PROMPT
from ._reporter import (
    _REMEDIATION_CORE_SECTION,
    _REMEDIATION_GITOPS_SECTION,
    _REMEDIATION_HELM_SECTION,
    _REMEDIATION_MANUAL_SECTION,
    HEALTH_REPORTER_PROMPT,
    build_reporter_prompt,
)
from ._shared import _CORE_TOOLS_TABLE, _DATADOG_API_STEP, _PRIORITY_HIERARCHY
from ._sub_gatherers import (
    build_datadog_gatherer_prompt,
    build_event_gatherer_prompt,
    build_logging_gatherer_prompt,
    build_node_gatherer_prompt,
    build_workload_gatherer_prompt,
)
from ._system import (
    _SYSTEM_INSTRUCTION_ANALYSIS,
    _SYSTEM_INSTRUCTION_UNIVERSAL,
    SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION_GATHERER,
)
from ._verifier import HEALTH_VERIFIER_PROMPT

__all__ = [
    # system
    "SYSTEM_INSTRUCTION",
    "SYSTEM_INSTRUCTION_GATHERER",
    "_SYSTEM_INSTRUCTION_ANALYSIS",
    "_SYSTEM_INSTRUCTION_UNIVERSAL",
    # gatherer
    "HEALTH_GATHERER_PROMPT",
    "build_gatherer_prompt",
    # analyzer
    "HEALTH_ANALYZER_PROMPT",
    "ANALYZER_AUTONOMOUS_OVERLAY",
    "_CONTRADICTION_RULES_PROMPT",
    "_CHANGE_CORRELATION_PROMPT",
    "_RECENT_CHANGES_PROMPT",
    # verifier
    "HEALTH_VERIFIER_PROMPT",
    # planner
    "HEALTH_PLANNER_PROMPT",
    # investigator
    "HEALTH_INVESTIGATOR_PROMPT",
    "INVESTIGATOR_AUTONOMOUS_OVERLAY",
    # reporter
    "HEALTH_REPORTER_PROMPT",
    "build_reporter_prompt",
    # sub-gatherers
    "build_node_gatherer_prompt",
    "build_workload_gatherer_prompt",
    "build_datadog_gatherer_prompt",
    "build_event_gatherer_prompt",
    "build_logging_gatherer_prompt",
    # phases
    "PHASE_PROMPTS",
    # shared internal (accessed by tests)
    "_CORE_TOOLS_TABLE",
    "_DATADOG_API_STEP",
    "_PRIORITY_HIERARCHY",
    "_GATHERER_PROMPT_TEMPLATE",
    "_REMEDIATION_CORE_SECTION",
    "_REMEDIATION_GITOPS_SECTION",
    "_REMEDIATION_HELM_SECTION",
    "_REMEDIATION_MANUAL_SECTION",
    # re-exported from prompt_defense (backward-compat)
    "ANTI_HALLUCINATION_RULES",
    "ANTI_INJECTION_RULE",
    "COT_INSTRUCTION",
    "DELIMITER_DATA_START",
    "DELIMITER_DATA_END",
    "_sanitize_namespace",
    "wrap_untrusted_content",
]

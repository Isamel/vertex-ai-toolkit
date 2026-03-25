"""Phase prompts for the service health skill.

Contains PHASE_PROMPTS: the per-phase system instructions used by the
legacy single-agent pipeline (analyze / execute / report phases).
"""

from vaig.core.prompt_defense import (
    ANTI_INJECTION_RULE,
    DELIMITER_DATA_END,
    DELIMITER_DATA_START,
)

PHASE_PROMPTS = {
    "analyze": f"""## Phase: Service Health Analysis

{ANTI_INJECTION_RULE}

Analyze the current health status of Kubernetes services.

{DELIMITER_DATA_START}
### Context (cluster data):
{{context}}
{DELIMITER_DATA_END}

# NOTE: user_input is placed OUTSIDE data delimiters intentionally.
# It is the user's trusted query, not external/untrusted data.
# Do NOT move it inside DELIMITER_DATA_START/END.
### User's request:
{{user_input}}

### Your Task:
1. Review the provided cluster data for health indicators
2. Identify any services showing degradation or failure
3. Check resource utilization patterns
4. List all observable health issues with severity
5. Note any gaps in monitoring or data

### CRITICAL RULES:
- Base ALL findings exclusively on the provided context data. NEVER invent pod names,
metrics, timestamps, or events.
- If the context data is empty or insufficient, state that clearly instead of fabricating
a health assessment.
- Every finding MUST cite specific evidence from the context data above.

Format your response as a structured health assessment.
""",
    "execute": f"""## Phase: Health Data Collection & Analysis

{ANTI_INJECTION_RULE}

Collect and analyze service health data from the Kubernetes cluster.

{DELIMITER_DATA_START}
### Context:
{{context}}
{DELIMITER_DATA_END}

# NOTE: user_input is placed OUTSIDE data delimiters intentionally.
# It is the user's trusted query, not external/untrusted data.
# Do NOT move it inside DELIMITER_DATA_START/END.
### User's request:
{{user_input}}

### Your Task:
1. Gather pod status, resource usage, events, and logs using available tools
2. Analyze the collected data for health issues
3. Identify patterns and correlations
4. Assess overall cluster health

### CRITICAL RULES:
- Report ONLY data returned by the tools. NEVER fabricate tool outputs, pod names,
metrics, or events.
- If a tool call fails or returns no data, record that fact — do NOT invent substitute data.
- Every claim in the assessment MUST be traceable to actual tool output.

Provide a comprehensive health assessment with evidence.
""",
    "report": f"""## Phase: Health Report Generation

{ANTI_INJECTION_RULE}

Generate a comprehensive service health report.

{DELIMITER_DATA_START}
### Context:
{{context}}

### Analysis results:
{{user_input}}
{DELIMITER_DATA_END}

### Your Task:
Generate a structured JSON report conforming to the HealthReport schema, including:
- Executive Summary
- Service Status Table
- Findings by severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
- Root Cause Hypotheses
- Recommended Actions with kubectl commands
- Event Timeline

### CRITICAL RULES:
- ONLY include data that appears in the analysis results above. NEVER invent pod names,
metrics, percentages, or timestamps.
- If the analysis results do not provide data for a report section, write "Data not
available" rather than fabricating content.
- A shorter, accurate report is always preferred over a longer report with fabricated details.

Make every finding specific and every recommendation actionable.
""",
}

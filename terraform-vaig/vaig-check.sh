#!/usr/bin/env bash
# vaig-check.sh — Terraform external data source wrapper for vaig check.
#
# Reads the Terraform query JSON from stdin, extracts parameters, invokes
# `vaig check`, and always writes valid JSON to stdout — even on error.
#
# Usage in HCL:
#   data "external" "vaig_health" {
#     program = ["bash", "${path.module}/vaig-check.sh"]
#     query   = { namespace = var.namespace, ... }
#   }
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

error_json() {
  local msg="${1:-unknown error}"
  cat <<EOF
{"status":"ERROR","critical_count":"0","warning_count":"0","issues_found":"0","services_checked":"0","summary_text":"${msg}","scope":"","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)","version":"unknown","cached":"false"}
EOF
  exit 2
}

# ── prerequisite checks ─────────────────────────────────────────────────────

command -v vaig >/dev/null 2>&1 || error_json "vaig not found — install with: pip install vaig"
command -v jq   >/dev/null 2>&1 || error_json "jq not found — install jq to parse Terraform input"

# ── read Terraform stdin JSON ────────────────────────────────────────────────

INPUT=$(cat)

NAMESPACE=$(echo "$INPUT" | jq -r '.namespace // empty')
CLUSTER=$(echo "$INPUT"   | jq -r '.cluster   // empty')
PROJECT=$(echo "$INPUT"   | jq -r '.project   // empty')
TIMEOUT=$(echo "$INPUT"   | jq -r '.timeout   // "120"')

# ── build vaig check arguments ──────────────────────────────────────────────

ARGS=()
[[ -n "$NAMESPACE" ]] && ARGS+=(--namespace "$NAMESPACE")
[[ -n "$CLUSTER"   ]] && ARGS+=(--cluster   "$CLUSTER")
[[ -n "$PROJECT"   ]] && ARGS+=(--project   "$PROJECT")
ARGS+=(--timeout "$TIMEOUT")
ARGS+=(--cached)

# ── invoke vaig check ───────────────────────────────────────────────────────

OUTPUT=$(vaig check "${ARGS[@]}" 2>/dev/null) || {
  rc=$?
  # If vaig check wrote valid JSON to stdout, forward it even on non-zero exit
  if echo "$OUTPUT" | jq . >/dev/null 2>&1; then
    echo "$OUTPUT"
    exit $rc
  fi
  error_json "vaig check failed with exit code ${rc}"
}

# Validate output is JSON before forwarding to Terraform
if echo "$OUTPUT" | jq . >/dev/null 2>&1; then
  echo "$OUTPUT"
else
  error_json "vaig check produced invalid JSON"
fi

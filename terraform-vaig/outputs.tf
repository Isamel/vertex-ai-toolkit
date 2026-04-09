# outputs.tf — Expose health check results for downstream modules.

output "health_status" {
  description = "Overall health status: HEALTHY, DEGRADED, CRITICAL, UNKNOWN, ERROR, or TIMEOUT."
  value       = local.health_status
}

output "critical_count" {
  description = "Number of critical findings."
  value       = local.critical_count
}

output "warning_count" {
  description = "Number of warning-level findings."
  value       = local.warning_count
}

output "summary" {
  description = "Human-readable summary of the health check result."
  value       = local.summary_text
}

output "is_cached" {
  description = "Whether the result was served from cache."
  value       = local.is_cached
}

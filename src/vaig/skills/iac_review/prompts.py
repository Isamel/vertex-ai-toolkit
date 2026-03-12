"""IaC Review Skill — prompts for Infrastructure-as-Code review and analysis."""

SYSTEM_INSTRUCTION = """You are a Senior Cloud Infrastructure Engineer with 15+ years of experience \
designing, reviewing, and hardening Infrastructure-as-Code across enterprise environments.

## Your Expertise
- Deep proficiency in Terraform, Pulumi, AWS CloudFormation, AWS CDK, and Crossplane
- Cloud security across AWS, GCP, and Azure (IAM, networking, encryption, secrets management)
- Cost optimization strategies (right-sizing, reserved capacity, spot/preemptible, storage tiering)
- Reliability engineering (multi-AZ, auto-scaling, health checks, disaster recovery, chaos testing)
- Compliance frameworks: SOC 2, HIPAA, PCI-DSS, CIS Benchmarks, NIST 800-53, GDPR
- IaC quality patterns: DRY modules, state management, drift detection, policy-as-code

## Review Methodology
1. **Security Posture Analysis**: Evaluate network exposure, IAM permissions, encryption at rest \
and in transit, secrets handling, and attack surface area
2. **Cost Impact Assessment**: Identify oversized resources, missing reservations, idle capacity, \
storage waste, and cross-region transfer costs
3. **Reliability Evaluation**: Assess redundancy, failover mechanisms, backup strategies, \
scaling policies, health checks, and blast radius containment
4. **Compliance Mapping**: Check resource configurations against applicable compliance frameworks \
and organizational policies
5. **IaC Quality Review**: Evaluate code structure, modularity, variable hygiene, state management, \
provider pinning, and testing coverage

## Output Standards
- Rate every finding with severity: CRITICAL / HIGH / MEDIUM / LOW / INFO
- Provide the exact resource path and line reference for each finding
- Include a concrete remediation code snippet for every actionable finding
- Distinguish between CONFIRMED misconfigurations and POTENTIAL risks (context-dependent)
- Never assume cloud provider — detect from the IaC code and state findings accordingly
- End every response with a prioritized action plan ordered by risk-to-effort ratio
- Assign an overall IaC Quality Score (0–100) based on security, cost, reliability, and code quality
"""

PHASE_PROMPTS = {
    "analyze": """## Phase: IaC Security & Configuration Analysis

Analyze the provided Infrastructure-as-Code for security misconfigurations, cost issues, \
reliability gaps, and IaC anti-patterns.

### IaC Code / Context:
{context}

### User's request:
{user_input}

### Your Task:

#### 1. Security Misconfigurations
- **Network Exposure**: Open security groups (0.0.0.0/0 ingress), missing NACLs, public subnets \
without justification, unrestricted egress rules
- **IAM & Access Control**: Overly permissive policies (wildcards in actions/resources), missing \
least-privilege boundaries, cross-account trust without conditions, missing MFA enforcement
- **Encryption Gaps**: Unencrypted storage volumes, databases without encryption at rest, missing \
TLS/SSL for in-transit data, default KMS keys instead of CMKs
- **Secrets Handling**: Hardcoded credentials, API keys, or tokens in IaC files; missing Secrets \
Manager or Vault integration; plaintext sensitive outputs
- **Logging & Monitoring**: Missing CloudTrail/Audit Logs, disabled VPC Flow Logs, no access \
logging on storage buckets, missing alerting on sensitive operations

#### 2. Cost Optimization Issues
- **Oversized Resources**: Instances larger than workload requires, over-provisioned databases, \
excessive EBS/disk allocations
- **Missing Cost Controls**: No reserved instances or savings plans for stable workloads, missing \
spot/preemptible for fault-tolerant jobs, no lifecycle policies on storage
- **Idle & Waste**: Unattached volumes, unused Elastic IPs, orphaned snapshots, load balancers \
with no targets, NAT gateways in unused AZs
- **Data Transfer**: Cross-region replication without justification, public endpoints where \
VPC endpoints would reduce cost, missing CDN for static assets

#### 3. Reliability Gaps
- **Single Points of Failure**: Single-AZ deployments for production, no read replicas, missing \
failover configurations, single NAT gateway
- **Scaling Deficiencies**: Missing auto-scaling policies, no scale-to-zero for non-production, \
fixed capacity without burst capability
- **Health & Recovery**: Missing health checks on load balancers and target groups, no automated \
backup policies, undefined RTO/RPO, missing dead-letter queues
- **Blast Radius**: No resource isolation between environments, shared VPCs without segmentation, \
missing circuit breakers or rate limiting

#### 4. IaC Anti-Patterns
- **Hardcoded Values**: Magic numbers, hardcoded AMI IDs, region-specific values without variables, \
embedded account IDs
- **State Management**: Missing remote state backend, no state locking (DynamoDB/GCS), state file \
in version control, missing state encryption
- **Module Hygiene**: Monolithic configurations without modules, unpinned module versions, modules \
sourced from unverified registries, copy-pasted resource blocks
- **Provider & Version Pinning**: Unpinned provider versions, missing required_version constraints, \
no .terraform.lock.hcl committed
- **Naming & Tagging**: Inconsistent naming conventions, missing mandatory tags (environment, \
owner, cost-center, terraform-managed), no tagging policy enforcement

Format your response as a structured analysis with a summary table of findings by category and \
severity. Include the exact resource identifier and file location for each finding.
""",

    "report": """## Phase: IaC Review Report

Generate a comprehensive Infrastructure-as-Code review report.

### IaC Code / Context:
{context}

### Analysis results:
{user_input}

### Generate Report:

# Infrastructure-as-Code Review Report

## Executive Summary
(3-5 sentences: overall IaC health assessment, critical risk count, estimated cost impact, \
compliance posture, and top recommendation)

## IaC Quality Score

| Dimension        | Score (0-100) | Grade | Key Factor                        |
|------------------|---------------|-------|-----------------------------------|
| Security         |               |       |                                   |
| Cost Efficiency  |               |       |                                   |
| Reliability      |               |       |                                   |
| Code Quality     |               |       |                                   |
| Compliance       |               |       |                                   |
| **Overall**      |               |       |                                   |

## Security Findings

### CRITICAL
(Findings that expose the infrastructure to immediate exploit or data breach risk)
| # | Resource | Finding | Impact | Remediation |
|---|----------|---------|--------|-------------|

### HIGH
(Findings that significantly weaken security posture)
| # | Resource | Finding | Impact | Remediation |
|---|----------|---------|--------|-------------|

### MEDIUM
(Findings that represent defense-in-depth gaps)
| # | Resource | Finding | Impact | Remediation |
|---|----------|---------|--------|-------------|

### LOW / INFO
(Best practice recommendations and hardening suggestions)
| # | Resource | Finding | Recommendation |
|---|----------|---------|----------------|

## Cost Optimization Opportunities

| # | Resource | Current Config | Recommended | Est. Monthly Savings | Effort |
|---|----------|----------------|-------------|----------------------|--------|

**Total Estimated Monthly Savings**: $___

## Reliability Assessment

| Component | Current State | Risk Level | Recommendation |
|-----------|--------------|------------|----------------|

### Single Points of Failure
(List any SPOF identified with mitigation strategies)

### Disaster Recovery Gaps
(RTO/RPO assessment and backup coverage)

### Scaling Readiness
(Auto-scaling configuration and capacity planning assessment)

## Compliance Gaps

| Framework | Control | Current State | Required State | Gap Description |
|-----------|---------|---------------|----------------|-----------------|

## IaC Code Quality

### Module Structure
(Assessment of modularity, reusability, and DRY principles)

### State Management
(Backend configuration, locking, encryption, and isolation assessment)

### Variable & Output Hygiene
(Validation rules, descriptions, sensitive marking, default values)

### Provider Management
(Version pinning, lock files, provider configuration)

### Tagging Strategy
(Mandatory tags coverage, consistency, policy enforcement)

## Destructive Change Warnings
(Any resources marked for destruction or replacement that could cause data loss or downtime)

| Resource | Action | Risk | Mitigation |
|----------|--------|------|------------|

## Action Items

### Immediate (block deployment)
| # | Action | Category | Severity | Effort |
|---|--------|----------|----------|--------|

### Short-term (next sprint)
| # | Action | Category | Severity | Effort |
|---|--------|----------|----------|--------|

### Long-term (next quarter)
| # | Action | Category | Severity | Effort |
|---|--------|----------|----------|--------|

## Appendix: Resources Reviewed
| Resource Type | Count | Provider | Issues Found |
|---------------|-------|----------|--------------|
""",
}

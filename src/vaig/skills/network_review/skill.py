"""Network Architecture Review Skill — network security, topology, and performance analysis."""

from __future__ import annotations

from typing import Any

from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.network_review.prompts import PHASE_PROMPTS, SYSTEM_INSTRUCTION


class NetworkReviewSkill(BaseSkill):
    """Network Architecture Review skill for security, topology, and performance analysis.

    Supports multi-agent execution with specialized agents:
    - Security Reviewer: Analyzes firewall rules, segmentation, DNS, and encryption
    - Topology Analyzer: Evaluates redundancy, routing, latency, and service mesh
    - Network Lead: Synthesizes into network health report with recommendations
    """

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="network-review",
            display_name="Network Architecture Review",
            description=(
                "Review network architecture for security vulnerabilities, topology "
                "weaknesses, DNS misconfigurations, and service mesh policy issues"
            ),
            version="1.0.0",
            tags=[
                "networking",
                "firewall",
                "dns",
                "load-balancer",
                "service-mesh",
                "security",
            ],
            supported_phases=[
                SkillPhase.ANALYZE,
                SkillPhase.PLAN,
                SkillPhase.EXECUTE,
                SkillPhase.VALIDATE,
                SkillPhase.REPORT,
            ],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        template = PHASE_PROMPTS.get(phase.value, PHASE_PROMPTS["analyze"])
        return template.format(context=context, user_input=user_input)

    def get_agents_config(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "security_reviewer",
                "role": "Network Security Reviewer",
                "system_instruction": (
                    "You are a network security analysis specialist. Your job is to audit "
                    "firewall rules, security groups, and NACLs for over-permissive ingress "
                    "and egress, missing deny-all defaults, unnecessary port exposure, and "
                    "wildcard CIDR ranges. Evaluate network segmentation between environments "
                    "(prod/staging/dev), workload tiers (web/app/data), and trust zones. "
                    "Assess lateral movement risk if a single workload is compromised. Review "
                    "DNS configuration for dangling CNAME records (subdomain takeover risk), "
                    "zone transfer restrictions (AXFR/IXFR), and DNSSEC deployment. Check for "
                    "unencrypted internal traffic between services handling sensitive data. "
                    "Evaluate VPN and peering security — site-to-site VPN configurations, "
                    "transit gateway architectures, and private connectivity to cloud services."
                ),
                "model": "",
            },
            {
                "name": "topology_analyzer",
                "role": "Network Topology Analyzer",
                "system_instruction": (
                    "You are a network topology and performance specialist. Your job is to "
                    "evaluate network architecture for single points of failure — single NAT "
                    "gateway, single availability zone, single ISP, single load balancer. "
                    "Identify asymmetric routing that could cause stateful firewall issues. "
                    "Detect suboptimal traffic paths including hairpinning, tromboning, and "
                    "unnecessary cross-region traffic. Assess latency-sensitive placement — "
                    "are tightly coupled services co-located, or are they separated by "
                    "unnecessary network hops? Evaluate load balancer configuration: health "
                    "check adequacy, connection draining, session persistence, and TLS "
                    "termination. If a service mesh is deployed, verify mTLS enforcement "
                    "mode, traffic policy correctness (timeouts, retries, circuit breakers), "
                    "traffic splitting configurations, and control plane security."
                ),
                "model": "",
            },
            {
                "name": "network_lead",
                "role": "Network Review Lead",
                "system_instruction": SYSTEM_INSTRUCTION,
                "model": "",
            },
        ]

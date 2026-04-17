"""Shared Pydantic schemas for structured LLM output parsing.

These models are used as structured output targets (parsed from JSON) when
Gemini returns structured verification or analysis results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerificationReport(BaseModel):
    """Structured output schema for coding pipeline verification results.

    Used to parse the verifier agent's JSON output when the LLM returns a
    structured response.  All fields except ``success`` have defaults so
    that a minimal ``{"success": true}`` payload parses without error.

    Attributes:
        success: True when verification passed, False when it failed.
        files_verified: List of file paths that were verified.
        issues: List of issue descriptions found during verification.
        summary: Human-readable summary of the verification result.
    """

    model_config = ConfigDict(extra="ignore")

    success: bool
    files_verified: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    summary: str = ""

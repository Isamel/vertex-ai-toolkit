"""Organization, user, and CLI instance models for the platform domain.

These Pydantic V2 models map to the Firestore document schema defined
in REQ-API-005 and are used by both the backend API and the CLI for
serialization / deserialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ConfigPolicy(BaseModel):
    """Organization-level configuration policy.

    Enforced fields override local CLI config unconditionally.
    User-configurable fields can be set locally.
    Blocked fields cannot be changed by users.
    """

    enforced_fields: dict[str, Any] = Field(default_factory=dict)
    user_configurable_fields: list[str] = Field(default_factory=list)
    blocked_fields: list[str] = Field(default_factory=list)


class Organization(BaseModel):
    """Top-level organization document (Firestore: ``organizations/{org_id}``)."""

    org_id: str
    name: str
    created_at: datetime | None = None
    admin_emails: list[str] = Field(default_factory=list)
    config_policy: ConfigPolicy = Field(default_factory=ConfigPolicy)
    quota_policy: dict[str, Any] = Field(default_factory=dict)


class User(BaseModel):
    """Organization member (Firestore: ``organizations/{org_id}/users/{email}``)."""

    email: str
    role: Literal["admin", "operator", "viewer"] = "operator"
    joined_at: datetime | None = None
    last_active: datetime | None = None
    status: Literal["active", "suspended"] = "active"


class CLIInstance(BaseModel):
    """Registered CLI instance (Firestore: ``organizations/{org_id}/cli_instances/{cli_id}``)."""

    cli_id: str
    user_email: str
    machine_id: str = ""
    hostname: str = ""
    os_user: str = ""
    vaig_version: str = ""
    gcp_project: str = ""
    cluster_name: str = ""
    registered_at: datetime | None = None
    last_heartbeat: datetime | None = None
    status: Literal["active", "revoked"] = "active"
    config_version: str = ""


class ConfigHistoryEntry(BaseModel):
    """Config policy change record (Firestore: ``organizations/{org_id}/config_history/{version_id}``)."""

    version_id: str
    pushed_by: str
    pushed_at: datetime | None = None
    config_policy: ConfigPolicy = Field(default_factory=ConfigPolicy)
    changelog: str = ""

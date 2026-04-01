"""Platform data models — Pydantic V2 models for the platform domain.

Re-exports commonly used models for convenient imports:

    from vaig.platform.models import Organization, User, CLIInstance
"""

from vaig.platform.models.auth import (
    AuthResult,
    JWTClaims,
    RefreshRequest,
    RegisterRequest,
    TokenRequest,
    TokenResponse,
    WhoamiResponse,
)
from vaig.platform.models.organization import (
    CLIInstance,
    ConfigHistoryEntry,
    ConfigPolicy,
    Organization,
    User,
)

__all__ = [
    "AuthResult",
    "CLIInstance",
    "ConfigHistoryEntry",
    "ConfigPolicy",
    "JWTClaims",
    "Organization",
    "RefreshRequest",
    "RegisterRequest",
    "TokenRequest",
    "TokenResponse",
    "User",
    "WhoamiResponse",
]

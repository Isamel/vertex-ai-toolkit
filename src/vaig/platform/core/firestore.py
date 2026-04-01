"""Firestore repository — CRUD operations for platform data.

Uses lazy imports for ``google-cloud-firestore`` so that existing users
who don't have it installed are not broken.  All Firestore access goes
through the repository interface — no direct client calls from API
endpoints (REQ-NFR-005).

Provides:
  - ``AbstractRepository``: Abstract base with CRUD methods
  - ``FirestoreRepository``: Concrete implementation using Firestore
  - ``InMemoryRepository``: In-memory implementation for testing
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from vaig.platform.models.organization import (
    CLIInstance,
    ConfigHistoryEntry,
    ConfigPolicy,
    Organization,
    User,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Abstract Repository Interface
# ══════════════════════════════════════════════════════════════


class AbstractRepository(abc.ABC):
    """Abstract base for platform data persistence.

    All methods operate on the Pydantic models from
    ``vaig.platform.models`` — the concrete backend translates
    between models and its storage format.
    """

    # ── Organization ──────────────────────────────────────────

    @abc.abstractmethod
    async def get_organization(self, org_id: str) -> Organization | None:
        """Get an organization by ID."""
        ...

    @abc.abstractmethod
    async def save_organization(self, org: Organization) -> None:
        """Create or update an organization."""
        ...

    # ── User ──────────────────────────────────────────────────

    @abc.abstractmethod
    async def get_user(self, org_id: str, email: str) -> User | None:
        """Get a user by email within an organization."""
        ...

    @abc.abstractmethod
    async def save_user(self, org_id: str, user: User) -> None:
        """Create or update a user in an organization."""
        ...

    @abc.abstractmethod
    async def list_users(self, org_id: str) -> list[User]:
        """List all users in an organization."""
        ...

    # ── CLI Instance ──────────────────────────────────────────

    @abc.abstractmethod
    async def get_cli_instance(self, org_id: str, cli_id: str) -> CLIInstance | None:
        """Get a CLI instance by ID."""
        ...

    @abc.abstractmethod
    async def save_cli_instance(self, org_id: str, instance: CLIInstance) -> None:
        """Create or update a CLI instance."""
        ...

    @abc.abstractmethod
    async def list_cli_instances(self, org_id: str) -> list[CLIInstance]:
        """List all CLI instances in an organization."""
        ...

    @abc.abstractmethod
    async def list_cli_instances_by_user(self, org_id: str, email: str) -> list[CLIInstance]:
        """List CLI instances owned by a specific user."""
        ...

    # ── Config Policy ─────────────────────────────────────────

    @abc.abstractmethod
    async def get_config_policy(self, org_id: str) -> ConfigPolicy | None:
        """Get the current config policy for an organization."""
        ...

    @abc.abstractmethod
    async def save_config_policy(self, org_id: str, policy: ConfigPolicy) -> None:
        """Update the config policy for an organization."""
        ...

    @abc.abstractmethod
    async def add_config_history(self, org_id: str, entry: ConfigHistoryEntry) -> None:
        """Add a config history entry."""
        ...

    @abc.abstractmethod
    async def list_config_history(self, org_id: str) -> list[ConfigHistoryEntry]:
        """List config history entries, newest first."""
        ...


# ══════════════════════════════════════════════════════════════
# In-Memory Repository (for testing)
# ══════════════════════════════════════════════════════════════


class InMemoryRepository(AbstractRepository):
    """In-memory repository for testing — no external dependencies."""

    def __init__(self) -> None:
        self._orgs: dict[str, Organization] = {}
        self._users: dict[str, dict[str, User]] = {}  # org_id -> {email -> User}
        self._cli_instances: dict[str, dict[str, CLIInstance]] = {}  # org_id -> {cli_id -> CLI}
        self._config_history: dict[str, list[ConfigHistoryEntry]] = {}  # org_id -> [entries]
        self._config_policies: dict[str, ConfigPolicy] = {}  # org_id -> ConfigPolicy (standalone)

    # ── Organization ──────────────────────────────────────────

    async def get_organization(self, org_id: str) -> Organization | None:
        return self._orgs.get(org_id)

    async def save_organization(self, org: Organization) -> None:
        self._orgs[org.org_id] = org

    # ── User ──────────────────────────────────────────────────

    async def get_user(self, org_id: str, email: str) -> User | None:
        return self._users.get(org_id, {}).get(email)

    async def save_user(self, org_id: str, user: User) -> None:
        if org_id not in self._users:
            self._users[org_id] = {}
        self._users[org_id][user.email] = user

    async def list_users(self, org_id: str) -> list[User]:
        return list(self._users.get(org_id, {}).values())

    # ── CLI Instance ──────────────────────────────────────────

    async def get_cli_instance(self, org_id: str, cli_id: str) -> CLIInstance | None:
        return self._cli_instances.get(org_id, {}).get(cli_id)

    async def save_cli_instance(self, org_id: str, instance: CLIInstance) -> None:
        if org_id not in self._cli_instances:
            self._cli_instances[org_id] = {}
        self._cli_instances[org_id][instance.cli_id] = instance

    async def list_cli_instances(self, org_id: str) -> list[CLIInstance]:
        return list(self._cli_instances.get(org_id, {}).values())

    async def list_cli_instances_by_user(self, org_id: str, email: str) -> list[CLIInstance]:
        return [
            cli
            for cli in self._cli_instances.get(org_id, {}).values()
            if cli.user_email == email
        ]

    # ── Config Policy ─────────────────────────────────────────

    async def get_config_policy(self, org_id: str) -> ConfigPolicy | None:
        # Check standalone storage first, then fall back to org's embedded policy
        if org_id in self._config_policies:
            return self._config_policies[org_id]
        org = self._orgs.get(org_id)
        return org.config_policy if org else None

    async def save_config_policy(self, org_id: str, policy: ConfigPolicy) -> None:
        # Always persist to standalone storage so saves succeed even without an org
        self._config_policies[org_id] = policy
        org = self._orgs.get(org_id)
        if org:
            org.config_policy = policy

    async def add_config_history(self, org_id: str, entry: ConfigHistoryEntry) -> None:
        if org_id not in self._config_history:
            self._config_history[org_id] = []
        self._config_history[org_id].insert(0, entry)  # newest first

    async def list_config_history(self, org_id: str) -> list[ConfigHistoryEntry]:
        return list(self._config_history.get(org_id, []))


# ══════════════════════════════════════════════════════════════
# Firestore Repository (lazy import — optional dependency)
# ══════════════════════════════════════════════════════════════


class FirestoreRepository(AbstractRepository):
    """Firestore-backed repository.

    Requires ``google-cloud-firestore`` to be installed. If it's not
    available, instantiation raises ``ImportError``.

    Collections:
      - ``organizations/{org_id}``
      - ``organizations/{org_id}/users/{email}``
      - ``organizations/{org_id}/cli_instances/{cli_id}``
      - ``organizations/{org_id}/config_history/{version_id}``
    """

    def __init__(self, project: str | None = None) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise ImportError(
                "google-cloud-firestore is required for FirestoreRepository. "
                "Install it with: pip install google-cloud-firestore"
            ) from exc

        self._db: Any = firestore.AsyncClient(project=project)

    # ── Organization ──────────────────────────────────────────

    async def get_organization(self, org_id: str) -> Organization | None:
        doc = await self._db.collection("organizations").document(org_id).get()
        if not doc.exists:
            return None
        return Organization(org_id=org_id, **doc.to_dict())

    async def save_organization(self, org: Organization) -> None:
        data = org.model_dump(exclude={"org_id"})
        await self._db.collection("organizations").document(org.org_id).set(data, merge=True)

    # ── User ──────────────────────────────────────────────────

    async def get_user(self, org_id: str, email: str) -> User | None:
        doc = (
            await self._db.collection("organizations")
            .document(org_id)
            .collection("users")
            .document(email)
            .get()
        )
        if not doc.exists:
            return None
        return User(**doc.to_dict())

    async def save_user(self, org_id: str, user: User) -> None:
        data = user.model_dump()
        await (
            self._db.collection("organizations")
            .document(org_id)
            .collection("users")
            .document(user.email)
            .set(data, merge=True)
        )

    async def list_users(self, org_id: str) -> list[User]:
        docs = (
            self._db.collection("organizations")
            .document(org_id)
            .collection("users")
            .stream()
        )
        return [User(**doc.to_dict()) async for doc in docs]

    # ── CLI Instance ──────────────────────────────────────────

    async def get_cli_instance(self, org_id: str, cli_id: str) -> CLIInstance | None:
        doc = (
            await self._db.collection("organizations")
            .document(org_id)
            .collection("cli_instances")
            .document(cli_id)
            .get()
        )
        if not doc.exists:
            return None
        return CLIInstance(**doc.to_dict())

    async def save_cli_instance(self, org_id: str, instance: CLIInstance) -> None:
        data = instance.model_dump()
        await (
            self._db.collection("organizations")
            .document(org_id)
            .collection("cli_instances")
            .document(instance.cli_id)
            .set(data, merge=True)
        )

    async def list_cli_instances(self, org_id: str) -> list[CLIInstance]:
        docs = (
            self._db.collection("organizations")
            .document(org_id)
            .collection("cli_instances")
            .stream()
        )
        return [CLIInstance(**doc.to_dict()) async for doc in docs]

    async def list_cli_instances_by_user(self, org_id: str, email: str) -> list[CLIInstance]:
        docs = (
            self._db.collection("organizations")
            .document(org_id)
            .collection("cli_instances")
            .where("user_email", "==", email)
            .stream()
        )
        return [CLIInstance(**doc.to_dict()) async for doc in docs]

    # ── Config Policy ─────────────────────────────────────────

    async def get_config_policy(self, org_id: str) -> ConfigPolicy | None:
        org = await self.get_organization(org_id)
        return org.config_policy if org else None

    async def save_config_policy(self, org_id: str, policy: ConfigPolicy) -> None:
        await (
            self._db.collection("organizations")
            .document(org_id)
            .set({"config_policy": policy.model_dump()}, merge=True)
        )

    async def add_config_history(self, org_id: str, entry: ConfigHistoryEntry) -> None:
        data = entry.model_dump()
        await (
            self._db.collection("organizations")
            .document(org_id)
            .collection("config_history")
            .document(entry.version_id)
            .set(data)
        )

    async def list_config_history(self, org_id: str) -> list[ConfigHistoryEntry]:
        docs = (
            self._db.collection("organizations")
            .document(org_id)
            .collection("config_history")
            .order_by("pushed_at", direction="DESCENDING")
            .stream()
        )
        return [ConfigHistoryEntry(**doc.to_dict()) async for doc in docs]

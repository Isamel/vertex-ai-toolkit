"""Integration tests for config policy API endpoints (Phase 4, Task 4.5).

Covers:
  - SC-API-004a: Get policy (any authenticated user)
  - SC-API-004b: Update policy as admin (success, returns version)
  - SC-API-004c: Update policy as non-admin (403 Forbidden)
  - SC-API-004d: Config history ordering (newest first)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

jwt = pytest.importorskip("jwt", reason="PyJWT not installed — skipping config policy tests")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from vaig.platform.app import create_platform_app
from vaig.platform.core.firestore import InMemoryRepository
from vaig.platform.core.jwt import ACCESS_TOKEN_LIFETIME, JWTService
from vaig.platform.models.organization import ConfigPolicy, Organization

# ── Test RSA Keypair ──────────────────────────────────────────
_rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PRIVATE_KEY = _rsa_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
).decode()
_TEST_PUBLIC_KEY = _rsa_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def jwt_service() -> JWTService:
    return JWTService(private_key=_TEST_PRIVATE_KEY, public_key=_TEST_PUBLIC_KEY)


@pytest.fixture()
def repo() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture()
def client(jwt_service: JWTService, repo: InMemoryRepository) -> TestClient:
    app = create_platform_app(jwt_service=jwt_service, repository=repo)
    return TestClient(app)


# ── Helpers ────────────────────────────────────────────────────


def _make_token(
    jwt_service: JWTService,
    *,
    sub: str = "user@example.com",
    org_id: str = "test-org",
    role: str = "operator",
    cli_id: str = "cli-test123",
    lifetime: int = ACCESS_TOKEN_LIFETIME,
) -> str:
    return jwt_service.issue_token(
        sub=sub, org_id=org_id, role=role, cli_id=cli_id, lifetime=lifetime,
    )


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_org(
    repo: InMemoryRepository,
    org_id: str = "test-org",
    *,
    policy: ConfigPolicy | None = None,
) -> None:
    """Seed an organization, optionally with a config policy."""
    kwargs: dict[str, Any] = {"org_id": org_id, "name": "Test Org"}
    if policy is not None:
        kwargs["config_policy"] = policy
    org = Organization(**kwargs)
    asyncio.run(repo.save_organization(org))


# ══════════════════════════════════════════════════════════════
# SC-API-004a: Get policy
# ══════════════════════════════════════════════════════════════


class TestGetPolicy:
    """Any authenticated user can read the org config policy."""

    def test_get_policy_empty_defaults(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """No org exists → returns empty default policy."""
        token = _make_token(jwt_service)
        resp = client.get("/api/v1/config/policy", headers=_auth_header(token))

        assert resp.status_code == 200
        data = resp.json()
        assert data["enforced_fields"] == {}
        assert data["user_configurable_fields"] == []
        assert data["blocked_fields"] == []

    def test_get_policy_with_existing_policy(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """Org with a policy → returns its enforced_fields."""
        policy = ConfigPolicy(
            enforced_fields={"models.default": "gemini-pro"},
            user_configurable_fields=["gcp.location"],
            blocked_fields=["budget.max_cost"],
        )
        _seed_org(repo, policy=policy)

        token = _make_token(jwt_service)
        resp = client.get("/api/v1/config/policy", headers=_auth_header(token))

        assert resp.status_code == 200
        data = resp.json()
        assert data["enforced_fields"] == {"models.default": "gemini-pro"}
        assert "gcp.location" in data["user_configurable_fields"]
        assert "budget.max_cost" in data["blocked_fields"]

    def test_get_policy_as_operator(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """Operator (non-admin) can also read policy."""
        token = _make_token(jwt_service, role="operator")
        resp = client.get("/api/v1/config/policy", headers=_auth_header(token))
        assert resp.status_code == 200

    def test_get_policy_unauthenticated_returns_401(
        self, client: TestClient
    ) -> None:
        resp = client.get("/api/v1/config/policy")
        assert resp.status_code == 401


# ══════════════════════════════════════════════════════════════
# SC-API-004b: Update policy as admin
# ══════════════════════════════════════════════════════════════


class TestUpdatePolicyAdmin:
    """Admin can update the config policy."""

    def test_update_policy_success(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)

        token = _make_token(jwt_service, role="admin")
        resp = client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={
                "enforced_fields": {"budget.max_cost_per_run": 5.0},
                "changelog": "Set budget limit",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "config_version" in data
        assert data["config_version"].startswith("v-")
        assert "updated_at" in data

    def test_update_policy_persists_in_repo(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)

        token = _make_token(jwt_service, role="admin")
        client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={"enforced_fields": {"models.default": "gemini-ultra"}},
        )

        # Read back from repo
        policy = asyncio.run(repo.get_config_policy("test-org"))
        assert policy is not None
        assert policy.enforced_fields == {"models.default": "gemini-ultra"}

    def test_update_policy_with_multiple_fields(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)

        token = _make_token(jwt_service, role="admin")
        resp = client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={
                "enforced_fields": {
                    "models.default": "gemini-pro",
                    "gcp.location": "us-central1",
                },
                "user_configurable_fields": ["models.temperature"],
                "blocked_fields": ["budget.max_cost"],
            },
        )

        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════
# SC-API-004c: Update policy as non-admin → 403
# ══════════════════════════════════════════════════════════════


class TestUpdatePolicyNonAdmin:
    """Non-admin users cannot update config policy."""

    def test_operator_update_forbidden(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        token = _make_token(jwt_service, role="operator")
        resp = client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={"enforced_fields": {}},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Admin role required"

    def test_unauthenticated_update_returns_401(
        self, client: TestClient
    ) -> None:
        resp = client.put(
            "/api/v1/config/policy",
            json={"enforced_fields": {}},
        )
        assert resp.status_code == 401


# ══════════════════════════════════════════════════════════════
# SC-API-004d: Config history ordering (newest first)
# ══════════════════════════════════════════════════════════════


class TestConfigHistory:
    """Config history returns entries in newest-first order."""

    def test_history_ordering(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        token = _make_token(jwt_service, role="admin")

        # Create multiple policy updates
        for i in range(3):
            client.put(
                "/api/v1/config/policy",
                headers=_auth_header(token),
                json={"enforced_fields": {f"key{i}": f"val{i}"}, "changelog": f"v{i}"},
            )

        resp = client.get("/api/v1/config/history", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

        # InMemoryRepository inserts newest first
        items = data["items"]
        assert items[0]["changelog"] == "v2"
        assert items[1]["changelog"] == "v1"
        assert items[2]["changelog"] == "v0"

    def test_empty_history(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        token = _make_token(jwt_service, role="admin")

        resp = client.get("/api/v1/config/history", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["items"] == []

    def test_history_as_operator_forbidden(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        token = _make_token(jwt_service, role="operator")
        resp = client.get("/api/v1/config/history", headers=_auth_header(token))
        assert resp.status_code == 403

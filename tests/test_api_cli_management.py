"""Integration tests for CLI management API endpoints (Phase 4, Task 4.4).

Covers:
  - SC-API-003a: List CLIs as admin (returns all instances)
  - SC-API-003b: List CLIs as non-admin (returns 403)
  - SC-API-003c: Heartbeat updates CLI instance metadata
  - SC-API-003d: Revoke flow marks CLI as revoked
  - SC-API-003e: Operator can view own CLI details
  - SC-API-003f: Operator cannot view another user's CLI
"""

from __future__ import annotations

import asyncio

import pytest

jwt = pytest.importorskip("jwt", reason="PyJWT not installed — skipping CLI management tests")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from vaig.platform.app import create_platform_app
from vaig.platform.core.firestore import InMemoryRepository
from vaig.platform.core.jwt import ACCESS_TOKEN_LIFETIME, JWTService
from vaig.platform.models.organization import CLIInstance, Organization

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


def _seed_org(repo: InMemoryRepository, org_id: str = "test-org") -> None:
    """Seed an organization in the repo."""
    asyncio.run(repo.save_organization(Organization(org_id=org_id, name="Test Org")))


def _seed_cli(
    repo: InMemoryRepository,
    *,
    org_id: str = "test-org",
    cli_id: str = "cli-1",
    user_email: str = "user@example.com",
    cli_status: str = "active",
) -> CLIInstance:
    """Seed a CLI instance and return it."""
    cli = CLIInstance(cli_id=cli_id, user_email=user_email, status=cli_status)
    asyncio.run(repo.save_cli_instance(org_id, cli))
    return cli


# ══════════════════════════════════════════════════════════════
# SC-API-003a: List CLIs as admin
# ══════════════════════════════════════════════════════════════


class TestListCLIsAdmin:
    """Admin can list all CLI instances in the org."""

    def test_list_returns_all_instances(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        _seed_cli(repo, cli_id="cli-a", user_email="a@test.com")
        _seed_cli(repo, cli_id="cli-b", user_email="b@test.com")
        _seed_cli(repo, cli_id="cli-c", user_email="c@test.com")

        token = _make_token(jwt_service, role="admin")
        resp = client.get("/api/v1/cli/list", headers=_auth_header(token))

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        cli_ids = {item["cli_id"] for item in data["items"]}
        assert cli_ids == {"cli-a", "cli-b", "cli-c"}

    def test_list_empty_org(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)

        token = _make_token(jwt_service, role="admin")
        resp = client.get("/api/v1/cli/list", headers=_auth_header(token))

        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        assert resp.json()["items"] == []


# ══════════════════════════════════════════════════════════════
# SC-API-003b: List CLIs as non-admin → 403
# ══════════════════════════════════════════════════════════════


class TestListCLIsNonAdmin:
    """Non-admin cannot list CLI instances."""

    def test_operator_list_forbidden(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        token = _make_token(jwt_service, role="operator")
        resp = client.get("/api/v1/cli/list", headers=_auth_header(token))
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Admin role required"


# ══════════════════════════════════════════════════════════════
# SC-API-003c: Heartbeat updates metadata
# ══════════════════════════════════════════════════════════════


class TestHeartbeat:
    """Heartbeat updates CLI instance last_heartbeat and metadata."""

    def test_heartbeat_updates_instance(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        cli = _seed_cli(repo, cli_id="cli-hb", user_email="user@example.com")
        assert cli.last_heartbeat is None

        token = _make_token(jwt_service, cli_id="cli-hb")
        resp = client.post(
            "/api/v1/cli/heartbeat",
            headers=_auth_header(token),
            json={"vaig_version": "2.0.0", "cluster_name": "my-cluster"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "config_version" in data

        # Verify instance was updated in repo
        updated = asyncio.run(repo.get_cli_instance("test-org", "cli-hb"))
        assert updated is not None
        assert updated.last_heartbeat is not None
        assert updated.vaig_version == "2.0.0"
        assert updated.cluster_name == "my-cluster"

    def test_heartbeat_nonexistent_cli(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """Heartbeat for non-existent CLI still returns 200 (no crash)."""
        token = _make_token(jwt_service, cli_id="cli-missing")
        resp = client.post(
            "/api/v1/cli/heartbeat",
            headers=_auth_header(token),
            json={},
        )
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════
# SC-API-003d: Revoke flow
# ══════════════════════════════════════════════════════════════


class TestRevokeCLI:
    """Admin revokes a CLI, status becomes 'revoked'."""

    def test_revoke_marks_cli_as_revoked(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        _seed_cli(repo, cli_id="cli-rev", user_email="user@test.com")

        token = _make_token(jwt_service, role="admin")
        resp = client.post("/api/v1/cli/cli-rev/revoke", headers=_auth_header(token))

        assert resp.status_code == 200
        assert resp.json() == {"revoked": True}

        # Verify status updated in repo
        instance = asyncio.run(repo.get_cli_instance("test-org", "cli-rev"))
        assert instance is not None
        assert instance.status == "revoked"

    def test_revoke_nonexistent_cli_returns_404(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        token = _make_token(jwt_service, role="admin")
        resp = client.post("/api/v1/cli/cli-nope/revoke", headers=_auth_header(token))
        assert resp.status_code == 404

    def test_revoke_as_operator_forbidden(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        _seed_cli(repo, cli_id="cli-rev2", user_email="user@test.com")

        token = _make_token(jwt_service, role="operator")
        resp = client.post("/api/v1/cli/cli-rev2/revoke", headers=_auth_header(token))
        assert resp.status_code == 403


# ══════════════════════════════════════════════════════════════
# SC-API-003e: Operator views own CLI
# ══════════════════════════════════════════════════════════════


class TestGetOwnCLI:
    """Operator can view their own CLI instance details."""

    def test_get_own_cli(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        _seed_cli(repo, cli_id="cli-mine", user_email="user@example.com")

        token = _make_token(jwt_service, sub="user@example.com", role="operator")
        resp = client.get("/api/v1/cli/cli-mine", headers=_auth_header(token))

        assert resp.status_code == 200
        data = resp.json()
        assert data["cli_id"] == "cli-mine"
        assert data["user_email"] == "user@example.com"

    def test_admin_can_view_any_cli(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """Admin can see any CLI, regardless of user_email."""
        _seed_org(repo)
        _seed_cli(repo, cli_id="cli-other", user_email="other@test.com")

        token = _make_token(jwt_service, sub="admin@test.com", role="admin")
        resp = client.get("/api/v1/cli/cli-other", headers=_auth_header(token))

        assert resp.status_code == 200
        assert resp.json()["cli_id"] == "cli-other"


# ══════════════════════════════════════════════════════════════
# SC-API-003f: Operator cannot view other user's CLI
# ══════════════════════════════════════════════════════════════


class TestGetOtherCLI:
    """Operator cannot access another user's CLI."""

    def test_operator_other_cli_forbidden(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        _seed_org(repo)
        _seed_cli(repo, cli_id="cli-theirs", user_email="other@test.com")

        token = _make_token(jwt_service, sub="user@example.com", role="operator")
        resp = client.get("/api/v1/cli/cli-theirs", headers=_auth_header(token))

        assert resp.status_code == 403
        assert "denied" in resp.json()["detail"].lower()

    def test_get_nonexistent_cli_returns_404(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        token = _make_token(jwt_service, role="operator")
        resp = client.get("/api/v1/cli/cli-nope", headers=_auth_header(token))
        assert resp.status_code == 404

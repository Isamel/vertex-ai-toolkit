"""Tests for the platform backend API scaffold (Phase 3).

Covers:
  - SC-API-001a: Health check endpoint
  - SC-API-006a/b: JWT issue/validate round-trip
  - SC-API-002f: Auth guard — 401 without token
  - SC-API-002g: Auth guard — 401 with expired token
  - REQ-NFR-002: Admin-only endpoints return 403 for non-admin
  - REQ-API-002: Auth endpoints (register, token, refresh, revoke, whoami)
  - REQ-API-003: CLI management endpoints
  - REQ-API-004: Config policy endpoints
  - REQ-NFR-005: Testable with mock repository (InMemoryRepository)
"""

from __future__ import annotations

import asyncio

import pytest

jwt = pytest.importorskip("jwt", reason="PyJWT not installed — skipping platform API tests")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from vaig.platform.app import create_platform_app
from vaig.platform.core.firestore import InMemoryRepository
from vaig.platform.core.jwt import ACCESS_TOKEN_LIFETIME, JWTError, JWTService
from vaig.platform.models.organization import CLIInstance, Organization

# ── Test RSA Keypair (generated at module load, for testing only) ──
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
    """JWTService with test keypair."""
    return JWTService(private_key=_TEST_PRIVATE_KEY, public_key=_TEST_PUBLIC_KEY)


@pytest.fixture()
def repo() -> InMemoryRepository:
    """Fresh in-memory repository for each test."""
    return InMemoryRepository()


@pytest.fixture()
def client(jwt_service: JWTService, repo: InMemoryRepository) -> TestClient:
    """FastAPI TestClient with test dependencies."""
    app = create_platform_app(jwt_service=jwt_service, repository=repo)
    return TestClient(app)


def _make_token(
    jwt_service: JWTService,
    *,
    sub: str = "user@example.com",
    org_id: str = "test-org",
    role: str = "operator",
    cli_id: str = "cli-test123",
    lifetime: int = ACCESS_TOKEN_LIFETIME,
) -> str:
    """Helper: issue a test JWT."""
    return jwt_service.issue_token(
        sub=sub,
        org_id=org_id,
        role=role,
        cli_id=cli_id,
        lifetime=lifetime,
    )


def _auth_header(token: str) -> dict[str, str]:
    """Helper: Authorization header with Bearer token."""
    return {"Authorization": f"Bearer {token}"}


# ══════════════════════════════════════════════════════════════
# Health Check (SC-API-001a)
# ══════════════════════════════════════════════════════════════


class TestHealthCheck:
    """SC-API-001a: GET /healthz returns 200 {status: ok}."""

    def test_healthz_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_healthz_content_type_json(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert "application/json" in resp.headers["content-type"]


# ══════════════════════════════════════════════════════════════
# JWT Service (SC-API-006a/b)
# ══════════════════════════════════════════════════════════════


class TestJWTService:
    """JWT issue/validate round-trip tests."""

    def test_issue_and_validate_round_trip(self, jwt_service: JWTService) -> None:
        """SC-API-006a: A token we issue can be validated back."""
        token = jwt_service.issue_token(
            sub="alice@example.com",
            org_id="org-abc",
            role="admin",
            cli_id="cli-xyz",
        )
        claims = jwt_service.validate_token(token)
        assert claims.sub == "alice@example.com"
        assert claims.org_id == "org-abc"
        assert claims.role == "admin"
        assert claims.cli_id == "cli-xyz"

    def test_validate_rejects_tampered_token(self, jwt_service: JWTService) -> None:
        """SC-API-006a: Tampered payload is rejected."""
        token = jwt_service.issue_token(sub="alice@example.com")
        # Tamper with the payload (flip a character in the middle segment)
        parts = token.split(".")
        tampered_payload = parts[1][:5] + ("A" if parts[1][5] != "A" else "B") + parts[1][6:]
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"
        with pytest.raises(JWTError, match="Invalid token"):
            jwt_service.validate_token(tampered_token)

    def test_validate_rejects_expired_token(self, jwt_service: JWTService) -> None:
        """SC-API-006b: Expired token is rejected."""
        token = jwt_service.issue_token(sub="alice@example.com", lifetime=-1)
        with pytest.raises(JWTError, match="Token expired"):
            jwt_service.validate_token(token)

    def test_issue_requires_private_key(self) -> None:
        """JWTService without private key cannot issue tokens."""
        svc = JWTService(private_key="", public_key=_TEST_PUBLIC_KEY)
        with pytest.raises(JWTError, match="Private key not configured"):
            svc.issue_token(sub="test@example.com")

    def test_validate_requires_public_key(self) -> None:
        """JWTService without public key cannot validate tokens."""
        svc = JWTService(private_key=_TEST_PRIVATE_KEY, public_key="")
        token = svc.issue_token(sub="test@example.com")
        with pytest.raises(JWTError, match="Public key not configured"):
            svc.validate_token(token)


# ══════════════════════════════════════════════════════════════
# Auth Guards (SC-API-002f/g, REQ-NFR-002)
# ══════════════════════════════════════════════════════════════


class TestAuthGuards:
    """Auth guard tests — 401 without token, 401 with expired, 403 without admin."""

    def test_401_without_token(self, client: TestClient) -> None:
        """SC-API-002f: Authenticated endpoint without token returns 401."""
        resp = client.get("/api/v1/auth/whoami")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated"

    def test_401_with_expired_token(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """SC-API-002g: Expired token returns 401."""
        token = _make_token(jwt_service, lifetime=-1)
        resp = client.get("/api/v1/auth/whoami", headers=_auth_header(token))
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_401_with_invalid_token(self, client: TestClient) -> None:
        """Invalid (garbage) token returns 401."""
        resp = client.get(
            "/api/v1/auth/whoami",
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
        assert resp.status_code == 401

    def test_403_non_admin_on_admin_endpoint(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """REQ-NFR-002: Non-admin on admin-only endpoint returns 403."""
        token = _make_token(jwt_service, role="operator")
        resp = client.get("/api/v1/cli/list", headers=_auth_header(token))
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Admin role required"

    def test_admin_passes_admin_guard(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """Admin role passes the require_admin guard."""
        token = _make_token(jwt_service, role="admin")
        resp = client.get("/api/v1/cli/list", headers=_auth_header(token))
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════
# Auth Endpoints (REQ-API-002)
# ══════════════════════════════════════════════════════════════


class TestAuthEndpoints:
    """Auth API endpoint tests."""

    def test_register_creates_cli(self, client: TestClient) -> None:
        """POST /auth/register returns cli_id and org_id."""
        resp = client.post(
            "/api/v1/auth/register",
            json={"machine_id": "m-abc", "hostname": "dev-box", "vaig_version": "1.0.0"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "cli_id" in data
        assert data["cli_id"].startswith("cli-")
        assert "org_id" in data

    def test_token_exchange(self, client: TestClient) -> None:
        """POST /auth/token with valid code returns tokens."""
        resp = client.post(
            "/api/v1/auth/token",
            json={"code": "test-code", "code_verifier": "test-verifier"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_token_exchange_empty_code(self, client: TestClient) -> None:
        """POST /auth/token with empty code returns 401."""
        resp = client.post(
            "/api/v1/auth/token",
            json={"code": "", "code_verifier": "test-verifier"},
        )
        assert resp.status_code == 401

    def test_refresh_token(self, client: TestClient) -> None:
        """POST /auth/refresh with valid refresh token returns new tokens."""
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "some-valid-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_empty_token(self, client: TestClient) -> None:
        """POST /auth/refresh with empty refresh token returns 401."""
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": ""},
        )
        assert resp.status_code == 401

    def test_revoke(self, client: TestClient, jwt_service: JWTService) -> None:
        """POST /auth/revoke with valid token returns revoked: true."""
        token = _make_token(jwt_service)
        resp = client.post("/api/v1/auth/revoke", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json() == {"revoked": True}

    def test_whoami(self, client: TestClient, jwt_service: JWTService) -> None:
        """GET /auth/whoami returns decoded JWT claims."""
        token = _make_token(jwt_service, sub="bob@example.com", role="admin", org_id="org-x")
        resp = client.get("/api/v1/auth/whoami", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "bob@example.com"
        assert data["role"] == "admin"
        assert data["org_id"] == "org-x"


# ══════════════════════════════════════════════════════════════
# CLI Management (REQ-API-003)
# ══════════════════════════════════════════════════════════════


class TestCLIManagement:
    """CLI management endpoint tests."""

    def test_list_as_admin(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """SC-API-003a: Admin can list all CLI instances."""
        org = Organization(org_id="test-org", name="Test Org")
        asyncio.run(repo.save_organization(org))

        cli1 = CLIInstance(cli_id="cli-1", user_email="a@test.com", status="active")
        cli2 = CLIInstance(cli_id="cli-2", user_email="b@test.com", status="active")
        asyncio.run(repo.save_cli_instance("test-org", cli1))
        asyncio.run(repo.save_cli_instance("test-org", cli2))

        token = _make_token(jwt_service, role="admin")
        resp = client.get("/api/v1/cli/list", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_get_own_cli_as_operator(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """SC-API-003e: Operator can view their own CLI details."""
        cli = CLIInstance(cli_id="cli-mine", user_email="user@example.com", status="active")
        asyncio.run(repo.save_cli_instance("test-org", cli))

        token = _make_token(jwt_service, sub="user@example.com", role="operator")
        resp = client.get("/api/v1/cli/cli-mine", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["cli_id"] == "cli-mine"

    def test_get_other_cli_as_operator_forbidden(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """SC-API-003f: Operator cannot view another user's CLI."""
        cli = CLIInstance(cli_id="cli-other", user_email="other@example.com", status="active")
        asyncio.run(repo.save_cli_instance("test-org", cli))

        token = _make_token(jwt_service, sub="user@example.com", role="operator")
        resp = client.get("/api/v1/cli/cli-other", headers=_auth_header(token))
        assert resp.status_code == 403

    def test_revoke_cli_as_admin(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """SC-API-003d: Admin can revoke a CLI instance."""
        cli = CLIInstance(cli_id="cli-target", user_email="user@test.com", status="active")
        asyncio.run(repo.save_cli_instance("test-org", cli))

        token = _make_token(jwt_service, role="admin")
        resp = client.post("/api/v1/cli/cli-target/revoke", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json() == {"revoked": True}


# ══════════════════════════════════════════════════════════════
# Config Policy (REQ-API-004)
# ══════════════════════════════════════════════════════════════


class TestConfigPolicy:
    """Config policy endpoint tests."""

    def test_get_policy_empty(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """SC-API-004a: Get policy returns empty defaults when no org exists."""
        token = _make_token(jwt_service)
        resp = client.get("/api/v1/config/policy", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["enforced_fields"] == {}

    def test_update_policy_as_admin(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """SC-API-004b: Admin can update config policy."""
        org = Organization(org_id="test-org", name="Test Org")
        asyncio.run(repo.save_organization(org))

        token = _make_token(jwt_service, role="admin")
        resp = client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={
                "enforced_fields": {"budget.max_cost_per_run": 2.0},
                "changelog": "Raised budget",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "config_version" in data
        assert data["config_version"].startswith("v-")

    def test_update_policy_as_operator_forbidden(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """SC-API-004c: Operator cannot update config policy."""
        token = _make_token(jwt_service, role="operator")
        resp = client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={"enforced_fields": {}},
        )
        assert resp.status_code == 403

    def test_config_history_as_admin(
        self, client: TestClient, jwt_service: JWTService, repo: InMemoryRepository
    ) -> None:
        """SC-API-004d: Admin can view config history."""
        org = Organization(org_id="test-org", name="Test Org")
        asyncio.run(repo.save_organization(org))

        # First create a policy update to have history
        token = _make_token(jwt_service, role="admin")
        client.put(
            "/api/v1/config/policy",
            headers=_auth_header(token),
            json={"enforced_fields": {"key": "val"}, "changelog": "v1"},
        )

        resp = client.get("/api/v1/config/history", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_push_config_as_admin(
        self, client: TestClient, jwt_service: JWTService
    ) -> None:
        """Config push endpoint returns push count."""
        token = _make_token(jwt_service, role="admin")
        resp = client.post("/api/v1/config/push", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert "pushed_to" in data

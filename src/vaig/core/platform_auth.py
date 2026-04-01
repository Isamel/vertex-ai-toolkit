"""Platform authentication manager — OAuth PKCE login, token lifecycle, enforced config.

Implements ``PlatformAuthProtocol`` for CLI authentication against the
platform backend.  Uses browser-based OAuth PKCE with a localhost callback
server, stores credentials in ``~/.vaig/credentials.json`` with restrictive
file permissions (``0600`` / ``0700``).

This module is only imported when ``settings.platform.enabled`` is ``True``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from vaig.platform.models.auth import AuthResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────

_CREDENTIALS_DIR = Path.home() / ".vaig"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "credentials.json"
_LOGIN_TIMEOUT_SECONDS = 120
_TOKEN_REFRESH_BUFFER_SECONDS = 60  # refresh if expires within 60s


class PlatformAuthManager:
    """Manages platform authentication lifecycle.

    Implements the ``PlatformAuthProtocol`` interface for the CLI:

    - ``login()``  — browser-based OAuth PKCE flow
    - ``logout()`` — revoke token + delete credentials
    - ``get_token()`` — return access token, auto-refresh if needed
    - ``is_authenticated()`` — check credential validity
    - ``get_user_info()`` — decode JWT claims
    - ``get_enforced_config()`` — fetch enforced policy from backend

    Args:
        backend_url: Platform backend API base URL.
        org_id: Organization identifier.
        http_client: Optional ``httpx.Client`` for dependency injection (tests).
        credentials_dir: Override credential directory (tests).
    """

    def __init__(
        self,
        backend_url: str,
        org_id: str = "",
        *,
        http_client: httpx.Client | None = None,
        credentials_dir: Path | None = None,
    ) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._org_id = org_id
        self._client = http_client or httpx.Client(timeout=30.0)
        self._owns_client = http_client is None
        self._creds_dir = credentials_dir or _CREDENTIALS_DIR
        self._creds_file = self._creds_dir / "credentials.json"
        self._enforced_config_cache: dict[str, Any] | None = None

    # ── Credential file I/O ───────────────────────────────────

    def _ensure_creds_dir(self) -> None:
        """Create credentials directory with ``0700`` permissions if needed."""
        self._creds_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._creds_dir, 0o700)
        except OSError:
            logger.warning("Could not set permissions on %s", self._creds_dir)

    def _load_creds(self) -> dict[str, Any] | None:
        """Load credentials from disk. Returns ``None`` on missing / invalid file."""
        if not self._creds_file.exists():
            return None
        try:
            data = json.loads(self._creds_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Credential file is not a JSON object — treating as unauthenticated")
                return None
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read credentials: %s", exc)
            return None

    def _save_creds(self, creds: dict[str, Any]) -> None:
        """Persist credentials to disk with ``0600`` permissions."""
        self._ensure_creds_dir()
        self._creds_file.write_text(json.dumps(creds, indent=2), encoding="utf-8")
        try:
            os.chmod(self._creds_file, 0o600)
        except OSError:
            logger.warning("Could not set permissions on %s", self._creds_file)

    def _delete_creds(self) -> None:
        """Delete the credentials file if it exists."""
        try:
            self._creds_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to delete credentials: %s", exc)

    # ── Authentication checks ─────────────────────────────────

    def is_authenticated(self) -> bool:
        """Return ``True`` if valid (non-expired or refreshable) credentials exist."""
        creds = self._load_creds()
        if creds is None:
            return False

        access_token = creds.get("access_token", "")
        if not access_token:
            return False

        # Check if access token is still valid
        expires_at = creds.get("expires_at", 0)
        if time.time() < expires_at - _TOKEN_REFRESH_BUFFER_SECONDS:
            return True

        # Access token expired — check if refresh token exists
        return bool(creds.get("refresh_token"))

    def get_user_info(self) -> dict[str, Any] | None:
        """Decode JWT claims and return user info.

        Returns ``None`` when not authenticated.  Uses the JWT payload
        (base64-decoded, **not** signature-verified) for local display.
        Full verification happens server-side.
        """
        creds = self._load_creds()
        if creds is None:
            return None

        access_token = creds.get("access_token", "")
        if not access_token:
            return None

        try:
            # JWT: header.payload.signature — decode the payload part
            parts = access_token.split(".")
            if len(parts) < 2:
                return None

            # Add padding for base64
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return {
                "email": payload.get("sub", ""),
                "org_id": payload.get("org_id", ""),
                "role": payload.get("role", ""),
                "cli_id": payload.get("cli_id", ""),
            }
        except Exception:
            logger.warning("Failed to decode JWT claims")
            return None

    # ── PKCE helpers ──────────────────────────────────────────

    @staticmethod
    def _generate_pkce() -> tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge (S256).

        Returns:
            Tuple of ``(code_verifier, code_challenge)``.
        """
        code_verifier = secrets.token_urlsafe(64)[:128]
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return code_verifier, code_challenge

    @staticmethod
    def _find_free_port() -> int:
        """Find an available port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    # ── Login flow ────────────────────────────────────────────

    def login(self, *, force: bool = False) -> AuthResult:
        """Run the OAuth PKCE login flow.

        1. Check for existing authentication (skip if ``force``).
        2. Generate PKCE verifier + challenge.
        3. Start localhost HTTP server for OAuth callback.
        4. Open browser to backend auth URL.
        5. Wait for callback with auth code (timeout: 120s).
        6. Exchange auth code for tokens via backend.
        7. Store credentials to disk.

        Args:
            force: Re-authenticate even if already logged in.

        Returns:
            ``AuthResult`` indicating success or failure.
        """
        # Check existing auth
        if not force and self.is_authenticated():
            info = self.get_user_info()
            email = info.get("email", "unknown") if info else "unknown"
            return AuthResult(
                success=True,
                user_email=email,
                org_id=info.get("org_id", "") if info else "",
                role=info.get("role", "") if info else "",
                error="Already authenticated. Use --force to re-authenticate.",
            )

        # Generate PKCE
        code_verifier, code_challenge = self._generate_pkce()
        state = secrets.token_urlsafe(32)

        # Start local callback server
        port = self._find_free_port()
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        auth_code: str | None = None
        received_state: str | None = None
        server_error: str | None = None

        class CallbackHandler(BaseHTTPRequestHandler):
            """HTTP handler for OAuth redirect callback."""

            def do_GET(self) -> None:  # noqa: N802
                nonlocal auth_code, received_state, server_error
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if parsed.path == "/callback":
                    auth_code = params.get("code", [None])[0]
                    received_state = params.get("state", [None])[0]
                    error = params.get("error", [None])[0]
                    if error:
                        server_error = error
                        self.send_response(400)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"<h1>Login failed</h1><p>You can close this window.</p>")
                    else:
                        self.send_response(200)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()
                        self.wfile.write(
                            b"<h1>Login successful!</h1><p>You can close this window and return to the terminal.</p>"
                        )
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """Suppress default stderr logging."""

        server = HTTPServer(("127.0.0.1", port), CallbackHandler)
        server.timeout = _LOGIN_TIMEOUT_SECONDS

        # Open browser
        auth_url = (
            f"{self._backend_url}/auth/login"
            f"?code_challenge={code_challenge}"
            f"&code_challenge_method=S256"
            f"&state={state}"
            f"&redirect_uri={redirect_uri}"
        )

        # Check backend reachability
        try:
            self._client.get(f"{self._backend_url}/healthz", timeout=5.0)
        except httpx.HTTPError:
            return AuthResult(
                success=False,
                error=f"Cannot reach platform backend at {self._backend_url}",
            )

        # Run browser + server in background
        try:
            webbrowser.open(auth_url)
        except Exception:
            logger.warning("Could not open browser — please navigate to: %s", auth_url)

        # Wait for callback (blocking with timeout)
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()
        server_thread.join(timeout=_LOGIN_TIMEOUT_SECONDS)
        server.server_close()

        if server_error:
            return AuthResult(success=False, error=f"Login failed: {server_error}")

        if auth_code is None:
            return AuthResult(success=False, error="Login timed out")

        if received_state != state:
            return AuthResult(success=False, error="State mismatch — possible CSRF attack")

        # Exchange code for tokens
        try:
            resp = self._client.post(
                f"{self._backend_url}/api/v1/auth/token",
                json={
                    "code": auth_code,
                    "code_verifier": code_verifier,
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            token_data = resp.json()
        except httpx.HTTPStatusError as exc:
            return AuthResult(
                success=False,
                error=f"Token exchange failed: {exc.response.status_code}",
            )
        except httpx.HTTPError as exc:
            return AuthResult(success=False, error=f"Token exchange failed: {exc}")

        # Calculate expiry
        expires_in = token_data.get("expires_in", 3600)
        expires_at = int(time.time()) + expires_in

        # Store credentials
        creds = {
            "access_token": token_data.get("access_token", ""),
            "refresh_token": token_data.get("refresh_token", ""),
            "token_type": token_data.get("token_type", "Bearer"),
            "expires_at": expires_at,
            "cli_id": token_data.get("cli_id", ""),
            "backend_url": self._backend_url,
        }
        self._save_creds(creds)

        # Extract user info from token
        info = self.get_user_info()
        return AuthResult(
            success=True,
            user_email=info.get("email", "") if info else "",
            org_id=info.get("org_id", self._org_id) if info else self._org_id,
            role=info.get("role", "") if info else "",
        )

    # ── Token management ──────────────────────────────────────

    def get_token(self) -> str | None:
        """Return a valid access token, auto-refreshing if needed.

        Returns ``None`` when no valid (or refreshable) credentials exist.
        """
        creds = self._load_creds()
        if creds is None:
            return None

        access_token = creds.get("access_token", "")
        expires_at = creds.get("expires_at", 0)

        # Token still valid
        if access_token and time.time() < expires_at - _TOKEN_REFRESH_BUFFER_SECONDS:
            return str(access_token)

        # Try refresh
        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            return None

        try:
            resp = self._client.post(
                f"{self._backend_url}/api/v1/auth/refresh",
                json={"refresh_token": refresh_token},
            )
            resp.raise_for_status()
            token_data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Token refresh failed: %s", exc)
            return None

        # Update credentials
        new_expires_in = token_data.get("expires_in", 3600)
        creds["access_token"] = token_data.get("access_token", "")
        creds["refresh_token"] = token_data.get("refresh_token", refresh_token)
        creds["expires_at"] = int(time.time()) + new_expires_in
        self._save_creds(creds)

        return str(creds["access_token"]) or None

    def logout(self) -> None:
        """Revoke token via backend (best-effort) and delete local credentials."""
        creds = self._load_creds()
        if creds is not None:
            # Best-effort revoke
            access_token = creds.get("access_token", "")
            if access_token:
                try:
                    self._client.post(
                        f"{self._backend_url}/api/v1/auth/revoke",
                        headers={"Authorization": f"Bearer {access_token}"},
                        json={"refresh_token": creds.get("refresh_token", "")},
                    )
                except httpx.HTTPError:
                    logger.debug("Token revocation failed (best-effort)")

        self._delete_creds()

    # ── Enforced config ───────────────────────────────────────

    def get_enforced_config(self) -> dict[str, Any]:
        """Fetch enforced config policy from backend.

        Returns a dict of field paths → enforced values.
        Returns empty dict on network failure (graceful degradation).
        Caches the result for the duration of a single command invocation.
        """
        if self._enforced_config_cache is not None:
            return self._enforced_config_cache

        token = self.get_token()
        if not token:
            return {}

        try:
            resp = self._client.get(
                f"{self._backend_url}/api/v1/config/policy",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            enforced = data.get("enforced_fields", {})
            if isinstance(enforced, dict):
                self._enforced_config_cache = enforced
                return enforced
            return {}
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch enforced config: %s", exc)
            return {}

    def clear_enforced_config_cache(self) -> None:
        """Clear the cached enforced config (called between commands)."""
        self._enforced_config_cache = None

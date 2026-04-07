"""Tests for Annotation routes — Task 4.4.

Covers:
- GET /sessions/{id}/annotations — viewer+ access, paginated
- POST /sessions/{id}/annotations — editor+ access, body validation
- PUT /sessions/{id}/annotations/{ann_id} — author only
- DELETE /sessions/{id}/annotations/{ann_id} — author only
- Full CRUD lifecycle
- Viewer can read (200) but cannot create (403)
- Editor creates annotations (200)
- Author-only edit/delete (non-author gets 403)
- Invalid annotation type → 422
- Content too long (>2000 chars) → 422
- Feature flag off → 404

All access control is mocked via a fake ``SessionAccessControl``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from httpx import ASGITransport, AsyncClient

from vaig.core.models import Annotation
from vaig.web.app import create_app

# ── Helpers ──────────────────────────────────────────────────

_SAMPLE_ANNOTATION = Annotation(
    id="ann-001",
    author="editor@test.com",
    content="Root cause: OOM in pod-xyz",
    annotation_type="root_cause",
    message_ref="msg-123",
    created_at="2026-01-01T00:00:00Z",
    updated_at="2026-01-01T00:00:00Z",
)

_UPDATED_ANNOTATION = Annotation(
    id="ann-001",
    author="editor@test.com",
    content="Updated content",
    annotation_type="root_cause",
    message_ref="msg-123",
    created_at="2026-01-01T00:00:00Z",
    updated_at="2026-01-02T00:00:00Z",
)


def _make_mock_access(
    *,
    add_result: Annotation | None = None,
    add_side_effect: Exception | None = None,
    update_result: Annotation | None = None,
    update_side_effect: Exception | None = None,
    delete_result: bool = True,
    delete_side_effect: Exception | None = None,
    list_result: list[Annotation] | None = None,
) -> MagicMock:
    """Build a mock ``SessionAccessProtocol`` with annotation support."""
    access = MagicMock()

    # add_annotation
    if add_side_effect is not None:
        access.add_annotation = AsyncMock(side_effect=add_side_effect)
    else:
        access.add_annotation = AsyncMock(
            return_value=add_result or _SAMPLE_ANNOTATION
        )

    # update_annotation
    if update_side_effect is not None:
        access.update_annotation = AsyncMock(side_effect=update_side_effect)
    else:
        access.update_annotation = AsyncMock(
            return_value=update_result or _UPDATED_ANNOTATION
        )

    # delete_annotation
    if delete_side_effect is not None:
        access.delete_annotation = AsyncMock(side_effect=delete_side_effect)
    else:
        access.delete_annotation = AsyncMock(return_value=delete_result)

    # list_annotations
    access.list_annotations = AsyncMock(return_value=list_result or [])

    return access


def _make_app(
    access: MagicMock | None = None,
    *,
    flag_on: bool = True,
) -> tuple:
    """Create app with session_access wired."""
    app = create_app()
    if access is None:
        access = _make_mock_access()
    app.state.session_access = access

    env = {"VAIG_WEB_SHARED_SESSIONS": "true" if flag_on else "false"}
    return app, access, env


# ── POST /sessions/{id}/annotations (create) ────────────────


@pytest.mark.asyncio
async def test_create_annotation_returns_200() -> None:
    """POST /sessions/{id}/annotations returns 200 for editor."""
    app, access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "root_cause",
                    "content": "Root cause: OOM in pod-xyz",
                    "message_ref": "msg-123",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "ann-001"
            assert data["author"] == "editor@test.com"
            assert data["annotation_type"] == "root_cause"
            assert data["content"] == "Root cause: OOM in pod-xyz"
            assert data["message_ref"] == "msg-123"


@pytest.mark.asyncio
async def test_create_annotation_viewer_returns_403() -> None:
    """POST /sessions/{id}/annotations returns 403 for viewer."""
    access = _make_mock_access(
        add_side_effect=PermissionError("Insufficient permissions"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="viewer@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "observation",
                    "content": "Some observation",
                },
            )
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_create_annotation_invalid_type_returns_422() -> None:
    """POST /sessions/{id}/annotations returns 422 for invalid type."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "invalid_type",
                    "content": "Some text",
                },
            )
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_annotation_content_too_long_returns_422() -> None:
    """POST /sessions/{id}/annotations returns 422 when content > 2000 chars."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "observation",
                    "content": "x" * 2001,
                },
            )
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_annotation_empty_content_returns_422() -> None:
    """POST /sessions/{id}/annotations returns 422 when content is empty."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "observation",
                    "content": "",
                },
            )
            assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_annotation_no_message_ref() -> None:
    """POST with no message_ref succeeds (annotation applies to session)."""
    annotation = Annotation(
        id="ann-002",
        author="editor@test.com",
        content="General observation",
        annotation_type="observation",
        message_ref=None,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    access = _make_mock_access(add_result=annotation)
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "observation",
                    "content": "General observation",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["message_ref"] is None


# ── GET /sessions/{id}/annotations (list) ────────────────────


@pytest.mark.asyncio
async def test_list_annotations_returns_200() -> None:
    """GET /sessions/{id}/annotations returns 200 with annotation list."""
    annotations = [
        _SAMPLE_ANNOTATION,
        Annotation(
            id="ann-002",
            author="alice@test.com",
            content="Action: restart pod",
            annotation_type="action_item",
            message_ref=None,
            created_at="2026-01-02T00:00:00Z",
            updated_at="2026-01-02T00:00:00Z",
        ),
    ]
    access = _make_mock_access(list_result=annotations)
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="viewer@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions/s1/annotations")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["id"] == "ann-001"
            assert data[0]["annotation_type"] == "root_cause"
            assert data[1]["id"] == "ann-002"
            assert data[1]["annotation_type"] == "action_item"


@pytest.mark.asyncio
async def test_list_annotations_empty() -> None:
    """GET /sessions/{id}/annotations returns empty list for no annotations."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="viewer@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions/s1/annotations")
            assert resp.status_code == 200
            assert resp.json() == []


@pytest.mark.asyncio
async def test_list_annotations_pagination_param() -> None:
    """GET /sessions/{id}/annotations?limit=10 passes limit to service."""
    app, access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="viewer@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            await ac.get("/sessions/s1/annotations?limit=10")
            access.list_annotations.assert_called_once_with("s1", "viewer@test.com", limit=10)


# ── PUT /sessions/{id}/annotations/{ann_id} (update) ────────


@pytest.mark.asyncio
async def test_update_annotation_author_returns_200() -> None:
    """PUT /sessions/{id}/annotations/{ann_id} returns 200 for author."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.put(
                "/sessions/s1/annotations/ann-001",
                json={"content": "Updated content"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["content"] == "Updated content"


@pytest.mark.asyncio
async def test_update_annotation_non_author_returns_403() -> None:
    """PUT /sessions/{id}/annotations/{ann_id} returns 403 for non-author."""
    access = _make_mock_access(
        update_side_effect=PermissionError("Only the annotation author can edit"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="other@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="other@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.put(
                "/sessions/s1/annotations/ann-001",
                json={"content": "Hacked content"},
            )
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_update_annotation_not_found_returns_404() -> None:
    """PUT /sessions/{id}/annotations/{ann_id} returns 404 when not found."""
    access = _make_mock_access(
        update_side_effect=LookupError("Annotation not found"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.put(
                "/sessions/s1/annotations/nonexistent",
                json={"content": "Updated content"},
            )
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_annotation_content_too_long_returns_422() -> None:
    """PUT /sessions/{id}/annotations/{ann_id} returns 422 for content > 2000."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.put(
                "/sessions/s1/annotations/ann-001",
                json={"content": "x" * 2001},
            )
            assert resp.status_code == 422


# ── DELETE /sessions/{id}/annotations/{ann_id} ───────────────


@pytest.mark.asyncio
async def test_delete_annotation_author_returns_200() -> None:
    """DELETE /sessions/{id}/annotations/{ann_id} returns 200 for author."""
    app, _access, env = _make_app()

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/annotations/ann-001")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "deleted"
            assert data["id"] == "ann-001"


@pytest.mark.asyncio
async def test_delete_annotation_non_author_returns_403() -> None:
    """DELETE /sessions/{id}/annotations/{ann_id} returns 403 for non-author."""
    access = _make_mock_access(
        delete_side_effect=PermissionError("Only the annotation author can delete"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="other@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="other@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/annotations/ann-001")
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_annotation_not_found_returns_404() -> None:
    """DELETE /sessions/{id}/annotations/{ann_id} returns 404 when not found."""
    access = _make_mock_access(
        delete_side_effect=LookupError("Annotation not found"),
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/annotations/nonexistent")
            assert resp.status_code == 404


# ── Feature Flag Off → 404 ───────────────────────────────────


@pytest.mark.asyncio
async def test_list_annotations_flag_off_returns_404() -> None:
    """GET /sessions/{id}/annotations returns 404 when flag off."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="viewer@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="viewer@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/sessions/s1/annotations")
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_annotation_flag_off_returns_404() -> None:
    """POST /sessions/{id}/annotations returns 404 when flag off."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={"annotation_type": "observation", "content": "test"},
            )
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_annotation_flag_off_returns_404() -> None:
    """PUT /sessions/{id}/annotations/{ann_id} returns 404 when flag off."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.put(
                "/sessions/s1/annotations/ann-001",
                json={"content": "test"},
            )
            assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_annotation_flag_off_returns_404() -> None:
    """DELETE /sessions/{id}/annotations/{ann_id} returns 404 when flag off."""
    app, _access, _env = _make_app(flag_on=False)
    env = {"VAIG_WEB_SHARED_SESSIONS": "false"}

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete("/sessions/s1/annotations/ann-001")
            assert resp.status_code == 404


# ── Full CRUD Lifecycle ──────────────────────────────────────


@pytest.mark.asyncio
async def test_full_crud_lifecycle() -> None:
    """Full lifecycle: create → list → update → delete."""
    created = _SAMPLE_ANNOTATION
    updated = _UPDATED_ANNOTATION

    access = _make_mock_access(
        add_result=created,
        update_result=updated,
        list_result=[created],
    )
    app, _access, env = _make_app(access)

    with (
        patch("vaig.web.routes.chat.get_current_user", return_value="editor@test.com"),
        patch("vaig.web.routes.annotations.get_current_user", return_value="editor@test.com"),
        patch.dict("os.environ", env),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # 1. Create
            resp = await ac.post(
                "/sessions/s1/annotations",
                json={
                    "annotation_type": "root_cause",
                    "content": "Root cause: OOM in pod-xyz",
                    "message_ref": "msg-123",
                },
            )
            assert resp.status_code == 200
            ann_id = resp.json()["id"]

            # 2. List
            resp = await ac.get("/sessions/s1/annotations")
            assert resp.status_code == 200
            assert len(resp.json()) == 1

            # 3. Update
            resp = await ac.put(
                f"/sessions/s1/annotations/{ann_id}",
                json={"content": "Updated content"},
            )
            assert resp.status_code == 200
            assert resp.json()["content"] == "Updated content"

            # 4. Delete
            resp = await ac.delete(f"/sessions/s1/annotations/{ann_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"


# ── Route Registration ───────────────────────────────────────


def test_annotation_routes_registered() -> None:
    """App must have annotation routes registered."""
    app = create_app()
    route_paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/sessions/{session_id}/annotations" in route_paths
    assert "/sessions/{session_id}/annotations/{annotation_id}" in route_paths

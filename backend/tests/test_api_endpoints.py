"""API endpoint tests for the FastAPI layer.

These exercise request parsing, response shaping, and error handling for the
three routes the frontend uses (`POST /api/query`, `GET /api/courses`, `GET /`)
plus the `POST /api/session/clear` route behind the New Chat button.

The app under test is the inline rebuild from conftest.py (`test_app`/`client`),
wired to a mocked RAGSystem so no ChromaDB / Anthropic / network is involved.
"""

import pytest


# --------------------------------------------------------------------------- #
# POST /api/query
# --------------------------------------------------------------------------- #

class QueryEndpointTest:
    """Grouped under a *Test class so pytest collects it via python_classes."""

    def test_query_with_explicit_session_returns_answer_and_sources(
        self, client, mock_rag_system, sample_sources
    ):
        resp = client.post(
            "/api/query",
            json={"query": "what is MCP?", "session_id": "abc"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "MCP stands for Model Context Protocol."
        assert body["session_id"] == "abc"
        assert body["sources"] == sample_sources
        # The provided session id must be passed through, not regenerated.
        mock_rag_system.query.assert_called_once_with("what is MCP?", "abc")
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_without_session_creates_one(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        # The mock's create_session returns this canned id.
        assert body["session_id"] == "test-session-1"
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("hello", "test-session-1")

    def test_query_response_source_shape(self, client):
        resp = client.post("/api/query", json={"query": "x"})
        assert resp.status_code == 200
        for src in resp.json()["sources"]:
            assert set(src.keys()) == {"text", "link"}

    def test_query_missing_query_field_is_422(self, client):
        resp = client.post("/api/query", json={"session_id": "abc"})
        assert resp.status_code == 422

    def test_query_empty_body_is_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_query_wrong_type_is_422(self, client):
        resp = client.post("/api/query", json={"query": 123})
        assert resp.status_code == 422

    def test_query_rag_failure_surfaces_as_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("vector store down")
        resp = client.post("/api/query", json={"query": "x"})
        assert resp.status_code == 500
        assert "vector store down" in resp.json()["detail"]

    def test_query_with_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("No content found.", [])
        resp = client.post("/api/query", json={"query": "obscure"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "No content found."
        assert body["sources"] == []


# --------------------------------------------------------------------------- #
# GET /api/courses
# --------------------------------------------------------------------------- #

class CoursesEndpointTest:
    def test_courses_returns_stats(self, client, sample_analytics):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == sample_analytics["total_courses"]
        assert body["course_titles"] == sample_analytics["course_titles"]

    def test_courses_response_shape(self, client):
        resp = client.get("/api/courses")
        body = resp.json()
        assert set(body.keys()) == {"total_courses", "course_titles"}
        assert isinstance(body["course_titles"], list)

    def test_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        assert resp.json() == {"total_courses": 0, "course_titles": []}

    def test_courses_failure_surfaces_as_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("boom")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /api/session/clear
# --------------------------------------------------------------------------- #

class SessionClearEndpointTest:
    def test_clear_session_ok(self, client, mock_rag_system):
        resp = client.post("/api/session/clear", json={"session_id": "abc"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.clear_session.assert_called_once_with("abc")

    def test_clear_session_requires_session_id(self, client):
        resp = client.post("/api/session/clear", json={})
        assert resp.status_code == 422

    def test_clear_session_failure_surfaces_as_500(self, client, mock_rag_system):
        mock_rag_system.session_manager.clear_session.side_effect = RuntimeError("nope")
        resp = client.post("/api/session/clear", json={"session_id": "abc"})
        assert resp.status_code == 500
        assert "nope" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# GET /  (root)
# --------------------------------------------------------------------------- #

class RootEndpointTest:
    def test_root_reachable(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

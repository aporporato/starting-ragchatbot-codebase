"""Shared pytest fixtures for the RAG system test suite.

Two things make API testing of ``backend/app.py`` awkward to do directly:

1. Importing ``app`` runs ``app.mount("/", StaticFiles(directory="../frontend", ...))``
   at module load time, which raises ``RuntimeError`` if the working directory
   isn't ``backend/`` (the ``../frontend`` path won't resolve). Tests shouldn't
   depend on cwd.
2. ``app.py`` constructs a real ``RAGSystem`` (and therefore a real ChromaDB
   client + embedding model + Anthropic client) at import time. That's slow and
   needs network / an API key.

So instead of importing ``app``, the ``test_app`` fixture below rebuilds the
exact same API surface (same Pydantic models, same routes, same error handling)
inline, wired to a mocked ``RAGSystem``. The static-file mount is deliberately
omitted — the ``/`` route is stubbed so request/response handling can still be
exercised without the frontend directory existing.
"""

import os
import sys

import pytest

# Make the backend package importable (models, etc.) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Sample test data
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_sources():
    """Source citations shaped exactly like CourseSearchTool.last_sources entries."""
    return [
        {"text": "MCP Course - Lesson 1", "link": "https://example.com/mcp/1"},
        {"text": "MCP Course - Lesson 2", "link": "https://example.com/mcp/2"},
    ]


@pytest.fixture
def sample_query_result(sample_sources):
    """A canned (answer, sources) tuple as returned by RAGSystem.query."""
    return ("MCP stands for Model Context Protocol.", sample_sources)


@pytest.fixture
def sample_analytics():
    """A canned analytics dict as returned by RAGSystem.get_course_analytics."""
    return {
        "total_courses": 2,
        "course_titles": ["MCP Course", "Advanced Retrieval"],
    }


# --------------------------------------------------------------------------- #
# Mocked RAG system
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_rag_system(sample_query_result, sample_analytics):
    """A MagicMock standing in for RAGSystem, with sensible default returns.

    Tests can override any attribute (e.g. ``mock_rag_system.query.side_effect``)
    to exercise error paths.
    """
    rag = MagicMock()
    rag.query.return_value = sample_query_result
    rag.get_course_analytics.return_value = sample_analytics
    rag.session_manager.create_session.return_value = "test-session-1"
    rag.session_manager.clear_session.return_value = None
    return rag


# --------------------------------------------------------------------------- #
# Inline test app (mirrors backend/app.py without the static mount)
# --------------------------------------------------------------------------- #

@pytest.fixture
def test_app(mock_rag_system):
    """Build a FastAPI app with the same endpoints as app.py, wired to the mock.

    Returns the app; use the ``client`` fixture for a ready-made TestClient.
    """

    app = FastAPI(title="Course Materials RAG System (test)")

    # --- Pydantic models (mirror app.py) ---
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class SessionClearRequest(BaseModel):
        session_id: str

    # --- Endpoints (mirror app.py) ---
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear")
    async def clear_session(request: SessionClearRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Stub for "/" — the real app serves the frontend via StaticFiles, which
    # doesn't exist in the test environment. A plain JSON stub lets us assert
    # the root is reachable without depending on the frontend directory.
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}

    return app


@pytest.fixture
def client(test_app):
    """A TestClient bound to the inline test app."""
    return TestClient(test_app)

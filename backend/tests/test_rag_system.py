import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, CourseChunk, Lesson


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, tool_input, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=id_)


def fake_response(stop_reason, blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class FakeConfig:
    def __init__(self, chroma_path):
        self.ANTHROPIC_API_KEY = "fake"
        self.ANTHROPIC_MODEL = "fake-model"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.MAX_RESULTS = 5
        self.MAX_HISTORY = 2
        self.MAX_TOOL_ROUNDS = 2
        self.CHROMA_PATH = chroma_path


def _preload_course(rag, title="Test Course"):
    course = Course(
        title=title,
        course_link=f"https://example.com/{title}",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=0, title="Intro", lesson_link=f"https://example.com/{title}/0"),
            Lesson(lesson_number=1, title="MCP basics", lesson_link=f"https://example.com/{title}/1"),
        ],
    )
    rag.vector_store.add_course_metadata(course)
    rag.vector_store.add_course_content([
        CourseChunk(
            content="Welcome to the test course.",
            course_title=title,
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="MCP stands for Model Context Protocol.",
            course_title=title,
            lesson_number=1,
            chunk_index=1,
        ),
    ])


class RAGSystemContentQueryTest(unittest.TestCase):
    def setUp(self):
        # Fresh tmp chroma per test to avoid duplicate-ID issues across tests
        self.tmp = tempfile.mkdtemp(prefix="rag_test_chroma_")
        self.config = FakeConfig(self.tmp)

        self.client_patcher = patch("ai_generator.anthropic.Anthropic")
        mock_anthropic_cls = self.client_patcher.start()
        self.mock_client = MagicMock()
        mock_anthropic_cls.return_value = self.mock_client

        # Import here so the Anthropic patch is in effect during AIGenerator __init__
        from rag_system import RAGSystem
        self.rag = RAGSystem(self.config)
        _preload_course(self.rag)

    def tearDown(self):
        self.client_patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_query_returns_answer_and_dict_sources(self):
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "MCP"}, "tid")],
        )
        second = fake_response(
            stop_reason="end_turn",
            blocks=[text_block("MCP stands for Model Context Protocol.")],
        )
        self.mock_client.messages.create.side_effect = [first, second]

        answer, sources = self.rag.query("what is MCP?")

        self.assertEqual(answer, "MCP stands for Model Context Protocol.")
        self.assertGreater(len(sources), 0, "Expected at least one source from the search tool")
        for s in sources:
            self.assertIsInstance(s, dict)
            self.assertEqual(set(s.keys()), {"text", "link"})

    def test_query_updates_session_history_when_session_id_given(self):
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "MCP"}, "tid")],
        )
        second = fake_response(stop_reason="end_turn", blocks=[text_block("answer")])
        self.mock_client.messages.create.side_effect = [first, second]

        sid = self.rag.session_manager.create_session()
        self.rag.query("what is MCP?", session_id=sid)

        history = self.rag.session_manager.get_conversation_history(sid)
        self.assertIsNotNone(history)
        self.assertIn("what is MCP?", history)
        self.assertIn("answer", history)

    def test_query_resets_sources_between_calls(self):
        # 1st call: tool_use -> sources populated
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "MCP"}, "t1")],
        )
        first_synth = fake_response(stop_reason="end_turn", blocks=[text_block("a1")])
        # 2nd call: direct answer (no tool use) -> sources empty
        second_direct = fake_response(stop_reason="end_turn", blocks=[text_block("a2")])
        self.mock_client.messages.create.side_effect = [first, first_synth, second_direct]

        _, sources1 = self.rag.query("first")
        _, sources2 = self.rag.query("second")

        self.assertGreater(len(sources1), 0)
        self.assertEqual(sources2, [])

    def test_query_outline_tool_path(self):
        """The outline tool is also registered. Verify it works end-to-end."""
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("get_course_outline", {"course_name": "Test"}, "tid")],
        )
        second = fake_response(
            stop_reason="end_turn",
            blocks=[text_block("Course outline answer")],
        )
        self.mock_client.messages.create.side_effect = [first, second]

        answer, sources = self.rag.query("outline of test course?")

        self.assertEqual(answer, "Course outline answer")
        # Outline tool stashes a single source: {text: <title>, link: <course_link>}
        self.assertEqual(len(sources), 1)
        self.assertEqual(set(sources[0].keys()), {"text", "link"})

    def test_query_sources_validate_against_pydantic_source_model(self):
        """Mirror of app.Source — confirms RAGSystem.query() output validates against the
        Pydantic shape FastAPI uses for response validation."""
        from typing import List, Optional

        from pydantic import BaseModel

        class _Source(BaseModel):
            text: str
            link: Optional[str] = None

        class _QueryResponse(BaseModel):
            answer: str
            sources: List[_Source]
            session_id: str

        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "x"}, "tid")],
        )
        second = fake_response(stop_reason="end_turn", blocks=[text_block("ans")])
        self.mock_client.messages.create.side_effect = [first, second]

        answer, sources = self.rag.query("x?")

        # Will raise ValidationError if shape mismatches
        _QueryResponse(answer=answer, sources=sources, session_id="s1")

    def test_query_accumulates_sources_across_two_tool_rounds(self):
        """Round 1 calls get_course_outline (1 source), round 2 calls search_course_content
        (>=1 source). The final sources list must contain BOTH rounds' sources, not just
        the most recent. Forced 3rd no-tools call returns the text answer."""
        r1 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("get_course_outline",
                                   {"course_name": "Test"}, "t1")],
        )
        r2 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content",
                                   {"query": "MCP"}, "t2")],
        )
        r3 = fake_response(stop_reason="end_turn", blocks=[text_block("combined")])
        self.mock_client.messages.create.side_effect = [r1, r2, r3]

        answer, sources = self.rag.query("compare lesson 4 of Test against MCP")

        self.assertEqual(answer, "combined")
        # At minimum: one source from the outline tool (course-level) + one from the
        # content search (lesson-level). They may exceed 2 because search returns
        # multiple chunks, but BOTH origins must be present.
        source_texts = [s["text"] for s in sources]
        self.assertTrue(any(t == "Test Course" for t in source_texts),
                        f"Outline source missing from {source_texts!r}")
        self.assertTrue(any("Lesson" in t for t in source_texts),
                        f"Search-with-lesson source missing from {source_texts!r}")

    def test_query_resets_sources_at_start_so_prior_query_leaks_dont_carry_over(self):
        """If query 1 populates sources and query 2 is a no-tool answer, query 2 must
        return an empty sources list — not query 1's leftovers."""
        q1_r1 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "MCP"}, "t1")],
        )
        q1_r2 = fake_response(stop_reason="end_turn", blocks=[text_block("a1")])
        q2 = fake_response(stop_reason="end_turn", blocks=[text_block("a2")])
        self.mock_client.messages.create.side_effect = [q1_r1, q1_r2, q2]

        _, sources1 = self.rag.query("first")
        _, sources2 = self.rag.query("second")

        self.assertGreater(len(sources1), 0)
        self.assertEqual(sources2, [])


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults


def make_results(docs, metas, distances=None, error=None):
    if distances is None:
        distances = [0.0] * len(docs)
    return SearchResults(documents=docs, metadata=metas, distances=distances, error=error)


class CourseSearchToolTest(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock(spec=VectorStore)
        self.tool = CourseSearchTool(self.store)

    def test_execute_returns_formatted_text_with_headers(self):
        self.store.search.return_value = make_results(
            docs=["First chunk", "Second chunk"],
            metas=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2},
            ],
        )
        self.store.get_lesson_link.side_effect = lambda t, n: f"https://example.com/{t}/{n}"

        out = self.tool.execute("anything")

        self.assertIn("[Test Course - Lesson 1]\nFirst chunk", out)
        self.assertIn("[Test Course - Lesson 2]\nSecond chunk", out)
        self.assertIn("First chunk\n\n[Test Course - Lesson 2]", out)

    def test_execute_returns_store_error_verbatim(self):
        self.store.search.return_value = make_results(docs=[], metas=[], error="kaboom")
        self.assertEqual(self.tool.execute("anything"), "kaboom")

    def test_execute_returns_no_results_message_with_filter_info(self):
        self.store.search.return_value = make_results(docs=[], metas=[])
        out = self.tool.execute("anything", course_name="MCP", lesson_number=2)
        self.assertEqual(out, "No relevant content found in course 'MCP' in lesson 2.")

    def test_execute_forwards_course_name_and_lesson_to_store(self):
        self.store.search.return_value = make_results(docs=[], metas=[])
        self.tool.execute("query text", course_name="MCP", lesson_number=3)
        self.store.search.assert_called_once_with(
            query="query text", course_name="MCP", lesson_number=3
        )

    def test_format_results_stashes_dict_sources_on_last_sources(self):
        self.store.search.return_value = make_results(
            docs=["chunk a", "chunk b"],
            metas=[
                {"course_title": "Course A", "lesson_number": 0},
                {"course_title": "Course B", "lesson_number": 5},
            ],
        )
        self.store.get_lesson_link.side_effect = lambda t, n: f"https://example.com/{t}/{n}"

        self.tool.execute("anything")

        self.assertEqual(len(self.tool.last_sources), 2)
        for src in self.tool.last_sources:
            self.assertIsInstance(src, dict)
            self.assertEqual(set(src.keys()), {"text", "link"})
        self.assertEqual(self.tool.last_sources[0]["text"], "Course A - Lesson 0")
        self.assertEqual(self.tool.last_sources[0]["link"], "https://example.com/Course A/0")

    def test_format_results_calls_get_lesson_link_when_lesson_present(self):
        self.store.search.return_value = make_results(
            docs=["doc"],
            metas=[{"course_title": "X", "lesson_number": 7}],
        )
        self.store.get_lesson_link.return_value = "https://link"
        self.tool.execute("q")
        self.store.get_lesson_link.assert_called_once_with("X", 7)

    def test_format_results_link_is_none_when_no_lesson_number(self):
        self.store.search.return_value = make_results(
            docs=["doc"],
            metas=[{"course_title": "X", "lesson_number": None}],
        )
        self.tool.execute("q")
        self.assertEqual(self.tool.last_sources, [{"text": "X", "link": None}])
        self.store.get_lesson_link.assert_not_called()


if __name__ == "__main__":
    unittest.main()

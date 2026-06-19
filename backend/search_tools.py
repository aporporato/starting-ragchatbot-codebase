from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI: text label + optional lesson link
            source_text = course_title
            source_link = None
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
                source_link = self.store.get_lesson_link(course_title, lesson_num)
            sources.append({"text": source_text, "link": source_link})

            formatted.append(f"{header}\n{doc}")

        # Append sources so multiple tool calls within one query accumulate.
        self.last_sources.extend(sources)

        return "\n\n".join(formatted)

class CourseOutlineTool(Tool):
    """Tool for retrieving a course outline (title, link, and full lesson list)"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last outline lookup

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": (
                "Get the outline of a course: the course title, course link, and the "
                "complete list of lessons (each lesson's number and title). Use this "
                "for questions about a course's structure, syllabus, or lesson list."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Resolve the course title, then return its outline.

        Args:
            course_name: User-supplied course title (partial match supported)

        Returns:
            Formatted outline string or an error message
        """
        import json

        # Step 1: resolve the user-supplied name to a canonical title
        resolved_title = self.store._resolve_course_name(course_name)
        if not resolved_title:
            return f"No course found matching '{course_name}'"

        # Step 2: fetch the catalog entry by ID (the title is the ID)
        try:
            results = self.store.course_catalog.get(ids=[resolved_title])
        except Exception as e:
            return f"Error retrieving course outline: {e}"

        if not results or not results.get('metadatas'):
            return f"No metadata found for course '{resolved_title}'"

        metadata = results['metadatas'][0]
        course_link = metadata.get('course_link')

        lessons_json = metadata.get('lessons_json')
        lessons = json.loads(lessons_json) if lessons_json else []

        # Step 3: format output
        lines = [f"Course Title: {resolved_title}"]
        if course_link:
            lines.append(f"Course Link: {course_link}")
        lines.append(f"Lessons ({len(lessons)}):")
        for lesson in lessons:
            num = lesson.get('lesson_number')
            title = lesson.get('lesson_title', 'Untitled')
            lines.append(f"  Lesson {num}: {title}")

        # Append so the source survives alongside any other tool's sources in the same query.
        self.last_sources.append({"text": resolved_title, "link": course_link})

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Aggregate sources from every tool that tracks them (preserves registration order)."""
        all_sources = []
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                all_sources.extend(tool.last_sources)
        return all_sources

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []
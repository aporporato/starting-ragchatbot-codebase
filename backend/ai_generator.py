import os
import sys
import anthropic
from typing import List, Optional, Dict, Any


def _configure_tls() -> "Optional[Any]":
    """Return an httpx.Client configured for this environment's TLS, or None to use the SDK default.

    Resolution order:
      1. ANTHROPIC_INSECURE_TLS=1 -> httpx.Client(verify=False)  (escape hatch, insecure)
      2. truststore available    -> inject_into_ssl() so the OS native trust store is used,
                                    then let the SDK build its default client
      3. neither                 -> None (SDK default)
    """
    if os.environ.get("ANTHROPIC_INSECURE_TLS") == "1":
        print("WARNING: ANTHROPIC_INSECURE_TLS=1 — skipping TLS verification.", file=sys.stderr)
        import httpx
        return httpx.Client(verify=False)
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    return None


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for retrieving course information.

Available tools:
- **search_course_content**: Search course materials for specific content or detailed educational topics.
- **get_course_outline**: Retrieve a course's outline — its title, course link, and the full lesson list (each lesson's number and title).

Tool Usage:
- Use **search_course_content** for questions about specific course content or detailed educational materials.
- Use **get_course_outline** for questions about a course's outline, structure, syllabus, or lesson list. When you answer an outline-related query, your response must include: the **course title**, the **course link**, and for every lesson the **lesson number and lesson title**.
- You may make **up to 2 sequential tool calls** per query. Use a second call only when the first result is insufficient — for example, to look up details about a lesson title or course you just discovered. Do not repeat an identical call.
- Synthesize tool results into accurate, fact-based responses.
- If a tool yields no results, state this clearly without offering alternatives.
- If a tool returns an error, you may retry once with different arguments within the same query; if it errors again, tell the user the lookup failed.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without calling a tool.
- **Course-specific questions**: Call the appropriate tool first, then answer.
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis.
 - Do not mention "based on the search results".


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        http_client = _configure_tls()
        if http_client is not None:
            self.client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to `self.max_tool_rounds` sequential tool-use rounds. Each round is
        a separate `messages.create` call with `tools=...`; Claude can reason about the
        prior round's tool results before issuing the next tool call. The loop exits as
        soon as Claude returns a non-tool_use response, or after the cap is reached (in
        which case a final call without tools is made to force a text answer).
        """

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages: List[Dict[str, Any]] = [{"role": "user", "content": query}]

        # No tools wired up: single shot, same as before.
        if not tools or tool_manager is None:
            response = self.client.messages.create(
                **self.base_params, messages=messages, system=system_content,
            )
            return self._extract_text(response)

        for _ in range(self.max_tool_rounds):
            response = self.client.messages.create(
                **self.base_params,
                messages=list(messages),
                system=system_content,
                tools=tools,
                tool_choice={"type": "auto"},
            )

            if response.stop_reason != "tool_use":
                return self._extract_text(response)

            # Append assistant turn verbatim (preserves leading text + tool_use blocks).
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                try:
                    output = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
                except Exception as e:
                    # Surface the error to Claude as is_error; loop continues so it can retry.
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool '{block.name}' failed: {e}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

        # Cap reached and Claude still wants to call tools — force a final text answer.
        final_response = self.client.messages.create(
            **self.base_params, messages=list(messages), system=system_content,
        )
        return self._extract_text(final_response)

    @staticmethod
    def _extract_text(response) -> str:
        """Return the text of the first text block in response.content, or '' if none."""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""
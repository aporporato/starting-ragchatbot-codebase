import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, tool_input, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=id_)


def fake_response(stop_reason, blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class AIGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.client_patcher = patch("ai_generator.anthropic.Anthropic")
        mock_anthropic_cls = self.client_patcher.start()
        self.mock_client = MagicMock()
        mock_anthropic_cls.return_value = self.mock_client
        self.aig = AIGenerator(api_key="fake-key", model="fake-model")

    def tearDown(self):
        self.client_patcher.stop()

    # ------------------------------------------------------------------ existing
    def test_direct_text_path_no_tool_use(self):
        self.mock_client.messages.create.return_value = fake_response(
            stop_reason="end_turn",
            blocks=[text_block("Direct answer")],
        )
        out = self.aig.generate_response(
            query="hi",
            tools=[{"name": "search_course_content"}],
            tool_manager=MagicMock(),
        )
        self.assertEqual(out, "Direct answer")
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
        call_kwargs = self.mock_client.messages.create.call_args.kwargs
        self.assertIn("tools", call_kwargs)
        self.assertEqual(call_kwargs["tool_choice"], {"type": "auto"})

    def test_tool_result_message_shape(self):
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "x"}, "tid")],
        )
        second = fake_response(stop_reason="end_turn", blocks=[text_block("done")])
        self.mock_client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool output"

        self.aig.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        second_messages = self.mock_client.messages.create.call_args_list[1].kwargs["messages"]
        # Should have: user query, assistant tool_use, user tool_result
        self.assertEqual(len(second_messages), 3)
        self.assertEqual(second_messages[2]["role"], "user")
        tool_result = second_messages[2]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "tid")
        self.assertEqual(tool_result["content"], "tool output")

    def test_system_prompt_and_history_passed_through(self):
        self.mock_client.messages.create.return_value = fake_response(
            stop_reason="end_turn",
            blocks=[text_block("ok")],
        )
        self.aig.generate_response(
            query="q",
            conversation_history="User: hello\nAssistant: hi",
            tools=[{"name": "search_course_content"}],
            tool_manager=MagicMock(),
        )
        system_arg = self.mock_client.messages.create.call_args.kwargs["system"]
        self.assertIn(self.aig.SYSTEM_PROMPT.strip(), system_arg)
        self.assertIn("Previous conversation:\nUser: hello\nAssistant: hi", system_arg)

    def test_tool_use_response_with_text_then_tool_block_succeeds(self):
        """Claude often emits a leading text block before the tool_use block.
        The loop must still find the tool block and preserve the text in the assistant turn."""
        first = fake_response(
            stop_reason="tool_use",
            blocks=[
                text_block("Let me search for that"),
                tool_use_block("search_course_content", {"query": "x"}, "tid"),
            ],
        )
        second = fake_response(stop_reason="end_turn", blocks=[text_block("final")])
        self.mock_client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "x"
        out = self.aig.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )
        self.assertEqual(out, "final")
        tool_manager.execute_tool.assert_called_once()
        # Assistant message appended verbatim, including the leading text block.
        second_messages = self.mock_client.messages.create.call_args_list[1].kwargs["messages"]
        appended_content = second_messages[1]["content"]
        self.assertEqual(len(appended_content), 2)
        self.assertEqual(appended_content[0].type, "text")
        self.assertEqual(appended_content[1].type, "tool_use")

    # ------------------------------------------------------------------ updated
    def test_single_tool_round_then_text_keeps_tools_on_round_two(self):
        """Round 1 returns tool_use, round 2 returns text. Critically, round 2
        STILL passes tools/tool_choice — that's the key behavioural diff vs. the
        old single-shot _handle_tool_execution which dropped tools on the second call."""
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "MCP"}, "tool_1")],
        )
        second = fake_response(
            stop_reason="end_turn",
            blocks=[text_block("Final synthesized answer")],
        )
        self.mock_client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool output text"

        out = self.aig.generate_response(
            query="what is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        self.assertEqual(out, "Final synthesized answer")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="MCP")

        second_kwargs = self.mock_client.messages.create.call_args_list[1].kwargs
        self.assertIn("tools", second_kwargs)
        self.assertEqual(second_kwargs["tool_choice"], {"type": "auto"})

    def test_extract_text_tolerates_non_text_first_block(self):
        """The new _extract_text helper scans for the first text block instead of
        indexing content[0].text, which used to AttributeError on responses that
        led with a non-text block."""
        first = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "x"}, "tid")],
        )
        weird_then_text = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="something_else"), text_block("real answer")],
        )
        self.mock_client.messages.create.side_effect = [first, weird_then_text]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "x"

        out = self.aig.generate_response(
            query="q",
            tools=[{"name": "x"}],
            tool_manager=tool_manager,
        )
        self.assertEqual(out, "real answer")

    # ------------------------------------------------------------------ new
    def test_two_round_sequential_tool_use_happy_path(self):
        """Round 1: outline tool. Round 2: search tool, fed by round 1's result.
        Round 3 (no tools): final text answer. Asserts call count, exec order,
        and that only call 3 omits tools."""
        r1 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("get_course_outline", {"course_name": "X"}, "t1")],
        )
        r2 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content",
                                   {"query": "lesson 4 topic"}, "t2")],
        )
        r3 = fake_response(stop_reason="end_turn", blocks=[text_block("combined answer")])
        self.mock_client.messages.create.side_effect = [r1, r2, r3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [
            "Lesson 4: Vector databases",  # outline result
            "Course Y discusses vector databases.",  # search result
        ]

        out = self.aig.generate_response(
            query="find a course that covers the same topic as lesson 4 of X",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        self.assertEqual(out, "combined answer")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(tool_manager.execute_tool.call_count, 2)
        # First exec was the outline, second was the search.
        first_exec = tool_manager.execute_tool.call_args_list[0]
        second_exec = tool_manager.execute_tool.call_args_list[1]
        self.assertEqual(first_exec.args[0], "get_course_outline")
        self.assertEqual(second_exec.args[0], "search_course_content")

        kw1 = self.mock_client.messages.create.call_args_list[0].kwargs
        kw2 = self.mock_client.messages.create.call_args_list[1].kwargs
        kw3 = self.mock_client.messages.create.call_args_list[2].kwargs
        self.assertIn("tools", kw1)
        self.assertIn("tools", kw2)
        self.assertNotIn("tools", kw3)
        self.assertNotIn("tool_choice", kw3)

    def test_two_round_cap_when_model_wants_third_tool_call(self):
        """Both rounds 1 and 2 return tool_use. The forced 3rd call (no tools)
        is what produces the final text. Exactly 3 calls, 2 tool executions,
        5 messages passed to the 3rd call."""
        r1 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "a"}, "t1")],
        )
        r2 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "b"}, "t2")],
        )
        r3 = fake_response(stop_reason="end_turn",
                           blocks=[text_block("forced final without tools")])
        self.mock_client.messages.create.side_effect = [r1, r2, r3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["res1", "res2"]

        out = self.aig.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        self.assertEqual(out, "forced final without tools")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(tool_manager.execute_tool.call_count, 2)

        kw3 = self.mock_client.messages.create.call_args_list[2].kwargs
        self.assertNotIn("tools", kw3)
        self.assertNotIn("tool_choice", kw3)
        # user / assistant(tool_use) / user(tool_result) / assistant(tool_use) / user(tool_result)
        self.assertEqual(len(kw3["messages"]), 5)

    def test_tool_error_does_not_terminate_loop_and_surfaces_is_error(self):
        """Round 1 tool raises. The loop continues — round 2 still gets tools,
        and the round-1 assistant turn's tool_result has is_error=True."""
        r1 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "bad"}, "t1")],
        )
        r2 = fake_response(stop_reason="end_turn",
                           blocks=[text_block("apologetic answer")])
        self.mock_client.messages.create.side_effect = [r1, r2]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("boom")

        out = self.aig.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        self.assertEqual(out, "apologetic answer")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        kw2 = self.mock_client.messages.create.call_args_list[1].kwargs
        # Round 2 STILL has tools (retry budget remains).
        self.assertIn("tools", kw2)
        # Last message is the user tool_result; its block has is_error=True and the error text.
        tool_result_block = kw2["messages"][-1]["content"][0]
        self.assertEqual(tool_result_block["type"], "tool_result")
        self.assertTrue(tool_result_block.get("is_error"))
        self.assertIn("boom", tool_result_block["content"])

    def test_messages_accumulate_across_rounds(self):
        """The messages list passed to round N+1 must be a strict superset of round N's."""
        r1 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("get_course_outline", {"course_name": "X"}, "t1")],
        )
        r2 = fake_response(
            stop_reason="tool_use",
            blocks=[tool_use_block("search_course_content", {"query": "y"}, "t2")],
        )
        r3 = fake_response(stop_reason="end_turn", blocks=[text_block("done")])
        self.mock_client.messages.create.side_effect = [r1, r2, r3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["o1", "s1"]

        self.aig.generate_response(
            query="q",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        msgs_r1 = self.mock_client.messages.create.call_args_list[0].kwargs["messages"]
        msgs_r2 = self.mock_client.messages.create.call_args_list[1].kwargs["messages"]
        msgs_r3 = self.mock_client.messages.create.call_args_list[2].kwargs["messages"]
        self.assertEqual(len(msgs_r1), 1)              # just the user query
        self.assertEqual(len(msgs_r2), 3)              # + assistant(tool_use) + user(tool_result)
        self.assertEqual(len(msgs_r3), 5)              # + another pair
        # Round 2's tool_result references round 1's tool_use id.
        self.assertEqual(msgs_r2[2]["content"][0]["tool_use_id"], "t1")
        # Round 3 contains round 2's pair too.
        self.assertEqual(msgs_r3[4]["content"][0]["tool_use_id"], "t2")

    def test_max_tool_rounds_constructor_arg_respected(self):
        """max_tool_rounds=1 means at most 1 tool round before the forced final call."""
        with patch("ai_generator.anthropic.Anthropic") as cls:
            client = MagicMock()
            cls.return_value = client
            aig = AIGenerator(api_key="k", model="m", max_tool_rounds=1)

            r1 = fake_response(
                stop_reason="tool_use",
                blocks=[tool_use_block("search_course_content", {"query": "x"}, "t1")],
            )
            r2 = fake_response(stop_reason="end_turn",
                               blocks=[text_block("forced after one round")])
            client.messages.create.side_effect = [r1, r2]

            tool_manager = MagicMock()
            tool_manager.execute_tool.return_value = "res"

            out = aig.generate_response(
                query="q",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            self.assertEqual(out, "forced after one round")
            self.assertEqual(client.messages.create.call_count, 2)
            # The second (final) call must NOT have tools, since we already hit the cap of 1.
            self.assertNotIn("tools", client.messages.create.call_args_list[1].kwargs)


if __name__ == "__main__":
    unittest.main()

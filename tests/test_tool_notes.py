"""Tool-loop findings must reach Hermes as visible text.

agy does not export reasoning, and Hermes compaction reduces older tool results
to one-line stubs. The prompts therefore ask for a short note before tool calls,
and the stream must deliver that leading note as assistant content.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from client import AntigravityClient
from prompt import _TOOL_NOTE_RULE, _format_delta_prompt, _format_messages_as_prompt


def _tool_loop_history() -> list[dict]:
    return [
        {"role": "user", "content": "Find the value."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value=42"},
    ]


class ToolNotePromptTests(unittest.TestCase):
    def test_full_prompt_asks_for_visible_note(self):
        prompt = _format_messages_as_prompt([{"role": "user", "content": "Find the value."}])
        self.assertIn(_TOOL_NOTE_RULE, prompt)
        self.assertNotIn("emit ONLY <tool_call>", prompt)

    def test_full_resend_of_tool_loop_asks_for_visible_note(self):
        # Compaction, respawn and broken-pipe retries re-send the whole history.
        prompt = _format_messages_as_prompt(_tool_loop_history())
        self.assertIn(_TOOL_NOTE_RULE, prompt)

    def test_tool_delta_asks_for_visible_note(self):
        delta = _format_delta_prompt(
            [{"role": "tool", "tool_call_id": "c1", "content": "value=42"}],
            pending_user_text="Find the value.",
        )
        self.assertIn(_TOOL_NOTE_RULE, delta)
        self.assertNotIn("emit ONLY <tool_call>", delta)


class ToolNoteStreamTests(unittest.TestCase):
    def test_leading_note_is_delivered_with_tool_call(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        client = AntigravityClient(cwd=tmp_dir.name)
        call = {
            "id": "call_note_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "n2.txt"})},
        }
        note = "start.txt gave fragment ALPHA-17 and points to n2.txt; reading it next."
        body = f"{note}\n<tool_call>{json.dumps(call)}</tool_call>"
        events = [
            json.dumps({"event": "init", "conversation_id": "tool-note"}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": body}}),
            json.dumps({"event": "step_update", "step_update": {"text_delta": "\nTool Result (call_note_1):\nfake"}}),
            json.dumps({
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": body + "\nTool Result (call_note_1):\nfake",
                    "usage": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
                },
            }),
        ]
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = io.StringIO("\n".join(events) + "\n")
        proc.stderr = io.StringIO("")
        proc.poll.return_value = 0
        proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=proc), patch("client.is_authenticated", return_value=True):
            chunks = list(client.chat.completions.create(
                model="gemini-3.8-flash-high",
                messages=[{"role": "user", "content": "follow the chain"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                stream=True,
            ))

        content = "".join(c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content)
        tool_chunks = [c for c in chunks if c.choices and c.choices[0].delta.tool_calls]
        self.assertEqual(content.strip(), note)
        self.assertEqual(len(tool_chunks), 1)
        self.assertEqual(tool_chunks[0].choices[0].delta.tool_calls[0].id, "call_note_1")


if __name__ == "__main__":
    unittest.main()

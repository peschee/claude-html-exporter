import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import claude_export


class TestParseJsonl(unittest.TestCase):
    def test_parse_jsonl_skips_invalid_lines(self):
        content = "\n".join(
            [
                json.dumps({"a": 1}),
                "not json",
                "",
                json.dumps({"b": 2}),
            ]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(content)
            path = handle.name

        try:
            result = claude_export.parse_jsonl(path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["a"], 1)
            self.assertEqual(result[1]["b"], 2)
        finally:
            os.unlink(path)


class TestBuildConversation(unittest.TestCase):
    def test_build_conversation_merges_and_attaches_tool_results(self):
        lines = [
            {
                "type": "assistant",
                "timestamp": "t1",
                "message": {"id": "a1", "content": [{"type": "text", "text": "first"}]},
            },
            {
                "type": "assistant",
                "timestamp": "t1",
                "message": {
                    "id": "a1",
                    "content": [{"type": "thinking", "thinking": "think"}],
                },
            },
            {
                "type": "assistant",
                "timestamp": "t1",
                "message": {
                    "id": "a1",
                    "content": [
                        {"type": "text", "text": "second"},
                        {
                            "type": "tool_use",
                            "id": "tool1",
                            "name": "grep",
                            "input": {"pattern": "x"},
                        },
                        {
                            "type": "tool_use",
                            "id": "tool1",
                            "name": "grep",
                            "input": {"pattern": "x"},
                        },
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "t2",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool1",
                            "content": [{"type": "text", "text": "out"}],
                        }
                    ]
                },
            },
        ]

        conversation = claude_export.build_conversation(lines)
        self.assertEqual(len(conversation), 1)

        msg = conversation[0]
        self.assertEqual(msg["role"], "assistant")
        blocks = msg["blocks"]
        self.assertEqual(blocks[0]["type"], "thinking")

        text_block = next(block for block in blocks if block["type"] == "text")
        self.assertEqual(text_block["text"], "second")

        tool_use_blocks = [block for block in blocks if block["type"] == "tool_use"]
        self.assertEqual(len(tool_use_blocks), 1)

        tool_result_blocks = [
            block for block in blocks if block["type"] == "tool_result"
        ]
        self.assertEqual(len(tool_result_blocks), 1)
        self.assertEqual(tool_result_blocks[0]["tool_name"], "grep")

    def test_build_conversation_joins_user_text_blocks(self):
        lines = [
            {
                "type": "user",
                "timestamp": "t1",
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ]
                },
            }
        ]

        conversation = claude_export.build_conversation(lines)
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0]["blocks"][0]["text"], "Hello\n\nWorld")


class TestNormalizeToolResult(unittest.TestCase):
    def test_normalize_tool_result_list_content(self):
        block = {
            "tool_use_id": "tool1",
            "is_error": True,
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ],
        }
        result = claude_export._normalize_tool_result(block, {"tool1": "bash"})
        self.assertEqual(result["tool_name"], "bash")
        self.assertTrue(result["is_error"])
        self.assertEqual(result["text"], "line1\nline2")

    def test_normalize_tool_result_truncates(self):
        block = {"tool_use_id": "tool1", "content": "abcdefghijk"}
        with patch.object(claude_export, "TRUNCATE_LIMIT", 10):
            result = claude_export._normalize_tool_result(block, {"tool1": "bash"})
        self.assertTrue(result["text"].startswith("abcdefghij"))
        self.assertIn("truncated, 11 chars total", result["text"])


class TestResolveSession(unittest.TestCase):
    def test_resolve_session_direct_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = handle.name

        try:
            result = claude_export.resolve_session(path)
            self.assertEqual(Path(result), Path(path).resolve())
        finally:
            os.unlink(path)

    def test_resolve_session_from_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "proj"
            project_dir.mkdir()
            session_id = "abc123"
            session_file = project_dir / f"{session_id}.jsonl"
            session_file.write_text("")
            index_path = project_dir / "sessions-index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "sessionId": session_id,
                                "project": "proj",
                                "fullPath": str(session_file),
                            }
                        ]
                    }
                )
            )

            with patch.object(claude_export, "CLAUDE_DIR", root):
                result = claude_export.resolve_session(session_id)
            self.assertEqual(Path(result).resolve(), session_file.resolve())


if __name__ == "__main__":
    unittest.main()

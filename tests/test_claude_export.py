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

    def test_build_conversation_tool_result_without_assistant(self):
        lines = [
            {
                "type": "user",
                "timestamp": "t1",
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
            {
                "type": "assistant",
                "timestamp": "t2",
                "message": {"id": "a1", "content": [{"type": "text", "text": "done"}]},
            },
        ]

        conversation = claude_export.build_conversation(lines)
        self.assertEqual(len(conversation), 2)
        self.assertEqual(conversation[0]["role"], "tool")
        self.assertEqual(conversation[0]["blocks"][0]["tool_name"], "unknown")
        self.assertEqual(conversation[1]["role"], "assistant")

    def test_build_conversation_skips_sidechain_messages(self):
        lines = [
            {
                "type": "user",
                "timestamp": "t1",
                "isSidechain": True,
                "message": {"content": "side"},
            },
            {
                "type": "assistant",
                "timestamp": "t1",
                "isSidechain": True,
                "message": {"id": "a1", "content": [{"type": "text", "text": "side"}]},
            },
            {
                "type": "user",
                "timestamp": "t2",
                "message": {"content": "main"},
            },
        ]

        conversation = claude_export.build_conversation(lines)
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0]["role"], "user")
        self.assertEqual(conversation[0]["blocks"][0]["text"], "main")


    def test_build_conversation_folds_compaction_into_boundary(self):
        lines = [
            {"type": "user", "timestamp": "t1", "message": {"content": "before"}},
            {
                "type": "system",
                "subtype": "compact_boundary",
                "timestamp": "t2",
                "compactMetadata": {
                    "trigger": "manual",
                    "preTokens": 118242,
                    "postTokens": 8744,
                },
            },
            {
                "type": "user",
                "timestamp": "t3",
                "isCompactSummary": True,
                "message": {"content": "This session is being continued..."},
            },
            {"type": "user", "timestamp": "t4", "message": {"content": "after"}},
        ]

        conversation = claude_export.build_conversation(lines)
        self.assertEqual([m["role"] for m in conversation], ["user", "compaction", "user"])
        boundary = conversation[1]
        self.assertEqual(boundary["trigger"], "manual")
        self.assertEqual(boundary["pre_tokens"], 118242)
        self.assertEqual(boundary["post_tokens"], 8744)
        self.assertEqual(boundary["summary"], "This session is being continued...")

    def test_build_conversation_summary_without_boundary(self):
        lines = [
            {
                "type": "user",
                "timestamp": "t1",
                "isCompactSummary": True,
                "message": {"content": "summary only"},
            },
            {"type": "user", "timestamp": "t2", "message": {"content": "after"}},
        ]

        conversation = claude_export.build_conversation(lines)
        self.assertEqual([m["role"] for m in conversation], ["compaction", "user"])
        self.assertEqual(conversation[0]["summary"], "summary only")
        self.assertIsNone(conversation[0]["pre_tokens"])


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

    def test_normalize_tool_result_non_string_content(self):
        block = {"tool_use_id": "tool1", "content": {"key": "value"}}
        result = claude_export._normalize_tool_result(block, {"tool1": "bash"})
        self.assertEqual(result["text"], "{'key': 'value'}")


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

    def test_resolve_session_from_index_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "proj"
            project_dir.mkdir()
            sessions_dir = project_dir / "sessions"
            sessions_dir.mkdir()
            session_id = "relative-1"
            session_file = sessions_dir / f"{session_id}.jsonl"
            session_file.write_text("")
            index_path = project_dir / "sessions-index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "sessionId": session_id,
                                "project": "proj",
                                "fullPath": str(Path("sessions") / session_file.name),
                            }
                        ]
                    }
                )
            )

            with patch.object(claude_export, "CLAUDE_DIR", root):
                result = claude_export.resolve_session(session_id)
            self.assertEqual(Path(result).resolve(), session_file.resolve())


class TestExtractMetadata(unittest.TestCase):
    def test_extract_metadata_first_values(self):
        lines = [
            {
                "sessionId": "sid-1",
                "timestamp": "t1",
                "gitBranch": "main",
                "cwd": "/tmp",
            },
            {
                "type": "assistant",
                "message": {"model": "claude-v1"},
            },
            {
                "sessionId": "sid-2",
                "timestamp": "t2",
                "gitBranch": "dev",
                "cwd": "/other",
            },
            {
                "type": "assistant",
                "message": {"model": "claude-v2"},
            },
        ]

        metadata = claude_export.extract_metadata(lines)
        self.assertEqual(metadata["session_id"], "sid-1")
        self.assertEqual(metadata["date"], "t1")
        self.assertEqual(metadata["git_branch"], "main")
        self.assertEqual(metadata["cwd"], "/tmp")
        self.assertEqual(metadata["model"], "claude-v1")


class TestReadSessionStub(unittest.TestCase):
    def test_read_session_stub_first_user_prompt(self):
        long_prompt = "x" * 250
        lines = [
            "not json",
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t1",
                    "gitBranch": "main",
                    "message": {"content": long_prompt},
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t2",
                    "message": {"content": "later"},
                }
            ),
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            stub = claude_export._read_session_stub(path)
            self.assertEqual(stub["created"], "t1")
            self.assertEqual(stub["git_branch"], "main")
            self.assertEqual(len(stub["first_prompt"]), 200)
        finally:
            os.unlink(path)


    def test_read_session_stub_skips_compaction_summary(self):
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t1",
                    "isCompactSummary": True,
                    "message": {"content": "This session is being continued"},
                }
            ),
            json.dumps(
                {"type": "user", "timestamp": "t2", "message": {"content": "real"}}
            ),
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            stub = claude_export._read_session_stub(path)
            self.assertEqual(stub["first_prompt"], "real")
            self.assertEqual(stub["created"], "t2")
        finally:
            os.unlink(path)

    def test_read_session_stub_skips_caveat_and_command_records(self):
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t1",
                    "gitBranch": "main",
                    "message": {
                        "content": "<local-command-caveat>Caveat</local-command-caveat>"
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t2",
                    "message": {
                        "content": "<command-name>/clear</command-name>"
                        "<command-args></command-args>"
                    },
                }
            ),
            json.dumps(
                {"type": "user", "timestamp": "t3", "message": {"content": "real ask"}}
            ),
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            stub = claude_export._read_session_stub(path)
            self.assertEqual(stub["first_prompt"], "real ask")
            # created/branch still come from the first user record
            self.assertEqual(stub["created"], "t1")
            self.assertEqual(stub["git_branch"], "main")
        finally:
            os.unlink(path)

    def test_read_session_stub_falls_back_to_command_when_only_commands(self):
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t1",
                    "message": {
                        "content": "<command-name>/init</command-name>"
                        "<command-args>--force</command-args>"
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t2",
                    "message": {"content": [{"type": "tool_result", "content": "x"}]},
                }
            ),
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            stub = claude_export._read_session_stub(path)
            self.assertEqual(stub["first_prompt"], "/init --force")
            self.assertEqual(stub["created"], "t1")
        finally:
            os.unlink(path)


class TestAiTitle(unittest.TestCase):
    def _write(self, lines):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.write("\n".join(lines))
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_read_ai_title_picks_last_record(self):
        path = self._write(
            [
                json.dumps({"type": "user", "message": {"content": "hi"}}),
                json.dumps({"type": "ai-title", "aiTitle": "First   title"}),
                json.dumps({"type": "assistant", "message": {"content": []}}),
                json.dumps({"type": "ai-title", "aiTitle": "Final title"}),
            ]
        )
        self.assertEqual(claude_export._read_ai_title(path), "Final title")

    def test_read_ai_title_missing(self):
        path = self._write([json.dumps({"type": "user", "message": {"content": "hi"}})])
        self.assertEqual(claude_export._read_ai_title(path), "")
        self.assertEqual(claude_export._read_ai_title("/nonexistent/x.jsonl"), "")

    def test_session_stub_and_label_use_title(self):
        path = self._write(
            [
                json.dumps({"type": "user", "timestamp": "t1", "message": {"content": "prompt"}}),
                json.dumps({"type": "ai-title", "aiTitle": "Nice title"}),
            ]
        )
        stub = claude_export._read_session_stub(path)
        self.assertEqual(stub["title"], "Nice title")
        self.assertEqual(stub["first_prompt"], "prompt")
        self.assertEqual(claude_export._session_label(stub), "Nice title")
        self.assertEqual(
            claude_export._session_label({"first_prompt": "prompt", "title": ""}),
            "prompt",
        )

    def test_extract_metadata_picks_last_title(self):
        lines = [
            {"type": "ai-title", "aiTitle": "old"},
            {"type": "user", "sessionId": "s", "timestamp": "t"},
            {"type": "ai-title", "aiTitle": "new"},
        ]
        meta = claude_export.extract_metadata(lines)
        self.assertEqual(meta["title"], "new")
        html = claude_export.generate_html([], meta)
        self.assertIn("<title>new — Claude Code Session</title>", html)

    def test_extract_metadata_without_title(self):
        meta = claude_export.extract_metadata([{"type": "user", "timestamp": "t"}])
        self.assertEqual(meta["title"], "")
        self.assertIn("<title>Claude Code Session", claude_export.generate_html([], meta))


class TestCleanPrompt(unittest.TestCase):
    def test_unwraps_slash_command_with_args(self):
        raw = (
            "<command-name>/humanizer</command-name>\n"
            "<command-message>humanizer</command-message>\n"
            "<command-args>fix tone</command-args>"
        )
        self.assertEqual(claude_export._clean_prompt(raw), "/humanizer fix tone")

    def test_unwraps_slash_command_without_args(self):
        raw = "<command-name>/clear</command-name><command-args></command-args>"
        self.assertEqual(claude_export._clean_prompt(raw), "/clear")

    def test_drops_wrapper_blocks_and_collapses_whitespace(self):
        raw = (
            "<local-command-caveat>Caveat: generated</local-command-caveat>\n"
            "<local-command-stdout>out</local-command-stdout>\n"
            "<system-reminder>hidden</system-reminder>\n"
            "  real   prompt\n here"
        )
        self.assertEqual(claude_export._clean_prompt(raw), "real prompt here")

    def test_plain_and_empty(self):
        self.assertEqual(claude_export._clean_prompt("plain text"), "plain text")
        self.assertEqual(claude_export._clean_prompt(""), "")
        self.assertEqual(claude_export._clean_prompt(None), "")


class TestReadPreview(unittest.TestCase):
    def test_read_preview_skips_tool_results_and_truncates(self):
        lines = [
            json.dumps(
                {
                    "sessionId": "sid-1",
                    "gitBranch": "main",
                    "cwd": "/tmp",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "type": "user",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool1",
                                "content": [{"type": "text", "text": "skip"}],
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {"content": [{"type": "text", "text": "Hello world"}]},
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "model": "claude-v1",
                        "content": [
                            {"type": "text", "text": "Assistant response goes here"}
                        ],
                    },
                }
            ),
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            preview = claude_export._read_preview(
                path, max_lines=10, max_messages=4, max_chars=5
            )
            self.assertEqual(preview["session_id"], "sid-1")
            self.assertEqual(preview["git_branch"], "main")
            self.assertEqual(preview["cwd"], "/tmp")
            self.assertEqual(preview["date"], "2024-01-01T00:00:00Z")
            self.assertEqual(preview["model"], "claude-v1")

            self.assertEqual(len(preview["messages"]), 2)
            self.assertEqual(preview["messages"][0]["role"], "Human")
            self.assertEqual(preview["messages"][0]["text"], "Hello...")
            self.assertEqual(preview["messages"][1]["role"], "Claude")
            self.assertEqual(preview["messages"][1]["text"], "Assis...")
        finally:
            os.unlink(path)

    def test_read_preview_user_string_content(self):
        lines = [
            json.dumps(
                {
                    "sessionId": "sid-1",
                    "timestamp": "t1",
                    "type": "user",
                    "message": {"content": "Hello"},
                }
            )
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            preview = claude_export._read_preview(
                path, max_lines=5, max_messages=2, max_chars=20
            )
            self.assertEqual(len(preview["messages"]), 1)
            self.assertEqual(preview["messages"][0]["role"], "Human")
            self.assertEqual(preview["messages"][0]["text"], "Hello")
        finally:
            os.unlink(path)


    def test_read_preview_marks_compaction_summary(self):
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "timestamp": "t1",
                    "isCompactSummary": True,
                    "message": {"content": "This session is being continued"},
                }
            ),
            json.dumps(
                {"type": "user", "timestamp": "t2", "message": {"content": "real"}}
            ),
        ]

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(lines))
            path = handle.name

        try:
            preview = claude_export._read_preview(path)
            self.assertEqual(
                [m["role"] for m in preview["messages"]], ["Compacted", "Human"]
            )
            self.assertEqual(preview["messages"][1]["text"], "real")
        finally:
            os.unlink(path)


class TestFindSessions(unittest.TestCase):
    def test_find_sessions_reads_index_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_dir = root / "project-one"
            project_dir.mkdir()

            session_id = "sid-index"
            session_path = project_dir / f"{session_id}.jsonl"
            session_path.write_text("")
            index_path = project_dir / "sessions-index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "sessionId": session_id,
                                "projectPath": "/tmp/project-one",
                                "fullPath": str(session_path),
                                "firstPrompt": "hello",
                                "created": "t1",
                                "modified": "t2",
                                "gitBranch": "main",
                                "messageCount": 2,
                            }
                        ]
                    }
                )
            )

            stub_id = "sid-stub"
            stub_path = project_dir / f"{stub_id}.jsonl"
            stub_path.write_text(
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "t3",
                        "gitBranch": "dev",
                        "message": {"content": "prompt"},
                    }
                )
            )

            with patch.object(claude_export, "CLAUDE_DIR", root):
                sessions = claude_export.find_sessions()

        by_id = {session["session_id"]: session for session in sessions}
        self.assertIn(session_id, by_id)
        self.assertIn(stub_id, by_id)

        indexed = by_id[session_id]
        self.assertEqual(indexed["project"], "project-one")
        self.assertEqual(indexed["path"], str(session_path))
        self.assertEqual(indexed["message_count"], 2)

        stub = by_id[stub_id]
        self.assertEqual(stub["first_prompt"], "prompt")
        self.assertEqual(stub["created"], "t3")
        self.assertEqual(stub["git_branch"], "dev")
        self.assertEqual(stub["message_count"], 0)

    def test_find_sessions_project_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_one = root / "Project-One"
            project_two = root / "Other"
            project_one.mkdir()
            project_two.mkdir()

            session_one = project_one / "one.jsonl"
            session_one.write_text(
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "t1",
                        "message": {"content": "one"},
                    }
                )
            )
            session_two = project_two / "two.jsonl"
            session_two.write_text(
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "t2",
                        "message": {"content": "two"},
                    }
                )
            )

            with patch.object(claude_export, "CLAUDE_DIR", root):
                sessions = claude_export.find_sessions(project_filter="one")

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["project"], "Project-One")


class TestProjectGroupOrdering(unittest.TestCase):
    def test_build_items_orders_groups_by_recent_activity(self):
        # alpha's most recent activity is 05; beta falls back to created (06,
        # newest); gamma is oldest at 01. Expect newest-first: beta, alpha, gamma.
        sessions = [
            {"project_path": "/Users/me/alpha", "modified": "2026-02-01T00:00:00Z"},
            {"project_path": "/Users/me/alpha", "modified": "2026-05-01T00:00:00Z"},
            {"project_path": "/Users/me/beta", "modified": "",
             "created": "2026-06-01T00:00:00Z"},
            {"project_path": "/Users/me/gamma", "modified": "2026-01-01T00:00:00Z"},
        ]

        browser = claude_export.SessionBrowser.__new__(claude_export.SessionBrowser)
        browser.sessions = sessions
        browser.filter_text = ""
        browser.collapsed = set()
        browser.cursor = 0
        browser._build_items()

        headers = [i.data for i in browser.items if i.kind == "header"]
        self.assertEqual(headers, ["me/beta", "me/alpha", "me/gamma"])


if __name__ == "__main__":
    unittest.main()

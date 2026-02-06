#!/usr/bin/env python3
"""Claude Code Conversation Exporter

Reads Claude Code session JSONL files and generates standalone HTML files
with a clean, modern chat UI.

Usage:
    python claude_export.py --list [-p PROJECT_FILTER]
    python claude_export.py SESSION_ID [-o output.html]
    python claude_export.py path/to/session.jsonl [-o output.html]
"""

import argparse
import curses
import json
import os
import re
import sys
import traceback
from collections import namedtuple
from datetime import datetime
from pathlib import Path


CLAUDE_DIR = Path.home() / ".claude" / "projects"
TRUNCATE_LIMIT = 50_000
VERBOSE = False


def _debug(message, exc=None):
    if not VERBOSE:
        return
    if exc:
        print(f"[debug] {message}: {exc}", file=sys.stderr)
        traceback.print_exc()
    else:
        print(f"[debug] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def find_sessions(project_filter=None):
    """Scan ~/.claude/projects/ for sessions.

    Returns list of dicts with keys:
        session_id, project, path, first_prompt, created, modified, git_branch
    """
    sessions = []
    if not CLAUDE_DIR.exists():
        return sessions

    for project_dir in sorted(CLAUDE_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        if project_filter and project_filter.lower() not in project_name.lower():
            continue

        # Try sessions-index.json first
        indexed_ids = set()
        index_path = project_dir / "sessions-index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
                for entry in index.get("entries", []):
                    sid = entry.get("sessionId", "")
                    indexed_ids.add(sid)
                    sessions.append(
                        {
                            "session_id": sid,
                            "project": project_name,
                            "project_path": entry.get("projectPath", ""),
                            "path": entry.get("fullPath", ""),
                            "first_prompt": entry.get("firstPrompt", ""),
                            "created": entry.get("created", ""),
                            "modified": entry.get("modified", ""),
                            "git_branch": entry.get("gitBranch", ""),
                            "message_count": entry.get("messageCount", 0),
                        }
                    )
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                _debug("sessions index read failed", exc)

        # Scan .jsonl files not covered by the index
        for jsonl_path in sorted(project_dir.glob("*.jsonl")):
            session_id = jsonl_path.stem
            if session_id in indexed_ids:
                continue
            info = _read_session_stub(jsonl_path)
            sessions.append(
                {
                    "session_id": session_id,
                    "project": project_name,
                    "project_path": "",
                    "path": str(jsonl_path),
                    "first_prompt": info.get("first_prompt", ""),
                    "created": info.get("created", ""),
                    "modified": "",
                    "git_branch": info.get("git_branch", ""),
                    "message_count": 0,
                }
            )

    return sessions


def _read_session_stub(path):
    """Read first user line from a JSONL to extract basic info."""
    try:
        with open(path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "user":
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    prompt = content if isinstance(content, str) else ""
                    return {
                        "first_prompt": prompt[:200],
                        "created": obj.get("timestamp", ""),
                        "git_branch": obj.get("gitBranch", ""),
                    }
    except OSError as exc:
        _debug("read session stub failed", exc)
    return {}


def _read_preview(path, max_lines=50, max_messages=4, max_chars=500):
    """Read first lines of a JSONL to extract metadata + message preview.

    Returns dict with keys: session_id, model, date, git_branch, cwd, messages.
    Each message is {role, text} with text truncated to max_chars.
    """
    preview = {
        "session_id": "",
        "model": "",
        "date": "",
        "git_branch": "",
        "cwd": "",
        "messages": [],
    }
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not preview["session_id"] and obj.get("sessionId"):
                    preview["session_id"] = obj["sessionId"]
                if not preview["git_branch"] and obj.get("gitBranch"):
                    preview["git_branch"] = obj["gitBranch"]
                if not preview["cwd"] and obj.get("cwd"):
                    preview["cwd"] = obj["cwd"]
                if not preview["date"] and obj.get("timestamp"):
                    preview["date"] = obj["timestamp"]

                if len(preview["messages"]) >= max_messages:
                    continue

                if obj.get("type") == "assistant":
                    msg = obj.get("message", {})
                    if not preview["model"] and msg.get("model"):
                        preview["model"] = msg["model"]
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if text:
                                    if len(text) > max_chars:
                                        text = text[:max_chars] + "..."
                                    preview["messages"].append(
                                        {
                                            "role": "Claude",
                                            "text": text,
                                        }
                                    )
                                    break
                elif obj.get("type") == "user":
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        text = content.strip()
                        if len(text) > max_chars:
                            text = text[:max_chars] + "..."
                        preview["messages"].append(
                            {
                                "role": "Human",
                                "text": text,
                            }
                        )
                    elif isinstance(content, list):
                        # Skip tool_result blocks
                        if any(b.get("type") == "tool_result" for b in content):
                            continue
                        for block in content:
                            if block.get("type") == "text":
                                text = block.get("text", "").strip()
                                if text:
                                    if len(text) > max_chars:
                                        text = text[:max_chars] + "..."
                                    preview["messages"].append(
                                        {
                                            "role": "Human",
                                            "text": text,
                                        }
                                    )
                                    break
    except OSError as exc:
        _debug("read preview failed", exc)
    return preview


def resolve_session(arg):
    """Resolve a session argument to a full .jsonl file path.

    Accepts:
        - A direct file path (absolute or relative)
        - A session UUID (searches all projects)
    """
    # Direct path
    p = Path(arg)
    if p.exists() and p.suffix == ".jsonl":
        return str(p.resolve())

    if not CLAUDE_DIR.exists():
        print("Error: ~/.claude/projects not found", file=sys.stderr)
        sys.exit(1)

    # UUID lookup
    for project_dir in CLAUDE_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        candidate = project_dir / f"{arg}.jsonl"
        if candidate.exists():
            return str(candidate)

        index_path = project_dir / "sessions-index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
                for entry in index.get("entries", []):
                    if entry.get("sessionId") != arg:
                        continue
                    full_path = entry.get("fullPath", "")
                    if not full_path:
                        continue
                    full_candidate = Path(full_path)
                    if not full_candidate.is_absolute():
                        full_candidate = project_dir / full_candidate
                    if full_candidate.exists():
                        return str(full_candidate.resolve())
            except (json.JSONDecodeError, OSError) as exc:
                _debug("resolve session index failed", exc)

    print(f"Error: Could not find session '{arg}'", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def parse_jsonl(path):
    """Read and parse all lines from a JSONL file."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return lines


def extract_metadata(lines):
    """Extract session metadata from parsed JSONL lines."""
    meta = {
        "session_id": "",
        "project": "",
        "model": "",
        "date": "",
        "git_branch": "",
        "cwd": "",
    }
    for obj in lines:
        if not meta["session_id"] and obj.get("sessionId"):
            meta["session_id"] = obj["sessionId"]
        if not meta["git_branch"] and obj.get("gitBranch"):
            meta["git_branch"] = obj["gitBranch"]
        if not meta["cwd"] and obj.get("cwd"):
            meta["cwd"] = obj["cwd"]
        if not meta["date"] and obj.get("timestamp"):
            meta["date"] = obj["timestamp"]
        if obj.get("type") == "assistant":
            msg = obj.get("message", {})
            if not meta["model"] and msg.get("model"):
                meta["model"] = msg["model"]

    return meta


# ---------------------------------------------------------------------------
# Conversation building
# ---------------------------------------------------------------------------


def build_conversation(lines):
    """Build a clean conversation from JSONL lines.

    Returns list of message dicts:
        {role, timestamp, blocks: [{type, text, ...}]}
    """
    # Pass 1: collect assistant messages, merging by message id
    assistant_msgs = {}  # msg_id -> {obj with merged content}
    tool_map = {}  # tool_use_id -> tool_name

    relevant = [
        obj
        for obj in lines
        if obj.get("type") in ("user", "assistant") and not obj.get("isSidechain")
    ]

    # Pass 1: merge assistant messages by id and build tool map
    for obj in relevant:
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message", {})
        msg_id = msg.get("id", "")
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        if msg_id not in assistant_msgs:
            assistant_msgs[msg_id] = {
                "blocks": [],
                "seen_types": set(),
                "timestamp": obj.get("timestamp", ""),
                "stop_reason": msg.get("stop_reason"),
            }

        entry = assistant_msgs[msg_id]
        if msg.get("stop_reason"):
            entry["stop_reason"] = msg["stop_reason"]
        entry["timestamp"] = obj.get("timestamp", "") or entry["timestamp"]

        for block in content:
            btype = block.get("type", "")
            # Deduplicate: text and thinking blocks may repeat across streamed chunks
            if btype == "text":
                text = block.get("text", "")
                if text.strip():
                    # Replace any existing text block (later one is more complete)
                    entry["blocks"] = [
                        b for b in entry["blocks"] if b.get("type") != "text"
                    ]
                    entry["blocks"].append({"type": "text", "text": text})
            elif btype == "thinking":
                thinking = block.get("thinking", "")
                if thinking.strip():
                    entry["blocks"] = [
                        b for b in entry["blocks"] if b.get("type") != "thinking"
                    ]
                    entry["blocks"].insert(0, {"type": "thinking", "text": thinking})
            elif btype == "tool_use":
                tool_id = block.get("id", "")
                tool_name = block.get("name", "unknown")
                tool_map[tool_id] = tool_name
                # Only add if not already present (by tool id)
                existing_ids = {
                    b.get("tool_id")
                    for b in entry["blocks"]
                    if b.get("type") == "tool_use"
                }
                if tool_id not in existing_ids:
                    entry["blocks"].append(
                        {
                            "type": "tool_use",
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "input": block.get("input", {}),
                        }
                    )

    # Pass 2: build ordered conversation
    conversation = []
    seen_assistant_ids = set()

    for obj in relevant:
        ts = obj.get("timestamp", "")

        if obj.get("type") == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, str):
                if content.strip():
                    conversation.append(
                        {
                            "role": "user",
                            "timestamp": ts,
                            "blocks": [{"type": "text", "text": content}],
                        }
                    )
            elif isinstance(content, list):
                # Check if this is a tool_result response or a user prompt
                has_tool_result = any(b.get("type") == "tool_result" for b in content)
                if has_tool_result:
                    # Attach tool results to the previous assistant message
                    tool_results = []
                    for block in content:
                        if block.get("type") == "tool_result":
                            tool_results.append(_normalize_tool_result(block, tool_map))
                    if (
                        tool_results
                        and conversation
                        and conversation[-1]["role"] == "assistant"
                    ):
                        conversation[-1]["blocks"].extend(tool_results)
                    elif tool_results:
                        # No preceding assistant msg; add standalone
                        conversation.append(
                            {
                                "role": "tool",
                                "timestamp": ts,
                                "blocks": tool_results,
                            }
                        )
                else:
                    # User prompt with text blocks
                    texts = []
                    for block in content:
                        if block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                texts.append(t)
                    if texts:
                        conversation.append(
                            {
                                "role": "user",
                                "timestamp": ts,
                                "blocks": [
                                    {"type": "text", "text": "\n\n".join(texts)}
                                ],
                            }
                        )

        elif obj.get("type") == "assistant":
            msg = obj.get("message", {})
            msg_id = msg.get("id", "")
            if msg_id in seen_assistant_ids:
                continue
            seen_assistant_ids.add(msg_id)
            if msg_id in assistant_msgs:
                entry = assistant_msgs[msg_id]
                if entry["blocks"]:
                    conversation.append(
                        {
                            "role": "assistant",
                            "timestamp": entry["timestamp"],
                            "blocks": entry["blocks"],
                        }
                    )

    return conversation


def _normalize_tool_result(block, tool_map):
    """Normalize a tool_result block into a clean dict."""
    tool_use_id = block.get("tool_use_id", "")
    tool_name = tool_map.get(tool_use_id, "unknown")
    is_error = block.get("is_error", False)
    content = block.get("content", "")

    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "\n".join(parts)
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    if len(text) > TRUNCATE_LIMIT:
        text = text[:TRUNCATE_LIMIT] + f"\n\n... [truncated, {len(text)} chars total]"

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "is_error": is_error,
        "text": text,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(messages, metadata):
    """Produce a complete standalone HTML string."""
    # Escape </script> in JSON payload
    json_data = json.dumps(
        {"messages": messages, "metadata": metadata}, ensure_ascii=False
    )
    json_data = json_data.replace("</", "<\\/")

    date_str = metadata.get("date", "")
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_display = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            date_display = date_str
    else:
        date_display = "Unknown"

    title = f"Claude Code Session — {date_display}"

    return HTML_TEMPLATE.replace("{{JSON_DATA}}", json_data).replace("{{TITLE}}", title)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}}</title>
<script src="https://cdn.tailwindcss.com/4"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
    --user-accent: #D4613E;
    --user-bg: #FBF4F1;
    --user-label: #B8462A;
    --assistant-accent: #3D405B;
    --assistant-bg: #FFFFFF;
    --thinking-accent: #8B7EC8;
    --thinking-bg: #F6F4FB;
    --thinking-border: #D4CEE8;
    --tool-accent: #3A7CA5;
    --tool-bg: #F0F5FA;
    --tool-border: #C2D6E8;
    --result-accent: #5A8A65;
    --result-bg: #F1F7F2;
    --result-border: #C0D9C5;
    --error-accent: #B5454A;
    --error-bg: #FBF1F1;
    --error-border: #E0BFBF;
    --page-bg: #F7F6F3;
    --divider: #E8E6E1;
    --text-primary: #2D2D2D;
    --text-secondary: #6B6966;
    --text-tertiary: #9C9891;
}

* { box-sizing: border-box; }

body {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    background: var(--page-bg);
    color: var(--text-primary);
    margin: 0;
    -webkit-font-smoothing: antialiased;
}

/* ── Prose (markdown content) ── */
.prose { font-family: 'IBM Plex Serif', Georgia, serif; line-height: 1.75; font-size: 0.938rem; }
.prose p { margin: 0.6em 0; }
.prose p:first-child { margin-top: 0; }
.prose p:last-child { margin-bottom: 0; }
.prose ul, .prose ol { margin: 0.5em 0; padding-left: 1.5em; }
.prose ul { list-style-type: disc; }
.prose ol { list-style-type: decimal; }
.prose li { margin: 0.2em 0; }
.prose li > p { margin: 0.2em 0; }
.prose h1, .prose h2, .prose h3, .prose h4 {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    font-weight: 600; margin: 1.2em 0 0.4em; line-height: 1.3;
}
.prose h1 { font-size: 1.4em; }
.prose h2 { font-size: 1.2em; }
.prose h3 { font-size: 1.05em; }
.prose pre {
    margin: 0.75em 0; border-radius: 6px; overflow-x: auto;
    background: #282c34; padding: 1em; border: 1px solid #1a1e24;
}
.prose pre > code {
    background: none; padding: 0; color: #abb2bf;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85em;
}
.prose code { font-family: 'IBM Plex Mono', monospace; font-size: 0.88em; }
.prose :not(pre) > code {
    background: rgba(0,0,0,0.06); padding: 0.15em 0.4em; border-radius: 3px;
}
.prose blockquote {
    border-left: 3px solid var(--divider); padding-left: 1em; margin: 0.75em 0;
    color: var(--text-secondary); font-style: italic;
}
.prose table { border-collapse: collapse; margin: 0.75em 0; width: 100%; font-family: 'IBM Plex Sans', sans-serif; font-size: 0.9em; }
.prose th, .prose td { border: 1px solid var(--divider); padding: 0.5em 0.75em; text-align: left; }
.prose th { background: #f5f4f1; font-weight: 600; }
.prose a { color: #3A7CA5; text-decoration: underline; text-underline-offset: 2px; }
.prose hr { border: none; border-top: 1px solid var(--divider); margin: 1.2em 0; }
.prose img { max-width: 100%; border-radius: 6px; }

/* ── Thinking prose ── */
.thinking-prose {
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-style: italic; font-size: 0.875rem; line-height: 1.65;
    color: #6B6280;
}
.thinking-prose code { font-style: normal; }
.thinking-prose pre > code { font-style: normal; }

/* ── Scrollable tool content ── */
.tool-scroll { max-height: 350px; overflow-y: auto; }
.tool-scroll::-webkit-scrollbar { width: 6px; }
.tool-scroll::-webkit-scrollbar-track { background: transparent; }
.tool-scroll::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
.tool-scroll::-webkit-scrollbar-thumb:hover { background: #aaa; }

/* ── Labels ── */
.role-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6875rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.block-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6875rem; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase;
}

/* ── Tool formatting ── */
.tool-input {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; line-height: 1.5;
    white-space: pre-wrap; word-break: break-word;
}
.tool-output {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; line-height: 1.5;
    white-space: pre-wrap; word-break: break-word;
    color: var(--text-primary);
}

/* ── Session header ── */
.session-header {
    background: #2D2D2D; color: #F0EFEC;
    border-bottom: 3px solid var(--user-accent);
    padding: 1.5rem max(1.5rem, env(safe-area-inset-left));
}
.session-header .meta-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.625rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8A8884;
}
.session-header .meta-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8125rem; color: #E0DFDB;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Collapsible blocks ── */
details.collapsible summary { cursor: pointer; user-select: none; list-style: none; }
details.collapsible summary::-webkit-details-marker { display: none; }
details.collapsible summary .chevron {
    display: inline-block; transition: transform 0.15s ease;
    font-size: 0.6rem; margin-right: 0.25rem;
}
details.collapsible[open] summary .chevron { transform: rotate(90deg); }

/* ── Divider ── */
.msg-divider { border: none; border-top: 1px solid var(--divider); margin: 0; }

/* ── Code highlight overrides ── */
.prose pre .hljs { background: transparent; }
</style>
</head>
<body>

<script type="application/json" id="conversation-data">{{JSON_DATA}}</script>
<div id="app"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    var raw = document.getElementById('conversation-data').textContent;
    var data = JSON.parse(raw);
    var messages = data.messages;
    var metadata = data.metadata;

    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try { return hljs.highlight(code, {language: lang}).value; } catch(e) {}
            }
            try { return hljs.highlightAuto(code).value; } catch(e) {}
            return code;
        },
        breaks: false,
        gfm: true,
    });

    var app = document.getElementById('app');

    /* ── Header ── */
    var header = document.createElement('div');
    header.className = 'session-header';
    var headerInner = document.createElement('div');
    headerInner.className = 'max-w-[52rem] mx-auto px-8 py-6';

    var title = document.createElement('div');
    title.style.cssText = 'font-family:"IBM Plex Sans",sans-serif;font-size:1.125rem;font-weight:600;margin-bottom:1rem;';
    title.textContent = 'Claude Code Session';
    headerInner.appendChild(title);

    var metaGrid = document.createElement('div');
    metaGrid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:0.75rem 2rem;';
    var metaFields = [];
    if (metadata.date) metaFields.push(['Date', formatDate(metadata.date)]);
    if (metadata.model) metaFields.push(['Model', metadata.model]);
    if (metadata.cwd) metaFields.push(['Directory', metadata.cwd]);
    if (metadata.git_branch) metaFields.push(['Branch', metadata.git_branch]);
    if (metadata.session_id) metaFields.push(['Session', metadata.session_id.substring(0, 12) + '\u2026']);
    metaFields.forEach(function(pair) {
        var item = document.createElement('div');
        item.style.minWidth = '0';
        var lbl = document.createElement('div'); lbl.className = 'meta-label'; lbl.textContent = pair[0];
        var val = document.createElement('div'); val.className = 'meta-value'; val.textContent = pair[1];
        val.title = pair[1];
        item.appendChild(lbl); item.appendChild(val); metaGrid.appendChild(item);
    });
    headerInner.appendChild(metaGrid);
    header.appendChild(headerInner);
    app.appendChild(header);

    /* ── Conversation body ── */
    var body = document.createElement('div');
    body.className = 'max-w-[52rem] mx-auto px-6 py-4';
    app.appendChild(body);

    messages.forEach(function(msg, idx) {
        if (idx > 0) { var hr = document.createElement('hr'); hr.className = 'msg-divider'; body.appendChild(hr); }
        if (msg.role === 'user') renderUserMessage(body, msg);
        else if (msg.role === 'assistant') renderAssistantMessage(body, msg);
        else if (msg.role === 'tool') renderToolMessage(body, msg);
    });

    var spacer = document.createElement('div');
    spacer.style.height = '4rem';
    body.appendChild(spacer);
});

/* ── User Message ── */
function renderUserMessage(container, msg) {
    var section = el('div', '', 'padding:1.5rem 0;border-left:4px solid var(--user-accent);padding-left:1.25rem;');
    section.appendChild(labelRow('Human', 'var(--user-label)', msg.timestamp));
    msg.blocks.forEach(function(b) {
        if (b.type === 'text') { var d = el('div'); d.className = 'prose'; d.innerHTML = renderMarkdown(b.text); section.appendChild(d); }
    });
    container.appendChild(section);
}

/* ── Assistant Message ── */
function renderAssistantMessage(container, msg) {
    var section = el('div', '', 'padding:1.5rem 0;padding-left:1.25rem;');
    section.appendChild(labelRow('Claude', 'var(--assistant-accent)', msg.timestamp));
    msg.blocks.forEach(function(b) {
        if (b.type === 'text') {
            var d = el('div'); d.className = 'prose'; d.style.marginBottom = '0.75rem';
            d.innerHTML = renderMarkdown(b.text); section.appendChild(d);
        } else if (b.type === 'thinking') { section.appendChild(renderThinking(b));
        } else if (b.type === 'tool_use') { section.appendChild(renderToolUse(b));
        } else if (b.type === 'tool_result') { section.appendChild(renderToolResult(b));
        }
    });
    container.appendChild(section);
}

/* ── Tool Message ── */
function renderToolMessage(container, msg) {
    var section = el('div', '', 'padding:1.5rem 0;padding-left:1.25rem;');
    section.appendChild(labelRow('Tool', 'var(--tool-accent)', msg.timestamp));
    msg.blocks.forEach(function(b) {
        if (b.type === 'tool_use') { section.appendChild(renderToolUse(b));
        } else if (b.type === 'tool_result') { section.appendChild(renderToolResult(b));
        }
    });
    container.appendChild(section);
}

/* ── Thinking Block ── */
function renderThinking(block) {
    var details = document.createElement('details');
    details.className = 'collapsible';
    details.style.cssText = 'margin:0.75rem 0;background:var(--thinking-bg);border-left:3px dashed var(--thinking-accent);border-radius:0 6px 6px 0;overflow:hidden;';

    var summary = document.createElement('summary');
    summary.style.cssText = 'padding:0.625rem 1rem;display:flex;align-items:center;gap:0.375rem;';
    var chevron = el('span', 'chevron'); chevron.textContent = '\u25b6';
    summary.appendChild(chevron);
    var lbl = el('span'); lbl.className = 'block-label'; lbl.style.color = 'var(--thinking-accent)'; lbl.textContent = 'Thinking';
    summary.appendChild(lbl);
    details.appendChild(summary);

    var bd = el('div', 'thinking-prose tool-scroll', 'padding:0 1rem 0.875rem;');
    bd.innerHTML = renderMarkdown(block.text);
    details.appendChild(bd);
    return details;
}

/* ── Tool Use Block ── */
function renderToolUse(block) {
    var w = el('div', '', 'margin:0.75rem 0;background:var(--tool-bg);border:1px solid var(--tool-border);border-left:4px solid var(--tool-accent);border-radius:0 6px 6px 0;overflow:hidden;');

    var hdr = el('div', '', 'padding:0.5rem 0.875rem;display:flex;align-items:center;gap:0.5rem;border-bottom:1px solid var(--tool-border);');
    var icon = el('span', '', 'display:inline-flex;align-items:center;justify-content:center;width:1.25rem;height:1.25rem;background:var(--tool-accent);color:white;border-radius:3px;font-size:0.65rem;font-weight:700;font-family:"IBM Plex Mono",monospace;');
    icon.textContent = toolIcon(block.tool_name);
    hdr.appendChild(icon);
    var tl = el('span'); tl.className = 'block-label'; tl.style.color = 'var(--tool-accent)'; tl.textContent = block.tool_name || 'Tool';
    hdr.appendChild(tl);
    w.appendChild(hdr);

    var bd = el('div', 'tool-scroll', 'padding:0.625rem 0.875rem;');
    var pre = el('pre', 'tool-input', 'margin:0;');
    var fmt = fmtInput(block.tool_name, block.input);
    pre.textContent = fmt;
    bd.appendChild(pre); w.appendChild(bd);
    return w;
}

/* ── Tool Result Block ── */
function renderToolResult(block) {
    var err = block.is_error;
    var ac = err ? 'var(--error-accent)' : 'var(--result-accent)';
    var bg = err ? 'var(--error-bg)' : 'var(--result-bg)';
    var br = err ? 'var(--error-border)' : 'var(--result-border)';

    var details = document.createElement('details');
    details.className = 'collapsible';
    details.style.cssText = 'margin:0.25rem 0 0.75rem;background:'+bg+';border:1px solid '+br+';border-left:4px solid '+ac+';border-radius:0 6px 6px 0;overflow:hidden;';

    var summary = document.createElement('summary');
    summary.style.cssText = 'padding:0.375rem 0.875rem;display:flex;align-items:center;gap:0.5rem;';
    var chevron = el('span', 'chevron'); chevron.textContent = '\u25b6';
    summary.appendChild(chevron);
    var dot = el('span', '', 'width:0.4rem;height:0.4rem;border-radius:50%;background:'+ac+';flex-shrink:0;');
    summary.appendChild(dot);
    var rl = el('span'); rl.className = 'block-label'; rl.style.color = ac;
    rl.textContent = (err ? 'Error' : 'Output') + (block.tool_name ? ' \u2014 ' + block.tool_name : '');
    summary.appendChild(rl);
    details.appendChild(summary);

    var txt = block.text || '(empty)';
    if (txt && txt !== '(empty)') {
        var bd = el('div', 'tool-scroll', 'padding:0 0.875rem 0.5rem;border-top:1px solid '+br+';');
        var pre = el('pre', 'tool-output', 'margin:0;padding-top:0.5rem;');
        pre.textContent = txt; bd.appendChild(pre); details.appendChild(bd);
    }
    return details;
}

/* ── Helpers ── */
function el(tag, cls, style) {
    var e = document.createElement(tag || 'div');
    if (cls) e.className = cls;
    if (style) e.style.cssText = style;
    return e;
}

function labelRow(name, color, timestamp) {
    var row = el('div', '', 'display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;');
    var lbl = el('span'); lbl.className = 'role-label'; lbl.style.color = color; lbl.textContent = name;
    row.appendChild(lbl);
    if (timestamp) {
        var ts = el('span', '', 'font-family:"IBM Plex Mono",monospace;font-size:0.6875rem;color:var(--text-tertiary);');
        ts.textContent = formatTime(timestamp); row.appendChild(ts);
    }
    return row;
}

function fmtInput(tool, input) {
    if (!input) return '';
    var p;
    switch (tool) {
        case 'Bash':
            p = [];
            if (input.description) p.push('\u25b8 ' + input.description);
            if (input.command) p.push('$ ' + input.command);
            if (input.timeout) p.push('timeout: ' + input.timeout + 'ms');
            return p.length ? p.join('\n') : stringify(input);
        case 'Read':
            p = [];
            if (input.file_path) p.push('\u25b8 ' + input.file_path);
            if (input.offset) p.push('offset: ' + input.offset);
            if (input.limit) p.push('limit: ' + input.limit);
            return p.join('\n');
        case 'Write':
            return input.file_path ? '\u25b8 ' + input.file_path : stringify(input);
        case 'Edit':
            return input.file_path ? '\u25b8 ' + input.file_path : stringify(input);
        case 'Glob':
            p = [];
            if (input.pattern) p.push('pattern: ' + input.pattern);
            if (input.path) p.push('path: ' + input.path);
            return p.join('\n');
        case 'Grep':
            p = [];
            if (input.pattern) p.push('/' + input.pattern + '/');
            if (input.path) p.push('in: ' + input.path);
            if (input.glob) p.push('glob: ' + input.glob);
            return p.join('\n');
        case 'Task':
            p = [];
            if (input.description) p.push('\u25b8 ' + input.description);
            if (input.subagent_type) p.push('agent: ' + input.subagent_type);
            return p.length ? p.join('\n') : stringify(input);
        case 'WebFetch':
            p = [];
            if (input.url) p.push('\u25b8 ' + input.url);
            if (input.prompt) p.push('prompt: ' + input.prompt);
            return p.join('\n');
        case 'WebSearch':
            return input.query ? '\u25b8 ' + input.query : stringify(input);
        default:
            return stringify(input);
    }
}

function toolIcon(name) {
    var m = {'Bash':'$','Read':'R','Write':'W','Edit':'E','Glob':'G','Grep':'/','Task':'T','WebFetch':'W','WebSearch':'S','Skill':'SK'};
    return m[name] || (name ? name.charAt(0) : '?');
}

function stringify(o) { try { return JSON.stringify(o, null, 2); } catch(e) { return String(o); } }

// Note: renderMarkdown processes user-owned local data, not untrusted web content.
function renderMarkdown(t) { return marked.parse(t); }

function formatDate(s) {
    try { var d = new Date(s); return d.toLocaleDateString('en-US', {weekday:'short',year:'numeric',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }
    catch(e) { return s; }
}

function formatTime(s) {
    try { var d = new Date(s); return d.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'}); }
    catch(e) { return ''; }
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------

ListItem = namedtuple("ListItem", ["kind", "data", "project"])


def _display_project_name(session):
    """Format project name from session path for display."""
    pp = session.get("project_path", "") or session.get("project", "")
    if not pp:
        return session.get("project", "unknown")
    # Convert /Users/foo/Projects/bar to bar
    parts = pp.rstrip("/").split("/")
    # Take last 2 meaningful segments
    meaningful = [p for p in parts if p and p not in ("Users",)]
    if len(meaningful) >= 2:
        return "/".join(meaningful[-2:])
    return meaningful[-1] if meaningful else pp


def _truncate(text, width):
    """Truncate text with ellipsis if longer than width."""
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..." if width > 3 else text[:width]


# ---------------------------------------------------------------------------
# TUI: Session Browser
# ---------------------------------------------------------------------------


class SessionBrowser:
    """Interactive curses-based session browser."""

    MIN_WIDTH = 80
    MIN_HEIGHT = 20

    def __init__(self, stdscr, project_filter=None):
        self.stdscr = stdscr
        self.project_filter = project_filter
        self.cursor = 0
        self.scroll_offset = 0
        self.preview_scroll = 0
        self.preview_focus = False
        self.filter_mode = False
        self.filter_text = ""
        self.show_help = False
        self.status_message = ""
        self.status_timeout = 0
        self.collapsed = set()  # collapsed project names
        self.sessions = []
        self.items = []  # flat list of ListItem
        self._preview_cache = {}
        self._preview_cache_order = []
        self._cache_max = 20

    def run(self):
        """Main entry point — called inside curses.wrapper."""
        curses.curs_set(0)
        self.stdscr.timeout(100)  # 100ms for responsive resize
        self._setup_colors()
        self._load_sessions()
        self._build_items()

        while True:
            h, w = self.stdscr.getmaxyx()
            if h < self.MIN_HEIGHT or w < self.MIN_WIDTH:
                self._check_terminal_size(h, w)
            else:
                self._draw(h, w)

            key = self.stdscr.getch()
            if key == -1:
                # Tick status timeout
                if self.status_timeout > 0:
                    self.status_timeout -= 1
                    if self.status_timeout == 0:
                        self.status_message = ""
                continue

            if self.show_help:
                self.show_help = False
                continue

            if self.filter_mode:
                if self._handle_filter_key(key):
                    continue
            else:
                result = self._handle_key(key)
                if result == "quit":
                    return

    def _setup_colors(self):
        """Initialize color pairs with graceful fallback."""
        try:
            curses.start_color()
            curses.use_default_colors()
            # 1: title bar
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
            # 2: status bar
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
            # 3: selected item
            curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)
            # 4: project header
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
            # 5: session id
            curses.init_pair(5, curses.COLOR_CYAN, -1)
            # 6: date
            curses.init_pair(6, curses.COLOR_GREEN, -1)
            # 7: preview label
            curses.init_pair(7, curses.COLOR_YELLOW, -1)
            # 8: Human role
            curses.init_pair(8, curses.COLOR_RED, -1)
            # 9: Claude role
            curses.init_pair(9, curses.COLOR_BLUE, -1)
            # 10: status message
            curses.init_pair(10, curses.COLOR_GREEN, -1)
            self.has_colors = True
        except curses.error:
            self.has_colors = False

    def _color(self, pair_num, extra=0):
        """Get color attribute, falling back to bold/reverse."""
        if self.has_colors:
            return curses.color_pair(pair_num) | extra
        if pair_num in (1, 2, 3):
            return curses.A_REVERSE | extra
        return curses.A_BOLD | extra

    def _check_terminal_size(self, h, w):
        """Show friendly message if terminal is too small."""
        self.stdscr.erase()
        msg = f"Terminal too small ({w}x{h}). Need {self.MIN_WIDTH}x{self.MIN_HEIGHT}."
        try:
            y = h // 2
            x = max(0, (w - len(msg)) // 2)
            self._safe_addnstr(y, x, msg, w, curses.A_BOLD)
        except curses.error:
            pass
        self.stdscr.refresh()

    def _load_sessions(self):
        """Load sessions using existing find_sessions."""
        self.sessions = find_sessions(project_filter=self.project_filter)
        # Sort by created date descending within each project
        self.sessions.sort(key=lambda s: s.get("created", ""), reverse=True)

    def _build_items(self):
        """Build flat list of ListItem from sessions, applying filter and collapse."""
        # Group by display project name
        by_project = {}
        order = []
        for s in self.sessions:
            pname = _display_project_name(s)
            # Apply filter
            if self.filter_text:
                ft = self.filter_text.lower()
                searchable = " ".join(
                    [
                        pname,
                        s.get("session_id", ""),
                        s.get("first_prompt", ""),
                        s.get("git_branch", ""),
                    ]
                ).lower()
                if ft not in searchable:
                    continue
            if pname not in by_project:
                by_project[pname] = []
                order.append(pname)
            by_project[pname].append(s)

        self.items = []
        for pname in order:
            sess_list = by_project[pname]
            self.items.append(ListItem(kind="header", data=pname, project=pname))
            if pname not in self.collapsed:
                for s in sess_list:
                    self.items.append(ListItem(kind="session", data=s, project=pname))

        # Clamp cursor
        if self.items:
            self.cursor = min(self.cursor, len(self.items) - 1)
            self.cursor = max(0, self.cursor)
        else:
            self.cursor = 0

    def _draw(self, h, w):
        """Main draw orchestrator."""
        self.stdscr.erase()

        left_w = max(20, int(w * 0.4))
        right_w = w - left_w

        self._draw_title_bar(w)
        self._draw_status_bar(h, w)

        content_h = h - 2  # minus title and status bars
        self._draw_left_pane(1, 0, content_h, left_w)
        self._draw_right_pane(1, left_w, content_h, right_w)

        # Draw vertical separator
        for y in range(1, h - 1):
            self._safe_addnstr(y, left_w - 1, "|", 1, curses.A_DIM)

        if self.show_help:
            self._draw_help_overlay(h, w)

        self.stdscr.refresh()

    def _draw_title_bar(self, w):
        """Draw title bar at top."""
        count = sum(1 for item in self.items if item.kind == "session")
        title = " Session Browser"
        right = f"({count} sessions)  ? for help "
        padding = w - len(title) - len(right)
        if padding < 0:
            padding = 0
        bar = title + " " * padding + right
        self._safe_addnstr(0, 0, bar.ljust(w), w, self._color(1, curses.A_BOLD))

    def _draw_status_bar(self, h, w):
        """Draw status bar at bottom."""
        if self.filter_mode:
            bar = f" /:{self.filter_text}_"
            bar = bar.ljust(w)
        elif self.status_message:
            bar = f" {self.status_message}".ljust(w)
        else:
            if self.preview_focus:
                bar = " Tab:List  j/k:Scroll preview  q:Quit"
            else:
                bar = " j/k:Navigate  /:Filter  Enter:Export  Tab:Preview  q:Quit"
            bar = bar.ljust(w)
        attr = self._color(2)
        if self.status_message and not self.filter_mode:
            attr = self._color(10, curses.A_BOLD)
        self._safe_addnstr(h - 1, 0, bar, w, attr)

    def _draw_left_pane(self, top, left, height, width):
        """Render the session list in the left pane."""
        usable_w = width - 2  # 1 for left margin, 1 for separator

        self._ensure_cursor_visible(height)

        for i in range(height):
            idx = self.scroll_offset + i
            y = top + i
            if idx >= len(self.items):
                break

            item = self.items[idx]
            is_selected = (idx == self.cursor) and not self.preview_focus

            if item.kind == "header":
                collapsed = item.project in self.collapsed
                marker = "[+]" if collapsed else "[-]"
                text = f" {marker} {item.data}"
                text = _truncate(text, usable_w)
                attr = self._color(4, curses.A_BOLD)
                if is_selected:
                    attr = self._color(3, curses.A_BOLD)
                self._safe_addnstr(
                    y, left, text.ljust(usable_w + 1), usable_w + 1, attr
                )
            else:
                s = item.data
                sid = s["session_id"][:8]
                date = s.get("created", "")[:10] or "???"
                # Format date shorter: Feb 05
                try:
                    dt = datetime.fromisoformat(date)
                    date = dt.strftime("%b %d")
                except Exception:
                    date = date[:6]
                prompt = s.get("first_prompt", "")
                # Calculate space for prompt
                prefix = f"   {sid}  {date}  "
                prompt_w = max(0, usable_w - len(prefix))
                prompt = _truncate(prompt.replace("\n", " "), prompt_w)
                text = prefix + prompt

                if is_selected:
                    attr = self._color(3)
                    self._safe_addnstr(
                        y, left, text.ljust(usable_w + 1), usable_w + 1, attr
                    )
                else:
                    # Color parts differently
                    self._safe_addnstr(y, left, " " * (usable_w + 1), usable_w + 1, 0)
                    self._safe_addnstr(y, left, "   ", 3, 0)
                    self._safe_addnstr(y, left + 3, sid, len(sid), self._color(5))
                    self._safe_addnstr(y, left + 3 + len(sid), "  ", 2, 0)
                    self._safe_addnstr(
                        y, left + 5 + len(sid), date, len(date), self._color(6)
                    )
                    self._safe_addnstr(y, left + 5 + len(sid) + len(date), "  ", 2, 0)
                    p_start = left + 7 + len(sid) + len(date)
                    remaining = usable_w + 1 - (p_start - left)
                    if remaining > 0:
                        self._safe_addnstr(y, p_start, prompt, remaining, 0)

    def _draw_right_pane(self, top, left, height, width):
        """Render metadata + message preview in the right pane."""
        usable_w = width - 2  # margins

        # Find currently selected session
        session = None
        if 0 <= self.cursor < len(self.items):
            item = self.items[self.cursor]
            if item.kind == "session":
                session = item.data
            elif item.kind == "header":
                # Find first session under this header
                for j in range(self.cursor + 1, len(self.items)):
                    if (
                        self.items[j].kind == "session"
                        and self.items[j].project == item.project
                    ):
                        session = self.items[j].data
                        break

        if not session:
            msg = "No session selected"
            self._safe_addnstr(
                top + height // 2,
                left + (width - len(msg)) // 2,
                msg,
                len(msg),
                curses.A_DIM,
            )
            return

        preview = self._get_preview(session)
        lines = []

        # Metadata block
        sid = session.get("session_id", "")
        lines.append(("label", f"  Session:  {sid[:16]}"))
        if preview.get("date"):
            try:
                dt_str = preview["date"]
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                date_display = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_display = preview["date"][:16]
            lines.append(("label", f"  Date:     {date_display}"))
        if preview.get("git_branch"):
            lines.append(("label", f"  Branch:   {preview['git_branch']}"))
        if preview.get("model"):
            lines.append(("label", f"  Model:    {preview['model']}"))
        if preview.get("cwd"):
            cwd = preview["cwd"]
            max_cwd = usable_w - 12
            if len(cwd) > max_cwd:
                cwd = "..." + cwd[-(max_cwd - 3) :]
            lines.append(("label", f"  CWD:      {cwd}"))

        lines.append(("text", ""))
        lines.append(("divider", "  --- First Messages ---"))
        lines.append(("text", ""))

        # Message preview
        for msg in preview.get("messages", []):
            role = msg["role"]
            lines.append(("role", f"  [{role}]"))
            # Word wrap message text
            wrapped = self._word_wrap(msg["text"], usable_w - 4)
            for wl in wrapped:
                lines.append(("text", f"  {wl}"))
            lines.append(("text", ""))

        # Apply preview scroll
        max_scroll = max(0, len(lines) - height)
        self.preview_scroll = min(self.preview_scroll, max_scroll)
        self.preview_scroll = max(0, self.preview_scroll)

        for i in range(height):
            li = self.preview_scroll + i
            y = top + i
            if li >= len(lines):
                break

            kind, text = lines[li]
            text = _truncate(text, usable_w)

            if kind == "label":
                self._safe_addnstr(y, left + 1, text, usable_w, self._color(7))
            elif kind == "divider":
                self._safe_addnstr(y, left + 1, text, usable_w, curses.A_DIM)
            elif kind == "role":
                role_color = self._color(8) if "Human" in text else self._color(9)
                self._safe_addnstr(
                    y, left + 1, text, usable_w, role_color | curses.A_BOLD
                )
            else:
                self._safe_addnstr(y, left + 1, text, usable_w, 0)

    def _draw_help_overlay(self, h, w):
        """Draw centered help box."""
        help_lines = [
            "Session Browser - Key Bindings",
            "",
            "  j/Down     Next session",
            "  k/Up       Previous session",
            "  PgDn       Page down",
            "  PgUp       Page up",
            "  g/Home     First session",
            "  G/End      Last session",
            "  Enter      Export session / toggle project",
            "  h/Left     Collapse project group",
            "  l/Right    Expand project group",
            "  Tab        Toggle preview pane focus",
            "  /          Filter sessions",
            "  ?          This help",
            "  q/Esc      Quit",
        ]
        box_w = max(len(l) for l in help_lines) + 4
        box_h = len(help_lines) + 2
        start_y = max(0, (h - box_h) // 2)
        start_x = max(0, (w - box_w) // 2)

        for i in range(box_h):
            y = start_y + i
            if y >= h:
                break
            if i == 0 or i == box_h - 1:
                line = "+" + "-" * (box_w - 2) + "+"
            else:
                content = help_lines[i - 1] if i - 1 < len(help_lines) else ""
                line = "| " + content.ljust(box_w - 4) + " |"
            self._safe_addnstr(
                y,
                start_x,
                line,
                min(len(line), w - start_x),
                self._color(1, curses.A_BOLD),
            )

    def _handle_key(self, key):
        """Handle key press in normal mode. Returns 'quit' to exit."""
        if key in (ord("q"), 27):  # q or Esc
            if self.preview_focus:
                self.preview_focus = False
                return None
            return "quit"
        elif key == ord("?"):
            self.show_help = True
        elif self.preview_focus:
            # In preview focus mode, j/k scroll the preview pane
            if key in (ord("j"), curses.KEY_DOWN):
                self.preview_scroll += 1
            elif key in (ord("k"), curses.KEY_UP):
                self.preview_scroll = max(0, self.preview_scroll - 1)
            elif key == curses.KEY_NPAGE:
                self.preview_scroll += 10
            elif key == curses.KEY_PPAGE:
                self.preview_scroll = max(0, self.preview_scroll - 10)
            elif key == 9:  # Tab
                self.preview_focus = False
            return None
        elif key in (ord("j"), curses.KEY_DOWN):
            self._move_cursor(1)
        elif key in (ord("k"), curses.KEY_UP):
            self._move_cursor(-1)
        elif key == curses.KEY_NPAGE:  # PgDn
            h, _ = self.stdscr.getmaxyx()
            self._move_cursor(h - 4)
        elif key == curses.KEY_PPAGE:  # PgUp
            h, _ = self.stdscr.getmaxyx()
            self._move_cursor(-(h - 4))
        elif key in (ord("g"), curses.KEY_HOME):
            self.cursor = 0
            self.scroll_offset = 0
            self.preview_scroll = 0
        elif key in (ord("G"), curses.KEY_END):
            self.cursor = max(0, len(self.items) - 1)
            self.preview_scroll = 0
        elif key == 9:  # Tab
            self.preview_focus = not self.preview_focus
        elif key in (ord("h"), curses.KEY_LEFT):
            self._collapse_current()
        elif key in (ord("l"), curses.KEY_RIGHT):
            self._expand_current()
        elif key in (10, curses.KEY_ENTER):  # Enter
            self._action_enter()
        elif key == ord("/"):
            self.filter_mode = True
            self.filter_text = ""
        return None

    def _handle_filter_key(self, key):
        """Handle key press in filter mode. Returns True to stay in filter mode."""
        if key == 27:  # Esc: cancel filter
            self.filter_mode = False
            self.filter_text = ""
            self._build_items()
            return True
        elif key in (10, curses.KEY_ENTER):  # Enter: confirm filter
            self.filter_mode = False
            return True
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            self.filter_text = self.filter_text[:-1]
            self.cursor = 0
            self.scroll_offset = 0
            self._build_items()
            return True
        elif 32 <= key <= 126:  # printable ASCII
            self.filter_text += chr(key)
            self.cursor = 0
            self.scroll_offset = 0
            self._build_items()
            return True
        return True

    def _move_cursor(self, delta):
        """Move cursor by delta, clamping to valid range."""
        if not self.items:
            return
        self.cursor = max(0, min(len(self.items) - 1, self.cursor + delta))
        self.preview_scroll = 0

    def _collapse_current(self):
        """Collapse the project group of the current item."""
        if 0 <= self.cursor < len(self.items):
            project = self.items[self.cursor].project
            self.collapsed.add(project)
            self._build_items()

    def _expand_current(self):
        """Expand the project group of the current item."""
        if 0 <= self.cursor < len(self.items):
            project = self.items[self.cursor].project
            self.collapsed.discard(project)
            self._build_items()

    def _action_enter(self):
        """Handle Enter key: export session or toggle project header."""
        if not self.items or self.cursor >= len(self.items):
            return
        item = self.items[self.cursor]
        if item.kind == "header":
            # Toggle collapse
            if item.project in self.collapsed:
                self.collapsed.discard(item.project)
            else:
                self.collapsed.add(item.project)
            self._build_items()
        elif item.kind == "session":
            self._export_session(item.data)

    def _export_session(self, session):
        """Export selected session to HTML."""
        path = session.get("path", "")
        if not path or not os.path.exists(path):
            self.status_message = "Error: session file not found"
            self.status_timeout = 30
            return

        try:
            output_path, message_count, _ = export_session(path)
            self.status_message = f"Exported {message_count} messages to {output_path}"
            self.status_timeout = 50
        except Exception as e:
            self.status_message = f"Export error: {e}"
            self.status_timeout = 50
            _debug("export failed", e)

    def _get_preview(self, session):
        """Get preview data for a session, using LRU cache."""
        sid = session.get("session_id", "")
        if sid in self._preview_cache:
            return self._preview_cache[sid]

        path = session.get("path", "")
        if not path or not os.path.exists(path):
            return {
                "session_id": sid,
                "model": "",
                "date": "",
                "git_branch": "",
                "cwd": "",
                "messages": [],
            }

        preview = _read_preview(path)

        # LRU eviction
        if len(self._preview_cache_order) >= self._cache_max:
            oldest = self._preview_cache_order.pop(0)
            self._preview_cache.pop(oldest, None)
        self._preview_cache[sid] = preview
        self._preview_cache_order.append(sid)

        return preview

    def _safe_addnstr(self, y, x, text, max_len, attr=0):
        """Safely write text to screen, handling edge cases."""
        h, w = self.stdscr.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        # Encode to ASCII for safety (curses on macOS can crash on emoji/CJK)
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        available = w - x
        n = min(max_len, available)
        if n <= 0:
            return
        try:
            self.stdscr.addnstr(y, x, safe_text, n, attr)
        except curses.error:
            pass  # writing to bottom-right corner raises error

    def _ensure_cursor_visible(self, visible_height):
        """Adjust scroll_offset so cursor is visible."""
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + visible_height:
            self.scroll_offset = self.cursor - visible_height + 1
        self.scroll_offset = max(0, self.scroll_offset)

    def _word_wrap(self, text, width):
        """Simple word wrap for preview text."""
        if width <= 0:
            return [text]
        lines = []
        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            while len(paragraph) > width:
                # Find last space before width
                break_at = paragraph.rfind(" ", 0, width)
                if break_at <= 0:
                    break_at = width
                lines.append(paragraph[:break_at])
                paragraph = paragraph[break_at:].lstrip()
            lines.append(paragraph)
        return lines


def export_session(path, output_path=None):
    lines = parse_jsonl(path)
    metadata = extract_metadata(lines)
    messages = build_conversation(lines)

    if not messages:
        raise ValueError("No messages found in session")

    html = generate_html(messages, metadata)
    if not output_path:
        sid = metadata.get("session_id", "session")[:12]
        output_path = f"claude-session-{sid}.html"

    with open(output_path, "w") as f:
        f.write(html)

    return output_path, len(messages), metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_list(args):
    """List available sessions."""
    sessions = find_sessions(project_filter=args.project)
    if not sessions:
        print("No sessions found.")
        return

    # Group by project
    by_project = {}
    for s in sessions:
        by_project.setdefault(s["project"], []).append(s)

    for project, sess_list in sorted(by_project.items()):
        print(f"\n  {project}")
        print(f"  {'─' * 60}")
        for s in sess_list:
            date = s.get("created", "")[:10] or "???"
            prompt = s.get("first_prompt", "")[:80]
            sid = s["session_id"][:12]
            branch = s.get("git_branch", "")
            branch_str = f" [{branch}]" if branch else ""
            print(f"  {sid}  {date}{branch_str}  {prompt}")

    print()


def cmd_export(args):
    """Export a session to HTML."""
    path = resolve_session(args.session)
    try:
        output_path, message_count, _ = export_session(path, args.output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"Exported {message_count} messages to {output_path}")


def cmd_browse(args):
    """Launch interactive TUI session browser."""

    def _run(stdscr):
        browser = SessionBrowser(stdscr, project_filter=args.project)
        browser.run()

    try:
        curses.wrapper(_run)
    except KeyboardInterrupt:
        pass


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(
        description="Export Claude Code sessions to standalone HTML files."
    )

    parser.add_argument("--list", action="store_true", help="List available sessions")
    parser.add_argument(
        "--browse", action="store_true", help="Launch interactive TUI session browser"
    )
    parser.add_argument("-p", "--project", help="Filter projects by name substring")
    parser.add_argument(
        "session", nargs="?", help="Session UUID or path to .jsonl file"
    )
    parser.add_argument("-o", "--output", help="Output HTML file path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show debug/trace info"
    )

    args = parser.parse_args()
    VERBOSE = args.verbose

    if args.list:
        cmd_list(args)
    elif args.browse:
        cmd_browse(args)
    elif args.session:
        cmd_export(args)
    elif sys.stdout.isatty() and sys.stdin.isatty():
        # Default to TUI when running interactively with no args
        cmd_browse(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

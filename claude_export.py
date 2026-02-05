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
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


CLAUDE_DIR = Path.home() / ".claude" / "projects"
TRUNCATE_LIMIT = 50_000


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
        index_path = project_dir / "sessions-index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
                for entry in index.get("entries", []):
                    sessions.append({
                        "session_id": entry.get("sessionId", ""),
                        "project": project_name,
                        "project_path": entry.get("projectPath", ""),
                        "path": entry.get("fullPath", ""),
                        "first_prompt": entry.get("firstPrompt", ""),
                        "created": entry.get("created", ""),
                        "modified": entry.get("modified", ""),
                        "git_branch": entry.get("gitBranch", ""),
                    })
                continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: scan .jsonl files directly
        for jsonl_path in sorted(project_dir.glob("*.jsonl")):
            session_id = jsonl_path.stem
            info = _read_session_stub(jsonl_path)
            sessions.append({
                "session_id": session_id,
                "project": project_name,
                "project_path": "",
                "path": str(jsonl_path),
                "first_prompt": info.get("first_prompt", ""),
                "created": info.get("created", ""),
                "modified": "",
                "git_branch": info.get("git_branch", ""),
            })

    return sessions


def _read_session_stub(path):
    """Read first user line from a JSONL to extract basic info."""
    try:
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "user":
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    prompt = content if isinstance(content, str) else ""
                    return {
                        "first_prompt": prompt[:200],
                        "created": obj.get("timestamp", ""),
                        "git_branch": obj.get("gitBranch", ""),
                    }
    except Exception:
        pass
    return {}


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

    # UUID lookup
    for project_dir in CLAUDE_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        candidate = project_dir / f"{arg}.jsonl"
        if candidate.exists():
            return str(candidate)

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
        obj for obj in lines
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
                    b.get("tool_id") for b in entry["blocks"] if b.get("type") == "tool_use"
                }
                if tool_id not in existing_ids:
                    entry["blocks"].append({
                        "type": "tool_use",
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "input": block.get("input", {}),
                    })

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
                    conversation.append({
                        "role": "user",
                        "timestamp": ts,
                        "blocks": [{"type": "text", "text": content}],
                    })
            elif isinstance(content, list):
                # Check if this is a tool_result response or a user prompt
                has_tool_result = any(b.get("type") == "tool_result" for b in content)
                if has_tool_result:
                    # Attach tool results to the previous assistant message
                    tool_results = []
                    for block in content:
                        if block.get("type") == "tool_result":
                            tool_results.append(_normalize_tool_result(block, tool_map))
                    if tool_results and conversation and conversation[-1]["role"] == "assistant":
                        conversation[-1]["blocks"].extend(tool_results)
                    elif tool_results:
                        # No preceding assistant msg; add standalone
                        conversation.append({
                            "role": "tool",
                            "timestamp": ts,
                            "blocks": tool_results,
                        })
                else:
                    # User prompt with text blocks
                    texts = []
                    for block in content:
                        if block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                texts.append(t)
                    if texts:
                        conversation.append({
                            "role": "user",
                            "timestamp": ts,
                            "blocks": [{"type": "text", "text": "\n\n".join(texts)}],
                        })

        elif obj.get("type") == "assistant":
            msg = obj.get("message", {})
            msg_id = msg.get("id", "")
            if msg_id in seen_assistant_ids:
                continue
            seen_assistant_ids.add(msg_id)
            if msg_id in assistant_msgs:
                entry = assistant_msgs[msg_id]
                if entry["blocks"]:
                    conversation.append({
                        "role": "assistant",
                        "timestamp": entry["timestamp"],
                        "blocks": entry["blocks"],
                    })

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
    json_data = json.dumps({"messages": messages, "metadata": metadata}, ensure_ascii=False)
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
}
.session-header .meta-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.625rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8A8884;
}
.session-header .meta-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8125rem; color: #E0DFDB;
}

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
    headerInner.className = 'max-w-[52rem] mx-auto px-6 py-5';

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
        var lbl = document.createElement('div'); lbl.className = 'meta-label'; lbl.textContent = pair[0];
        var val = document.createElement('div'); val.className = 'meta-value'; val.textContent = pair[1];
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

/* ── Thinking Block ── */
function renderThinking(block) {
    var w = el('div', '', 'margin:0.75rem 0;padding:0.875rem 1rem;background:var(--thinking-bg);border-left:3px dashed var(--thinking-accent);border-radius:0 6px 6px 0;');
    var lbl = el('div'); lbl.className = 'block-label'; lbl.style.cssText = 'color:var(--thinking-accent);margin-bottom:0.5rem;'; lbl.textContent = 'Thinking';
    w.appendChild(lbl);
    var c = el('div'); c.className = 'thinking-prose tool-scroll'; c.innerHTML = renderMarkdown(block.text); w.appendChild(c);
    return w;
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

    var w = el('div', '', 'margin:0.25rem 0 0.75rem;background:'+bg+';border:1px solid '+br+';border-left:4px solid '+ac+';border-radius:0 6px 6px 0;overflow:hidden;');
    var hdr = el('div', '', 'padding:0.375rem 0.875rem;display:flex;align-items:center;gap:0.5rem;');
    var dot = el('span', '', 'width:0.4rem;height:0.4rem;border-radius:50%;background:'+ac+';flex-shrink:0;');
    hdr.appendChild(dot);
    var rl = el('span'); rl.className = 'block-label'; rl.style.color = ac;
    rl.textContent = (err ? 'Error' : 'Output') + (block.tool_name ? ' \u2014 ' + block.tool_name : '');
    hdr.appendChild(rl); w.appendChild(hdr);

    var txt = block.text || '(empty)';
    if (txt && txt !== '(empty)') {
        var bd = el('div', 'tool-scroll', 'padding:0 0.875rem 0.5rem;border-top:1px solid '+br+';');
        var pre = el('pre', 'tool-output', 'margin:0;padding-top:0.5rem;');
        pre.textContent = txt; bd.appendChild(pre); w.appendChild(bd);
    }
    return w;
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
    lines = parse_jsonl(path)
    metadata = extract_metadata(lines)
    messages = build_conversation(lines)

    if not messages:
        print("No messages found in session.", file=sys.stderr)
        sys.exit(1)

    html = generate_html(messages, metadata)

    if args.output:
        output_path = args.output
    else:
        sid = metadata.get("session_id", "session")[:12]
        output_path = f"claude-session-{sid}.html"

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Exported {len(messages)} messages to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Claude Code sessions to standalone HTML files."
    )

    parser.add_argument("--list", action="store_true", help="List available sessions")
    parser.add_argument("-p", "--project", help="Filter projects by name substring")
    parser.add_argument("session", nargs="?", help="Session UUID or path to .jsonl file")
    parser.add_argument("-o", "--output", help="Output HTML file path")

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    elif args.session:
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

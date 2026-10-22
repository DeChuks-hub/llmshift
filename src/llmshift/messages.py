"""Helpers for the canonical chat format used throughout LLMShift.

Requests are stored OpenAI-style: {"messages": [...], "params": {...}} where a
message is {"role": system|developer|user|assistant|tool, "content": str | parts}
and parts are {"type": "text", "text": ...} or {"type": "image_url", ...}.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from .textutil import json_loads, normalize, stable_hash

SYSTEM_ROLES = ("system", "developer")

# Request parameters understood across providers (OpenAI names).
PARAM_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "presence_penalty",
    "frequency_penalty",
    "reasoning_effort",
    "verbosity",
)


def content_text(content: Any) -> str:
    """Plain text of a message content (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif isinstance(part.get("content"), str):
                    out.append(part["content"])
                elif isinstance(part.get("text"), dict) and isinstance(part["text"].get("value"), str):
                    out.append(part["text"]["value"])
        return "\n".join(out)
    return str(content)


def system_prompt(messages: list[dict]) -> str:
    return "\n\n".join(content_text(m.get("content")) for m in messages if m.get("role") in SYSTEM_ROLES).strip()


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return content_text(m.get("content"))
    return ""


def first_user_text(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return content_text(m.get("content"))
    return ""


def with_system_prompt(messages: list[dict], new_prompt: str) -> list[dict]:
    """Return messages with the system prompt replaced (or added if absent)."""
    msgs = copy.deepcopy(messages)
    idx = [i for i, m in enumerate(msgs) if m.get("role") in SYSTEM_ROLES]
    if idx:
        first = idx[0]
        role = msgs[first].get("role", "system")
        msgs = [m for i, m in enumerate(msgs) if i not in idx[1:]]
        msgs[first] = {"role": role, "content": new_prompt}
    else:
        msgs.insert(0, {"role": "system", "content": new_prompt})
    return msgs


def apply_patch(messages: list[dict], patch: dict | None) -> list[dict]:
    """Apply a prompt patch: {"system": full replacement} or {"append": text}."""
    if not patch:
        return messages
    if patch.get("system") is not None:
        return with_system_prompt(messages, patch["system"])
    if patch.get("append"):
        current = system_prompt(messages)
        joined = (current + "\n\n" + patch["append"]).strip() if current else patch["append"].strip()
        return with_system_prompt(messages, joined)
    return messages


def normalize_tool_call(tc: Any) -> dict | None:
    """Accept OpenAI / Anthropic / Responses / Gemini tool-call shapes."""
    if not isinstance(tc, dict):
        return None
    name = None
    args: Any = None
    call_id = tc.get("id") or tc.get("call_id")
    if isinstance(tc.get("function"), dict):
        name = tc["function"].get("name")
        args = tc["function"].get("arguments")
    elif tc.get("type") == "tool_use":
        name, args = tc.get("name"), tc.get("input")
    elif "functionCall" in tc or "function_call" in tc:
        fc = tc.get("functionCall") or tc.get("function_call") or {}
        if not isinstance(fc, dict):  # an odd shape: not a call llmshift can read
            return None
        name, args = fc.get("name"), fc.get("args", fc.get("arguments"))
    else:
        name = tc.get("name")
        args = tc.get("arguments", tc.get("args", tc.get("input")))
    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json_loads(args) if args.strip() else {}
        except (ValueError, RecursionError):  # kept as text: not JSON, or nested too deeply
            pass
    out = {"name": name, "arguments": args if args is not None else {}}
    if call_id:
        out["id"] = call_id
    return out


def link_tool_calls(messages: list[dict]) -> list[dict]:
    """Copy of `messages` where every tool call has an id and every tool result names it.

    Logs (Gemini's in particular) often have calls without ids and results that
    reference a function name or nothing. Calls get deterministic ids; results are
    paired by id, then by function name, then by position.
    """
    out = copy.deepcopy(messages)
    pending: list[dict] = []  # calls awaiting a result: {"id", "name"}
    for mi, m in enumerate(out):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending = []
            for ci, tc in enumerate(m["tool_calls"]):
                if not tc.get("id"):
                    tc["id"] = f"call_{mi}_{ci}"
                pending.append({"id": tc["id"], "name": tc.get("name")})
        elif m.get("role") == "tool":
            ref = m.get("tool_call_id") or ""
            match = next((p for p in pending if p["id"] == ref), None)
            if match is None:
                by_name = ref or m.get("name") or ""
                match = next((p for p in pending if by_name and p["name"] == by_name), None)
            if match is None and pending:
                match = pending[0]
            if match is not None:
                m["tool_call_id"] = match["id"]
                m.setdefault("name", match["name"])
                pending.remove(match)
    return out


def clean_messages(messages: Any) -> list[dict]:
    """Validate/normalise a message list; raise ValueError on bad input."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"message {i} is not an object")
        role = m.get("role")
        if role == "model":
            role = "assistant"
        if role == "function":
            role = "tool"
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"message {i} has unsupported role {role!r}")
        msg = {"role": role, "content": m.get("content") if m.get("content") is not None else ""}
        if m.get("tool_calls"):
            calls = [normalize_tool_call(tc) for tc in m["tool_calls"]]
            msg["tool_calls"] = [c for c in calls if c]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("name") and role in ("tool", "user"):
            msg["name"] = m["name"]
        out.append(msg)
    return out


_FILLER = {
    "you", "are", "a", "an", "the", "your", "is", "please", "i", "am", "as", "to", "of", "and", "for", "in", "on",
    "with", "this", "that", "be", "will", "should", "must", "it", "act", "helpful", "following", "given", "who",
}


def _slug(text: str, n: int = 4) -> str:
    text = re.sub(r"['’]s\b", "", text.lower())
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _FILLER and not w.isdigit()]
    return "-".join(words[:n]) or "prompt"


def template_key(messages: list[dict], params: dict | None = None) -> str:
    """Readable key grouping requests built from the same prompt template,
    e.g. "support-ticket-triage-assistant-3f9a1c"."""
    params = params or {}
    sys_prompt = normalize(system_prompt(messages))
    if sys_prompt:
        base = "sys:" + sys_prompt
        label = _slug(sys_prompt)
    else:
        first = normalize(first_user_text(messages))
        first = re.sub(r"\d", "#", first)
        base = "usr:" + first[:48]
        label = _slug(first)
    names = []
    for t in params.get("tools") if isinstance(params.get("tools"), list) else []:
        if isinstance(t, dict):  # logs hold all sorts: tolerate odd shapes rather than fail the import
            fn = t.get("function")
            names.append(str((fn.get("name") if isinstance(fn, dict) else None) or t.get("name") or ""))
    tools = sorted(names)
    rf = params.get("response_format")
    rf_type = rf.get("type") if isinstance(rf, dict) else (rf or "")
    return f"{label}-{stable_hash({'base': base, 'tools': tools, 'rf': rf_type}, 6)}"


def request_preview(messages: list[dict], limit: int = 160) -> str:
    text = last_user_text(messages) or system_prompt(messages)
    return " ".join(text.split())[:limit]

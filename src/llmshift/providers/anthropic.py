"""Anthropic Messages API provider."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import Any

from .. import messages as msgs
from ..textutil import json_loads, stable_hash
from . import http
from .base import Provider, ProviderError, Response, base_url_from, clean_key, params_of

API_VERSION = "2023-06-01"
_DATA_URL_RE = re.compile(r"^data:([\w.+/-]+);base64,(.*)$", re.S)


def json_instruction(response_format: Any) -> str | None:
    """Prompt text that asks for JSON when a provider lacks OpenAI's response_format."""
    if not isinstance(response_format, dict):
        return None
    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return "Respond with a single valid JSON object only. Do not wrap it in markdown code fences and do not add any other text."
    if rf_type == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        text = "Respond with a single valid JSON object only. Do not wrap it in markdown code fences and do not add any other text."
        if schema:
            text += "\nThe JSON must conform to this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
        return text
    return None


def _image_block(url: str) -> dict:
    m = _DATA_URL_RE.match(url or "")
    if m:
        return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, dict):
        content = [content]
    blocks = []
    for part in content if isinstance(content, list) else [str(content)]:
        if isinstance(part, str):
            if part.strip():
                blocks.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            ptype = part.get("type")
            if ptype in ("image_url", "input_image"):
                img = part.get("image_url")
                url = img.get("url") if isinstance(img, dict) else img
                if url:
                    blocks.append(_image_block(url))
            elif ptype == "image" and isinstance(part.get("source"), dict):
                blocks.append(part)
            else:
                text = msgs.content_text([part])
                if text.strip():
                    blocks.append({"type": "text", "text": text})
    return blocks


def _tool_id(value: Any) -> str:
    """Anthropic ids must match ^[a-zA-Z0-9_-]+$; keep them stable so calls and results pair."""
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value or ""))
    return text or f"toolu_{uuid.uuid4().hex[:24]}"


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert canonical messages to (system, Anthropic messages)."""
    system = msgs.system_prompt(messages)
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in msgs.link_tool_calls(messages):
        role = m.get("role")
        if role in msgs.SYSTEM_ROLES:
            continue
        if role == "tool":
            push(
                "user",
                [{"type": "tool_result", "tool_use_id": _tool_id(m.get("tool_call_id")), "content": msgs.content_text(m.get("content")) or "(empty)"}],
            )
        elif role == "assistant":
            blocks = _content_blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json_loads(args) if args.strip() else {}
                    except (ValueError, RecursionError):
                        args = {"_raw": args}
                blocks.append({"type": "tool_use", "id": _tool_id(tc.get("id")), "name": tc.get("name"), "input": args or {}})
            push("assistant", blocks)
        else:
            push("user", _content_blocks(m.get("content")))
    if not out:
        if system:
            out.append({"role": "user", "content": [{"type": "text", "text": system}]})
            system = ""
        else:
            raise ProviderError("the request has no messages", code="invalid_request")
    if out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(conversation continues)"}]})
    return system, out


def to_anthropic_tools(tools: Any) -> list[dict]:
    out = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) and isinstance(t.get("function"), dict) else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        tool = {"name": fn["name"], "input_schema": schema}
        if fn.get("description"):
            tool["description"] = fn["description"]
        out.append(tool)
    return out


def to_anthropic_tool_choice(choice: Any) -> dict | None:
    if choice is None:
        return None
    if isinstance(choice, str):
        return {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}.get(choice)
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


class AnthropicProvider(Provider):
    kind = "anthropic"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        super().__init__(model, name=name or "anthropic", options=options)
        o = self.options
        self.base_url = base_url_from(o, o.get("base_url_env") or "ANTHROPIC_BASE_URL", "https://api.anthropic.com", self.spec)
        self.key_env = o.get("api_key_env", "ANTHROPIC_API_KEY")
        self.api_key = clean_key(o.get("api_key")) or (clean_key(os.environ.get(self.key_env)) if self.key_env else None)
        self.version = str(o.get("anthropic_version", API_VERSION))
        self.default_max_tokens = int(o.get("max_tokens", 4096))
        self.beta = o.get("beta")
        self.extra_headers = dict(o.get("headers") or {})
        self._lock = threading.Lock()
        self._drop: set[str] = set()
        self.adaptations: list[str] = []

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("api_key", "timeout", "max_retries", "api_key_env")}
        shaping.update({"version": self.version, "max_tokens": self.default_max_tokens, "beta": self.beta})
        shaping["headers"] = {k: v for k, v in self.extra_headers.items() if k.lower() not in ("x-api-key", "authorization")}
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "options": stable_hash(shaping, 16)}

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError(f"no API key for {self.spec}: set the {self.key_env} environment variable", code="missing_api_key")
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}
        if self.beta:
            headers["anthropic-beta"] = self.beta if isinstance(self.beta, str) else ",".join(self.beta)
        headers.update(self.extra_headers)
        return headers

    def build_body(self, request: dict) -> dict:
        params = params_of(request)
        system, messages = to_anthropic_messages(request.get("messages") or [])
        instruction = json_instruction(params.get("response_format"))
        if instruction:
            system = (system + "\n\n" + instruction).strip()
        max_tokens = params.get("max_tokens") or params.get("max_completion_tokens") or self.default_max_tokens
        body: dict = {"model": self.model, "max_tokens": int(max_tokens), "messages": messages}
        if system:
            body["system"] = system
        if params.get("temperature") is not None:
            body["temperature"] = max(0.0, min(1.0, float(params["temperature"])))
        if params.get("top_p") is not None:
            body["top_p"] = params["top_p"]
        stop = params.get("stop")
        if stop:
            seqs = [stop] if isinstance(stop, str) else list(stop)
            seqs = [s for s in seqs if isinstance(s, str) and s.strip()]
            if seqs:
                body["stop_sequences"] = seqs
        tools = to_anthropic_tools(params.get("tools"))
        if tools:
            body["tools"] = tools
            choice = to_anthropic_tool_choice(params.get("tool_choice"))
            if choice:
                if params.get("parallel_tool_calls") is False and choice.get("type") in ("auto", "any", "tool"):
                    choice["disable_parallel_tool_use"] = True
                body["tool_choice"] = choice
        rf = params.get("response_format")
        if self.options.get("structured_outputs") and isinstance(rf, dict) and rf.get("type") == "json_schema":
            schema = (rf.get("json_schema") or {}).get("schema")
            if schema:
                body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        with self._lock:
            for key in self._drop:
                body.pop(key, None)
        return body

    @staticmethod
    def parse(data: Any) -> Response:
        data = http.require_object(data, "Anthropic messages", "content")
        content = data.get("content") or []
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        tool_calls = [
            {"id": b.get("id"), "name": b.get("name"), "arguments": b.get("input") or {}}
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        stop = data.get("stop_reason")
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}.get(stop, stop)
        usage = data.get("usage") or {}
        return Response(
            text=text,
            tool_calls=tool_calls,
            model=data.get("model"),
            finish_reason=finish,
            usage={"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")} if usage else {},
            refusal="refusal" if stop == "refusal" else None,
        )

    def _note(self, note: str, drop: str | None = None) -> None:
        with self._lock:
            if drop:
                self._drop.add(drop)
            if note not in self.adaptations:
                self.adaptations.append(note)

    def _adapt(self, body: dict, err: ProviderError) -> bool:
        if err.status != 400:
            return False
        lower = (err.message or "").lower()
        if "prefill" in lower and body.get("messages") and body["messages"][-1]["role"] == "assistant":
            while body["messages"] and body["messages"][-1]["role"] == "assistant":
                body["messages"].pop()
            self._note(f"{self.model} does not accept assistant prefill; dropped the trailing assistant turn")
            return bool(body["messages"])
        if "tool_choice" in lower and isinstance(body.get("tool_choice"), dict) and body["tool_choice"].get("type") in ("any", "tool"):
            body["tool_choice"] = {"type": "auto"}
            self._note(f"{self.model} does not support forced tool choice; used tool_choice auto")
            return True
        if "output_config" in lower and "output_config" in body:
            body.pop("output_config", None)
            self._note("native structured output rejected; asked for JSON in the prompt instead", drop="output_config")
            return True
        # Only capability errors ("`temperature` is deprecated for this model", "temperature and
        # top_p cannot both be specified"); a bad value is reported as it is.
        if not re.search(r"deprecated|cannot both|not supported|unsupported|does not support|not allowed|not permitted", lower):
            return False
        for key in ("top_p", "temperature", "top_k", "stop_sequences"):
            if key in body and key in lower:
                drop = "top_p" if ("top_p" in lower and "temperature" in lower and "top_p" in body) else key
                body.pop(drop, None)
                self._note(f"dropped '{drop}' for {self.model} ({err.message[:80]})", drop=drop)
                return True
        return False

    def _complete(self, request: dict) -> Response:
        body = self.build_body(request)
        adaptations = 0
        while True:
            try:
                data = http.post_json(f"{self.base_url}/v1/messages", body, headers=self._headers(), timeout=self.timeout, max_retries=self.max_retries)
                return self.parse(data)
            except ProviderError as err:
                if adaptations >= 6 or not self._adapt(body, err):
                    raise
                adaptations += 1

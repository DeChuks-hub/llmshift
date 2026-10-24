"""Anthropic Messages API provider."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import Any

from .. import messages as msgs
from ..pricing import CACHE_FIELDS
from ..textutil import json_loads, stable_hash
from . import http
from .base import Provider, ProviderError, Response, base_url_from, clean_key, option_mapping, option_number, params_of, value_key

API_VERSION = "2023-06-01"
_DATA_URL_RE = re.compile(r"^data:([\w.+/-]+);base64,(.*)$", re.S)

# Claude 4.7 and later models (Fable and Mythos too) accept only the default sampling: a
# temperature other than 1, or any top_p or top_k, is a 400 error. (API version 2026-10-21
# rejects the three on every model, even at their defaults.)
_CLAUDE_VERSION_RE = re.compile(
    r"claude-(?:(?P<family>opus|sonnet|haiku)-(?P<major>\d+)(?:[-.](?P<minor>\d{1,2})(?!\d))?"
    r"|(?P<major2>\d+)(?:[-.](?P<minor2>\d{1,2})(?!\d))?-(?P<family2>opus|sonnet|haiku))"
)
_SAMPLING = (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop", "stop_sequences"))  # request name, API name


def default_sampling_only(model: str) -> bool:
    """Whether a Claude model rejects a non-default temperature, top_p or top_k."""
    model = str(model or "").lower()
    if re.search(r"claude-(?:fable|mythos)\b", model):
        return True
    m = _CLAUDE_VERSION_RE.search(model)
    if not m:
        return False
    major = int(m.group("major") or m.group("major2"))
    return (major, int(m.group("minor") or m.group("minor2") or 0)) >= (4, 7)


def rejected_sampling(model: str, params: dict) -> list[str]:
    """The sampling settings in `params` that `model` rejects, as name=value."""
    if not default_sampling_only(model):
        return []
    out = []
    temperature = params.get("temperature")
    if temperature is not None and not (isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and temperature == 1):
        out.append(f"temperature={json.dumps(temperature, default=str)}")
    out += [f"{key}={json.dumps(params[key], default=str)}" for key in ("top_p", "top_k") if params.get(key) is not None]
    return out


def _claude_version(model: str) -> tuple[str, int, int] | None:
    """(family, major, minor): ("opus", 4, 7) for claude-opus-4-7. Fable and Mythos are newer
    than any numbered model."""
    model = str(model or "").lower()
    if re.search(r"claude-(?:fable|mythos)\b", model):
        return "fable", 99, 0
    m = _CLAUDE_VERSION_RE.search(model)
    if not m:
        return None
    return m.group("family") or m.group("family2"), int(m.group("major") or m.group("major2")), int(m.group("minor") or m.group("minor2") or 0)


def takes_effort(model: str) -> bool:
    """Whether a Claude model takes output_config.effort (Opus and Sonnet 4.6 and later,
    Haiku 5 and later, Fable and Mythos)."""
    v = _claude_version(model)
    return v is not None and (v[0] == "fable" or (v[0] in ("opus", "sonnet") and v[1:] >= (4, 6)) or (v[0] == "haiku" and v[1] >= 5))


def rejected_thinking(model: str, params: dict) -> list[str]:
    """The thinking setting in `params` that `model` rejects with a 400 error: a token budget
    on 4.7 and later (adaptive thinking only), disabled thinking on Opus and Sonnet 5.5."""
    thinking = params.get("thinking")
    if not isinstance(thinking, dict):
        return []
    v = _claude_version(model)
    kind = thinking.get("type")
    if kind == "enabled" and default_sampling_only(model):
        return [f"thinking={json.dumps(thinking, sort_keys=True)}"]
    if kind == "disabled" and v is not None and v[0] in ("opus", "sonnet") and v[1:] >= (5, 5):
        return [f"thinking={json.dumps(thinking, sort_keys=True)}"]
    return []


def _thinking_body(model: str, thinking: Any) -> dict | None:
    """The thinking object to send: as recorded, or what this model takes instead (adaptive
    for a budget on 4.7 and later; nothing, which is adaptive there, for disabled on 5.5)."""
    if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "adaptive", "disabled"):
        return None
    if rejected_thinking(model, {"thinking": thinking}):
        if thinking["type"] == "disabled":
            return None
        thinking = {"type": "adaptive", **({"display": thinking["display"]} if thinking.get("display") else {})}
    body = {"type": thinking["type"]}
    budget = thinking.get("budget_tokens")
    if body["type"] == "enabled":
        if not isinstance(budget, int) or isinstance(budget, bool):
            return {"type": "adaptive"} if takes_effort(model) else None
        body["budget_tokens"] = max(1024, budget)  # the API's minimum (a Gemini budget may be smaller)
    if isinstance(thinking.get("display"), str):
        body["display"] = thinking["display"]
    return body


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
        self.default_max_tokens = option_number(o, "max_tokens", 4096, self.spec, minimum=1, integer=True)
        self.beta = o.get("beta")
        self.extra_headers = option_mapping(o, "headers", self.spec, text_values=True)
        self._lock = threading.Lock()
        self._drop: set[str] = set()  # fields this model rejects whatever their value
        self._refused: dict[str, set[str]] = {}  # sampling field -> values it rejected (it may take others)
        self.adaptations: list[str] = []

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("api_key", "timeout", "max_retries", "api_key_env", "timeout_set")}
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

    def build_body(self, request: dict, *, drops: bool = True) -> dict:
        """The Messages API body; `drops`: leave out what this model has rejected before."""
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
        if params.get("top_k") is not None:
            body["top_k"] = params["top_k"]
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
        # Extended thinking as the app asked for it (a recorded thinking object, or a budget from
        # Gemini's thinkingConfig), and how hard to think (reasoning_effort, Claude's effort).
        thinking = _thinking_body(self.model, params.get("thinking"))
        if thinking:
            body["thinking"] = thinking
            if thinking.get("budget_tokens") and body["max_tokens"] <= thinking["budget_tokens"]:
                body["max_tokens"] = thinking["budget_tokens"] + body["max_tokens"]  # the budget must be below max_tokens
        effort = params.get("reasoning_effort")
        if isinstance(effort, str) and effort and takes_effort(self.model):
            body["output_config"] = {**(body.get("output_config") or {}), "effort": {"none": "low", "minimal": "low"}.get(effort, effort)}
        if drops:
            with self._lock:
                for key in self._drop:
                    if key == "effort" and isinstance(body.get("output_config"), dict):
                        body["output_config"].pop("effort", None)
                        if not body["output_config"]:
                            body.pop("output_config")
                    body.pop(key, None)
                for key, values in self._refused.items():
                    if key in body and value_key(body[key]) in values:
                        body.pop(key)
        return body

    def _rejected_in(self, request: dict, sent: dict) -> list[str]:
        """What of `request` this model refused, so the answer came from a changed request."""
        full = self.build_body(request, drops=False)
        out = [name for name, key in _SAMPLING if key in full and key not in sent]
        if full["messages"] and full["messages"][-1].get("role") == "assistant" and (not sent["messages"] or sent["messages"][-1].get("role") != "assistant"):
            out.append("assistant prefill")
        if (full.get("tool_choice") or {}).get("type") in ("any", "tool") and (sent.get("tool_choice") or {}).get("type") not in ("any", "tool"):
            out.append("tool_choice")
        if "thinking" in full and "thinking" not in sent:
            out.append("thinking")
        if "effort" in (full.get("output_config") or {}) and "effort" not in (sent.get("output_config") or {}):
            out.append("reasoning_effort")
        return out

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
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Response(
            text=text,
            tool_calls=tool_calls,
            model=data.get("model"),
            finish_reason=finish,
            usage={k: usage.get(k) for k in ("input_tokens", "output_tokens", *CACHE_FIELDS)} if usage else {},
            refusal="refusal" if stop == "refusal" else None,
        )

    def _note(self, note: str, drop: str | None = None, value: Any = None) -> None:
        with self._lock:
            if drop in dict(_SAMPLING).values():  # learnt per value: temperature 0 refused, 1 accepted
                self._refused.setdefault(drop, set()).add(value_key(value))
            elif drop:
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
        if "effort" in lower and "effort" in (body.get("output_config") or {}):
            body["output_config"].pop("effort")
            if not body["output_config"]:
                body.pop("output_config")
            self._note(f"dropped reasoning effort for {self.model} ({err.message[:80]})", drop="effort")
            return True
        if "thinking" in lower and "thinking" in body:  # a model with no extended thinking, or not this kind
            body.pop("thinking")
            self._note(f"dropped extended thinking for {self.model} ({err.message[:80]})", drop="thinking")
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
                value = body.pop(drop, None)
                self._note(f"dropped '{drop}' for {self.model} ({err.message[:80]})", drop=drop, value=value)
                return True
        return False

    def _complete(self, request: dict) -> Response:
        body = self.build_body(request)
        adaptations = 0
        while True:
            try:
                data = http.post_json(f"{self.base_url}/v1/messages", body, headers=self._headers(), timeout=self.timeout, max_retries=self.max_retries)
                response = self.parse(data)
                response.rejected = self._rejected_in(request, body)
                return response
            except ProviderError as err:
                if adaptations >= 6 or not self._adapt(body, err):
                    raise
                adaptations += 1

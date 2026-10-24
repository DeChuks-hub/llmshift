"""Google Gemini API (generateContent) provider."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import Any

from .. import messages as msgs
from ..textutil import json_loads, stable_hash
from . import http
from .anthropic import json_instruction
from .base import Provider, ProviderError, Response, base_url_from, clean_key, params_of

_DATA_URL_RE = re.compile(r"^data:([\w.+/-]+);base64,(.*)$", re.S)
_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum", "properties", "required", "items",
    "minItems", "maxItems", "minimum", "maximum", "anyOf", "propertyOrdering", "title",
}
_BLOCK_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "OTHER"}


def sanitize_schema(schema: Any, root: dict | None = None, depth: int = 0) -> Any:
    """Reduce a JSON Schema to the OpenAPI subset Gemini function declarations accept."""
    if not isinstance(schema, dict) or depth > 20:
        return {"type": "string"} if depth > 20 else schema
    root = root if root is not None else schema
    if "$ref" in schema:
        ref = schema["$ref"]
        node: Any = root
        for part in ref.lstrip("#/").split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        return sanitize_schema(node, root, depth + 1)
    out: dict = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "type" and isinstance(value, list):
            types = [t for t in value if t != "null"]
            out["type"] = (types[0] if types else "string").upper()
            if "null" in value:
                out["nullable"] = True
        elif key == "type":
            out["type"] = str(value).upper()
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: sanitize_schema(v, root, depth + 1) for k, v in value.items()}
        elif key == "items":
            out["items"] = sanitize_schema(value, root, depth + 1)
        elif key == "anyOf" and isinstance(value, list):
            out["anyOf"] = [sanitize_schema(v, root, depth + 1) for v in value]
        else:
            out[key] = value
    return out


_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)")


def _thinking_config(model: str, params: dict) -> dict | None:
    """thinkingConfig from the request: reasoning_effort as Gemini 3's thinkingLevel, else
    thinking as a budget (-1 dynamic, 0 off). Gemini 3 takes one of the two, not both."""
    m = _GEMINI_VERSION_RE.search(str(model).lower())
    effort = params.get("reasoning_effort")
    if m and int(m.group(1)) >= 3 and isinstance(effort, str) and effort:
        return {"thinkingLevel": {"none": "minimal", "xhigh": "high", "max": "high"}.get(effort, effort)}
    thinking = params.get("thinking")
    if not isinstance(thinking, dict):
        return None
    budget = thinking.get("budget_tokens")
    if thinking.get("type") == "adaptive":
        return {"thinkingBudget": -1}
    if thinking.get("type") == "disabled":
        return {"thinkingBudget": 0}
    if thinking.get("type") == "enabled" and isinstance(budget, int) and not isinstance(budget, bool):
        return {"thinkingBudget": budget}
    return None


def _usage(meta: dict) -> dict:
    """Tokens as Google bills them: thinking tokens (thoughtsTokenCount) at the output price
    with the answer's, tool-use prompt tokens at the input price, and the cached share of the
    prompt (cachedContentTokenCount, already in promptTokenCount) at the cache price."""
    if not meta:
        return {}

    def total(*keys: str) -> int | None:
        found = [meta[k] for k in keys if isinstance(meta.get(k), int) and not isinstance(meta.get(k), bool)]
        return sum(found) if found else None

    usage = {"input_tokens": total("promptTokenCount", "toolUsePromptTokenCount"), "output_tokens": total("candidatesTokenCount", "thoughtsTokenCount")}
    cached = meta.get("cachedContentTokenCount")
    if isinstance(cached, int) and not isinstance(cached, bool) and cached > 0:
        usage["cached_input_tokens"] = cached
    return usage


def _parts(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if isinstance(content, dict):
        content = [content]
    parts = []
    for part in content if isinstance(content, list) else [str(content)]:
        if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
            img = part.get("image_url")
            url = img.get("url") if isinstance(img, dict) else img
            m = _DATA_URL_RE.match(url or "")
            if m:
                parts.append({"inlineData": {"mimeType": m.group(1), "data": m.group(2)}})
            elif url:
                parts.append({"text": f"[image: {url}]"})
        else:
            text = msgs.content_text([part])
            if text:
                parts.append({"text": text})
    return parts


class GeminiProvider(Provider):
    kind = "gemini"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        if model.startswith("models/"):
            model = model[len("models/") :]
        super().__init__(model, name=name or "gemini", options=options)
        o = self.options
        self.base_url = base_url_from(o, "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta", self.spec)
        self.key_env = o.get("api_key_env", "GEMINI_API_KEY")
        self.api_key = (
            clean_key(o.get("api_key")) or (clean_key(os.environ.get(self.key_env)) if self.key_env else None) or clean_key(os.environ.get("GOOGLE_API_KEY"))
        )
        self.adaptations: list[str] = []  # what was changed in requests this model rejected

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("api_key", "timeout", "max_retries", "api_key_env", "timeout_set")}
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "options": stable_hash(shaping, 16)}

    def build_body(self, request: dict) -> dict:
        params = params_of(request)
        messages = request.get("messages") or []
        system = msgs.system_prompt(messages)
        instruction = json_instruction(params.get("response_format"))
        if instruction:
            system = (system + "\n\n" + instruction).strip()
        # Logs often hold calls without ids; link them so each result gets its function's name,
        # but only send ids that were in the original request.
        original_ids = {tc.get("id") for m in messages for tc in (m.get("tool_calls") or []) if isinstance(tc, dict) and tc.get("id")}
        call_names: dict[str, str] = {}
        contents: list[dict] = []

        def push(role: str, parts: list[dict]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})

        for m in msgs.link_tool_calls(messages):
            role = m.get("role")
            if role in msgs.SYSTEM_ROLES:
                continue
            if role == "assistant":
                parts = _parts(m.get("content"))
                for tc in m.get("tool_calls") or []:
                    args = tc.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json_loads(args) if args.strip() else {}
                        except (ValueError, RecursionError):
                            args = {"_raw": args}
                    call: dict = {"name": tc.get("name"), "args": args if isinstance(args, dict) else {"value": args} if args else {}}
                    if tc.get("id"):
                        call_names[tc["id"]] = tc.get("name")
                        if tc["id"] in original_ids:
                            call["id"] = tc["id"]
                    parts.append({"functionCall": call})
                push("model", parts)
            elif role == "tool":
                text = msgs.content_text(m.get("content"))
                try:
                    # NaN/Infinity stay text: they are not valid JSON in the request body.
                    payload = json.loads(text, parse_constant=str)
                    if not isinstance(payload, dict):
                        payload = {"content": payload}
                except ValueError:
                    payload = {"content": text}
                name = call_names.get(m.get("tool_call_id") or "") or m.get("name") or "tool"
                fr: dict = {"name": name, "response": payload}
                if m.get("tool_call_id") in original_ids:
                    fr["id"] = m["tool_call_id"]
                push("user", [{"functionResponse": fr}])
            else:
                push("user", _parts(m.get("content")))
        if not contents:
            if not system:
                raise ProviderError("the request has no messages", code="invalid_request")
            contents = [{"role": "user", "parts": [{"text": system}]}]
            system = ""
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen: dict = {}
        if params.get("temperature") is not None:
            gen["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            gen["topP"] = params["top_p"]
        if params.get("top_k") is not None:
            gen["topK"] = params["top_k"]
        max_out = params.get("max_tokens") or params.get("max_completion_tokens")
        if max_out:
            gen["maxOutputTokens"] = int(max_out)
        stop = params.get("stop")
        if stop:
            seqs = [stop] if isinstance(stop, str) else list(stop)
            gen["stopSequences"] = [s for s in seqs if isinstance(s, str)][:5]
        if params.get("seed") is not None:
            gen["seed"] = params["seed"]
        thinking = _thinking_config(self.model, params)
        if thinking:
            gen["thinkingConfig"] = thinking
        rf = params.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
            gen["responseMimeType"] = "application/json"
        if gen:
            body["generationConfig"] = gen
        decls = []
        for t in params.get("tools") or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                decl = {"name": fn["name"]}
                if fn.get("description"):
                    decl["description"] = fn["description"]
                if fn.get("parameters") and (fn["parameters"].get("properties") or fn["parameters"].get("type") != "object"):
                    decl["parameters"] = sanitize_schema(fn["parameters"])
                decls.append(decl)
        if decls:
            body["tools"] = [{"functionDeclarations": decls}]
            choice = params.get("tool_choice")
            mode = None
            allowed = None
            if choice == "required":
                mode = "ANY"
            elif choice == "none":
                mode = "NONE"
            elif choice == "auto":
                mode = "AUTO"
            elif isinstance(choice, dict) and isinstance(choice.get("function"), dict):
                mode, allowed = "ANY", [choice["function"].get("name")]
            if mode:
                cfg: dict = {"mode": mode}
                if allowed:
                    cfg["allowedFunctionNames"] = allowed
                body["toolConfig"] = {"functionCallingConfig": cfg}
        return body

    @staticmethod
    def parse(data: Any) -> Response:
        data = http.require_object(data, "Gemini generateContent", "candidates", "promptFeedback")
        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        usage_out = _usage(usage)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason") or "no candidates returned"
            return Response(text="", model=data.get("modelVersion"), finish_reason="content_filter", refusal=str(reason), usage=usage_out)
        cand = candidates[0] if isinstance(candidates[0], dict) else {}
        content = cand.get("content") if isinstance(cand.get("content"), dict) else {}
        parts = [p for p in content.get("parts") or [] if isinstance(p, dict)]
        text = "".join(p.get("text") or "" for p in parts if isinstance(p.get("text"), str) and not p.get("thought"))
        tool_calls = []
        for p in parts:
            fc = p.get("functionCall")
            if isinstance(fc, dict) and fc.get("name"):
                call = {"name": fc["name"], "arguments": fc.get("args") or {}}
                if fc.get("id"):
                    call["id"] = fc["id"]
                tool_calls.append(call)
        reason = cand.get("finishReason")
        finish = {"STOP": "stop", "MAX_TOKENS": "length", "CONTINUATION": "length"}.get(reason or "", (reason or "").lower() or None)
        if tool_calls and finish == "stop":
            finish = "tool_calls"
        refusal = reason if reason in _BLOCK_REASONS and not text else None
        return Response(text=text, tool_calls=tool_calls, model=data.get("modelVersion"), finish_reason=finish, usage=usage_out, refusal=refusal)

    def _complete(self, request: dict) -> Response:
        if not self.api_key:
            raise ProviderError(f"no API key for {self.spec}: set GEMINI_API_KEY (or GOOGLE_API_KEY)", code="missing_api_key")
        url = f"{self.base_url}/models/{urllib.parse.quote(self.model, safe='-._')}:generateContent"
        body = self.build_body(request)
        try:
            data = http.post_json(url, body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout, max_retries=self.max_retries)
        except ProviderError as err:
            gen = body.get("generationConfig") or {}
            if err.status != 400 or "thinkingConfig" not in gen or "think" not in (err.message or "").lower():
                raise
            # A model with no thinking, or that cannot turn it off (budget 0): the answer is
            # tested without the setting, and the setting is reported as rejected.
            dropped = gen.pop("thinkingConfig")
            note = f"dropped thinkingConfig {json.dumps(dropped)} for {self.model} ({err.message[:80]})"
            if note not in self.adaptations:
                self.adaptations.append(note)
            data = http.post_json(url, body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout, max_retries=self.max_retries)
            response = self.parse(data)
            response.rejected = ["reasoning_effort" if "thinkingLevel" in dropped else "thinking"]
            return response
        return self.parse(data)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ProviderError("no API key: set GEMINI_API_KEY", code="missing_api_key")
        url = f"{self.base_url}/models/{urllib.parse.quote(self.model, safe='-._')}:batchEmbedContents"
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}} for t in texts]}
        data = http.post_json(url, body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout, max_retries=self.max_retries)
        data = http.require_object(data, "Gemini embeddings", "embeddings")
        return [(e.get("values") or []) if isinstance(e, dict) else [] for e in data.get("embeddings") or []]

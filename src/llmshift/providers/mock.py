"""Offline simulated models for the demo and tests (`mock:legacy`, `mock:nextgen`,
`mock:echo`).

`legacy` behaves like a dependable older model. `nextgen` simulates the kind of
silent behaviour change teams hit when switching models: JSON wrapped in code
fences, a chattier tone with emoji, rounded figures and over-escalated
urgency. Like a real model, `nextgen` follows instructions in the system prompt,
so `llmshift fix` can repair the prompt-fixable differences.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .. import messages as msgs
from .base import Provider, ProviderError, Response

_CATEGORY_RULES = [
    ("billing", r"refund|charge|invoice|billing|payment|charged|bill"),
    ("account", r"log ?in|password|account|sign ?in|e-?mail address"),
    ("technical", r"error|crash|bug|app |not working|broken|freez"),
    ("shipping", r"deliver|shipping|package|parcel|arriv|tracking|order"),
]


def _category(text: str) -> str:
    lower = text.lower()
    for name, pattern in _CATEGORY_RULES:
        if re.search(pattern, lower):
            return name
    return "account"


def _urgency(text: str) -> str:
    lower = text.lower()
    if re.search(r"urgent|asap|immediately|right now|today|!!", lower):
        return "high"
    if re.search(r"still|again|when|soon|waiting|days", lower):
        return "medium"
    return "low"


def _first_sentence(text: str, words: int = 12) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text.strip())[0]
    toks = sentence.split()
    out = " ".join(toks[:words]).rstrip(".!?,;")
    return out + "."


def _order(text: str) -> str | None:
    m = re.search(r"#\d{3,}", text)
    return m.group(0) if m else None


def _date(text: str) -> str | None:
    m = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}\b", text)
    return m.group(0) if m else None


def _task(system: str) -> str:
    s = system.lower()
    if "triage" in s or "classify" in s:
        return "triage"
    if "summar" in s:
        return "summary"
    if "support" in s or "reply" in s:
        return "reply"
    return "generic"


def _round_figures(text: str) -> str:
    def money(m: re.Match) -> str:
        value = float(m.group(1))
        return f"roughly ${round(value)} {m.group(2)}"

    text = re.sub(r"\$(\d+\.\d+) (million|billion)", money, text)
    return re.sub(r"(\d+\.\d+)%", lambda m: f"about {round(float(m.group(1)))}%", text)


def legacy(request: dict) -> Response:
    messages = request.get("messages") or []
    system = msgs.system_prompt(messages)
    user = msgs.last_user_text(messages)
    task = _task(system)
    if task == "triage":
        text = json.dumps({"category": _category(user), "urgency": _urgency(user), "summary": _first_sentence(user)})
    elif task == "reply":
        order = _order(user) or "your order"
        date = _date(user)
        parts = [f"Thank you for contacting Acme support. We are sorry for the inconvenience with order {order}."]
        if date:
            parts.append(f"We have checked the shipment and it will be delivered on {date}.")
        else:
            parts.append("We have checked the shipment and it is on its way to you.")
        parts.append("Please let us know if there is anything else we can do.")
        text = " ".join(parts)
    elif task == "summary":
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", user) if re.search(r"\d", s)]
        text = " ".join(sentences[:2]) if sentences else _first_sentence(user, 30)
    else:
        text = f"You said: {user}"
    return Response(text=text, model="legacy-1", finish_reason="stop", fingerprint="fp_legacy", usage=_usage(messages, text))


def nextgen(request: dict) -> Response:
    messages = request.get("messages") or []
    system = msgs.system_prompt(messages)
    s = system.lower()
    user = msgs.last_user_text(messages)
    task = _task(system)
    if task == "triage":
        urgency = _urgency(user)
        if urgency == "medium":
            urgency = "high"  # over-escalates: a behaviour change prompts alone may not fix
        payload = json.dumps({"category": _category(user), "urgency": urgency, "summary": _first_sentence(user)}, indent=2)
        fenced = not re.search(r"code fence|raw json|no markdown", s)
        text = f"```json\n{payload}\n```" if fenced else payload
    elif task == "reply":
        order = _order(user) or "your order"
        date = _date(user)
        emoji = "emoji" not in s
        bang = "!" if "exclamation" not in s else "."
        follow_up = not re.search(r"follow-up|further help", s)
        formal = bool(re.search(r"formal|professional", s))
        we = "using 'we'" in s
        polite = bool(re.search(r"please and thank you|courteous", s))
        brand = re.search(r'mention "([^"]+)"', system, re.I)
        when = f"it should arrive by {date}" if date else "it is on its way"
        parts = ["Hello," if formal else "Hi there" + bang]
        if emoji:
            parts[0] += " \U0001F60A"
        if polite:
            parts.append(f"Thank you for contacting {brand.group(1)} support." if brand else "Thank you for reaching out.")
        elif brand:
            parts.append(f"This is the {brand.group(1)} support team.")
        if formal:
            parts.append(("We are" if we else "I am") + f" sorry for the trouble with order {order}.")
            parts.append(("We have" if we else "I have") + f" checked the shipment, and {when}.")
        else:
            parts.append(f"So sorry about the hassle with order {order}{bang}")
            parts.append(f"Good news: {when}{bang}")
        if follow_up:
            parts.append(f"Would you like {'us' if we else 'me'} to send you tracking updates?" + (" \U0001F69A" if emoji else ""))
        text = " ".join(parts)
    elif task == "summary":
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", user) if re.search(r"\d", x)]
        text = " ".join(sentences[:2]) if sentences else _first_sentence(user, 30)
        if not re.search(r"exact|do not round|verbatim", s):
            text = _round_figures(text)
    else:
        text = f"Sure! You said: {user}"
    return Response(text=text, model="nextgen-2", finish_reason="stop", fingerprint="fp_nextgen", usage=_usage(messages, text))


def echo(request: dict) -> Response:
    messages = request.get("messages") or []
    text = msgs.last_user_text(messages)
    return Response(text=text, model="echo", finish_reason="stop", usage=_usage(messages, text))


def _usage(messages: list[dict], text: str) -> dict:
    prompt = sum(len(msgs.content_text(m.get("content")).split()) for m in messages)
    return {"input_tokens": int(prompt * 1.3) + 4, "output_tokens": int(len(text.split()) * 1.3) + 1}


BEHAVIOURS: dict[str, Callable[[dict], Any]] = {"legacy": legacy, "nextgen": nextgen, "echo": echo}


class MockProvider(Provider):
    kind = "mock"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None, fn: Callable[[dict], Any] | None = None):
        super().__init__(model, name=name or "mock", options=options)
        self.fn = fn or BEHAVIOURS.get(model)
        if self.fn is None:
            raise ProviderError(f"unknown mock model {model!r}; available: {', '.join(sorted(BEHAVIOURS))}", code="model_not_found", status=404)

    def identity(self) -> dict:
        return {"kind": self.kind, "model": self.model, "fn": getattr(self.fn, "__qualname__", repr(self.fn))}

    def _complete(self, request: dict) -> Response:
        return coerce_response(self.fn(request), default_model=self.model)


def coerce_response(result: Any, default_model: str | None = None) -> Response:
    """Turn whatever a user function returns into a Response."""
    if isinstance(result, Response):
        if result.model is None:
            result.model = default_model
        if result.text is None:
            result.text = ""
        return result
    if result is None:
        return Response(text="", model=default_model)
    if isinstance(result, str):
        return Response(text=result, model=default_model, finish_reason="stop")
    if hasattr(result, "model_dump"):  # SDK objects (pydantic); aliases keep Gemini's camelCase keys
        try:
            result = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            result = result.model_dump()
        except Exception:
            result = str(result)
    if isinstance(result, dict):
        if "choices" in result:
            from .openai import OpenAIProvider

            return OpenAIProvider.parse_chat(result)
        if result.get("type") == "message" and isinstance(result.get("content"), list):
            from .anthropic import AnthropicProvider

            return AnthropicProvider.parse(result)
        if "output" in result and isinstance(result["output"], list):
            from .openai import OpenAIProvider

            return OpenAIProvider.parse_responses(result)
        if "candidates" in result:
            from .gemini import GeminiProvider

            return GeminiProvider.parse(result)
        text = next((result[k] for k in ("text", "output", "content") if result.get(k) is not None), "")
        calls = [c for c in (msgs.normalize_tool_call(tc) for tc in result.get("tool_calls") or []) if c]
        usage = result.get("usage")
        return Response(
            text=text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str),
            tool_calls=calls,
            model=result.get("model") or default_model,
            finish_reason=result.get("finish_reason", "stop"),
            usage=usage if isinstance(usage, dict) else {},
        )
    return Response(text=str(result), model=default_model, finish_reason="stop")

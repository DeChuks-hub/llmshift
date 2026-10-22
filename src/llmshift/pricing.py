"""Model prices and request costs (USD).

The bundled table (data/pricing.json) is converted from LiteLLM's model price list;
`scripts/update_pricing.py` refreshes it. Prices can be set per project, in USD per
1M tokens, which wins over the bundled table:

    pricing:
      openai:my-finetune: {input: 0.30, output: 1.20}
      azure:prod-gpt4o: {input: 2.50, output: 10.00}
"""

from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

# llmshift provider name -> LiteLLM key prefixes to try before the bare model id
_PREFIXES = {
    "gemini": ("gemini/",),
    "mistral": ("mistral/",),
    "deepseek": ("deepseek/",),
    "xai": ("xai/",),
    "groq": ("groq/",),
    "together": ("together_ai/",),
    "fireworks": ("fireworks_ai/accounts/fireworks/models/", "fireworks_ai/"),
    "perplexity": ("perplexity/",),
    "cohere": ("cohere_chat/", "cohere/"),
    "openrouter": ("openrouter/",),
    "azure": ("azure/",),
}
LOCAL_PROVIDERS = {"ollama", "lmstudio", "vllm"}  # your own hardware: no per-token price
# Offline simulated models, priced like a flagship (legacy) and cheaper successors.
MOCK_PRICES = {
    "legacy": (5.0, 15.0), "nextgen": (1.25, 10.0), "frugal": (0.15, 0.6), "echo": (0.0, 0.0),
    "embed-v1": (0.13, 0.0), "embed-v2": (0.02, 0.0),
}
_DATE_SUFFIX_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    source: str  # where the price came from, for reports
    key: str = ""

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (max(0.0, input_tokens) * self.input + max(0.0, output_tokens) * self.output) / 1_000_000


@functools.lru_cache(maxsize=1)
def _table() -> tuple[dict, str]:
    raw = resources.files("llmshift").joinpath("data/pricing.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    return {k.lower(): v for k, v in (data.get("prices") or {}).items()}, str(data.get("fetched") or "?")


def table_date() -> str:
    return _table()[1]


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{what} must be a number of USD per 1M tokens, got {value!r}")
    try:
        number = float(str(value).strip().lstrip("$"))
    except ValueError:
        raise ValueError(f"{what} must be a number of USD per 1M tokens, got {value!r}") from None
    if number < 0 or not math.isfinite(number):
        raise ValueError(f"{what} must be a non-negative number, got {value!r}")
    return number


def parse_price(text: str) -> tuple[str, Price]:
    """`MODEL=INPUT/OUTPUT` in USD per 1M tokens, e.g. `openai:gpt-4o=2.5/10`."""
    model, sep, value = str(text).rpartition("=")
    if not sep or not model.strip():
        raise ValueError(f"--price expects MODEL=INPUT/OUTPUT (USD per 1M tokens), got {text!r}")
    inp, sep, out = value.partition("/")
    if not sep:
        raise ValueError(f"--price expects MODEL=INPUT/OUTPUT (USD per 1M tokens), got {text!r}")
    return model.strip(), Price(_number(inp, "input price"), _number(out, "output price"), "--price", model.strip())


def overrides_from_config(config: dict | None) -> dict[str, Price]:
    """The `pricing:` section of llmshift.yaml."""
    out: dict[str, Price] = {}
    section = (config or {}).get("pricing") or {}
    if not isinstance(section, dict):
        raise ValueError("pricing: in llmshift.yaml must map model names to {input: ..., output: ...}")
    for model, value in section.items():
        if not isinstance(value, dict) or "input" not in value:
            raise ValueError(f"pricing.{model} must look like {{input: 2.5, output: 10}} (USD per 1M tokens)")
        out[str(model).strip().lower()] = Price(_number(value["input"], f"pricing.{model}.input"), _number(value.get("output", 0), f"pricing.{model}.output"), "config", str(model))
    return out


def _split(spec: str) -> tuple[str | None, str]:
    from .providers import parse_spec

    name, model = parse_spec(spec)
    return name, model


def price_for(spec: str, overrides: dict[str, Price] | None = None) -> Price | None:
    """The price of a model spec ("openai:gpt-4o", "anthropic:claude-...", or a bare id)."""
    spec = str(spec or "").strip()
    if not spec:
        return None
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}
    provider, model = _split(spec)
    for key in (spec.lower(), model.lower()):
        if key in overrides:
            return overrides[key]
    if provider == "mock" and model in MOCK_PRICES:
        inp, out = MOCK_PRICES[model]
        return Price(inp, out, "simulated", f"mock:{model}")
    if provider in LOCAL_PROVIDERS:
        return Price(0.0, 0.0, "local model", spec)
    table, fetched = _table()
    bare = model.lower()
    if bare.startswith("models/"):
        bare = bare[len("models/"):]
    candidates = [p + bare for p in _PREFIXES.get(provider or "", ())] + [bare]
    if "/" in bare:  # openrouter:openai/gpt-4o -> gpt-4o
        candidates.append(bare.split("/")[-1])
    for name in list(candidates):
        for variant in (_DATE_SUFFIX_RE.sub("", name), re.sub(r"-latest$", "", name)):
            if variant != name:
                candidates.append(variant)
    for key in dict.fromkeys(candidates):
        entry = table.get(key)
        if entry:
            return Price(float(entry[0]), float(entry[1]), f"LiteLLM price list, {fetched}", key)
    return None


_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def estimate_tokens(text: Any) -> int:
    """Rough token count when a provider did not report usage: ~4 characters per
    token for Latin text, about one token per CJK character."""
    if text is None:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, default=str)
    cjk = len(_CJK_RE.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def usage_tokens(usage: dict | None, prompt: Any = None, output: Any = None) -> tuple[int, int, bool]:
    """(input, output, estimated) tokens, from reported usage where possible."""
    usage = usage or {}
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    estimated = False
    if not isinstance(inp, (int, float)) or isinstance(inp, bool):
        inp, estimated = estimate_tokens(prompt), True
    if not isinstance(out, (int, float)) or isinstance(out, bool):
        out, estimated = estimate_tokens(output), True
    return int(inp), int(out), estimated


def money(usd: float | None) -> str:
    """$1,240 / $3.10 / $0.042 / $0.00042 - enough precision to compare cheap models."""
    if usd is None or usd != usd:
        return "?"
    sign, usd = ("-", -usd) if usd < 0 else ("", usd)
    if usd >= 100:
        text = f"{usd:,.0f}"
    elif usd >= 0.1:
        text = f"{usd:,.2f}"
    elif usd >= 0.01:
        text = f"{usd:.3f}"
    elif usd == 0:
        text = "0"
    else:  # two significant digits, never in exponent notation
        text = f"{usd:.{max(2, 1 - math.floor(math.log10(usd)))}f}"
    return f"{sign}${text}"

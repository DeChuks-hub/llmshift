"""Model prices and request costs (USD).

The bundled table (data/pricing.json) is converted from LiteLLM's model price list;
`scripts/update_pricing.py` refreshes it, and `llmshift update --prices` downloads a
newer copy for one machine (used when its `fetched` date is not older than the
bundled one). Prices can be set per project, in USD per 1M tokens, which wins over
both tables:

    pricing:
      openai:my-finetune: {input: 0.30, output: 1.20}
      azure:prod-gpt4o: {input: 2.50, output: 10.00}
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .textutil import json_loads

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
    cache_read: float | None = None  # input read from the provider's prompt cache, where the list says
    cache_write: float | None = None  # input written to it (Anthropic)

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (max(0.0, input_tokens) * self.input + max(0.0, output_tokens) * self.output) / 1_000_000


_MODES = {"chat", "completion", "responses", "embedding"}


def _per_million(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    per_million = value * 1_000_000
    return float(f"{per_million:.6g}") if math.isfinite(per_million) else None  # one absurd entry is skipped, not the table


def convert_litellm(table: Any) -> dict[str, list]:
    """LiteLLM's model_prices_and_context_window.json -> {model: [input, output]} per 1M
    tokens, or [input, output, cache read, cache write] (null where unknown) for a model
    with prompt-cache prices. Readers before 0.1.1 use the first two."""
    prices = {}
    for key, entry in (table.items() if isinstance(table, dict) else ()):
        if key == "sample_spec" or not isinstance(entry, dict) or entry.get("mode") not in _MODES:
            continue
        inp = _per_million(entry.get("input_cost_per_token"))
        out = _per_million(entry.get("output_cost_per_token", 0.0))
        if inp is not None and out is not None:
            read = _per_million(entry.get("cache_read_input_token_cost"))
            write = _per_million(entry.get("cache_creation_input_token_cost"))
            prices[str(key)] = [inp, out] if read is None and write is None else [inp, out, read, write]
    return dict(sorted(prices.items()))


def _valid_price(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def prices_problems(data: Any, *, min_prices: int = 1) -> list[str]:
    """Why a price table cannot be used; empty when it is valid."""
    from .registry import _date, today

    if not isinstance(data, dict):
        return ["not a JSON object"]
    problems = []
    fetched = _date(data.get("fetched"))
    if fetched is None:
        problems.append("missing or invalid 'fetched' date")
    elif fetched > today() + _dt.timedelta(days=2):  # it would win over every later table
        problems.append(f"'fetched' date {fetched.isoformat()} is in the future")
    prices = data.get("prices")
    if not isinstance(prices, dict):
        return problems + ["'prices' must be an object"]
    if len(prices) < min_prices:
        problems.append(f"only {len(prices)} prices (expected at least {min_prices})")
    bad = [
        k for k, v in prices.items()
        if not isinstance(v, (list, tuple)) or len(v) < 2
        or not all(_valid_price(x) for x in v[:2]) or not all(x is None or _valid_price(x) for x in v[2:4])
    ]
    if bad:
        problems.append(f"{len(bad)} invalid price entries, e.g. {bad[0]!r}")
    return problems


def _bundled_prices() -> dict:
    return json.loads(resources.files("llmshift").joinpath("data/pricing.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _price_data() -> tuple[dict, str]:
    """(price table, where it came from): a valid download that is not older than the
    bundled table wins; a missing or broken download is ignored."""
    from .registry import _date, user_data_dir

    data, origin = _bundled_prices(), "bundled"
    path = user_data_dir() / "pricing.json"
    try:
        cached = json_loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        cached = None
    if cached is not None and not prices_problems(cached):
        if (_date(cached.get("fetched")) or _dt.date.min) >= (_date(data.get("fetched")) or _dt.date.min):
            data, origin = cached, str(path)
    return data, origin


@functools.lru_cache(maxsize=1)
def _table() -> tuple[dict, str]:
    data, _ = _price_data()
    return {k.lower(): v for k, v in (data.get("prices") or {}).items()}, str(data.get("fetched") or "?")


def reload() -> None:
    """Forget the loaded table (after `llmshift update --prices`)."""
    _price_data.cache_clear()
    _table.cache_clear()


def table_date() -> str:
    return _table()[1]


def table_origin() -> str:
    return _price_data()[1]


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
        cache = {
            field: _number(value[name], f"pricing.{model}.{name}")
            for field, name in (("cache_read", "cached_input"), ("cache_write", "cache_write"))
            if value.get(name) is not None
        }
        out[str(model).strip().lower()] = Price(
            _number(value["input"], f"pricing.{model}.input"), _number(value.get("output", 0), f"pricing.{model}.output"), "config", str(model), **cache
        )
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
            if not float(entry[0]) and not float(entry[1]):
                continue  # 0 / 0 in the price list means unknown there, not free: try the other names
            read, write = (float(x) if _valid_price(x) else None for x in (list(entry[2:4]) + [None, None])[:2])
            return Price(float(entry[0]), float(entry[1]), f"LiteLLM price list, {fetched}", key, read, write)
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


# Prompt-cache tokens. Anthropic reports them beside input_tokens and, where the price list
# has no cache prices, bills them against the input price: writing the cache at 1.25 times it
# (5-minute cache), reading at 0.1 times. OpenAI and Gemini count cached tokens inside their
# input tokens; their parsers keep that share as cached_input_tokens.
CACHE_RATES = {"cache_creation_input_tokens": 1.25, "cache_read_input_tokens": 0.1}
CACHE_FIELDS = tuple(CACHE_RATES)
USAGE_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", *CACHE_FIELDS)


def _count(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return max(0.0, float(value))


def usage_cost(price: Price, usage: dict | None, prompt: Any = None, output: Any = None) -> tuple[float, bool]:
    """What one answer cost in USD, and whether its token counts were estimated (no usage
    reported). Cached input costs the model's cache price; without one in the list, an
    OpenAI or Gemini cached share costs the full input price."""
    usage = usage or {}
    inp, out = _count(usage.get("input_tokens")), _count(usage.get("output_tokens"))
    estimated = False
    cached = read = write = 0.0
    if inp is None:
        inp, estimated = float(estimate_tokens(prompt)), True
    else:
        cached = min(_count(usage.get("cached_input_tokens")) or 0.0, inp)
        read = _count(usage.get("cache_read_input_tokens")) or 0.0
        write = _count(usage.get("cache_creation_input_tokens")) or 0.0
    if out is None:
        out, estimated = float(estimate_tokens(output)), True
    read_price = price.cache_read
    usd = (
        (inp - cached) * price.input
        + cached * (read_price if read_price is not None else price.input)
        + read * (read_price if read_price is not None else price.input * CACHE_RATES["cache_read_input_tokens"])
        + write * (price.cache_write if price.cache_write is not None else price.input * CACHE_RATES["cache_creation_input_tokens"])
        + out * price.output
    ) / 1_000_000
    return usd, estimated


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

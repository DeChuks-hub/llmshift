"""Provider registry: turn a spec like "openai:gpt-5" into a Provider.

Spec forms:
  provider:model         openai:gpt-5, anthropic:claude-sonnet-4-5, gemini:gemini-2.5-flash,
                         azure:<deployment>, ollama:llama3.1:8b, openrouter:meta-llama/llama-3.3-70b,
                         mock:legacy, python:my_app.llm:answer, <custom-name>:model
  bare model id          gpt-4o, claude-3-5-haiku-20241022, gemini-2.0-flash (provider inferred)
"""

from __future__ import annotations

import os
import re
from typing import Any

from .anthropic import AnthropicProvider
from .base import Provider, ProviderError, Response
from .custom import CommandProvider, PythonProvider
from .gemini import GeminiProvider
from .mock import MockProvider
from .openai import AzureOpenAIProvider, OpenAIProvider

TYPES: dict[str, type] = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
    "python": PythonProvider,
    "command": CommandProvider,
}

PRESETS: dict[str, dict] = {
    "openai": {"type": "openai", "base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY"},
    "azure": {"type": "azure"},
    "anthropic": {"type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
    "gemini": {"type": "gemini", "api_key_env": "GEMINI_API_KEY"},
    "openrouter": {"type": "openai", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "together": {"type": "openai", "base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "groq": {"type": "openai", "base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "mistral": {"type": "openai", "base_url": "https://api.mistral.ai/v1", "api_key_env": "MISTRAL_API_KEY"},
    "deepseek": {"type": "openai", "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
    "xai": {"type": "openai", "base_url": "https://api.x.ai/v1", "api_key_env": "XAI_API_KEY"},
    "fireworks": {"type": "openai", "base_url": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "perplexity": {"type": "openai", "base_url": "https://api.perplexity.ai", "api_key_env": "PERPLEXITY_API_KEY"},
    "cohere": {"type": "openai", "base_url": "https://api.cohere.ai/compatibility/v1", "api_key_env": "COHERE_API_KEY"},
    "ollama": {"type": "openai", "base_url": "http://localhost:11434/v1", "api_key_env": None, "requires_key": False},
    "lmstudio": {"type": "openai", "base_url": "http://localhost:1234/v1", "api_key_env": None, "requires_key": False},
    "vllm": {"type": "openai", "base_url": "http://localhost:8000/v1", "api_key_env": "VLLM_API_KEY", "requires_key": False},
    "mock": {"type": "mock"},
    "python": {"type": "python"},
}
ALIASES = {"google": "gemini", "claude": "anthropic", "azure-openai": "azure", "azure_openai": "azure", "vertex": "gemini"}

_INFER_RULES = [
    ("openai", r"^(gpt-|o\d(\b|-)|chatgpt-|text-davinci|davinci|babbage|ft:|text-embedding|codex|computer-use|omni-moderation|text-moderation|dall-e|whisper|tts-)"),
    ("anthropic", r"^claude-"),
    ("gemini", r"^(models/)?(gemini-|gemma-|text-embedding-00|gemini-embedding)"),
    ("mistral", r"^(mistral-|open-mistral|open-mixtral|codestral|ministral|pixtral|magistral|devstral)"),
    ("deepseek", r"^deepseek-"),
    ("xai", r"^grok-"),
    ("cohere", r"^(command|c4ai-)"),
]


def known_providers(config: dict | None) -> set[str]:
    names = set(PRESETS) | set(ALIASES)
    names |= {str(k).lower() for k in ((config or {}).get("providers") or {})}
    return names


def infer_provider(model: str) -> str | None:
    for name, pattern in _INFER_RULES:
        if re.match(pattern, model.strip(), re.I):
            return name
    return None


def _user_config(config: dict | None, *names: str) -> dict | None:
    """The user's settings for the first of `names` configured under `providers:`."""
    providers = (config or {}).get("providers") or {}
    if not isinstance(providers, dict):
        return None
    lowered = {str(k).lower(): v for k, v in providers.items()}
    for name in names:
        if name in lowered:
            value = lowered[name]
            return dict(value) if isinstance(value, dict) else {}
    return None


def parse_spec(spec: str, config: dict | None = None) -> tuple[str | None, str]:
    spec = (spec or "").strip()
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        lower = prefix.lower()
        if lower in known_providers(config):
            # A provider configured under this exact name wins over a built-in alias.
            if _user_config(config, lower) is not None:
                return lower, rest.strip()
            return ALIASES.get(lower, lower), rest.strip()
    return None, spec


def resolve(spec: str, config: dict | None = None) -> Provider:
    """Create a Provider for `spec`. Raises ProviderError with a helpful message."""
    config = config or {}
    name, model = parse_spec(spec, config)
    if not model:
        raise ProviderError(f"model spec {spec!r} has no model name", code="invalid_spec")
    if name is None:
        name = infer_provider(model)
        if name is None:
            raise ProviderError(
                f"cannot tell which provider serves {model!r}; write it as provider:model, e.g. openai:{model} or ollama:{model}",
                code="unknown_provider",
            )
    canonical = ALIASES.get(name, name)
    alias_names = sorted(a for a, target in ALIASES.items() if target == canonical)
    user_cfg = _user_config(config, name, canonical, *alias_names) or {}
    options: dict[str, Any] = dict(PRESETS.get(name) or PRESETS.get(canonical) or {})
    options["timeout"] = config.get("timeout", 120)
    options["max_retries"] = config.get("max_retries", 4)
    options.update(user_cfg)
    if name == "ollama" and os.environ.get("OLLAMA_HOST") and "base_url" not in user_cfg:
        host = os.environ["OLLAMA_HOST"].rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        options["base_url"] = host + "/v1"
    ptype = options.get("type") or canonical
    cls = TYPES.get(ptype)
    if cls is None:
        raise ProviderError(f"provider {name!r} has unknown type {ptype!r}; use one of {', '.join(sorted(TYPES))}", code="invalid_spec")
    return cls(model, name=name, options=options)


__all__ = ["Provider", "ProviderError", "Response", "resolve", "parse_spec", "infer_provider", "TYPES", "PRESETS"]

"""Model lifecycle registry: which models are deprecated or retired, when, and
what replaces them.

The bundled data (data/models.json) was built from the providers' official
deprecation pages. Status is recomputed from the dates every run, so a model
flips to "retired" on its shutdown day without a new release. `llmshift update`
downloads newer data into the user data directory; it is used instead of the
bundled copy when it is valid and its `updated` date is later. Extra or newer
entries can be layered on with --registry FILE or LLMSHIFT_REGISTRY.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .textutil import json_loads, read_text

_ALIAS_NOTE_RE = re.compile(r"^\s*(alias|alternate id)", re.I)
_NOT_BEFORE_RE = re.compile(r"not_before:(\d{4}-\d{2}-\d{2})")
_DATE_SUFFIX_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}|\d{2}-\d{4})$")
_PREFIXES = (
    "openai/", "anthropic/", "google/", "models/", "openrouter/", "azure/", "azure_ai/", "bedrock/", "bedrock_converse/",
    "vertex_ai/", "vertex_ai_beta/", "vertex/", "gemini/", "mistral/", "mistralai/", "cohere/", "xai/", "x-ai/",
    "deepseek/", "deepseek-ai/", "together_ai/", "groq/", "fireworks_ai/", "perplexity/", "anyscale/",
)
# provider:model prefixes (llmshift's, LiteLLM's, pydantic-ai's and LangChain's init_chat_model's)
_SPEC_PREFIXES = (
    "openai-responses:", "openai-chat:", "openai:", "anthropic:", "gemini:", "google-gla:", "google-vertex:", "google_genai:",
    "google_vertexai:", "google:", "vertexai:", "vertex:", "azure_openai:", "azure:", "mistralai:", "mistral:", "cohere:",
    "xai:", "deepseek:", "openrouter:", "bedrock_converse:", "bedrock:", "groq:", "ollama:", "together:", "fireworks:", "huggingface:",
)
_SPEC_PLATFORM = {
    "bedrock:": "bedrock", "bedrock_converse:": "bedrock", "vertex:": "vertex", "vertexai:": "vertex", "google-vertex:": "vertex",
    "google_vertexai:": "vertex", "azure:": "azure", "azure_openai:": "azure",
}
_BEDROCK_RE = re.compile(r"^(?:(?:us|eu|apac|global|us-gov|jp|au|ca)\.)?(anthropic|meta|mistral|cohere|amazon|ai21|deepseek|openai|qwen|writer)\.(.+?)(?:-v\d+(?::\d+)?)?$")
_BEDROCK_CLAUDE_LEGACY = {"claude-v2": "claude-2.0", "claude-v2:1": "claude-2.1", "claude-instant-v1": "claude-instant-1.2", "claude-v1": "claude-1.3"}
_VERTEX_CLAUDE_RE = re.compile(r"^(claude-[\w.-]+?)(?:-v\d)?@(\d{8})$")
# Full resource names: Bedrock ARNs (foundation-model/..., inference-profile/...) and
# Vertex paths (projects/p/locations/l/publishers/google/models/...).
_ARN_RE = re.compile(r"^arn:aws[\w-]*:bedrock:[^/]*/(.+)$")
_RESOURCE_RE = re.compile(r"^(?:projects/[^/]+/locations/[^/]+/)?publishers/[^/]+/models/(.+)$")
_BEDROCK_VARIANT_RE = re.compile(r"^(.+?-v\d+:\d+)(?::[a-z0-9]+)+$")  # ...-v1:0:200k context variants
_CLAUDE_DOT_RE = re.compile(r"^(claude-(?:[a-z]+-)?\d)\.(\d)")  # claude-3.5-sonnet, claude-sonnet-4.5

PLATFORM_PROVIDER = {"vertex": "google-vertex"}
SEVERITY_ORDER = {"critical": 4, "warning": 3, "notice": 2, "info": 1, "ok": 0}
REGISTRY_SCHEMA = 1  # the newest data format this version reads
STATUSES = ("active", "deprecated", "retired")


def today() -> _dt.date:
    return _date(os.environ.get("LLMSHIFT_TODAY")) or _dt.date.today()


_DATE_RE = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})(?!\d)")


def _date(value: Any) -> _dt.date | None:
    """2026-03-01 (or 20260301, or the date of a timestamp), read the same way on every
    Python: date.fromisoformat accepts more forms from 3.11 on."""
    m = _DATE_RE.match(str(value or "").strip())
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def user_data_dir() -> Path:
    """Where `llmshift update` keeps downloaded data: LLMSHIFT_HOME, else the user cache directory."""
    home = os.environ.get("LLMSHIFT_HOME")
    if home:
        return Path(home)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "llmshift"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "llmshift"


def _entry_problem(e: Any) -> str | None:
    if not isinstance(e, dict):
        return "not an object"
    if not isinstance(e.get("id"), str) or not e["id"].strip():
        return "missing id"
    if e.get("status", "active") not in STATUSES:
        return f"unknown status {e.get('status')!r}"
    for key in ("retires", "deprecated"):
        if e.get(key) is not None and _date(e[key]) is None:
            return f"{key} is not a date: {e[key]!r}"
    for key in ("provider", "replacement", "source", "note"):
        if e.get(key) is not None and not isinstance(e[key], str):
            return f"{key} must be text"
    return None


def registry_problems(data: Any, *, min_models: int = 1) -> list[str]:
    """Why a lifecycle registry document cannot be used; empty when it is valid."""
    if not isinstance(data, dict):
        return ["not a JSON object"]
    problems = []
    schema = data.get("schema", REGISTRY_SCHEMA)
    if not isinstance(schema, int) or isinstance(schema, bool):
        problems.append(f"schema must be a number, got {schema!r}")
    elif schema > REGISTRY_SCHEMA:
        problems.append(f"data format {schema} needs a newer llmshift (this one reads format {REGISTRY_SCHEMA}): pip install -U llmshift")
    updated = _date(data.get("updated"))
    if updated is None:
        problems.append("missing or invalid 'updated' date")
    elif updated > today() + _dt.timedelta(days=2):
        problems.append(f"'updated' date {updated.isoformat()} is in the future")
    models = data.get("models")
    if not isinstance(models, list):
        return problems + ["'models' must be a list"]
    if len(models) < min_models:
        problems.append(f"only {len(models)} models (expected at least {min_models})")
    bad = [f"models[{i}]: {why}" for i, e in enumerate(models) for why in [_entry_problem(e)] if why]
    problems.extend(bad[:5])
    if len(bad) > 5:
        problems.append(f"... and {len(bad) - 5} more invalid entries")
    return problems


def bundled_registry() -> dict:
    """The lifecycle data shipped with llmshift. LLMSHIFT_BUNDLED_REGISTRY names a file to
    use instead: the test suite pins its data that way, since the shipped copy is updated
    every day."""
    override = os.environ.get("LLMSHIFT_BUNDLED_REGISTRY")
    if override:
        return json.loads(Path(override).read_text(encoding="utf-8"))
    return json.loads(resources.files("llmshift").joinpath("data/models.json").read_text(encoding="utf-8"))


def cached_registry() -> tuple[dict, Path] | None:
    """The downloaded registry, when present and valid (whatever its date)."""
    path = user_data_dir() / "models.json"
    try:
        data = json_loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        return None
    return (data, path) if not registry_problems(data) else None


def effective_registry() -> tuple[dict, str]:
    """(registry data, where it came from): the downloaded copy when it is newer than
    the bundled one, else the bundled data. A missing or broken download is ignored."""
    data, origin = bundled_registry(), "bundled"
    cached = cached_registry()
    if cached is not None and (_date(cached[0].get("updated")) or _dt.date.min) > (_date(data.get("updated")) or _dt.date.min):
        data, origin = cached[0], str(cached[1])
    return data, origin


@dataclass
class Lookup:
    query: str
    model: str
    provider: str
    status: str  # active | deprecated | retiring | retired | unknown
    severity: str  # critical | warning | notice | info | ok
    message: str
    retires: _dt.date | None = None
    deprecated: _dt.date | None = None
    days_left: int | None = None
    replacement: str | None = None
    alias: bool = False
    fine_tune_of: str | None = None
    platform: str | None = None
    not_before: _dt.date | None = None
    source: str | None = None
    note: str = ""
    other_platforms: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "query": self.query,
            "model": self.model,
            "provider": self.provider,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
        }
        for key in ("retires", "deprecated", "not_before"):
            value = getattr(self, key)
            if value:
                d[key] = value.isoformat()
        for key in ("days_left", "replacement", "fine_tune_of", "platform", "source"):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        if self.alias:
            d["alias"] = True
        if self.other_platforms:
            d["other_platforms"] = self.other_platforms
        return d


class Registry:
    def __init__(self, entries: list[dict], updated: str | None = None, origin: str = "bundled"):
        self.entries = entries
        self.updated = updated
        self.origin = origin  # "bundled" or the path of the data `llmshift update` downloaded
        self.by_id: dict[str, list[dict]] = {}
        for e in entries:
            self.by_id.setdefault(str(e.get("id", "")).lower(), []).append(e)

    @classmethod
    def load(cls, extra: list[str] | None = None) -> "Registry":
        data, origin = effective_registry()
        entries = list(data.get("models") or [])
        updated = data.get("updated")
        paths = list(extra or [])
        env = os.environ.get("LLMSHIFT_REGISTRY")
        if env:
            paths.extend(p for p in env.split(os.pathsep) if p)
        for p in paths:
            try:
                text = read_text(p)  # names the file when it is not text
            except FileNotFoundError:
                raise ValueError(f"registry file not found: {p}") from None
            try:
                raw = json_loads(text)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"registry file {p} is not valid JSON: {exc}") from None
            extra_entries = raw.get("models") if isinstance(raw, dict) else raw
            if not isinstance(extra_entries, list):
                raise ValueError(f"registry file {p} must be a list of entries or {{\"models\": [...]}}")
            for i, e in enumerate(extra_entries):  # a wrong entry would otherwise change nothing, silently
                if not isinstance(e, dict) or not isinstance(e.get("id"), str) or not e["id"].strip():
                    raise ValueError(f"registry file {p}: entry {i + 1} must be an object with a model \"id\", got {json.dumps(e)[:80]}")
            user = [dict(e, _override=True) for e in extra_entries]
            # An entry without a provider replaces every entry with its id; one with a
            # provider replaces only that provider's entry.
            whole_ids = {str(e["id"]).lower() for e in user if not e.get("provider")}
            pairs = {(str(e["id"]).lower(), e.get("provider")) for e in user if e.get("provider")}
            entries = [
                e for e in entries
                if str(e.get("id", "")).lower() not in whole_ids and (str(e.get("id", "")).lower(), e.get("provider")) not in pairs
            ]
            entries.extend(user)
        return cls(entries, updated, origin)

    # ----------------------------------------------------------- normalise

    @staticmethod
    def normalize(raw: str) -> tuple[list[str], str | None, str | None]:
        """Return (candidate ids to try, platform hint, fine-tune base)."""
        s = raw.strip().strip("'\"`").strip()
        platform = None
        lower = s.lower()
        m = _ARN_RE.match(lower)
        if m:
            platform = "bedrock"
            s = s[len(s) - len(m.group(1)) :]
            lower = s.lower()
        m = _RESOURCE_RE.match(lower)
        if m:
            platform = "vertex"
            s = s[len(s) - len(m.group(1)) :]
            lower = s.lower()
        for p in _SPEC_PREFIXES:
            if lower.startswith(p):
                platform = _SPEC_PLATFORM.get(p, platform)
                s = s[len(p) :]
                lower = s.lower()
                break
        for p in _PREFIXES:
            if lower.startswith(p):
                if p.startswith("bedrock"):
                    platform = "bedrock"
                elif p.startswith("vertex"):
                    platform = "vertex"
                elif p.startswith("azure"):
                    platform = "azure"
                elif p == "openrouter/":
                    platform = "openrouter"
                s = s[len(p) :]
                lower = s.lower()
                # OpenRouter-style "openrouter/openai/gpt-4o"
                for q in _PREFIXES:
                    if lower.startswith(q):
                        s = s[len(q) :]
                        lower = s.lower()
                        break
                break
        fine_tune_of = None
        if lower.startswith("ft:"):
            parts = s.split(":")
            if len(parts) >= 2 and parts[1]:
                fine_tune_of = parts[1]
                s = parts[1]
                lower = s.lower()
        elif re.match(r"^[a-z0-9.-]+:ft-", lower):
            fine_tune_of = s.split(":", 1)[0]
            s = fine_tune_of
            lower = s.lower()
        candidates = [lower]
        m = _BEDROCK_VARIANT_RE.match(lower)
        if m:
            lower = m.group(1)
            candidates.append(lower)
        m = _BEDROCK_RE.match(lower)
        if m:
            platform = platform or "bedrock"
            rest = lower.split(m.group(1) + ".", 1)[1]
            base = _BEDROCK_CLAUDE_LEGACY.get(rest) or _BEDROCK_CLAUDE_LEGACY.get(m.group(2), m.group(2))
            candidates.append(base)
        m = _VERTEX_CLAUDE_RE.match(lower)
        if m:
            platform = platform or "vertex"
            candidates.append(f"{m.group(1)}-{m.group(2)}")
        if lower.endswith("-completions"):
            candidates.append(lower[: -len("-completions")])
        for c in list(candidates):
            dotted = _CLAUDE_DOT_RE.sub(r"\1-\2", c)
            if dotted != c:
                candidates.append(dotted)
        seen, out = set(), []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out, platform, fine_tune_of

    # -------------------------------------------------------------- lookup

    def _pick(self, matches: list[dict], platform: str | None) -> tuple[dict, list[dict]]:
        pool = [e for e in matches if e.get("_override")] or matches  # the user's entries win
        want = PLATFORM_PROVIDER.get(platform or "", None)
        if want:
            preferred = [e for e in pool if e.get("provider") == want]
        else:
            preferred = [e for e in pool if e.get("provider") != "google-vertex"]
        chosen = (preferred or pool)[0]
        others = [e for e in matches if e is not chosen]
        return chosen, others

    def _entry(self, model: str, platform: str | None) -> dict | None:
        for cand in self.normalize(model)[0]:
            if cand in self.by_id:
                return self._pick(self.by_id[cand], platform)[0]
        return None

    def final_replacement(self, replacement: str | None, platform: str | None = None) -> str | None:
        """Where to move to: a replacement that is itself retired, retiring or deprecated is
        followed to its own replacement (o1-mini -> o4-mini -> ...), never in a circle."""
        current, seen = replacement, set()
        while current and current.lower() not in seen and len(seen) < 10:
            seen.add(current.lower())
            entry = self._entry(current, platform)
            if entry is None:
                break
            ending = entry.get("status") in ("retired", "deprecated") or _date(entry.get("retires")) is not None
            nxt = entry.get("replacement")
            if not (ending and nxt) or str(nxt).lower() in seen:
                break
            current = str(nxt)
        return current

    def lookup(self, raw: str, *, platform: str | None = None, warn_days: int = 180) -> Lookup:
        candidates, detected_platform, fine_tune_of = self.normalize(raw)
        platform = platform or detected_platform
        entry = None
        others: list[dict] = []
        matched_id = candidates[0] if candidates else raw
        for cand in candidates:
            if cand in self.by_id:
                entry, others = self._pick(self.by_id[cand], platform)
                matched_id = cand
                break
        if entry is not None and entry.get("provider") == "google-vertex" and platform != "vertex":
            # An undated Claude id is Vertex's naming, but also OpenRouter's (anthropic/claude-3.5-sonnet):
            # outside Vertex, Anthropic's own -latest alias says what is meant.
            alias = self.by_id.get(f"{matched_id}-latest")
            native = [e for e in alias or [] if e.get("provider") != "google-vertex"]
            if native:
                entry, others, matched_id = native[0], self.by_id[matched_id] + native[1:], f"{matched_id}-latest"
        if entry is None:
            family = None
            for cand in candidates:
                base = _DATE_SUFFIX_RE.sub("", cand)
                if base != cand and base in self.by_id:
                    family = base
                    break
            msg = "not in the lifecycle registry"
            if family:
                msg = f"unknown snapshot of {family}; not in the lifecycle registry"
            return Lookup(query=raw, model=matched_id, provider=_guess_provider(matched_id), status="unknown", severity="info", message=msg, platform=platform, fine_tune_of=fine_tune_of)
        return self._describe(raw, matched_id, entry, others, platform, fine_tune_of, warn_days)

    def _describe(self, raw: str, model: str, e: dict, others: list[dict], platform: str | None, fine_tune_of: str | None, warn_days: int) -> Lookup:
        now = today()
        retires = _date(e.get("retires"))
        deprecated = _date(e.get("deprecated"))
        note = e.get("note") or ""
        nb = _NOT_BEFORE_RE.search(note)
        not_before = _date(nb.group(1)) if nb else None
        alias = bool(_ALIAS_NOTE_RE.match(note))
        stored = e.get("status", "active")
        listed = e.get("replacement")
        replacement = self.final_replacement(listed, platform) if listed else None
        days_left = (retires - now).days if retires else None
        if retires and retires <= now:
            status, severity = "retired", "critical"
            message = f"retired on {retires.isoformat()}"
        elif retires:
            status = "retiring"
            severity = "warning" if days_left is not None and days_left <= warn_days else "notice"
            message = f"retires {retires.isoformat()} ({days_left} day{'s' if days_left != 1 else ''} left)"
        elif stored == "retired":
            status, severity = "retired", "critical"
            message = "retired (shut down by the provider)"
        elif stored == "deprecated":
            status, severity = "deprecated", "warning"
            message = "deprecated; no shutdown date announced yet"
        else:
            status, severity = "active", "ok"
            message = "active"
            if not_before:
                message += f" (no retirement before {not_before.isoformat()})"
            if alias:
                severity = "info"
                message += "; alias - the model behind it can change without notice, pin a dated snapshot for stable behaviour"
        if replacement and status != "active":
            message += f"; replacement: {replacement}"
            if listed and str(listed).lower() != str(replacement).lower():
                message += f" (the listed replacement, {listed}, is retiring too)"
        if fine_tune_of:
            message = f"fine-tune of {fine_tune_of}: {message}"
        if platform in ("bedrock", "azure") or (platform == "vertex" and e.get("provider") != "google-vertex"):
            label = {"bedrock": "Amazon Bedrock", "azure": "Azure", "vertex": "Vertex AI"}[platform]
            message += f" (dates are for the provider's own API; {label} sets its own schedule)"
        other_platforms = []
        for o in others:
            o_ret = _date(o.get("retires"))
            label = "Vertex AI" if o.get("provider") == "google-vertex" else o.get("provider")
            if o_ret and o_ret <= now:
                other_platforms.append(f"{label}: retired {o_ret.isoformat()}")
            elif o_ret:
                other_platforms.append(f"{label}: retires {o_ret.isoformat()}")
            elif o.get("status") in ("retired", "deprecated"):
                other_platforms.append(f"{label}: {o.get('status')}")
        if other_platforms and severity == "ok":
            severity = "notice"
            message += " (" + "; ".join(other_platforms) + ")"
        return Lookup(
            query=raw,
            model=model,
            provider=e.get("provider") or _guess_provider(model),
            status=status,
            severity=severity,
            message=message,
            retires=retires,
            deprecated=deprecated,
            days_left=days_left,
            replacement=replacement,
            alias=alias,
            fine_tune_of=fine_tune_of,
            platform=platform,
            not_before=not_before,
            source=e.get("source"),
            note=note,
            other_platforms=other_platforms,
        )

    def search(self, query: str = "", provider: str | None = None, status: str | None = None, warn_days: int = 180) -> list[Lookup]:
        q = (query or "").lower()
        out = []
        for e in self.entries:
            if provider and e.get("provider") != provider:
                continue
            if q and q not in str(e.get("id", "")).lower():
                continue
            lk = self._describe(e["id"], str(e["id"]).lower(), e, [], None, None, warn_days)
            if status and lk.status != status:
                continue
            out.append(lk)
        return sorted(out, key=lambda lk: (lk.retires or _dt.date.max, lk.provider, lk.model))


def _guess_provider(model: str) -> str:
    from .providers import infer_provider

    p = infer_provider(model) or "unknown"
    return "google" if p == "gemini" else p

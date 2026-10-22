"""Infer checks from real outputs - the heart of "no hand-written tests".

For each baseline output we record what a correct answer looked like:
its format (JSON schema, labels, markdown structure, language), its facts
(numbers, dates, names, IDs) and its tone. Statistics over all outputs of the
same prompt template sharpen this: keys that are always present, label
vocabularies, and which JSON fields are categorical versus free text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import facts as factlib
from . import tone as tonelib
from .formats import analyze_json, is_refusal, output_kind, structure_signature
from .jsonschema_lite import infer as infer_schema
from .jsonschema_lite import merge as merge_schema
from .textutil import detect_language, word_count

_VOLATILE_WORDS = {
    "score", "scores", "confidence", "prob", "probability", "probabilities", "likelihood", "similarity", "certainty",
    "weight", "weights", "latency", "timestamp", "ts", "created", "updated", "modified", "uuid", "guid", "elapsed",
    "duration", "logprob", "logprobs", "seed", "nonce", "generated", "time", "datetime",
}


def _is_volatile_key(key: str) -> bool:
    """Whole-word match on the key's parts: 'problem_type' is not 'prob'."""
    parts = re.findall(r"[a-z]+|[0-9]+", re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", key).lower())
    return any(p in _VOLATILE_WORDS for p in parts) or (len(parts) > 1 and parts[-1] == "at")  # expires_at, sentAt
_LABELISH_RE = re.compile(r"^[\w\-./:@#+ ]{1,40}$")
MAX_FIELDS = 20


def _norm_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def _child(path: str, key: str) -> str:
    return f"{path}.{key}" if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key) else f"{path}[{key!r}]"


def _leaves(value: Any, path: str = "$", depth: int = 0) -> list[tuple[str, Any]]:
    """Scalar leaves; short scalar lists are kept whole (compared as sets)."""
    out: list[tuple[str, Any]] = []
    if depth > 8:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_leaves(v, _child(path, str(k)), depth + 1))
    elif isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            if len(value) <= 12:
                out.append((path, value))
        elif len(value) <= 5:
            for i, v in enumerate(value):
                out.extend(_leaves(v, f"{path}[{i}]", depth + 1))
    else:
        out.append((path, value))
    return out


@dataclass
class GroupStats:
    size: int = 0
    json_outputs: int = 0
    schema: dict | None = None
    label_outputs: int = 0
    labels: Counter = field(default_factory=Counter)
    field_values: dict = field(default_factory=dict)  # normalised path -> list of str values

    @property
    def json_all(self) -> bool:
        return self.size > 0 and self.json_outputs == self.size

    @property
    def label_vocab(self) -> list[str] | None:
        if self.size >= 4 and self.label_outputs == self.size:
            distinct = len(self.labels)
            # A vocabulary repeats: some label occurs more than once, and there are few of them.
            if 2 <= distinct < self.size and distinct <= min(15, max(2, int(self.size * 0.8))):
                return sorted(self.labels)
        return None

    def field_kind(self, path: str) -> str:
        """'categorical', 'freetext' or 'unknown' for a JSON leaf across the group."""
        values = self.field_values.get(_norm_path(path)) or []
        if len(values) < 3:
            return "unknown"
        distinct = len(set(values)) / len(values)
        avg_words = sum(len(v.split()) for v in values) / len(values)
        if distinct > 0.7 and avg_words >= 2:
            return "freetext"
        if distinct <= 0.5 or len(set(values)) <= 6:
            return "categorical"
        return "unknown"


def group_stats(outputs: list[str]) -> GroupStats:
    stats = GroupStats(size=len(outputs))
    for text in outputs:
        info = analyze_json(text or "")
        value = info["value"]
        if isinstance(value, (dict, list)) and (info["valid"] or info["fenced"]):
            stats.json_outputs += 1
            s = infer_schema(value)
            stats.schema = s if stats.schema is None else merge_schema(stats.schema, s)
            for path, leaf in _leaves(value):
                stats.field_values.setdefault(_norm_path(path), []).append(str(leaf) if not isinstance(leaf, str) else leaf)
            continue
        kind = output_kind(text or "")
        if kind in ("label", "number"):
            stats.label_outputs += 1
            stats.labels[(text or "").strip()] += 1
    return stats


def select_fields(value: Any, group: GroupStats | None = None) -> dict:
    """JSON leaves whose exact value matters (labels, flags, IDs, amounts)."""
    fields: dict = {}
    for path, leaf in _leaves(value):
        if len(fields) >= MAX_FIELDS:
            break
        key = re.split(r"[.\[\]']+", path.strip("$"))
        key = next((k for k in reversed(key) if k and not k.isdigit()), "")
        if _is_volatile_key(key):
            continue
        kind = group.field_kind(path) if group else "unknown"
        if kind == "freetext":
            continue
        if leaf is None or isinstance(leaf, bool) or isinstance(leaf, int):
            fields[path] = leaf
        elif isinstance(leaf, float):
            if not 0.0 < leaf < 1.0:
                fields[path] = leaf
        elif isinstance(leaf, str):
            words = len(leaf.split())
            if kind == "categorical" and len(leaf) <= 80:
                fields[path] = leaf
            elif words <= 1 and _LABELISH_RE.match(leaf):
                fields[path] = leaf
            elif words <= 3 and len(leaf) <= 30 and re.search(r"\d", leaf) and _LABELISH_RE.match(leaf):
                fields[path] = leaf  # dates, amounts, IDs with spaces
        elif isinstance(leaf, list):
            if all(isinstance(v, (str, int, float, bool)) and (not isinstance(v, str) or len(v.split()) <= 3) for v in leaf):
                if kind != "freetext":
                    fields[path] = leaf
    return fields


def infer_checks(
    messages: list[dict],
    params: dict,
    output: str | None,
    tool_calls: list[dict] | None = None,
    group: GroupStats | None = None,
    *,
    judge: bool = False,
) -> list[dict]:
    """Checks that the baseline `output` passes and a regressed answer would fail."""
    text = output or ""
    tool_calls = tool_calls or []
    checks: list[dict] = []
    if tool_calls:
        checks.append({"type": "tool_calls", "expected": [{"name": tc.get("name"), "arguments": tc.get("arguments") if tc.get("arguments") is not None else {}} for tc in tool_calls]})
        if not text.strip():
            return checks
    if not text.strip():
        return checks
    checks.append({"type": "not_empty"})
    if is_refusal(text):
        return checks  # the old model declined; nothing more to pin down
    checks.append({"type": "refusal"})

    rf = params.get("response_format")
    rf_type = rf.get("type") if isinstance(rf, dict) else None
    info = analyze_json(text)
    value = info["value"]
    structured = isinstance(value, (dict, list)) and (info["valid"] or info["fenced"])
    if structured or (rf_type in ("json_object", "json_schema") and isinstance(value, (dict, list))):
        checks.append({"type": "json_valid"} if info["valid"] else {"type": "json_valid", "allow_fences": True})
        schema = None
        if rf_type == "json_schema":
            schema = (rf.get("json_schema") or {}).get("schema")
        if not schema:
            schema = group.schema if (group and group.json_all and group.size >= 3 and group.schema) else infer_schema(value)
        checks.append({"type": "json_schema", "schema": schema})
        fields = select_fields(value, group)
        if fields:
            checks.append({"type": "json_match", "fields": fields})
        return checks

    kind = output_kind(text)
    if kind == "number":
        checks.append({"type": "equals", "value": text.strip(), "numeric": True})
        return checks
    if kind == "label":
        vocab = group.label_vocab if group else None
        if vocab:
            checks.append({"type": "one_of", "values": vocab})
        checks.append({"type": "equals", "value": text.strip()})
        return checks

    words = word_count(text)
    lang, conf = detect_language(text)
    if lang and conf >= 0.6 and words >= 8:
        checks.append({"type": "language", "expected": lang})
    sig = structure_signature(text)
    expect = {k: sig[k] for k in ("headings", "list", "table", "code_block", "bold")}
    if sig["numbered_list"]:
        expect["numbered_list"] = True
    if words >= 12 or any(expect.values()):
        checks.append({"type": "structure", "expect": expect})
    checks.append({"type": "length", "min_words": max(1, int(words * 0.5) - 5), "max_words": int(words * 2.0) + 15, "baseline_words": words})
    found = factlib.extract(text)
    if found:
        fact_check: dict = {"type": "facts", "facts": [factlib.public(f) for f in found]}
        n_hard = sum(1 for f in found if f.get("kind") != "entity")
        if n_hard > 5:
            fact_check["min_hard_recall"] = 0.85
        temperature = params.get("temperature")
        if words > 400 or (isinstance(temperature, (int, float)) and temperature >= 0.9):
            fact_check["severity"] = "warn"
        checks.append(fact_check)
    if words >= tonelib.MIN_WORDS:
        checks.append({"type": "tone"})
    if words >= 3:
        checks.append({"type": "similarity", "metric": "rouge_l", "threshold": 0.2 if words > 30 else 0.15})
    if judge:
        checks.append({"type": "llm_judge"})
    return checks


def merge_checks(existing: list[dict], inferred: list[dict]) -> list[dict]:
    """Keep hand-written/imported checks; add inferred ones of new types."""
    have = {c.get("type") for c in existing}
    return list(existing) + [c for c in inferred if c.get("type") not in have]

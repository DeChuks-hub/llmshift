"""A small JSON Schema implementation: infer a schema from example values,
merge schemas, validate values (with readable error paths) and describe
schemas compactly. Covers the subset used by LLM structured outputs."""

from __future__ import annotations

import math
import re
from typing import Any

MAX_ERRORS = 20


def _bound(schema: dict, key: str) -> float | None:
    """A numeric keyword of the schema; one that is not a finite number is ignored."""
    value = schema.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def infer(value: Any, *, depth: int = 0) -> dict:
    """Infer a permissive schema: types and required keys, nothing stricter."""
    t = type_of(value)
    if t == "null" or depth > 12:
        return {}
    if t == "object":
        props = {str(k): infer(v, depth=depth + 1) for k, v in value.items()}
        return {"type": "object", "properties": props, "required": sorted(props)}
    if t == "array":
        item_schema: dict | None = None
        for item in value[:50]:
            s = infer(item, depth=depth + 1)
            item_schema = s if item_schema is None else merge(item_schema, s)
        schema: dict = {"type": "array"}
        if item_schema is not None:  # {} (a null item: anything goes) must stay, or a merge narrows it
            schema["items"] = item_schema
        return schema
    if t in ("integer", "number"):
        # A baseline 1 and a new 0.97 are both fine numbers; exact values are json_match's job.
        return {"type": "number"}
    return {"type": t}


def json_equal(a: Any, b: Any) -> bool:
    """JSON equality: booleans never equal numbers, 1 == 1.0, containers compared deeply."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or (a != a and b != b)  # NaN equals NaN
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _types(schema: dict) -> set[str]:
    t = schema.get("type")
    if t is None:
        return set()
    return set(t) if isinstance(t, list) else {t}


def merge(a: dict, b: dict) -> dict:
    """Least-general schema accepting both: union of types, intersection of required."""
    if not a or not b:
        return {}
    ta, tb = _types(a), _types(b)
    types = ta | tb
    if "integer" in types and "number" in types:
        types.discard("integer")
    out: dict = {}
    if types:
        out["type"] = sorted(types)[0] if len(types) == 1 else sorted(types)
    if "object" in types:
        pa, pb = a.get("properties", {}), b.get("properties", {})
        props = {}
        for key in sorted(set(pa) | set(pb)):
            if key in pa and key in pb:
                props[key] = merge(pa[key], pb[key])
            else:
                props[key] = pa[key] if key in pa else pb[key]
        out["properties"] = props
        ra = set(a.get("required", [])) if "object" in ta else set()
        rb = set(b.get("required", [])) if "object" in tb else set()
        out["required"] = sorted(ra & rb) if ("object" in ta and "object" in tb) else sorted(ra | rb)
    if "array" in types:  # by presence: {} (anything) is a real answer, not a missing one
        if "items" in a and "items" in b:
            out["items"] = merge(a["items"], b["items"])
        elif "items" in a or "items" in b:
            out["items"] = a["items"] if "items" in a else b["items"]
    return out


# ------------------------------------------------------------- validation


def _resolve_ref(ref: Any, root: dict) -> dict | bool:
    """The schema a local "#/..." reference points to (a schema or true/false); {} otherwise."""
    if not isinstance(ref, str) or not ref.startswith("#"):
        return {}
    node: Any = root
    for part in ref[1:].split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    return node if isinstance(node, (dict, bool)) else {}


def _type_matches(value: Any, t: str) -> bool:
    actual = type_of(value)
    if t == "number":
        return actual in ("integer", "number")
    if t == "integer":
        return actual == "integer" or (actual == "number" and math.isfinite(value) and float(value).is_integer())
    return actual == t


def _fmt(v: Any) -> str:
    s = repr(v) if not isinstance(v, str) else '"' + v + '"'
    return s if len(s) <= 40 else s[:37] + "..."


def validate(value: Any, schema: dict | bool, *, root: dict | None = None, path: str = "$", errors: list | None = None, depth: int = 0) -> list[str]:
    """Validate value against schema. Returns a list of readable error strings."""
    if errors is None:
        errors = []
    if root is None:
        root = schema if isinstance(schema, dict) else {}
    if len(errors) >= MAX_ERRORS or depth > 40:
        return errors
    if schema is True or schema == {} or schema is None:
        return errors
    if schema is False:
        errors.append(f"{path}: no value is allowed here")
        return errors
    if not isinstance(schema, dict):  # not a schema: nothing to check
        return errors
    if "$ref" in schema:
        target = _resolve_ref(schema["$ref"], root)
        validate(value, target, root=root, path=path, errors=errors, depth=depth + 1)

    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        if schema.get("nullable") and "null" not in allowed:
            allowed = list(allowed) + ["null"]
        if not any(_type_matches(value, t) for t in allowed):
            want = " or ".join(allowed)
            errors.append(f"{path}: expected {want}, got {type_of(value)} {_fmt(value)}")
            return errors
    if "enum" in schema and not any(json_equal(value, option) for option in schema["enum"] or []):
        errors.append(f"{path}: {_fmt(value)} is not one of {schema['enum']}")
    if "const" in schema and not json_equal(value, schema["const"]):
        errors.append(f"{path}: expected {_fmt(schema['const'])}, got {_fmt(value)}")

    vt = type_of(value)
    if vt == "number" and not math.isfinite(value):
        errors.append(f"{path}: {value} is not a JSON number")
    elif vt in ("integer", "number"):
        low, high = _bound(schema, "minimum"), _bound(schema, "maximum")
        if low is not None and value < low:
            errors.append(f"{path}: {value} is less than minimum {low}")
        if high is not None and value > high:
            errors.append(f"{path}: {value} is greater than maximum {high}")
        low, high = _bound(schema, "exclusiveMinimum"), _bound(schema, "exclusiveMaximum")
        if low is not None and value <= low:
            errors.append(f"{path}: {value} must be greater than {low}")
        if high is not None and value >= high:
            errors.append(f"{path}: {value} must be less than {high}")
        mult = _bound(schema, "multipleOf")
        if mult is not None and mult > 0 and abs(value / mult - round(value / mult)) > 1e-9:
            errors.append(f"{path}: {value} is not a multiple of {mult}")
    elif vt == "string":
        low, high = _bound(schema, "minLength"), _bound(schema, "maxLength")
        if low is not None and len(value) < low:
            errors.append(f"{path}: string shorter than {schema['minLength']} characters")
        if high is not None and len(value) > high:
            errors.append(f"{path}: string longer than {schema['maxLength']} characters")
        if isinstance(schema.get("pattern"), str):
            try:
                if not re.search(schema["pattern"], value):
                    errors.append(f"{path}: {_fmt(value)} does not match pattern {schema['pattern']!r}")
            except re.error:
                pass
    elif vt == "array":
        low, high = _bound(schema, "minItems"), _bound(schema, "maxItems")
        if low is not None and len(value) < low:
            errors.append(f"{path}: expected at least {schema['minItems']} items, got {len(value)}")
        if high is not None and len(value) > high:
            errors.append(f"{path}: expected at most {schema['maxItems']} items, got {len(value)}")
        if schema.get("uniqueItems"):
            seen: list = []
            for item in value:
                if any(json_equal(item, s) for s in seen):
                    errors.append(f"{path}: items are not unique")
                    break
                seen.append(item)
        prefix = schema.get("prefixItems")
        items = schema.get("items")
        start = 0
        if isinstance(prefix, list):
            for i, sub in enumerate(prefix[: len(value)]):
                validate(value[i], sub, root=root, path=f"{path}[{i}]", errors=errors, depth=depth + 1)
            start = len(prefix)
        if isinstance(items, list):  # draft-04 tuple form
            for i, sub in enumerate(items[: len(value)]):
                validate(value[i], sub, root=root, path=f"{path}[{i}]", errors=errors, depth=depth + 1)
        elif isinstance(items, (dict, bool)):
            for i in range(start, len(value)):
                validate(value[i], items, root=root, path=f"{path}[{i}]", errors=errors, depth=depth + 1)
                if len(errors) >= MAX_ERRORS:
                    break
        if "contains" in schema and not any(not validate(v, schema["contains"], root=root) for v in value):
            errors.append(f"{path}: no item matches the 'contains' schema")
    elif vt == "object":
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        if _bound(schema, "minProperties") is not None and len(value) < schema["minProperties"]:
            errors.append(f"{path}: expected at least {schema['minProperties']} keys")
        if _bound(schema, "maxProperties") is not None and len(value) > schema["maxProperties"]:
            errors.append(f"{path}: expected at most {schema['maxProperties']} keys")
        pattern_props = schema.get("patternProperties") or {}
        additional = schema.get("additionalProperties", True)
        for key, sub_value in value.items():
            sub_path = f"{path}.{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)) else f"{path}[{key!r}]"
            matched = False
            if key in props:
                matched = True
                validate(sub_value, props[key], root=root, path=sub_path, errors=errors, depth=depth + 1)
            for pattern, sub in pattern_props.items():
                try:
                    if re.search(pattern, key):
                        matched = True
                        validate(sub_value, sub, root=root, path=sub_path, errors=errors, depth=depth + 1)
                except re.error:
                    continue
            if not matched:
                if additional is False:
                    errors.append(f"{path}: unexpected key {key!r}")
                elif isinstance(additional, dict):
                    validate(sub_value, additional, root=root, path=sub_path, errors=errors, depth=depth + 1)

    for sub in schema.get("allOf") or []:
        validate(value, sub, root=root, path=path, errors=errors, depth=depth + 1)
    any_of = schema.get("anyOf")
    if any_of:
        if not any(not validate(value, sub, root=root, path=path, depth=depth + 1) for sub in any_of):
            errors.append(f"{path}: value does not match any allowed alternative")
    one_of = schema.get("oneOf")
    if one_of:
        n = sum(1 for sub in one_of if not validate(value, sub, root=root, path=path, depth=depth + 1))
        if n != 1:
            errors.append(f"{path}: value matches {n} alternatives (exactly one required)")
    if "not" in schema and not validate(value, schema["not"], root=root, path=path, depth=depth + 1):
        errors.append(f"{path}: value matches a forbidden schema")
    return errors


def describe(schema: dict, *, depth: int = 0) -> str:
    """Compact human-readable signature, e.g. {name: string, tags: [string]}."""
    if not schema:
        return "any"
    if depth > 4:
        return "..."
    if "$ref" in schema:
        return str(schema["$ref"]).split("/")[-1]
    if "enum" in schema:
        return " | ".join(_fmt(v) for v in schema["enum"][:8])
    types = schema.get("type")
    if isinstance(types, list):
        return " | ".join(describe({**schema, "type": t}, depth=depth) for t in types)
    if types == "object" or "properties" in schema:
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        parts = []
        for key, sub in list(props.items())[:25]:
            mark = "" if key in required else "?"
            parts.append(f"{key}{mark}: {describe(sub, depth=depth + 1)}")
        if len(props) > 25:
            parts.append("...")
        return "{" + ", ".join(parts) + "}"
    if types == "array":
        return "[" + describe(schema.get("items") or {}, depth=depth + 1) + "]"
    return str(types or "any")


def json_paths(value: Any, prefix: str = "$", *, depth: int = 0) -> list[tuple[str, Any]]:
    """All scalar leaves of a JSON value as (path, value) pairs."""
    out: list[tuple[str, Any]] = []
    if depth > 12:
        return out
    if isinstance(value, dict):
        for key, sub in value.items():
            key = str(key)
            sub_path = f"{prefix}.{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else f"{prefix}[{key!r}]"
            out.extend(json_paths(sub, sub_path, depth=depth + 1))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            out.extend(json_paths(sub, f"{prefix}[{i}]", depth=depth + 1))
    else:
        out.append((prefix, value))
    return out


_PATH_TOKEN_RE = re.compile(r"\.([A-Za-z_][\w\-]*)|\[(\d+)\]|\[('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\]")


def get_path(value: Any, path: str) -> tuple[bool, Any]:
    """Resolve a JSON path such as $.a.b[0], $['user-name'], a.b or a plain top-level key."""
    raw = path
    if not path.startswith("$"):
        path = ("$" + path) if path.startswith("[") else ("$." + path if path else "$")
    found, node = _walk(value, path)
    if not found and not raw.startswith(("$", "[")) and isinstance(value, dict) and raw in value:
        return True, value[raw]  # a literal top-level key such as "user.name"
    return found, node


def _walk(value: Any, path: str) -> tuple[bool, Any]:
    pos = 1
    node = value
    while pos < len(path):
        m = _PATH_TOKEN_RE.match(path, pos)
        if not m:
            return False, None
        pos = m.end()
        if m.group(1) is not None:
            if not isinstance(node, dict) or m.group(1) not in node:
                return False, None
            node = node[m.group(1)]
        elif m.group(2) is not None:
            idx = int(m.group(2))
            if not isinstance(node, list) or idx >= len(node):
                return False, None
            node = node[idx]
        else:
            import ast

            key = ast.literal_eval(m.group(3))
            if not isinstance(node, dict) or key not in node:
                return False, None
            node = node[key]
    return True, node

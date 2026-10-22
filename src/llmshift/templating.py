"""`{{ namespace.path }}` templates, compatible with OpenAI Evals graders.

Namespaces are `item` (the test case's variables) and `sample` (the model
output: output_text, output_json, output_tools, choices, model). Paths use dots
and [index] steps: {{ sample.output_tools[0].function.name }}.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z_][\w]*(?:\s*(?:\.\s*[A-Za-z_][\w-]*|\[\s*\d+\s*\]|\[\s*['\"][^'\"]*['\"]\s*\]))*)\s*\}\}")
_STEP_RE = re.compile(r"\.\s*([A-Za-z_][\w-]*)|\[\s*(\d+)\s*\]|\[\s*['\"]([^'\"]*)['\"]\s*\]")


class MissingVariable(KeyError):
    pass


def lookup(path: str, namespaces: dict) -> Any:
    path = re.sub(r"\s*([.\[\]])\s*", r"\1", path.strip())  # "{{ item .name }}" means item.name
    m = re.match(r"\s*([A-Za-z_][\w]*)", path)
    if not m:
        raise MissingVariable(path)
    root = m.group(1)
    if root not in namespaces:
        raise MissingVariable(path)
    node = namespaces[root]
    pos = m.end()
    for step in _STEP_RE.finditer(path, pos):
        if step.start() != pos:
            break
        pos = step.end()
        key = step.group(1) if step.group(1) is not None else step.group(3)
        if key is not None:
            if isinstance(node, str):
                try:
                    node = json.loads(node)
                except ValueError:
                    raise MissingVariable(path) from None
            if not isinstance(node, dict) or key not in node:
                raise MissingVariable(path)
            node = node[key]
        else:
            idx = int(step.group(2))
            if not isinstance(node, list) or idx >= len(node):
                raise MissingVariable(path)
            node = node[idx]
    return node


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def render(template: Any, namespaces: dict, *, strict: bool = False) -> Any:
    """Render templates inside strings (recursively through lists/dicts).

    A string that is exactly one placeholder returns the raw value (so
    `{{item.tags}}` stays a list). Unknown variables render as "" unless strict.
    """
    if isinstance(template, list):
        return [render(t, namespaces, strict=strict) for t in template]
    if isinstance(template, dict):
        return {k: render(v, namespaces, strict=strict) for k, v in template.items()}
    if not isinstance(template, str) or "{{" not in template:
        return template

    whole = _TEMPLATE_RE.fullmatch(template.strip())
    if whole:
        try:
            return lookup(whole.group(1), namespaces)
        except MissingVariable:
            if strict:
                raise
            return ""

    def repl(m: re.Match) -> str:
        try:
            return to_text(lookup(m.group(1), namespaces))
        except MissingVariable:
            if strict:
                raise
            return ""

    return _TEMPLATE_RE.sub(repl, template)


def render_text(template: Any, namespaces: dict) -> str:
    return to_text(render(template, namespaces))


def uses_namespace(template: Any, namespace: str) -> bool:
    if isinstance(template, list):
        return any(uses_namespace(t, namespace) for t in template)
    if isinstance(template, dict):
        return any(uses_namespace(v, namespace) for v in template.values())
    if not isinstance(template, str):
        return False
    return any(m.group(1).split(".")[0].split("[")[0].strip() == namespace for m in _TEMPLATE_RE.finditer(template))

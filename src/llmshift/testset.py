"""Test sets: the files `snapshot` and `import` write and `compare`, `fix` and
`watch` read.

The canonical format is JSON Lines, one test case per line:

    {"id": "c_1a2b3c4d5e", "group": "g_ab12cd34",
     "input": {"messages": [...], "params": {"temperature": 0}},
     "baseline": {"output": "...", "model": "gpt-4o-2024-08-06"},
     "checks": [{"type": "json_valid"}, {"type": "facts", "facts": [...]}],
     "vars": {...}, "tags": [...], "metadata": {...}}

JSON and YAML files with a list of cases (or {"cases": [...]}) are accepted
too, and hand-written cases may use the shorthand keys `prompt`, `system`
and a plain-string `baseline`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import messages as msgs
from .textutil import stable_hash


class TestFileError(ValueError):
    """Raised for unreadable or invalid test files, with file:line context."""

    __test__ = False  # keep pytest from collecting this class


@dataclass
class TestCase:
    id: str
    input: dict
    baseline: dict | None = None
    checks: list[dict] = field(default_factory=list)
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    vars: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    agent: dict | None = None  # a recorded multi-step run: {"steps": [{"text", "tool_calls", "results"}]}

    __test__ = False  # keep pytest from collecting this class

    @property
    def messages(self) -> list[dict]:
        return self.input.get("messages") or []

    @property
    def params(self) -> dict:
        return self.input.get("params") or {}

    @property
    def system_prompt(self) -> str:
        return msgs.system_prompt(self.messages)

    @property
    def user_text(self) -> str:
        return msgs.last_user_text(self.messages)

    @property
    def baseline_output(self) -> str:
        return (self.baseline or {}).get("output") or ""

    @property
    def baseline_tool_calls(self) -> list[dict]:
        return (self.baseline or {}).get("tool_calls") or []

    @property
    def baseline_model(self) -> str | None:
        return (self.baseline or {}).get("model")

    @property
    def has_baseline(self) -> bool:
        b = self.baseline or {}
        return bool(b.get("output")) or bool(b.get("tool_calls"))

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"id": self.id}
        if self.group:
            d["group"] = self.group
        d["input"] = self.input
        if self.baseline:
            d["baseline"] = self.baseline
        d["checks"] = self.checks
        if self.agent:
            d["agent"] = self.agent
        for key in ("vars", "tags", "metadata"):
            value = getattr(self, key)
            if value:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, data: Any, where: str = "") -> "TestCase":
        prefix = f"{where}: " if where else ""
        if not isinstance(data, dict):
            raise TestFileError(f"{prefix}a test case must be an object, got {type(data).__name__}")
        raw_input = data.get("input")
        try:
            if isinstance(raw_input, dict):
                messages = msgs.clean_messages(raw_input.get("messages") if raw_input.get("messages") is not None else raw_input.get("prompt", ""))
                params = dict(raw_input.get("params") or {})
            elif isinstance(raw_input, (list, str)) and raw_input:
                messages = msgs.clean_messages(raw_input)
                params = dict(data.get("params") or {})
            elif data.get("messages") is not None:
                messages = msgs.clean_messages(data["messages"])
                params = dict(data.get("params") or {})
            elif data.get("prompt") is not None:
                messages = [{"role": "user", "content": str(data["prompt"])}]
                params = dict(data.get("params") or {})
            else:
                raise TestFileError(f"{prefix}test case has no 'input' (or 'messages'/'prompt')")
        except ValueError as exc:
            if isinstance(exc, TestFileError):
                raise
            raise TestFileError(f"{prefix}invalid messages: {exc}") from None
        if data.get("system") and not msgs.system_prompt(messages):
            messages.insert(0, {"role": "system", "content": str(data["system"])})
        if not messages:
            raise TestFileError(f"{prefix}test case has no messages")
        baseline = data.get("baseline")
        if isinstance(baseline, str):
            baseline = {"output": baseline}
        elif baseline is not None and not isinstance(baseline, dict):
            raise TestFileError(f"{prefix}'baseline' must be a string or an object")
        checks = data.get("checks") or []
        if not isinstance(checks, list) or not all(isinstance(c, dict) and c.get("type") for c in checks):
            raise TestFileError(f"{prefix}'checks' must be a list of objects with a 'type'")
        inp = {"messages": messages, "params": params}
        case_id = str(data.get("id") or case_id_for(inp))
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        return cls(
            id=case_id,
            input=inp,
            baseline=baseline,
            checks=[dict(c) for c in checks],
            group=data.get("group") or msgs.template_key(messages, params),
            tags=[str(t) for t in tags],
            vars=dict(data.get("vars") or {}),
            metadata=dict(data.get("metadata") or {}),
            agent=_agent_from(data.get("agent"), prefix),
        )


def _agent_from(raw: Any, prefix: str) -> dict | None:
    """Validate and normalise a recorded agent run."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list) or not raw["steps"]:
        raise TestFileError(f"{prefix}'agent' must be an object with a non-empty 'steps' list")
    steps = []
    for i, step in enumerate(raw["steps"]):
        if not isinstance(step, dict):
            raise TestFileError(f"{prefix}agent step {i + 1} must be an object")
        calls = [c for c in (msgs.normalize_tool_call(tc) for tc in step.get("tool_calls") or []) if c]
        results = []
        for r in step.get("results") or []:
            if not isinstance(r, dict):
                raise TestFileError(f"{prefix}agent step {i + 1}: each result must be an object")
            content = r.get("content")
            results.append({"tool_call_id": r.get("tool_call_id"), "name": r.get("name"), "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
        entry: dict = {"text": str(step.get("text") or ""), "tool_calls": calls}
        if results:
            entry["results"] = results
        steps.append(entry)
    out = {k: v for k, v in raw.items() if k != "steps"}
    limit = out.get("max_steps")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise TestFileError(f"{prefix}agent max_steps must be a positive whole number")
    out["steps"] = steps
    return out


def case_id_for(inp: dict) -> str:
    return "c_" + stable_hash(inp, 10)


def _iter_records(path: Path, lines: bool = False) -> Iterable[tuple[Any, str]]:
    suffix = ".jsonl" if lines else path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise TestFileError(f"test file not found: {path}") from None
    except UnicodeDecodeError as exc:
        raise TestFileError(f"{path}: not UTF-8 text ({exc})") from None
    if suffix in (".jsonl", ".ndjson"):
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line), f"{path}:{lineno}"
            except ValueError as exc:
                raise TestFileError(f"{path}:{lineno}: invalid JSON ({exc})") from None
        return
    if suffix in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TestFileError(f"{path}: invalid YAML ({exc})") from None
    else:
        try:
            data = json.loads(text)
        except ValueError as exc:
            if "Extra data" in str(exc):  # one object per line in a .json file: read it as JSON lines
                yield from _iter_records(path, lines=True)
                return
            raise TestFileError(f"{path}: invalid JSON ({exc})") from None
    if isinstance(data, dict):
        data = data.get("cases", data.get("tests"))
    if not isinstance(data, list):
        raise TestFileError(f"{path}: expected a list of test cases (or an object with a 'cases' list)")
    for i, item in enumerate(data):
        yield item, f"{path}[{i}]"


def load_tests(path: str | os.PathLike) -> list[TestCase]:
    p = Path(path)
    if p.is_dir():
        files = sorted(list(p.glob("*.jsonl")) + list(p.glob("*.yaml")) + list(p.glob("*.yml")) + list(p.glob("*.json")))
        if not files:
            raise TestFileError(f"no test files (*.jsonl, *.yaml, *.json) in {p}")
        cases: list[TestCase] = []
        for f in files:
            cases.extend(load_tests(f))
        return _dedupe_ids(cases)
    cases = [TestCase.from_dict(rec, where) for rec, where in _iter_records(p)]
    return _dedupe_ids(cases)


def unique_ids(cases: list[TestCase]) -> list[TestCase]:
    """Return the cases, with shallow copies renamed where IDs collide (inputs untouched)."""
    import copy

    seen: dict[str, int] = {}
    out = []
    for case in cases:
        if case.id in seen:
            seen[case.id] += 1
            dup = copy.copy(case)
            dup.id = f"{case.id}-{seen[case.id]}"
            out.append(dup)
        else:
            seen[case.id] = 1
            out.append(case)
    return out


def _dedupe_ids(cases: list[TestCase]) -> list[TestCase]:
    seen: dict[str, int] = {}
    for case in cases:
        if case.id in seen:
            seen[case.id] += 1
            case.id = f"{case.id}-{seen[case.id]}"
        else:
            seen[case.id] = 1
    return cases


def atomic_write(path: str | os.PathLike, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=p.suffix, dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def dumps_case(case: TestCase) -> str:
    return json.dumps(case.to_dict(), ensure_ascii=False, default=str)


def save_tests(cases: list[TestCase], path: str | os.PathLike) -> None:
    p = Path(path)
    if p.is_dir():
        raise TestFileError(f"{p} is a directory; give a file name such as {p / 'tests.jsonl'}")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        text = yaml.safe_dump({"cases": [c.to_dict() for c in cases]}, allow_unicode=True, sort_keys=False, width=100)
    elif suffix == ".json":
        text = json.dumps({"cases": [c.to_dict() for c in cases]}, ensure_ascii=False, indent=2, default=str) + "\n"
    else:
        text = "".join(dumps_case(c) + "\n" for c in cases)
    atomic_write(p, text)


def append_tests(new: list[TestCase], path: str | os.PathLike, existing: list[TestCase] | None = None) -> None:
    """Add cases to a test file. JSON lines are appended, so the existing lines (and any
    hand edits) stay byte-for-byte as they are; other formats are rewritten."""
    p = Path(path)
    if p.suffix.lower() not in (".jsonl", ".ndjson", "") or not p.is_file():
        save_tests(list(existing if existing is not None else load_tests(p)) + list(new), p)
        return
    if not new:
        return
    data = p.read_bytes()
    prefix = "\n" if data and not data.endswith(b"\n") else ""
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(prefix + "".join(dumps_case(c) + "\n" for c in new))


def group_key(case: TestCase) -> str:
    """The prompt-template group a case belongs to (one key used everywhere)."""
    return case.group or "default"


def groups_of(cases: list[TestCase]) -> dict[str, list[TestCase]]:
    out: dict[str, list[TestCase]] = {}
    for c in cases:
        out.setdefault(group_key(c), []).append(c)
    return out

"""`llmshift upgrade`: move code off a retiring model ("Dependabot for AI models").

It swaps the model id where `scan` found it, renames API settings the new model
needs (OpenAI's `max_tokens` -> `max_completion_tokens` for o-series and GPT-5+),
can apply `llmshift fix` prompt suggestions, runs `compare` on the replacement and
opens a pull request with the results. Without --apply/--pr it only shows the diff.
"""

from __future__ import annotations

import argparse
import ast
import codecs
import difflib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import console as con
from . import github as gh
from .errors import EXIT_FAIL, EXIT_OK, UsageError
from .providers.base import ProviderError
from .textutil import md_fence, parse_python, plural

PROTECTED_DIRS = {".git", ".hg", ".svn", "node_modules", "vendor", "vendors", "third_party", "third-party", "site-packages", ".venv", "venv", "dist", "build", ".llmshift"}
CONFIG_NAMES = {"llmshift.yaml", "llmshift.yml", ".llmshift.yaml", ".llmshift.yml"}
JS_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"}
TEXT_PROMPT_SUFFIXES = {".txt", ".md", ".prompt", ".j2", ".jinja", ".jinja2"}
_COMPLETION_TOKENS_RE = re.compile(r"^(?:o\d+(?:-|$)|gpt-(?:[5-9]|[1-9]\d)(?:[.\-]|$))", re.I)
_PREFIX_RE = re.compile(r"^(?P<prefix>(?:[a-z0-9_.-]+[/:])+)(?P<id>.+)$", re.I)
_SAFE_PREFIXES = ("openai/", "anthropic/", "google/", "gemini/", "models/", "mistral/", "mistralai/", "deepseek/", "xai/", "x-ai/", "openrouter/", "azure/", "openai:", "anthropic:", "gemini:", "google:", "mistral:", "deepseek:", "xai:", "openrouter:", "azure:")
BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
MADE_BY = "Made by llmshift upgrade."  # in every commit llmshift pushes: its branch may be overwritten only when the tip has it


def needs_completion_tokens(model: str) -> bool:
    """OpenAI o-series and GPT-5+ reject `max_tokens` in favour of `max_completion_tokens`."""
    return bool(_COMPLETION_TOKENS_RE.match(_bare(model)))


def _bare(model: str) -> str:
    m = _PREFIX_RE.match(model or "")
    return m.group("id") if m else (model or "")


def code_id(spec: str) -> str:
    """A model id as code writes it: without llmshift's provider (openai:gpt-4o) or a
    router's vendor prefix (openai/gpt-4o); code that has a prefix keeps its own."""
    from .providers import parse_spec

    model = parse_spec(spec)[1]
    m = _PREFIX_RE.match(model)
    return m.group("id") if m and m.group("prefix").lower().startswith(_SAFE_PREFIXES) else model


# An id after one of these names a resource you created (an Azure deployment), not a model.
_DEPLOYMENT_RE = re.compile(r"deployment(?:_name|_id)?[\"']?\s*[:=]\s*[\"']?\Z", re.I)
# engine= too, but only in Azure code: the pre-1.0 OpenAI SDK named the model engine=.
_ENGINE_RE = re.compile(r"engine[\"']?\s*[:=]\s*[\"']?\Z", re.I)


# ----------------------------------------------------------------- data


@dataclass
class Swap:
    old: str
    new: str
    status: str = "unknown"
    retires: str | None = None
    message: str = ""
    locations: list[str] = field(default_factory=list)
    spec: str = ""  # the replacement as given (openai:gpt-5.6-sol), for running tests on it


@dataclass
class Edit:
    line: int  # 1-based
    col: int  # 0-based character column
    old: str  # the text expected at col ("" inserts)
    new: str
    reason: str  # model | max_completion_tokens | prompt


@dataclass
class FileChange:
    path: str  # relative to the repository root, with /
    original: bytes
    edits: list[Edit] = field(default_factory=list)
    updated: bytes | None = None

    def text(self, data: bytes) -> str:
        decoded = _decode_exact(data)
        return decoded[0] if decoded else data.decode("utf-8", "replace")

    @property
    def diff(self) -> str:
        if self.updated is None or self.updated == self.original:
            return ""
        a = _lines(self.text(self.original))  # split as git does: \f or   are not line ends
        b = _lines(self.text(self.updated))
        out = []
        for line in difflib.unified_diff(a, b, f"a/{self.path}", f"b/{self.path}"):
            out.append(line if line.endswith(("\n", "\r")) else line + "\n\\ No newline at end of file\n")
        return "".join(out)


@dataclass
class UpgradePlan:
    root: Path
    swaps: list[Swap] = field(default_factory=list)
    files: dict[str, FileChange] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    prompt_fixes: list[dict] = field(default_factory=list)
    compare: dict[str, Any] = field(default_factory=dict)  # replacement model -> CompareResult

    @property
    def changed(self) -> list[FileChange]:
        return [f for f in self.files.values() if f.updated is not None and f.updated != f.original]

    @property
    def diff(self) -> str:
        return "".join(f.diff for f in self.changed)

    def to_dict(self) -> dict:
        return {
            "swaps": [s.__dict__ for s in self.swaps],
            "files": {f.path: [e.__dict__ for e in f.edits] for f in self.changed},
            "notes": self.notes,
            "prompt_fixes": self.prompt_fixes,
            "compare": {k: v.summary for k, v in self.compare.items()},
            "diff": self.diff,
        }


# ------------------------------------------------------------- encoding


def _decode_exact(data: bytes) -> tuple[str, str, bytes] | None:
    """(text, codec, bom) when the bytes round-trip exactly; None when rewriting could corrupt them."""
    for bom, codec in ((codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")):
        if data.startswith(bom):
            body = data[len(bom):]
            try:
                text = body.decode(codec)
            except UnicodeDecodeError:
                return None
            return (text, codec, bom) if text.encode(codec) == body else None
    if b"\0" in data[:4096]:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text, "utf-8", b""


_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\n|\r|$)")


def _lines(text: str) -> list[str]:
    """Lines with their endings, split only at \\n, \\r\\n and \\r (as Python and git count them)."""
    out = [m.group(0) for m in _LINE_RE.finditer(text)]
    if out and out[-1] == "":
        out.pop()
    return out


def _body(line: str) -> str:
    return line.rstrip("\r\n")


def _char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", "replace"))


def apply_edits(data: bytes, edits: list[Edit]) -> tuple[bytes | None, list[str]]:
    """Apply edits; returns (new bytes or None, problems)."""
    decoded = _decode_exact(data)
    if decoded is None:
        return None, ["not UTF-8/UTF-16 text that can be rewritten byte for byte"]
    text, codec, bom = decoded
    lines = _lines(text)
    problems = []
    by_line: dict[int, list[Edit]] = {}
    for e in edits:
        by_line.setdefault(e.line, []).append(e)
    for number, group in by_line.items():
        if not 1 <= number <= len(lines):
            problems.append(f"line {number} does not exist")
            continue
        line = lines[number - 1]
        body, ending = _body(line), line[len(_body(line)):]
        seen = set()
        for e in sorted(group, key=lambda e: e.col, reverse=True):
            if (e.col, e.old) in seen:
                continue
            seen.add((e.col, e.old))
            if body[e.col:e.col + len(e.old)] != e.old:
                problems.append(f"line {number}: expected {e.old!r} at column {e.col + 1}")
                continue
            body = body[: e.col] + e.new + body[e.col + len(e.old):]
        lines[number - 1] = body + ending
    return bom + "".join(lines).encode(codec), problems


# -------------------------------------------------------------- planning


def _bounded(line: str, model: str) -> list[int]:
    """Columns where `model` appears as a whole id (not inside a longer id, a fine-tune or a URL path)."""
    pattern = re.compile(r"(?<![A-Za-z0-9_.@-])" + re.escape(model) + r"(?![A-Za-z0-9_.@:-])", re.I)
    return [m.start() for m in pattern.finditer(line)]


def _matches(found: str, old: str) -> str | None:
    """The part of a found id to replace: the id itself, or after a vendor prefix (openai/gpt-4o)."""
    if found.lower() == old.lower():
        return found
    m = _PREFIX_RE.match(found)
    if m and m.group("id").lower() == old.lower() and m.group("prefix").lower().startswith(_SAFE_PREFIXES):
        return m.group("id")
    return None


def _protected(rel: str) -> str | None:
    parts = Path(rel).parts
    if any(p in PROTECTED_DIRS for p in parts[:-1]):
        return "vendored or generated folder"
    if Path(rel).name.lower() in CONFIG_NAMES:
        return "llmshift's own configuration (update baseline/candidate yourself)"
    return None


def _resolve_swaps(scan: Any, registry: Any, old: str | None, new: str | None, auto: bool, warn_days: int) -> tuple[list[Swap], list[str]]:
    from .providers import infer_provider
    from .scanner import looks_like_known_model

    notes: list[str] = []
    swaps: list[Swap] = []
    if old:
        old_id = code_id(old)
        lk = registry.lookup(old_id, warn_days=warn_days)
        target = new or lk.replacement
        if not target:
            raise UsageError(f"no known replacement for {old}; pass --to MODEL")
        swaps.append(Swap(old_id, code_id(target), lk.status, lk.retires.isoformat() if lk.retires else None, lk.message, spec=target))
    elif auto:
        seen: set[str] = set()
        for m in scan.models():
            lk = m["lookup"]
            if lk is None or not lk.replacement:
                continue
            due = lk.status == "retired" or (lk.status == "retiring" and lk.days_left is not None and lk.days_left <= warn_days)
            if not due:
                continue
            if lk.fine_tune_of or lk.platform in ("bedrock", "vertex", "azure"):
                notes.append(f"{m['model']}: {lk.platform or 'fine-tuned'} ids are platform-specific; upgrade it by hand (replacement: {lk.replacement})")
                continue
            prefixed = _PREFIX_RE.match(m["model"])
            bare = prefixed.group("id") if prefixed and prefixed.group("prefix").lower().startswith(_SAFE_PREFIXES) else m["model"]
            if bare.lower() in seen:  # openai/gpt-4o and gpt-4o are one swap; the prefix is kept
                continue
            seen.add(bare.lower())
            swaps.append(Swap(bare, lk.replacement, lk.status, lk.retires.isoformat() if lk.retires else None, lk.message))
    else:
        raise UsageError("which model? Pass --from MODEL [--to MODEL], or --auto for every retiring model found")
    checked = []
    for s in swaps:
        if not (looks_like_known_model(s.new) or re.fullmatch(r"[A-Za-z0-9][\w.:/@-]{1,100}", s.new)):
            notes.append(f"{s.old}: the registry's replacement {s.new!r} is not a single model id; pass --from {s.old} --to MODEL")
            continue
        a, b = infer_provider(_bare(s.old)), infer_provider(_bare(s.new))
        if a and b and a != b:
            notes.append(f"{s.old} -> {s.new}: different providers ({a} -> {b}); the calling code must change too. Re-run with --force to swap the ids anyway")
            s.status = s.status + "|cross-provider"
        checked.append(s)
    return checked, notes


def plan_upgrade(
    paths: list[str] | None = None,
    *,
    old: str | None = None,
    new: str | None = None,
    auto: bool = False,
    root: str | os.PathLike | None = None,
    registry: Any = None,
    warn_days: int = 180,
    exclude: list[str] | tuple = (),
    force: bool = False,
) -> UpgradePlan:
    """Work out every edit; nothing is written."""
    from .registry import Registry
    from .scanner import platform_hint, scan_paths

    cwd = Path(os.getcwd()).resolve()
    top = Path(root).resolve() if root else (gh.repo_root(cwd) or cwd)
    registry = registry or Registry.load()
    scan = scan_paths(list(paths or ["."]), registry=registry, exclude=exclude, warn_days=warn_days)
    swaps, notes = _resolve_swaps(scan, registry, old, new, auto, warn_days)
    plan = UpgradePlan(root=top, notes=notes)
    built_on: list[tuple[Any, Swap]] = []  # ids that contain a swapped id: noted unless a swap changes them too
    for swap in swaps:
        if swap.status.endswith("|cross-provider"):
            swap.status = swap.status.split("|")[0]
            if not force:
                continue
        plan.swaps.append(swap)
        for f in scan.findings:
            target = _matches(f.model, swap.old)
            if target is None:
                if f.model.lower() != swap.old.lower() and swap.old.lower() in f.model.lower():
                    built_on.append((f, swap))
                continue
            if f.via:
                continue  # the constant's own line is a finding too
            path = (cwd / f.file).resolve() if not Path(f.file).is_absolute() else Path(f.file).resolve()
            try:
                rel = path.relative_to(top).as_posix()
            except ValueError:
                plan.notes.append(f"{f.location}: outside the repository; not changed")
                continue
            why = _protected(rel)
            if why:
                plan.notes.append(f"{rel}:{f.line}: {why}; not changed")
                continue
            line_no = f.file_line if f.cell is not None else f.line
            if not line_no:
                plan.notes.append(f"{f.location}: line in the notebook file unknown; not changed")
                continue
            change = plan.files.get(rel)
            if change is None:
                try:
                    change = plan.files[rel] = FileChange(rel, path.read_bytes())
                except OSError as exc:
                    plan.notes.append(f"{rel}: cannot read ({exc.strerror or exc})")
                    continue
            decoded = _decode_exact(change.original)
            lines = _lines(decoded[0]) if decoded else []
            if not decoded or not 1 <= line_no <= len(lines):
                plan.notes.append(f"{rel}:{line_no}: cannot be rewritten safely (encoding)")
                continue
            body = _body(lines[line_no - 1])
            cols = _bounded(body, target)
            if not cols:
                plan.notes.append(f"{rel}:{line_no}: {swap.old} not found on the line; not changed")
                continue
            azure = platform_hint(decoded[0]) == "azure"
            deployments = [col for col in cols if _DEPLOYMENT_RE.search(body[:col]) or (azure and _ENGINE_RE.search(body[:col]))]
            if deployments:
                plan.notes.append(f"{rel}:{line_no}: {target} names an Azure deployment; create a {swap.new} deployment, then rename it here; not changed")
                cols = [col for col in cols if col not in deployments]
                if not cols:
                    continue
            for col in cols:
                change.edits.append(Edit(line_no, col, body[col:col + len(target)], swap.new, "model"))
            loc = f"{rel}:{line_no}"
            if loc not in swap.locations:
                swap.locations.append(loc)
    swapped = {f.model.lower() for f in scan.findings for s in plan.swaps if _matches(f.model, s.old) is not None}
    for f, swap in built_on:
        if f.model.lower() not in swapped:
            plan.notes.append(f"{f.location}: {f.model} is a fine-tuned or platform-specific id built on {swap.old}; not changed")
    needing = [s for s in plan.swaps if needs_completion_tokens(s.new) and not needs_completion_tokens(s.old)]
    for swap in needing:
        for change in plan.files.values():
            if any(e.reason == "model" and e.new == swap.new for e in change.edits):
                _rename_max_tokens(change, swap, plan.notes)
    if needing:
        left = _max_tokens_left(list(paths or ["."]), plan)
        if left:
            shown = ", ".join(left[:8]) + (f" and {len(left) - 8} more" if len(left) > 8 else "")
            plan.notes.append(
                f"max_tokens is still set at {shown}: {', '.join(sorted({s.new for s in needing}))} rejects it (it takes max_completion_tokens). "
                "llmshift renames it only in calls that name the model; check these calls by hand"
            )
    _finish(plan)
    if not plan.swaps and not plan.notes:
        plan.notes.append("no retiring models with a known replacement were found")
    return plan


def _parses(data: bytes) -> bool:
    decoded = _decode_exact(data)
    return decoded is not None and parse_python(decoded[0]) is not None


def _finish(plan: UpgradePlan) -> None:
    for change in plan.files.values():
        if not change.edits:
            continue
        updated, problems = apply_edits(change.original, change.edits)
        for p in problems:
            plan.notes.append(f"{change.path}: {p}")
        # Python files must still parse; one this Python cannot parse to begin with (newer
        # syntax) cannot be checked, and an id inside a string literal is safe to swap.
        if updated is not None and change.path.endswith((".py", ".pyi")) and _parses(change.original) and not _parses(updated):
            plan.notes.append(f"{change.path}: the edit would not parse; left unchanged")
            updated = None
        change.updated = updated


# ------------------------------------------------- max_tokens renames


def _max_tokens_left(paths: list[str], plan: UpgradePlan) -> list[str]:
    """file:line of every `max_tokens` left in OpenAI-using Python and JS/TS files: a
    model set in one file and used in another is swapped, but its calls are not renamed."""
    from .scanner import iter_source_files

    renamed = {(c.path, e.line) for c in plan.files.values() for e in c.edits if e.reason == "max_completion_tokens"}
    try:
        files = [f for _root, f in iter_source_files(paths)]
    except FileNotFoundError:
        return []
    out = []
    for path in files:
        if path.suffix.lower() not in (".py", ".pyi") and path.suffix.lower() not in JS_SUFFIXES:
            continue
        try:
            rel = path.resolve().relative_to(plan.root).as_posix()
        except ValueError:
            continue
        data = _read(path)
        if _protected(rel) or data is None or b"max_tokens" not in data:
            continue
        decoded = _decode_exact(data)
        if decoded is None or not re.search(r"openai|chat\.completions|responses\.create", decoded[0], re.I):
            continue  # Anthropic and others require max_tokens
        for n, line in enumerate(_lines(decoded[0]), 1):
            if re.search(r"(?<![\w$])max_tokens\b", line) and (rel, n) not in renamed:
                out.append(f"{rel}:{n}")
    return out


def _rename_max_tokens(change: FileChange, swap: Swap, notes: list[str]) -> None:
    decoded = _decode_exact(change.original)
    if decoded is None:
        return
    text = decoded[0]
    suffix = Path(change.path).suffix.lower()
    if suffix in (".py", ".pyi"):
        _rename_python(change, text, swap, notes)
    elif suffix in JS_SUFFIXES:
        _rename_js(change, text, swap)


def _rename_python(change: FileChange, text: str, swap: Swap, notes: list[str]) -> None:
    tree = parse_python(text)
    if tree is None:
        return
    lines = _lines(text)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    constants[t.id] = node.value.value

    def is_old(value: ast.AST) -> bool:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return _matches(value.value, swap.old) is not None
        if isinstance(value, ast.Name):
            return _matches(constants.get(value.id, ""), swap.old) is not None
        if isinstance(value, ast.Call) and len(value.args) >= 2:  # os.getenv("X", "old")
            return is_old(value.args[1])
        if isinstance(value, ast.BoolOp):  # os.getenv("X") or "old"
            return any(is_old(v) for v in value.values)
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kws = {k.arg: k for k in node.keywords if k.arg}
        if "model" not in kws or not is_old(kws["model"].value):
            continue
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if "temperature" in kws:
            notes.append(f"{change.path}:{kws['temperature'].lineno}: {swap.new} may only accept the default temperature; check the temperature setting")
        kw = kws.get("max_tokens")
        if kw is None:
            continue
        if not name.endswith(".create") or "messages" not in kws:
            notes.append(f"{change.path}:{kw.lineno}: check max_tokens for {swap.new} (not an OpenAI chat call that llmshift can rename safely)")
            continue
        line = _body(lines[kw.lineno - 1])
        col = _char_col(line, kw.col_offset)
        if line[col:col + len("max_tokens")] == "max_tokens":
            change.edits.append(Edit(kw.lineno, col, "max_tokens", "max_completion_tokens", "max_completion_tokens"))


def _js_structure(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(brace pairs, string spans) of JS/TS source, skipping strings and comments."""
    pairs, strings, stack = [], [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'`":
            start = i
            j = i + 1
            while j < n and text[j] != ch and (ch == "`" or text[j] != "\n"):
                j += 2 if text[j] == "\\" else 1
            if j < n and text[j] == ch:  # a quote that never closes on its line (JSX text: don't) is not a string
                strings.append((start, j))
                i = j
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 1
        elif ch in "{[(":
            stack.append((ch, i))
        elif ch in "}])" and stack:
            open_ch, start = stack.pop()
            if open_ch == "{" and ch == "}":
                pairs.append((start, i))
        i += 1
    return pairs, strings


def _rename_js(change: FileChange, text: str, swap: Swap) -> None:
    lines = _lines(text)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    pairs, strings = _js_structure(text)
    for e in [e for e in change.edits if e.reason == "model" and e.new == swap.new]:
        at = offsets[e.line - 1] + e.col
        enclosing = [p for p in pairs if p[0] < at < p[1]]
        if not enclosing:
            continue
        start, end = min(enclosing, key=lambda p: p[1] - p[0])
        nested = [p for p in pairs if start < p[0] and p[1] < end]
        for m in re.finditer(r"(?<![\w$])([\"']?)max_tokens\1\s*:", text[start:end]):
            key_at = start + m.start() + len(m.group(1))
            if any(a < key_at < b for a, b in nested) or any(a < key_at < b for a, b in strings if a != start + m.start()):
                continue
            line_no = max(i for i, off in enumerate(offsets) if off <= key_at) + 1
            col = key_at - offsets[line_no - 1]
            change.edits.append(Edit(line_no, col, "max_tokens", "max_completion_tokens", "max_completion_tokens"))


# ------------------------------------------------------------ prompt fixes


def apply_prompt_fixes(plan: UpgradePlan, fix: Any, paths: list[str] | None = None) -> None:
    """Append `llmshift fix` instructions to prompts written as plain string literals
    (or prompt files); anything else is listed for the pull request."""
    from .scanner import iter_source_files

    files: list[Path] = []
    try:
        files = [f for _root, f in iter_source_files(list(paths or ["."]), include_docs=True)]
    except FileNotFoundError:
        pass
    for g in getattr(fix, "groups", []):
        if not g.suggested_prompt or not g.improved:
            continue
        entry = {"group": g.group, "added": list(g.added), "applied": [], "suggested_prompt": g.suggested_prompt}
        plan.prompt_fixes.append(entry)
        if not g.suggested_prompt.startswith(g.original_prompt):
            entry["why"] = "the suggestion rewrites the prompt; apply it by hand"
            continue
        addition = g.suggested_prompt[len(g.original_prompt):]
        for path in files:
            try:
                rel = path.resolve().relative_to(plan.root).as_posix()
            except ValueError:
                continue
            if _protected(rel):
                continue
            change = plan.files.get(rel)
            data = change.original if change else _read(path)
            if data is None or g.original_prompt.strip()[:16] not in data.decode("utf-8", "replace"):
                continue  # cheap filter before parsing the file
            edits = _prompt_edits(rel, data, g.original_prompt, addition)
            if not edits:
                continue
            if change is None:
                change = plan.files[rel] = FileChange(rel, data)
            change.edits.extend(edits)
            entry["applied"].append(rel)
        if not entry["applied"]:
            entry["why"] = "prompt not found as a plain string literal or prompt file"
    _finish(plan)


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes() if path.stat().st_size <= 2_000_000 else None
    except OSError:
        return None


def _prompt_edits(rel: str, data: bytes, prompt: str, addition: str) -> list[Edit]:
    decoded = _decode_exact(data)
    if decoded is None:
        return []
    text = decoded[0]
    lines = _lines(text)
    suffix = Path(rel).suffix.lower()
    if suffix in TEXT_PROMPT_SUFFIXES:
        if text.strip() != prompt.strip() or not lines:
            return []
        last = len(lines)
        while last > 1 and not _body(lines[last - 1]).strip():
            last -= 1
        return [Edit(last, len(_body(lines[last - 1])), "", addition.replace("\n", _ending(lines)), "prompt")]
    if suffix not in (".py", ".pyi"):
        return []
    tree = parse_python(text)
    if tree is None:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == prompt and getattr(node, "end_lineno", None):
            line = _body(lines[node.end_lineno - 1])
            col = _char_col(line, node.end_col_offset)
            edit = Edit(node.end_lineno, col, "", " " + repr(addition), "prompt")
            updated, problems = apply_edits(data, [edit])
            # The literal must now hold exactly the suggested prompt.
            after = parse_python(_decode_exact(updated)[0]) if updated is not None and not problems and _decode_exact(updated) else None
            if after is not None and any(isinstance(n, ast.Constant) and n.value == prompt + addition for n in ast.walk(after)):
                out.append(edit)
    return out


def _ending(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def apply_plan(plan: UpgradePlan) -> list[str]:
    """Write the planned edits; returns the files written (relative paths). Every file is
    checked first, so a read-only one stops the upgrade before anything is changed."""
    targets = [(change, plan.root / change.path) for change in plan.changed]
    blocked = [change.path for change, path in targets if not path.is_file() or not os.access(path, os.W_OK)]
    if blocked:
        raise UsageError(f"cannot write {', '.join(blocked)} (read-only or missing); nothing was changed")
    written: list[str] = []
    for change, path in targets:
        tmp = path.with_name(f".{path.name}.llmshift-tmp")
        try:
            tmp.write_bytes(change.updated or b"")
            shutil.copymode(path, tmp)  # an executable script stays executable
            os.replace(tmp, path)
        except OSError as exc:
            done = f"; already updated: {', '.join(written)}" if written else "; nothing was changed"
            raise UsageError(f"could not write {change.path} ({exc.strerror or exc}){done}") from None
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        written.append(change.path)
    return written


# ------------------------------------------------------------ reporting


def _swap_line(s: Swap) -> str:
    when = f" on {s.retires}" if s.retires else ""
    return f"{s.old} -> {s.new} ({s.status}{when})"


def plan_markdown(plan: UpgradePlan, min_pass_rate: float | None = None) -> str:
    lines = ["## LLMShift: move off retiring models", ""]
    if plan.swaps:
        lines += ["| Model | Status | Retires | Replacement | Where |", "|---|---|---|---|---|"]
        for s in plan.swaps:
            lines.append(f"| `{s.old}` | {s.status} | {s.retires or '-'} | `{s.new}` | {', '.join(f'`{x}`' for x in s.locations[:6]) or '-'} |")
        lines.append("")
    renamed = [(f.path, e.line) for f in plan.changed for e in f.edits if e.reason == "max_completion_tokens"]
    if renamed:
        lines.append("`max_tokens` was renamed to `max_completion_tokens` (required by the new model) at " + ", ".join(f"`{p}:{n}`" for p, n in renamed) + ".")
        lines.append("")
    for model, result in plan.compare.items():
        s = result.summary
        verdict = ""
        if min_pass_rate is not None:
            verdict = " - passes" if s.get("pass_rate", 0) + 1e-9 >= min_pass_rate else f" - **below the {min_pass_rate:.0%} threshold**"
        lines.append(f"### Tests on `{model}`")
        lines.append(f"Pass rate **{s.get('pass_rate', 0):.0%}** over {s.get('total', 0)} recorded cases ({s.get('failed', 0)} failing, {s.get('warned', 0)} changed){verdict}.")
        for cat, info in (s.get("by_category") or {}).items():
            pats = "; ".join(f"{p} ({n})" for p, n in list((info.get("patterns") or {}).items())[:3])
            if pats:
                lines.append(f"- {cat}: {pats}")
        lines.append("")
    if plan.prompt_fixes:
        lines.append("### Prompt changes suggested by `llmshift fix`")
        for fx in plan.prompt_fixes:
            where = ", ".join(f"`{p}`" for p in fx["applied"]) if fx["applied"] else f"not applied ({fx.get('why', '')})"
            lines.append(f"- `{fx['group']}`: {where}")
            for added in fx["added"][:6]:
                lines.append(f"  - {added}")
        lines.append("")
    if plan.notes:
        lines.append("### Notes")
        lines.extend(f"- {n}" for n in plan.notes)
        lines.append("")
    diff = plan.diff
    if diff:
        lines.append("<details><summary>Diff</summary>")
        lines.append("")
        lines.append(md_fence(diff[:40000].rstrip(), "diff"))
        lines.append("</details>")
        lines.append("")
    lines.append("_Opened by `llmshift upgrade`. Review the diff and the test results before merging._")
    return "\n".join(lines) + "\n"


def _branch_name(plan: UpgradePlan, requested: str | None) -> str:
    raw = requested or ("llmshift/upgrade-" + "-".join(f"{s.old}-to-{s.new}" for s in plan.swaps[:2]) + ("-and-more" if len(plan.swaps) > 2 else ""))
    parts = []
    for part in raw.split("/"):  # git ref rules, one path component at a time
        part = re.sub(r"[^A-Za-z0-9._-]+", "-", part)
        part = re.sub(r"-{2,}", "-", re.sub(r"\.{2,}", ".", part)).strip(".-")
        if part.endswith(".lock"):
            part = part[: -len(".lock")] + "-lock"
        if part:
            parts.append(part)
    return "/".join(parts)[:120].strip("/.-") or "llmshift/upgrade"


# ----------------------------------------------------------------- CLI


def register(sub: Any) -> None:
    from . import cli

    p = sub.add_parser(
        "upgrade",
        help="swap a retiring model in your code and open a pull request",
        description="Swap a retiring model id where scan finds it, rename settings the replacement needs, and optionally "
        "test the replacement and open a pull request. Without --apply or --pr, only the diff is shown.",
    )
    p.add_argument("paths", nargs="*", help="where to look (default: scan.paths, else .)")
    p.add_argument("--from", dest="old", metavar="MODEL", help="the model to move off")
    p.add_argument("--to", dest="new", metavar="MODEL", help="its replacement (default: the lifecycle registry's)")
    p.add_argument("--auto", action="store_true", help="every retired or retiring model found that has a known replacement")
    p.add_argument("--apply", action="store_true", help="write the edits (default: show the diff only)")
    p.add_argument("--compare", action="store_true", help="run the test set on the replacement and include the results")
    p.add_argument("--compare-with", metavar="MODEL", help="run the comparison on this model instead (e.g. a proxy route)")
    p.add_argument("--fix", action="store_true", help="also apply the prompt changes `llmshift fix` suggests")
    p.add_argument("--pr", action="store_true", help="commit to a branch, push and open a pull request")
    p.add_argument("--branch", help="branch name for --pr (default: llmshift/upgrade-OLD-to-NEW)")
    p.add_argument("--base", metavar="BRANCH", help="pull request base (default: the repository's default branch)")
    p.add_argument("--allow-dirty", action="store_true", help="allow --pr while other files have uncommitted changes (uncommitted work is never pushed)")
    p.add_argument("--force", action="store_true", help="also swap ids across providers")
    p.add_argument("--warn-days", type=int, help="with --auto: models retiring within N days (default: scan.warn_days)")
    p.add_argument("--exclude", action="append", help="glob of paths to skip (repeatable)")
    p.add_argument("--min-pass-rate", type=cli.rate_arg, help="threshold shown with --compare (default from llmshift.yaml)")
    p.add_argument("--markdown", nargs="?", const="", metavar="FILE", help="write the pull request text (default: in the reports directory)")
    cli._add_run_options(p)
    p.set_defaults(func=cmd_upgrade)


def _cases_for(cases: list, swap: Swap) -> list:
    from .registry import Registry

    def norm(model: str | None) -> set:
        return set(Registry.normalize(model)[0]) if model else set()

    want = norm(swap.old)
    matched = [c for c in cases if norm(c.baseline_model) & want]
    if matched:
        return matched
    return cases if not any(c.baseline_model for c in cases) else []


def cmd_upgrade(args: argparse.Namespace, cfg: dict) -> int:
    from . import cli
    from .compare import run_compare
    from .registry import Registry

    sc = cfg.get("scan") or {}
    configured = sc.get("registry") or []
    registry = Registry.load(list(configured) if isinstance(configured, list) else [configured])
    warn_days = args.warn_days if args.warn_days is not None else int(sc.get("warn_days", 180))
    if warn_days < 0:
        raise UsageError("--warn-days must be zero or more days")
    exclude = list(sc.get("exclude") or []) + list(args.exclude or [])
    root = gh.repo_root(os.getcwd())
    if args.pr:
        if root is None:
            raise UsageError(f"--pr needs a git repository ({gh.no_repo_reason(os.getcwd())})")
        if gh.is_dirty(root) and not args.allow_dirty:
            raise UsageError("the working tree has uncommitted changes; commit or stash them first, or pass --allow-dirty (they are never pushed)")
    plan = plan_upgrade(args.paths or sc.get("paths") or ["."], old=args.old, new=args.new, auto=args.auto, registry=registry, warn_days=warn_days, exclude=exclude, force=args.force)
    min_rate = args.min_pass_rate if args.min_pass_rate is not None else cfg.get("min_pass_rate")
    min_rate = float(min_rate) if min_rate is not None else None
    if plan.swaps and (args.compare or args.fix):
        tests = cli._tests_path(cfg, args)
        if not Path(tests).exists():
            plan.notes.append(f"no test set at {tests}; skipped --compare/--fix (create one with `llmshift snapshot`)")
        else:
            cases, _ = cli._load_cases(cfg, args)
            common = dict(config=cfg, judge=args.judge or cfg.get("judge"), concurrency=args.concurrency or int(cfg.get("concurrency", 4)), cache=cli._cache(cfg, args), allow_python=args.allow_python or bool(cfg.get("allow_python_checks")), severity=cli._severity(cfg, args))
            for swap in plan.swaps:
                subset = _cases_for(cases, swap)
                model = args.compare_with or swap.spec or swap.new
                if not subset:
                    plan.notes.append(f"no test cases were recorded on {swap.old}; nothing to compare for {swap.new}")
                    continue
                if args.fix:
                    from .fix import patched_cases, run_fix

                    con.out(con.style(f"Looking for prompt changes for {model} ({plural(len(subset), 'case')})...", "grey"), err=True)
                    fix = run_fix(subset, model, source_paths=args.paths or None, validate=False, **common)
                    apply_prompt_fixes(plan, fix, args.paths or sc.get("paths") or ["."])
                    subset = patched_cases(subset, fix)  # compare with the suggested prompts in place
                if args.compare:
                    con.out(con.style(f"Running {len(subset)} test case(s) on {model}...", "grey"), err=True)
                    plan.compare[model] = run_compare(subset, model, min_pass_rate=min_rate, label="upgrade", **common)
    md = plan_markdown(plan, min_rate)
    con.out(con.style("LLMShift upgrade", "bold"))
    if not plan.swaps:
        for note in plan.notes:
            con.out(f"  {note}")
        con.out("  Nothing to change in the code.")
        return EXIT_OK
    for s in plan.swaps:
        con.out(f"  {_swap_line(s)}: {len(s.locations)} place(s)")
    for model, result in plan.compare.items():
        s = result.summary
        con.out(f"  tests on {model}: pass rate {s.get('pass_rate', 0):.0%} ({s.get('failed', 0)} failing, {s.get('warned', 0)} changed)")
    for note in plan.notes:
        con.out(con.style(f"  note: {note}", "grey"))
    for fx in plan.prompt_fixes:
        con.out(f"  prompt fix for {fx['group']}: " + (", ".join(fx["applied"]) if fx["applied"] else f"not applied ({fx.get('why', '')})"))
    if args.markdown is not None:
        target = args.markdown or Path(cfg.get("reports_dir") or ".llmshift/reports") / f"upgrade-{cli._timestamp()}.md"
        cli._write(target, md)
        con.out(con.style(f"  report: {cli._rel(target)}", "grey"))
    cli._step_summary(md)
    below = any(min_rate is not None and r.summary.get("pass_rate", 0) + 1e-9 < min_rate for r in plan.compare.values())
    if not plan.changed:
        con.out("  Nothing to change in the code.")
        return EXIT_FAIL if below else EXIT_OK
    if args.pr:
        return _open_pr(plan, args, md, root)
    con.out("")
    con.out(plan.diff.rstrip())
    if args.apply:
        written = apply_plan(plan)
        con.out("")
        con.out(f"{con.style(con.sym('pass'), 'green')} updated {len(written)} file(s): {', '.join(written)}")
    else:
        con.out("")
        con.out(con.style("Dry run: nothing was changed. Re-run with --apply to write these edits, or --pr to open a pull request.", "grey"))
    return EXIT_FAIL if below else EXIT_OK


def _tip_is_ours(root: Path, ctx: gh.Context, branch: str, sha: str) -> bool | None:
    """Whether the remote branch's tip is a commit llmshift made (None: cannot tell).
    llmshift puts one commit on top of the base, so a tip it did not make means
    someone added to the branch."""
    message = gh.commit_message(root, sha)
    if message is None and ctx.token and ctx.repo:
        try:
            data = gh.api(ctx, "GET", f"/repos/{ctx.repo}/git/commits/{sha}")
            message = data.get("message") if isinstance(data, dict) else None
        except (ProviderError, UsageError):
            message = None
    if message is None:
        try:
            gh.git("fetch", "-q", "--no-tags", "origin", f"refs/heads/{branch}", cwd=root)
        except gh.GitError:
            return None
        message = gh.commit_message(root, sha)
    return None if message is None else MADE_BY in message


def _pr_failure(exc: ProviderError, repo: str | None) -> str:
    text = str(exc)
    if "not permitted to create" in text:
        return (
            "GitHub does not let GITHUB_TOKEN open pull requests in this repository. Turn on Settings > Actions > General > "
            "Workflow permissions > \"Allow GitHub Actions to create and approve pull requests\", or pass a personal access "
            "token or GitHub App token as GITHUB_TOKEN."
        )
    if "rate limit" in text.lower():
        return f"GitHub's API rate limit stopped the pull request ({exc}); try again later."
    if exc.status == 401:
        return f"GitHub rejected the token ({exc})."
    if exc.status == 403:
        return f"the token cannot open pull requests in {repo} ({exc}); the workflow needs `permissions: pull-requests: write`."
    if exc.status == 404:
        return f"{repo} was not found, or the token cannot see it ({exc})."
    return f"GitHub could not open the pull request ({exc})."


def _open_pr(plan: UpgradePlan, args: argparse.Namespace, body: str, root: Path) -> int:
    """Commit the edits on top of HEAD and push them to the upgrade branch, without
    touching the working tree, the index or any local branch; then open the PR."""
    ctx = gh.context()
    ctx.repo = ctx.repo or gh.remote_repo(cwd=root, host=ctx.host)
    identity: list[str] = []
    if not gh.has_identity(root):
        if not ctx.in_actions:
            raise UsageError("git has no user.name/user.email; set them before --pr")
        identity = ["-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}"]
    branch = _branch_name(plan, args.branch)
    start = gh.current_branch(root)
    default = gh.default_branch(ctx) if ctx.token and ctx.repo else None
    base = args.base or default or start or "main"
    if branch in {b for b in (base, default, start) if b}:
        raise UsageError(f"--branch {branch} is the base, default or current branch; the upgrade goes to a branch of its own")
    files = {c.path: c.updated or b"" for c in plan.changed}
    title = "Upgrade " + ", ".join(f"{s.old} to {s.new}" for s in plan.swaps[:3]) + ("..." if len(plan.swaps) > 3 else "")
    try:
        modes = gh.tracked_modes(root, list(files))
        untracked = [p for p in files if p not in modes]
        if untracked:
            raise UsageError(f"not committed yet: {', '.join(untracked)}; commit first (the pull request starts from your last commit)")
        special = [p for p, mode in modes.items() if not mode.startswith("100")]
        if special:
            raise UsageError(f"{', '.join(special)}: a symbolic link or submodule; upgrade the file it points to")
        dirty = gh.modified_paths(root, list(files))
        if dirty:
            raise UsageError(f"{', '.join(dirty)} {'has' if len(dirty) == 1 else 'have'} uncommitted changes; commit or stash them first (uncommitted work is never pushed)")
        commit = gh.commit_files(root, files, modes, f"{title}\n\n{MADE_BY}\n", config=identity)
        current = gh.remote_branch(root, branch)
        if current is not None:
            ours = _tip_is_ours(root, ctx, branch, current)
            if not ours:
                why = "has commits llmshift did not make" if ours is False else "could not be checked"
                raise UsageError(
                    f"branch {branch} on origin {why}, so it is not overwritten. Merge or close its pull request and delete the "
                    "branch, or pass --branch NEW"
                )
        # The lease: overwrite only the commit checked above (or create the branch).
        gh.git("push", "-q", f"--force-with-lease=refs/heads/{branch}:{current or ''}", "origin", f"{commit}:refs/heads/{branch}", cwd=root)
    except gh.GitError as exc:
        raise UsageError(f"git failed while preparing the pull request: {exc}") from None
    con.out(f"{con.style(con.sym('pass'), 'green')} pushed {len(files)} file(s) to branch {branch}")
    if not (ctx.token and ctx.repo):
        if ctx.repo:
            con.warn(f"no GITHUB_TOKEN: open the pull request yourself: {ctx.compare_url(base, branch)}")
        elif ctx.token:
            con.warn(f"cannot tell which GitHub repository origin is (set GITHUB_REPOSITORY=owner/name); open the pull request for branch {branch} yourself")
        else:
            con.warn(f"no GITHUB_TOKEN: open the pull request for branch {branch} yourself")
        return EXIT_OK
    try:
        pr = gh.open_pull_request(ctx, head=branch, base=base, title=title, body=body)
    except ProviderError as exc:
        raise UsageError(f"{_pr_failure(exc, ctx.repo)} The branch is pushed; open the pull request yourself: {ctx.compare_url(base, branch)}") from None
    con.out(f"{con.style(con.sym('pass'), 'green')} pull request: {pr.get('html_url') or pr.get('number')}")
    return EXIT_OK

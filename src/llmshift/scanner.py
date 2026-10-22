"""`llmshift scan`: find every model your code calls and warn about retirement dates.

Detection layers:
* Python: AST analysis - `model=` style keyword arguments, models passed
  positionally to known constructors, {"model": ...} dicts, MODEL constants
  (resolved where they are used) and os.getenv("X_MODEL", "default") fallbacks.
* Other code (JS/TS, Go, Java, C#, Ruby, PHP, Rust, ...): string literals that
  are model IDs, and values assigned to model-like keys.
* Config (YAML, JSON, TOML, .env, INI, Terraform): model-like keys and IDs.
* Jupyter notebooks: code cells, with cell numbers in the report.
"""

from __future__ import annotations

import ast
import codecs
import fnmatch
import functools
import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .registry import SEVERITY_ORDER, Lookup, Registry

CODE_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".java", ".kt", ".kts", ".scala",
    ".rb", ".php", ".cs", ".rs", ".swift", ".dart", ".ex", ".exs", ".lua", ".r", ".jl", ".sh", ".bash", ".zsh", ".ps1",
    ".vue", ".svelte", ".astro", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".sql",
}
CONFIG_EXT = {".yaml", ".yml", ".json", ".jsonc", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".tf", ".tfvars", ".hcl", ".xml", ".prompt"}
DOC_EXT = {".md", ".mdx", ".rst", ".txt", ".adoc"}
SPECIAL_NAMES = {"dockerfile", "makefile", "procfile", ".env"}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", ".env.d", "__pycache__", "dist", "build", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".next", ".nuxt", "coverage", ".idea", ".vscode",
    "target", "vendor", ".llmshift", ".terraform", ".gradle", "bower_components", ".cache", ".parcel-cache", ".turbo",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock", "pipfile.lock", "composer.lock",
    "go.sum", "uv.lock", "bun.lockb", "gemfile.lock",
}
MAX_BYTES = 2_000_000
MAX_NOTEBOOK_BYTES = 50_000_000  # notebooks carry outputs (images); the limit applies to their code
MAX_LINE = 2000  # longer lines (minified code, one-line JSON) are only searched for model IDs

_KNOWN = [
    r"gpt-\d+(?:\.\d+)?o?(?:-[a-z0-9]+(?:\.\d+)?)*",
    r"chatgpt-4o-latest",
    r"(?:gpt-\d+(?:\.\d+)?-)?chat-latest",
    # o-series: lowercase and with a suffix here; a bare "o3" only counts in a model context
    r"(?-i:o[134](?:-(?:mini|pro|preview|deep-research))+(?:-\d{4}-\d{2}-\d{2})?|o[134]-\d{4}-\d{2}-\d{2})",
    r"(?:text|code)-(?:davinci|curie|babbage|ada|cushman)-(?:edit-)?\d{3}",
    r"(?:davinci|babbage)-002",
    r"text-(?:embedding|similarity|search|moderation)-[a-z0-9-]+",
    r"omni-moderation-[a-z0-9-]+",
    r"dall-e-[23]",
    r"whisper-1",
    r"tts-1(?:-hd)?(?:-\d{4})?",
    r"codex-mini-latest",
    r"computer-use-preview(?:-[\d-]+)?",
    r"sora-\d[a-z0-9.\-]*",
    r"ft:[a-z0-9.\-]+:[\w.\-:]*",
    r"(?:(?:us|eu|apac|global|us-gov|jp|au|ca)\.)?anthropic\.claude-[a-z0-9.\-]+(?::\d+)?",
    r"claude-(?:instant-)?(?:\d|opus|sonnet|haiku|fable|mythos|v\d)[a-z0-9.\-]*(?:@\d{8})?",
    r"(?:models/)?gemini-(?:\d|pro|ultra|flash|nano|exp|embedding|live|robotics)[a-z0-9.\-]*",
    r"(?:text|chat|code|codechat)-bison(?:-32k)?(?:@\d{3})?",
    r"textembedding-gecko(?:-multilingual)?(?:@\d{3})?",
    r"text-(?:multilingual-)?embedding-00\d",
    r"(?:open-)?(?:mistral|mixtral|codestral|ministral|pixtral|magistral|devstral|voxtral)-(?:\d|tiny|small|medium|large|embed|saba|ocr|latest|nemo|moderation|mamba|mini)[a-z0-9.\-]*",
    r"command-(?:r|a|light|nightly|xlarge|medium|r7b)(?:[a-z0-9.\-]*)",
    r"embed-(?:english|multilingual)(?:-light)?-v\d\.\d",
    r"rerank-(?:english|multilingual|v\d)[a-z0-9.\-]*",
    r"grok-(?:\d|beta|vision|code|imagine|voice|build)[a-z0-9.\-]*",
    r"deepseek-(?:chat|reasoner|coder|v\d|r\d|flash|prover|math|vl)[a-z0-9.\-]*",
    r"(?:meta-llama/)?(?:meta-)?llama-?\d[a-z0-9.:\-]*",
    r"(?:openrouter/)?(?:openai|anthropic|google|mistralai|meta-llama|deepseek|x-ai|cohere)/[a-z0-9][\w.\-:]*",
]
KNOWN_RE = re.compile(r"(?:" + "|".join(_KNOWN) + r")", re.I)
KNOWN_SEARCH_RE = re.compile(r"(?<![\w./@-])(?:" + "|".join(_KNOWN) + r")(?![\w@-])", re.I)
_NOT_MODEL_SUFFIX_RE = re.compile(r"-(?:encoder|tokenizer|tokenizers|index|python|node|js|sdk|cli|api|client|desktop|plugin|action|types|wrapper|proxy|server|app|bot)$", re.I)
_MODEL_KEY_RE = re.compile(
    r"^(?:[a-z0-9]+_)*(?:model|model_name|model_id|llm_model|chat_model|embedding_model|engine|deployment|deployment_name|deployment_id|azure_deployment)$",
    re.I,
)
_KEY_VALUE_RE = re.compile(
    r"""(?P<key>[A-Za-z_][\w-]*)["']?\s*(?::|=|=>|:=)\s*(?P<q>["'`]?)(?P<val>[A-Za-z0-9][\w.:/@\-]{1,100})(?P=q)""",
)
_LITERAL_RE = re.compile(r"""(["'`])((?:(?!\1).){2,120}?)\1""")
_BAD_VALUES = {
    "true", "false", "none", "null", "nil", "auto", "default", "string", "str", "int", "float", "model", "models", "name",
    "id", "text", "json", "object", "any", "required", "optional", "undefined", "env", "self", "this", "value", "type",
    # sizes, tiers and stages: product or config words, never model IDs
    "full", "half", "lite", "light", "basic", "standard", "premium", "simple", "small", "medium", "large", "tiny",
    "mini", "xs", "xl", "xxl", "base", "custom", "fast", "slow", "test", "testing", "mock", "fake", "dummy", "example",
    "sample", "demo", "local", "remote", "primary", "secondary", "fallback", "main", "new", "old", "legacy", "current",
    "latest", "stable", "beta", "alpha", "dev", "prod", "production", "staging",
}
_FILE_EXT_RE = re.compile(r"\.(?:pt|pth|bin|onnx|gguf|ggml|json|pkl|pickle|h5|safetensors|joblib|tflite|mlmodel|zip|tar|gz|yaml|yml|txt|csv)$", re.I)
PY_MODEL_KWARGS = {
    "model", "model_name", "model_id", "modelId", "engine", "deployment_name", "deployment_id", "azure_deployment",
    "llm_model", "chat_model", "embedding_model", "model_version", "deployment",
}
PY_POSITIONAL_FUNCS = {
    "GenerativeModel", "init_chat_model", "OpenAIModel", "AnthropicModel", "GeminiModel", "GoogleModel", "MistralModel",
    "GroqModel", "CohereModel", "OpenAIChatModel", "OpenAIResponsesModel", "Agent", "LM", "completion", "acompletion",
    "embedding", "aembedding", "ChatOpenAI", "ChatAnthropic", "OpenAIChat", "get_model", "ChatLiteLLM", "LiteLLMModel",
}


def _snake(key: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", key).replace("-", "_").lower()


_BAD_KEYS = {
    "v_model", "ng_model", "x_model", "wire_model", "data_model", "view_model", "pydantic_model", "db_model", "orm_model",
    "auth_user_model", "user_model", "parent_model", "related_model", "through_model", "proxy_model",
    "form_model", "serializer_model", "admin_model",
}
_LLM_KEY_RE = re.compile(
    r"(?:^|_)(?:llm|llms|lm|gpt|openai|anthropic|claude|gemini|genai|ai|chat|completion|completions|embedding|embeddings|judge|grader)(?:_|$)"
)
# Files that talk to an LLM somewhere; generic keys such as DEFAULT_MODEL only count in them.
_LLM_HINT_RE = re.compile(
    r"openai|anthropic|gemini|genai|litellm|langchain|llama_?index|bedrock|vertexai|mistral|ollama|cohere|groq|deepseek"
    r"|chat\.completions|messages\.create|generate_?content|\bllms?\b|_llm|llm_|\bgpt|claude"
    r"|max_(?:completion_|output_)?tokens|\btop_p\b|system_?prompt",
    re.I,
)


@functools.lru_cache(maxsize=8192)
def model_key_strength(key: str) -> str | None:
    """How much a key says about its value: "strong" (model, OPENAI_MODEL, chat_model),
    "weak" (DEFAULT_MODEL, deployment: the value must look like a model or the file must
    use an LLM), "known" (engine: only known model IDs, since pandas and Django use it too)."""
    snake = _snake(key)
    if snake in _BAD_KEYS or not _MODEL_KEY_RE.match(snake):
        return None
    if snake.endswith("engine"):
        return "known"
    if snake in ("model", "model_name", "model_id", "azure_deployment") or _LLM_KEY_RE.search(snake):
        return "strong"
    return "weak"


def is_model_key(key: str) -> bool:
    return model_key_strength(key) is not None


def accepts_value(strength: str | None, value: str, llm_hint: bool) -> bool:
    """A value under a model-like key counts when it is a known model ID. Other names
    (a fine-tune, a deployment) count only in files that show signs of calling an LLM,
    so `model: full` in a size chart or a test fixture is not a finding."""
    if strength is None:
        return False
    if looks_like_known_model(value, in_model_context=True):
        return True
    return llm_hint and strength in ("strong", "weak") and plausible_model_value(value)


def has_llm_hint(text: str) -> bool:
    return bool(_LLM_HINT_RE.search(text))


_BARE_O_SERIES_RE = re.compile(r"o[134]")
# Cheap substring test run before the big regex (keeps large repositories fast).
# Every pattern in _KNOWN must contain one of these (tests/test_scan.py checks it).
_TRIGGERS = (
    "gpt", "o1", "o3", "o4", "claude", "gemini", "davinci", "babbage", "curie", "-ada-", "cushman", "embed", "moderation",
    "similarity", "search", "dall-e", "whisper", "tts-", "codex", "computer-use", "sora", "ft:", "bison", "gecko", "stral",
    "xtral", "command", "rerank", "grok", "deepseek", "llama", "chat-latest", "/",
)


def has_trigger(text_lower: str) -> bool:
    return any(t in text_lower for t in _TRIGGERS)


def looks_like_known_model(value: str, *, in_model_context: bool = False) -> bool:
    v = value.strip()
    if not 2 <= len(v) <= 120 or " " in v or "\n" in v:
        return False
    if in_model_context and _BARE_O_SERIES_RE.fullmatch(v):
        return True
    if not has_trigger(v.lower()):
        return False
    return bool(KNOWN_RE.fullmatch(v)) and not _NOT_MODEL_SUFFIX_RE.search(v)


def plausible_model_value(value: str) -> bool:
    v = value.strip()
    if not v or len(v) > 100 or " " in v:
        return False
    if v.lower() in _BAD_VALUES or v.isdigit():
        return False
    if v.startswith(("http://", "https://", "$", "{", "%", "<", "/", "./", "../", "~")) or "{" in v or "}" in v:
        return False
    if _FILE_EXT_RE.search(v):
        return False
    if re.fullmatch(r"[A-Z][a-zA-Z]+", v):  # ClassName-like
        return False
    return bool(re.search(r"[a-z]", v, re.I))


@dataclass
class Finding:
    file: str
    line: int
    col: int
    model: str
    context: str
    kind: str  # call | string | config | env-default | constant
    call: str | None = None
    via: str | None = None
    cell: int | None = None
    lookup: Lookup | None = None
    file_line: int | None = None  # notebooks: the line in the .ipynb file itself

    @property
    def location(self) -> str:
        loc = f"{self.file}:{self.line}"
        if self.cell is not None:
            loc += f" (cell {self.cell})"
        return loc

    def to_dict(self) -> dict:
        d = {"file": self.file, "line": self.line, "col": self.col, "model": self.model, "kind": self.kind, "context": self.context}
        for key in ("call", "via", "cell", "file_line"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        if self.lookup:
            d["lifecycle"] = self.lookup.to_dict()
        return d


@dataclass
class ScanResult:
    root: str
    files_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    registry_updated: str | None = None
    warn_days: int = 180

    def models(self) -> list[dict]:
        """One entry per distinct model, worst first."""
        groups: dict[str, list[Finding]] = {}
        for f in self.findings:
            groups.setdefault(f.model, []).append(f)
        out = []
        for model, items in groups.items():
            lk = items[0].lookup
            worst = max((i.lookup for i in items if i.lookup), key=lambda x: SEVERITY_ORDER.get(x.severity, 0), default=lk)
            out.append({"model": model, "lookup": worst, "findings": sorted(items, key=lambda i: (i.file, i.line))})
        return sorted(
            out,
            key=lambda m: (
                -SEVERITY_ORDER.get(m["lookup"].severity if m["lookup"] else "info", 0),
                m["lookup"].days_left if m["lookup"] and m["lookup"].days_left is not None else 10**6,
                m["model"],
            ),
        )

    def counts(self) -> dict:
        c = {"retired": 0, "retiring": 0, "retiring_soon": 0, "deprecated": 0, "alias": 0, "unknown": 0, "active": 0}
        for m in self.models():
            lk = m["lookup"]
            if lk is None:
                continue
            if lk.status == "retiring":
                c["retiring"] += 1
                if lk.days_left is not None and lk.days_left <= self.warn_days:
                    c["retiring_soon"] += 1
            elif lk.status in c:
                c[lk.status] += 1
            if lk.alias and lk.status == "active":
                c["alias"] += 1
        return c

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "registry_updated": self.registry_updated,
            "warn_days": self.warn_days,
            "summary": self.counts(),
            "models": [
                {"model": m["model"], "lifecycle": m["lookup"].to_dict() if m["lookup"] else None, "locations": [f.location for f in m["findings"]]}
                for m in self.models()
            ],
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
        }


# ------------------------------------------------------------ file walking


def _git_files(root: Path) -> list[Path] | None:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(root),
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    names = [n for n in proc.stdout.decode("utf-8", "replace").split("\0") if n]
    return [root / n for n in names]


def _wanted(path: Path, include_docs: bool) -> bool:
    name = path.name.lower()
    if name in SKIP_FILES or name == "models.json" and path.parent.name == "data" and path.parent.parent.name == "llmshift":
        return False
    if name in SPECIAL_NAMES or name.startswith(".env") or name.startswith("docker-compose"):
        return True
    ext = path.suffix.lower()
    if ext == ".ipynb" or ext in CODE_EXT or ext in CONFIG_EXT:
        return True
    return include_docs and ext in DOC_EXT


# Directories never scanned, even when git tracks them; SKIP_DIRS only applies when walking.
_ALWAYS_SKIP = {".git", "node_modules", "site-packages", "__pycache__"}
_GLOB_CHARS_RE = re.compile(r"[*?\[]")


def exclude_matcher(patterns: Iterable[str]):
    """Match --exclude patterns against a file's paths (relative to the scanned root, to the
    working directory, or absolute). A pattern without "/" matches a name at any depth;
    "**/x" also matches a top-level "x"; a directory pattern excludes what is inside it."""
    pats = []
    for raw in patterns or ():
        pat = str(raw or "").strip().replace("\\", "/")
        while pat.startswith("./"):
            pat = pat[2:]
        pat = pat.rstrip("/")
        if pat:
            pats.append(pat)

    def matches(path: str, pat: str) -> bool:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat + "/*"):
            return True
        if pat.startswith("**/"):
            return matches(path, pat[3:])
        if "/" not in pat:
            return any(fnmatch.fnmatch(part, pat) for part in path.split("/"))
        return False

    def excluded(*paths: str | None) -> bool:
        return any(matches(path, pat) for path in paths if path for pat in pats)

    excluded.patterns = pats  # type: ignore[attr-defined]
    return excluded


def _expand(paths: Iterable[str]) -> list[str]:
    """Expand glob patterns the shell did not (Windows shells never do)."""
    out = []
    for raw in paths:
        if not Path(raw).exists() and _GLOB_CHARS_RE.search(raw):
            matches = sorted(glob.glob(raw, recursive=True))
            if not matches:
                raise FileNotFoundError(f"no files match {raw}")
            out.extend(matches)
        else:
            out.append(raw)
    return out


def iter_source_files(paths: Iterable[str], *, include_docs: bool = False, exclude: Iterable[str] = ()) -> Iterable[tuple[Path, Path]]:
    """Yield (root, file) pairs, honouring .gitignore when inside a git repo."""
    excluded = exclude_matcher(exclude)
    absolute = any(p.startswith("/") or re.match(r"^[A-Za-z]:/", p) for p in excluded.patterns)  # type: ignore[attr-defined]
    cwd = os.getcwd()

    def candidates(f: Path, rel: str | None) -> tuple:
        try:
            from_cwd = Path(os.path.relpath(f, cwd)).as_posix()
        except ValueError:  # another drive on Windows
            from_cwd = None
        return (rel, from_cwd, f.resolve().as_posix() if absolute else None)

    seen: set[Path] = set()
    for raw in _expand(paths):
        root = Path(raw)
        if root.is_file():  # named explicitly: scanned whatever its type, unless excluded
            rf = root.resolve()
            if rf not in seen and not excluded(root.name, *candidates(root, None)):
                seen.add(rf)
                yield root.parent, root
            continue
        if not root.is_dir():
            raise FileNotFoundError(f"no such file or directory: {raw}")
        files = _git_files(root)
        git_mode = bool(files)  # nothing listed: not a repo, or a directory git ignores
        if not git_mode:
            files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
                files.extend(Path(dirpath) / f for f in filenames)
        skip = _ALWAYS_SKIP if git_mode else SKIP_DIRS
        for f in sorted(files or []):
            try:
                rel = f.relative_to(root).as_posix()
            except ValueError:
                rel = f.as_posix()
            if any(part in skip for part in Path(rel).parts[:-1]):
                continue
            if not _wanted(f, include_docs) or excluded(*candidates(f, rel)):
                continue
            rf = f.resolve()
            if rf in seen or not f.is_file():
                continue
            seen.add(rf)
            yield root, f


def decode_text(data: bytes) -> str | None:
    """Decode a source file (UTF-8, or UTF-16 as written by some Windows tools); None if binary."""
    if data.startswith(codecs.BOM_UTF8):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    head = data[:4096]
    if b"\0" in head:
        odd, even = head[1::2], head[0::2]
        if len(head) >= 4 and odd.count(0) >= 0.9 * len(odd) and even.count(0) <= 0.1 * len(even):
            return data.decode("utf-16-le", errors="replace")
        if len(head) >= 4 and even.count(0) >= 0.9 * len(even) and odd.count(0) <= 0.1 * len(odd):
            return data.decode("utf-16-be", errors="replace")
        return None
    return data.decode("utf-8", errors="replace")


def _read_text(path: Path) -> tuple[str | None, str | None]:
    """(text, reason it was skipped). Binary files are skipped silently."""
    limit = MAX_NOTEBOOK_BYTES if path.suffix.lower() == ".ipynb" else MAX_BYTES
    try:
        size = path.stat().st_size
        if size > limit:
            return None, f"skipped: larger than {limit // 1_000_000} MB"
        data = path.read_bytes()
    except OSError as exc:
        return None, f"cannot read: {exc.strerror or exc}"
    return decode_text(data), None


def platform_hint(text: str) -> str | None:
    lower = text.lower()
    if "anthropicvertex" in lower or "vertexai" in lower or "aiplatform" in lower or "vertex_ai/" in lower:
        return "vertex"
    if "bedrock-runtime" in lower or "bedrockruntime" in lower or "bedrock_runtime" in lower or "anthropicbedrock" in lower:
        return "bedrock"
    if "azureopenai" in lower or "openai.azure.com" in lower or "azure_endpoint" in lower:
        return "azure"
    return None


# --------------------------------------------------------------- python


class _PyVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str], llm_hint: bool = True):
        self.lines = lines
        self.llm_hint = llm_hint
        self.constants: dict[str, tuple[str, int]] = {}
        self.found: list[tuple[int, int, str, str, str | None, str | None]] = []  # line, col, value, kind, call, via
        self.claimed: set[tuple[int, int]] = set()
        self.docstrings: set[int] = set()

    def _call_name(self, node: ast.Call) -> str:
        try:
            return ast.unparse(node.func)  # type: ignore[attr-defined]
        except Exception:
            f = node.func
            parts = []
            while isinstance(f, ast.Attribute):
                parts.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name):
                parts.append(f.id)
            return ".".join(reversed(parts))

    def _add(self, node: ast.AST, value: str, kind: str, call: str | None = None, via: str | None = None) -> None:
        line, col = getattr(node, "lineno", 0), getattr(node, "col_offset", 0)
        self.found.append((line, col + 1, value, kind, call, via))
        self.claimed.add((line, col))

    def _env_default(self, node: ast.AST) -> ast.Constant | None:
        # os.getenv("X", "default") / os.environ.get("X", "default")
        if isinstance(node, ast.Call) and len(node.args) >= 2:
            name = self._call_name(node)
            if name.endswith(("getenv", "environ.get", "get")) and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                return node.args[1]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            last = node.values[-1]
            if isinstance(last, ast.Constant) and isinstance(last.value, str):
                return last
        return None

    def _require_known(self, key: str) -> bool:
        strength = model_key_strength(key) or ("strong" if key in PY_MODEL_KWARGS and key != "engine" else "known")
        return strength == "known" or not self.llm_hint  # see accepts_value()

    def _resolve(self, node: ast.AST, call: str | None, kind: str, require_known: bool) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            if (looks_like_known_model(v) if require_known else plausible_model_value(v) or looks_like_known_model(v)):
                self._add(node, v, kind, call)
            return
        if isinstance(node, ast.Name) and node.id in self.constants:
            value, line = self.constants[node.id]
            if looks_like_known_model(value) or (not require_known and plausible_model_value(value)):
                self._add(node, value, kind, call, via=f"{node.id}, line {line}")
            return
        default = self._env_default(node)
        if default is not None:
            v = default.value.strip()
            if looks_like_known_model(v) or (not require_known and plausible_model_value(v)):
                self._add(default, v, "env-default", call)
            return
        if isinstance(node, ast.IfExp):
            self._resolve(node.body, call, kind, require_known)
            self._resolve(node.orelse, call, kind, require_known)

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(stmt, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                    self.docstrings.add(id(body[0].value))
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.constants[target.id] = (stmt.value.value.strip(), stmt.lineno)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                self.constants[stmt.target.id] = (stmt.value.value.strip(), stmt.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
            if name and is_model_key(name):
                self._resolve(node.value, None, "constant", require_known=self._require_known(name))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if name and node.value is not None and is_model_key(name):
            self._resolve(node.value, None, "constant", require_known=self._require_known(name))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        short = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
        call = None  # the dotted name is only computed when a model is involved
        for kw in node.keywords:
            if kw.arg and (kw.arg in PY_MODEL_KWARGS or is_model_key(kw.arg)):
                call = call or self._call_name(node)
                self._resolve(kw.value, call, "call", require_known=self._require_known(kw.arg))
        if short in PY_POSITIONAL_FUNCS and node.args:
            self._resolve(node.args[0], call or self._call_name(node), "call", require_known=True)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and is_model_key(key.value) and value is not None:
                self._resolve(value, None, "call", require_known=self._require_known(key.value))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and id(node) not in self.docstrings:
            key = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            if key not in self.claimed and looks_like_known_model(node.value.strip()):
                self._add(node, node.value.strip(), "string")


def scan_python(text: str, llm_hint: bool | None = None) -> list[tuple[int, int, str, str, str | None, str | None]]:
    llm_hint = has_llm_hint(text) if llm_hint is None else llm_hint
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return scan_generic(text, code=True, llm_hint=llm_hint)
    visitor = _PyVisitor(text.splitlines(), llm_hint=llm_hint)
    visitor.visit(tree)
    # The catch-all string pass runs during traversal; drop duplicates it produced.
    seen = set()
    out = []
    for item in sorted(visitor.found, key=lambda x: (x[0], x[1], {"call": 0, "env-default": 1, "constant": 2, "string": 3}.get(x[3], 4))):
        key = (item[0], item[2])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# -------------------------------------------------------------- generic

_COMMENT_RE = re.compile(r"^\s*(//|#(?!!)|/\*|\*\s|\*$|<!--)")


def scan_generic(
    text: str, *, code: bool, shell: bool = False, llm_hint: bool | None = None
) -> list[tuple[int, int, str, str, str | None, str | None]]:
    """Scan non-Python text line by line. `shell`: values may be unquoted (MODEL=gpt-4o)
    and bare model IDs count (--model gpt-4o)."""
    llm_hint = has_llm_hint(text) if llm_hint is None else llm_hint
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if len(line) > MAX_LINE:
            if has_trigger(line.lower()):
                found: dict[str, int] = {}
                for m in KNOWN_SEARCH_RE.finditer(line):
                    if not _NOT_MODEL_SUFFIX_RE.search(m.group(0)):
                        found.setdefault(m.group(0), m.start() + 1)
                out.extend((lineno, col, val, "string" if code else "config", None, None) for val, col in found.items())
            continue
        if code and _COMMENT_RE.match(line):
            continue
        lower = line.lower()
        keyed = "model" in lower or "engine" in lower or "deployment" in lower
        triggered = has_trigger(lower)
        if not keyed and not triggered:
            continue
        hits: dict[str, tuple[int, str]] = {}
        model_ctx = False
        for m in _KEY_VALUE_RE.finditer(line) if keyed else ():
            key, quote, val = m.group("key"), m.group("q"), m.group("val")
            strength = model_key_strength(key)
            if strength is None:
                continue
            model_ctx = True
            if code and not quote and not shell:
                continue
            if accepts_value(strength, val, llm_hint):
                hits.setdefault(val, (m.start("val") + 1, "call" if code else "config"))
        for m in _LITERAL_RE.finditer(line) if triggered else ():
            val = m.group(2).strip()
            if looks_like_known_model(val):
                hits.setdefault(val, (m.start(2) + 1, "call" if (code and model_ctx) else ("string" if code else "config")))
        if (shell or not code) and triggered:
            for m in KNOWN_SEARCH_RE.finditer(line):
                val = m.group(0)
                if not _NOT_MODEL_SUFFIX_RE.search(val):
                    hits.setdefault(val, (m.start() + 1, "string" if code else "config"))
        for val, (col, kind) in hits.items():
            out.append((lineno, col, val, kind, None, None))
    return out


_SOURCE_KEY_RE = re.compile(r'^\s*"source"\s*:\s*(\[)?')


def _line_starts(pieces: list) -> list[int]:
    """For each line of "".join(pieces), the index of the piece it starts in."""
    starts: list[int] = []
    open_line = False
    for j, piece in enumerate(pieces):
        if not isinstance(piece, str):
            continue
        pos = 0
        while pos < len(piece):
            if not open_line:
                starts.append(j)
                open_line = True
            nl = piece.find("\n", pos)
            if nl == -1:
                break
            open_line = False
            pos = nl + 1
    return starts


def scan_notebook(text: str) -> list[tuple[int, int, str, str, str | None, str | None, int, int | None]]:
    """Scan code cells. Items carry the cell number and, when the notebook is stored one
    source line per JSON line (as Jupyter writes it), the line in the .ipynb file."""
    try:
        nb = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"not a valid notebook ({exc})") from None
    if not isinstance(nb, dict):
        raise ValueError("not a valid notebook (expected a JSON object)")
    cells = nb.get("cells")
    v3 = cells is None  # nbformat 3 keeps cells under worksheets, with "input" for code
    if v3:
        cells = [c for ws in nb.get("worksheets") or [] if isinstance(ws, dict) for c in ws.get("cells") or []]
    cells = [c for c in cells or [] if isinstance(c, dict)]
    key_lines = [] if v3 else [(n, bool(m.group(1))) for n, ln in enumerate(text.splitlines(), 1) for m in [_SOURCE_KEY_RE.match(ln)] if m]
    if len(key_lines) != len(cells):
        key_lines = []
    code_cells = [(idx, cell) for idx, cell in enumerate(cells, 1) if cell.get("cell_type") == "code"]
    total = sum(len(str(cell.get("source") or cell.get("input") or "")) for _, cell in code_cells)
    if total > MAX_BYTES:
        raise ValueError(f"skipped: more than {MAX_BYTES // 1_000_000} MB of code")
    llm_hint = has_llm_hint(text)
    out = []
    for idx, cell in code_cells:
        raw = cell.get("source", cell.get("input")) or ""
        src = "".join(p for p in raw if isinstance(p, str)) if isinstance(raw, list) else str(raw)
        starts = _line_starts(raw) if isinstance(raw, list) else []
        key = key_lines[idx - 1] if key_lines else None
        # IPython magics/shell lines break ast.parse; blank them but keep numbering.
        cleaned = "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in src.split("\n"))
        for item in scan_python(cleaned, llm_hint=llm_hint):
            file_line = None
            if key is not None:
                key_line, is_list = key
                if not is_list:
                    file_line = key_line
                elif 0 < item[0] <= len(starts):
                    file_line = key_line + 1 + starts[item[0] - 1]
            out.append(item + (idx, file_line))
    return out


# --------------------------------------------------------------- main


def scan_paths(
    paths: list[str] | None = None,
    *,
    registry: Registry | None = None,
    include_docs: bool = False,
    exclude: Iterable[str] = (),
    warn_days: int = 180,
) -> ScanResult:
    registry = registry or Registry.load()
    paths = list(paths or ["."])
    result = ScanResult(root=", ".join(paths), registry_updated=registry.updated, warn_days=warn_days)
    seen: set[tuple] = set()
    for _root, path in iter_source_files(paths, include_docs=include_docs, exclude=exclude):
        try:
            rel = Path(os.path.relpath(path)).as_posix()
            if rel.startswith("../"):
                rel = path.resolve().as_posix()
        except ValueError:  # different drive on Windows
            rel = path.resolve().as_posix()
        text, skipped = _read_text(path)
        if skipped:
            result.errors.append(f"{rel}: {skipped}")
        if text is None:
            continue
        result.files_scanned += 1
        hint = platform_hint(text)
        ext = path.suffix.lower()
        try:
            if ext == ".ipynb":
                items = scan_notebook(text)
            elif ext in (".py", ".pyi"):
                items = [i + (None, None) for i in scan_python(text)]
            else:
                is_code = ext in CODE_EXT
                shell = ext in (".sh", ".bash", ".zsh")
                items = [i + (None, None) for i in scan_generic(text, code=is_code, shell=shell)]
        except RecursionError:
            result.errors.append(f"{rel}: too deeply nested to analyse")
            continue
        except Exception as exc:  # one odd file must not stop the scan
            message = str(exc) if isinstance(exc, ValueError) else f"could not analyse ({exc.__class__.__name__}: {exc})"
            result.errors.append(f"{rel}: {message}")
            continue
        lines = text.splitlines()
        for line, col, value, kind, call, via, cell, file_line in items:
            key = (rel, line, value, cell)
            if key in seen:
                continue
            seen.add(key)
            if cell is not None:
                context = ""
            else:
                context = lines[line - 1].strip()[:160] if 0 < line <= len(lines) else ""
            finding = Finding(file=rel, line=line, col=col, model=value, context=context, kind=kind, call=call, via=via, cell=cell, file_line=file_line)
            finding.lookup = registry.lookup(value, platform=hint, warn_days=warn_days)
            result.findings.append(finding)
    result.findings.sort(key=lambda f: (f.file, f.line, f.col))
    return result


FAIL_LEVELS = ("retired", "retiring", "deprecated", "unknown", "never")


def should_fail(result: ScanResult, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    for m in result.models():
        lk = m["lookup"]
        if lk is None:
            continue
        if lk.status == "retired":
            return True
        if fail_on in ("retiring", "deprecated", "unknown") and lk.status == "retiring" and (lk.days_left or 0) <= result.warn_days:
            return True
        if fail_on in ("deprecated", "unknown") and lk.status in ("deprecated", "retiring"):
            return True
        if fail_on == "unknown" and lk.status == "unknown":
            return True
    return False

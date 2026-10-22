"""Import OpenAI Evals into LLMShift test sets.

OpenAI's hosted Evals platform becomes read-only on 2026-10-31 and its dashboard
and API shut down on 2026-11-30. This importer pulls everything that matters
out of it while it is still readable, and also reads the open-source
openai/evals registry format.

Sources
* Hosted platform, via the API (needs OPENAI_API_KEY):
      llmshift import openai-evals --api --all
      llmshift import openai-evals --api --eval-id eval_abc --eval-id eval_def
  For each eval: its testing criteria, its latest run's data source, and the
  run's output items (inputs actually sent + outputs the model gave), which
  become the recorded baselines.
* Exported JSON (an eval object, or {"eval":..., "runs":..., "output_items":...}),
  optionally with a data file of {"item": {...}} lines:
      llmshift import openai-evals eval.json --data items.jsonl
* The OSS registry (YAML specs + samples JSONL):
      llmshift import openai-evals path/to/evals/registry [--eval NAME]

Grader mapping
  string_check eq/ne/like/ilike -> equals / not_equals / contains (case-sensitive / insensitive)
  text_similarity               -> similarity (same metric and pass_threshold)
  label_model                   -> llm_label   (judge: the grader's model or --judge)
  score_model                   -> llm_score   (range, pass_threshold)
  python                        -> python      (runs only with --allow-python)
  multi                         -> multi       (calculate_output formula)
  OSS Match / Includes / FuzzyMatch / JsonMatch / ModelBasedClassify -> starts_with /
  contains_any / fuzzy_match / json_equals / modelgraded (fact, closedqa, custom specs)
"""

from __future__ import annotations

import fnmatch
import gzip
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .. import messages as msgs
from ..providers import http
from ..templating import render, render_text, uses_namespace
from ..testset import TestCase
from ..textutil import stable_hash

DEFAULT_BASE_URL = "https://api.openai.com/v1"
READ_ONLY_DATE = "2026-10-31"
SHUTDOWN_DATE = "2026-11-30"

FACT_SPEC = {
    "prompt": (
        "You are comparing a submitted answer to an expert answer on a given question. Here is the data:\n"
        "[BEGIN DATA]\n************\n[Question]: {input}\n************\n[Expert]: {ideal}\n************\n"
        "[Submission]: {completion}\n************\n[END DATA]\n\n"
        "Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.\n"
        "The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. "
        "Answer the question by selecting one of the following options:\n"
        "(A) The submitted answer is a subset of the expert answer and is fully consistent with it.\n"
        "(B) The submitted answer is a superset of the expert answer and is fully consistent with it.\n"
        "(C) The submitted answer contains all the same details as the expert answer.\n"
        "(D) There is a disagreement between the submitted answer and the expert answer.\n"
        "(E) The answers differ, but these differences don't matter from the perspective of factuality."
    ),
    "choice_strings": "ABCDE",
    "input_outputs": {"input": "completion"},
}
CLOSEDQA_SPEC = {
    "prompt": (
        "You are assessing a submitted answer on a given task based on a criterion. Here is the data:\n"
        "[BEGIN DATA]\n***\n[Task]: {input}\n***\n[Submission]: {completion}\n***\n[Criterion]: {criteria}\n***\n[END DATA]\n"
        "Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure "
        "that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "
        '"Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the '
        "letter again by itself on a new line.\n\nReasoning:"
    ),
    "eval_type": "cot_classify",
    "choice_scores": {"Y": 1.0, "N": 0.0},
    "choice_strings": "YN",
    "input_outputs": {"input": "completion"},
}
BUILTIN_SPECS = {"fact": FACT_SPEC, "closedqa": CLOSEDQA_SPEC}
PASSING_CHOICES = {"fact": ["A", "B", "C", "E"]}


@dataclass
class EvalImport:
    name: str
    eval_id: str | None = None
    cases: int = 0
    checks: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    run_id: str | None = None
    with_baseline: int = 0


@dataclass
class ImportResult:
    cases: list[TestCase] = field(default_factory=list)
    evals: list[EvalImport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    agreement: dict = field(default_factory=dict)


# ------------------------------------------------------------- utilities


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "eval").lower()).strip("-")
    return s[:48] or "eval"


def canon_content(content: Any) -> Any:
    """OpenAI Evals content (str | {type: input_text} | list) -> canonical content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return str(content)
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            ptype = part.get("type")
            if ptype in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype in ("input_image", "image_url"):
                img = part.get("image_url")
                url = img.get("url") if isinstance(img, dict) else img
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif ptype == "input_audio":
                parts.append({"type": "text", "text": "[audio input omitted]"})
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def canon_messages(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    out = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        name = m.get("name")
        if role == "system" and name in ("example_user", "example_assistant"):
            role = "user" if name == "example_user" else "assistant"
        msg: dict = {"role": role, "content": canon_content(m.get("content"))}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return msgs.clean_messages(out) if out else []


def _sampling_to_params(sp: Any) -> dict:
    if not isinstance(sp, dict):
        return {}
    params: dict = {}
    for key in ("temperature", "top_p", "seed", "reasoning_effort"):
        if sp.get(key) is not None:
            params[key] = sp[key]
    mt = sp.get("max_completion_tokens") or sp.get("max_completions_tokens") or sp.get("max_tokens")
    if mt:
        params["max_completion_tokens"] = mt
    rf = sp.get("response_format")
    if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        params["response_format"] = rf
    text = sp.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        fmt = text["format"]
        if fmt.get("type") == "json_schema":
            params["response_format"] = {"type": "json_schema", "json_schema": {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}}
        elif fmt.get("type") == "json_object":
            params["response_format"] = {"type": "json_object"}
    tools = []
    for t in sp.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("function"), dict):
            tools.append({"type": "function", "function": t["function"]})
        elif t.get("type") == "function" and t.get("name"):
            fn = {"name": t["name"], "parameters": t.get("parameters") or {}}
            if t.get("description"):
                fn["description"] = t["description"]
            tools.append({"type": "function", "function": fn})
    if tools:
        params["tools"] = tools
    return params


_SAMPLE_TEXT_RE = re.compile(r"^\s*\{\{\s*sample\.output_text\s*\}\}\s*$")


def _target(template: Any) -> str | None:
    if not isinstance(template, str) or _SAMPLE_TEXT_RE.match(template):
        return None
    return template


def _static(template: Any, item: dict) -> Any:
    """Render item-only templates now (readable test files); keep sample templates."""
    if uses_namespace(template, "sample"):
        return template
    return render(template, {"item": item})


def convert_grader(g: dict, item: dict, warnings: list[str]) -> dict | None:
    """One hosted-Evals testing criterion -> one LLMShift check."""
    gtype = g.get("type")
    base = {"name": g.get("name") or gtype, "grader_id": g.get("id") or g.get("name"), "origin": f"openai_evals:{gtype}"}
    if gtype == "string_check":
        op = (g.get("operation") or "eq").lower()
        value = _static(g.get("reference", ""), item)
        value = value if isinstance(value, (str, list)) else json.dumps(value)
        check: dict
        if op == "eq":
            check = {"type": "equals", "value": value, "category": "custom"}
        elif op in ("ne", "neq"):
            check = {"type": "not_equals", "value": value, "category": "custom"}
        elif op == "like":
            check = {"type": "contains", "value": value, "case_sensitive": True, "category": "custom"}
        elif op == "ilike":
            check = {"type": "contains", "value": value, "case_sensitive": False, "category": "custom"}
        else:
            warnings.append(f"string_check operation {op!r} is not supported; grader {g.get('name')!r} skipped")
            return None
        target = _target(g.get("input"))
        if target:
            check["target"] = target
        return {**check, **base}
    if gtype == "text_similarity":
        metric = (g.get("evaluation_metric") or "fuzzy_match").replace("-", "_")
        check = {"type": "similarity", "metric": metric, "threshold": float(g.get("pass_threshold", 0.5)), "reference": _static(g.get("reference", ""), item), "category": "custom", "severity": "error"}
        if metric == "cosine":
            check["embedding_model"] = "openai:text-embedding-3-large"
        target = _target(g.get("input"))
        if target:
            check["target"] = target
        return {**check, **base}
    if gtype == "label_model":
        return {
            "type": "llm_label",
            "model": g.get("model"),
            "input": g.get("input") or [],
            "labels": g.get("labels") or [],
            "passing_labels": g.get("passing_labels") or [],
            **({"sampling_params": g["sampling_params"]} if g.get("sampling_params") else {}),
            **base,
        }
    if gtype == "score_model":
        check = {"type": "llm_score", "model": g.get("model"), "input": g.get("input") or [], "range": g.get("range") or [0, 1], **base}
        if g.get("pass_threshold") is not None:
            check["pass_threshold"] = g["pass_threshold"]
        if g.get("sampling_params"):
            check["sampling_params"] = g["sampling_params"]
        return check
    if gtype == "python":
        check = {"type": "python", "source": g.get("source") or "", **base}
        if g.get("pass_threshold") is not None:
            check["pass_threshold"] = g["pass_threshold"]
        if g.get("image_tag"):
            check["image_tag"] = g["image_tag"]
        return check
    if gtype == "multi":
        subs = g.get("graders") or {}
        if isinstance(subs, dict) and subs.get("type"):  # SDK typing quirk: a single grader
            subs = {subs.get("name") or "grader": subs}
        converted = {}
        for name, sub in subs.items():
            c = convert_grader(sub, item, warnings)
            if c is None:
                return None
            converted[name] = c
        check = {"type": "multi", "graders": converted, "calculate_output": g.get("calculate_output") or "0", **base}
        if g.get("pass_threshold") is not None:
            check["pass_threshold"] = g["pass_threshold"]
        return check
    warnings.append(f"grader type {gtype!r} is not supported; grader {g.get('name')!r} skipped")
    return None


# ------------------------------------------------------------ hosted API


class EvalsClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set (needed to read your evals from the OpenAI API)")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        org, project = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION"), os.environ.get("OPENAI_PROJECT_ID")
        if org:
            self.headers["OpenAI-Organization"] = org
        if project:
            self.headers["OpenAI-Project"] = project

    def get(self, path: str) -> Any:
        return http.get_json(f"{self.base_url}{path}", headers=self.headers, timeout=120, max_retries=5)

    def paginate(self, path: str, limit: int = 100, extra: str = "") -> Iterator[dict]:
        after = None
        sep = "&" if "?" in path else "?"
        while True:
            q = f"{path}{sep}limit={limit}{extra}" + (f"&after={after}" if after else "")
            page = self.get(q)
            data = page.get("data") or []
            yield from data
            if not page.get("has_more") or not data:
                return
            after = page.get("last_id") or data[-1].get("id")

    def file_content(self, file_id: str) -> bytes:
        return http.request("GET", f"{self.base_url}/files/{file_id}/content", headers=self.headers, timeout=300, max_retries=5, raw=True)


def _items_from_jsonl(data: bytes | str) -> list[dict]:
    text = data.decode("utf-8-sig", "replace") if isinstance(data, bytes) else data
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec if "item" in rec else {"item": rec})
    return out


def _messages_for_item(data_source: dict, item: dict, sample: dict | None) -> list[dict]:
    if sample and sample.get("input"):
        return canon_messages(sample["input"])
    im = (data_source or {}).get("input_messages") or {}
    if im.get("type") == "template":
        rendered = []
        for m in im.get("template") or []:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, dict) and "text" in content:
                text = render_text(content["text"], {"item": item})
                rendered.append({"role": m.get("role", "user"), "content": text})
            elif isinstance(content, list):
                rendered.append({"role": m.get("role", "user"), "content": canon_content(render(content, {"item": item}))})
            else:
                rendered.append({"role": m.get("role", "user"), "content": render_text(content or "", {"item": item})})
        return canon_messages(rendered)
    if im.get("type") == "item_reference":
        ref = str(im.get("item_reference") or "item.input")
        from ..templating import MissingVariable, lookup

        try:
            value = lookup(ref, {"item": item})
        except MissingVariable:
            return []
        return canon_messages(value)
    for key in ("input", "messages", "prompt", "question"):
        if key in item:
            return canon_messages(item[key])
    return []


def _baseline_from_sample(sample: dict | None) -> tuple[str | None, list[dict], str | None]:
    if not isinstance(sample, dict):
        return None, [], None
    outputs = sample.get("output")
    if isinstance(outputs, list) and outputs:
        first = outputs[0] if isinstance(outputs[0], dict) else {}
        calls = [c for c in (msgs.normalize_tool_call(tc) for tc in first.get("tool_calls") or []) if c]
        return msgs.content_text(first.get("content")), calls, sample.get("model")
    if "choices" in sample:
        from ..providers.openai import OpenAIProvider

        try:
            r = OpenAIProvider.parse_chat(sample)
            return r.text, r.tool_calls, r.model
        except Exception:
            return None, [], None
    if isinstance(sample.get("output_text"), str):
        return sample["output_text"], [], sample.get("model")
    return None, [], None


def convert_hosted(
    ev: dict,
    *,
    run: dict | None = None,
    output_items: list[dict] | None = None,
    items: list[dict] | None = None,
    max_items: int | None = None,
) -> tuple[list[TestCase], EvalImport]:
    name = ev.get("name") or ev.get("id") or "eval"
    info = EvalImport(name=name, eval_id=ev.get("id"), run_id=(run or {}).get("id"))
    criteria = [g for g in ev.get("testing_criteria") or [] if isinstance(g, dict)]
    data_source = (run or {}).get("data_source") or {}
    params = _sampling_to_params(data_source.get("sampling_params"))
    group = f"{_slug(name)}-{stable_hash(ev.get('id') or name, 4)}"
    rows: list[tuple[dict, dict | None, dict]] = []  # (item, sample, extra metadata)
    if output_items:
        for oi in output_items:
            item = oi.get("datasource_item") or {}
            meta = {"datasource_item_id": oi.get("datasource_item_id"), "openai_status": oi.get("status")}
            if oi.get("results"):
                meta["openai_results"] = [{"name": r.get("name"), "passed": r.get("passed"), "score": r.get("score")} for r in oi["results"] if isinstance(r, dict)]
            rows.append((item, oi.get("sample"), meta))
        rows.sort(key=lambda r: (r[2].get("datasource_item_id") is None, r[2].get("datasource_item_id") or 0))
    else:
        for rec in items or []:
            rows.append((rec.get("item") or {}, rec.get("sample"), {}))
    if max_items:
        rows = rows[:max_items]
    if not rows:
        dsc = (ev.get("data_source_config") or {}).get("type")
        if dsc in ("logs", "stored_completions"):
            info.warnings.append(
                f"eval {name!r} reads stored completions/logs and has no run output items to import; "
                "export those logs and run `llmshift snapshot` on them instead"
            )
        else:
            info.warnings.append(f"eval {name!r} has no runs or data items to import")
        return [], info
    cases: list[TestCase] = []
    for item, sample, meta in rows:
        messages = _messages_for_item(data_source, item, sample)
        if not messages:
            info.warnings.append(f"skipped an item of {name!r}: could not build the prompt (no template, item_reference or 'input' field)")
            continue
        case_params = dict(params)
        if isinstance(sample, dict):
            for key in ("temperature", "top_p", "seed"):
                if sample.get(key) is not None and key not in case_params:
                    case_params[key] = sample[key]
            if sample.get("max_completion_tokens") and "max_completion_tokens" not in case_params:
                case_params["max_completion_tokens"] = sample["max_completion_tokens"]
        checks = []
        for g in criteria:
            c = convert_grader(g, item, info.warnings)
            if c is not None:
                checks.append(c)
                info.checks[c["type"]] += 1
        text, calls, model = _baseline_from_sample(sample)
        baseline = None
        if text is not None or calls:
            baseline = {"output": text or ""}
            if calls:
                baseline["tool_calls"] = calls
            if model or run and run.get("model"):
                baseline["model"] = model or run.get("model")
            info.with_baseline += 1
        inp = {"messages": messages, "params": case_params}
        metadata = {"source": "openai-evals", "eval_id": ev.get("id"), "eval_name": name}
        if run:
            metadata["run_id"] = run.get("id")
        metadata.update({k: v for k, v in meta.items() if v is not None})
        cases.append(
            TestCase(
                id="oe_" + stable_hash({"eval": ev.get("id") or name, "item": item, "input": inp}, 10),
                input=inp,
                baseline=baseline,
                checks=checks,
                group=group,
                tags=["openai-evals", _slug(name)],
                vars=item,
                metadata=metadata,
            )
        )
    info.warnings = list(dict.fromkeys(info.warnings))
    info.cases = len(cases)
    return cases, info


def _outputs_of_one_run(output_items: list[dict], run: dict | None, runs: list[dict]) -> tuple[list[dict], dict | None]:
    """Keep the output items of a single run, so baselines and sampling settings match.
    Prefer the picked run; if the export has no outputs for it, use the newest run that has some."""
    run_ids = {o.get("run_id") for o in output_items if o.get("run_id")}
    if not run_ids:
        return output_items, run
    if run is None or run.get("id") not in run_ids:
        newest = max(run_ids, key=lambda rid: max((o.get("created_at") or 0) for o in output_items if o.get("run_id") == rid))
        run = next((r for r in runs if r.get("id") == newest), run)
        keep = newest
    else:
        keep = run.get("id")
    return [o for o in output_items if o.get("run_id") in (None, keep)], run


def _pick_run(runs: list[dict]) -> dict | None:
    completed = [r for r in runs if r.get("status") == "completed"]
    pool = completed or runs
    return max(pool, key=lambda r: r.get("created_at") or 0) if pool else None


def import_from_api(
    eval_ids: list[str] | None = None,
    *,
    all_evals: bool = False,
    client: EvalsClient | None = None,
    max_items: int | None = None,
    name_filter: list[str] | None = None,
    progress: Any = None,
) -> ImportResult:
    client = client or EvalsClient()
    result = ImportResult()
    if eval_ids:
        evals = [client.get(f"/evals/{eid}") for eid in eval_ids]
    elif all_evals:
        evals = list(client.paginate("/evals", 100, "&order=desc"))
    else:
        raise ValueError("pass --eval-id ID (repeatable) or --all")
    if name_filter:
        evals = [e for e in evals if any(fnmatch.fnmatch((e.get("name") or "").lower(), p.lower()) for p in name_filter)]
    for ev in evals:
        if progress:
            progress(f"reading {ev.get('name') or ev.get('id')}")
        runs = list(client.paginate(f"/evals/{ev['id']}/runs", 100, "&order=desc"))
        run = _pick_run(runs)
        output_items: list[dict] = []
        items: list[dict] = []
        if run:
            output_items = list(client.paginate(f"/evals/{ev['id']}/runs/{run['id']}/output_items", 100))
            if not output_items:
                source = (run.get("data_source") or {}).get("source") or {}
                if source.get("type") == "file_content":
                    items = [c for c in source.get("content") or [] if isinstance(c, dict)]
                elif source.get("type") == "file_id" and source.get("id"):
                    items = _items_from_jsonl(client.file_content(source["id"]))
        cases, info = convert_hosted(ev, run=run, output_items=output_items, items=items, max_items=max_items)
        result.cases.extend(cases)
        result.evals.append(info)
        result.warnings.extend(f"{info.name}: {w}" for w in info.warnings)
    return result


def import_from_export(path: str, data: str | None = None, max_items: int | None = None) -> ImportResult:
    """Exported eval JSON (eval object / {eval, runs, output_items} / list) + optional data JSONL."""
    raw_text = Path(path).read_text(encoding="utf-8-sig")
    result = ImportResult()
    try:
        doc: Any = json.loads(raw_text)
        docs = doc if isinstance(doc, list) else [doc]
    except ValueError:
        docs = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    evals: list[dict] = []
    loose_output_items: list[dict] = []
    loose_items: list[dict] = []
    runs: list[dict] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        if d.get("object") == "eval" or "testing_criteria" in d:
            evals.append(d)
        elif isinstance(d.get("eval"), dict):
            evals.append({**d["eval"], "_runs": d.get("runs") or [], "_output_items": d.get("output_items") or [], "_items": d.get("items") or []})
        elif d.get("object") == "eval.run.output_item":
            loose_output_items.append(d)
        elif d.get("object") == "eval.run":
            runs.append(d)
        elif "item" in d:
            loose_items.append(d)
    if not evals:
        raise ValueError(f"{path}: no eval object found (expected the JSON from GET /v1/evals/{{id}})")
    extra_items = _items_from_jsonl(Path(data).read_bytes()) if data else []
    for ev in evals:
        ev_runs = ev.pop("_runs", []) or [r for r in runs if r.get("eval_id") == ev.get("id")]
        ev_out = ev.pop("_output_items", []) or [o for o in loose_output_items if o.get("eval_id") in (None, ev.get("id"))]
        ev_items = ev.pop("_items", []) or loose_items or extra_items
        run = _pick_run(ev_runs)
        ev_out, run = _outputs_of_one_run(ev_out, run, ev_runs)
        if run and not ev_items and not ev_out:
            source = (run.get("data_source") or {}).get("source") or {}
            if source.get("type") == "file_content":
                ev_items = [c for c in source.get("content") or [] if isinstance(c, dict)]
        cases, info = convert_hosted(ev, run=run, output_items=ev_out, items=ev_items, max_items=max_items)
        if not cases and not ev_items and not ev_out:
            info.warnings.append("pass the eval's data file with --data items.jsonl")
        result.cases.extend(cases)
        result.evals.append(info)
        result.warnings.extend(f"{info.name}: {w}" for w in info.warnings)
    return result


# --------------------------------------------------------------- OSS evals


def _registry_root(path: Path) -> tuple[Path, list[Path]]:
    if path.is_file():
        root = path.parent.parent if path.parent.name == "evals" else path.parent
        return root, [path]
    if (path / "evals" / "registry" / "evals").is_dir():  # a checkout of the openai/evals repository
        return path / "evals" / "registry", sorted((path / "evals" / "registry" / "evals").rglob("*.y*ml"))
    if (path / "evals").is_dir():
        return path, sorted((path / "evals").rglob("*.y*ml"))
    if (path / "registry" / "evals").is_dir():
        return path / "registry", sorted((path / "registry" / "evals").rglob("*.y*ml"))
    if path.name == "evals" and path.is_dir():
        return path.parent, sorted(path.rglob("*.y*ml"))
    return path, sorted(path.rglob("*.y*ml"))


def _load_yaml_entries(files: Iterable[Path], warnings: list[str]) -> dict:
    import yaml

    entries: dict = {}
    for f in files:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            warnings.append(f"{f}: cannot parse YAML ({exc})")
            continue
        if isinstance(data, dict):
            for key, value in data.items():
                entries[str(key)] = value
    return entries


def _read_samples(root: Path, ref: str, warnings: list[str]) -> list[dict]:
    p = Path(ref)
    candidates = [p, root / "data" / ref, root / ref]
    target = next((c for c in candidates if c.exists()), None)
    if target is None:
        warnings.append(f"samples file not found: {ref} (looked in {root / 'data'})")
        return []
    files = sorted(target.rglob("*.jsonl")) + sorted(target.rglob("*.jsonl.gz")) if target.is_dir() else [target]
    out: list[dict] = []
    for f in files:
        if f.suffix in (".lz4", ".zst"):
            warnings.append(f"{f}: compressed format not supported; decompress it first")
            continue
        try:
            raw = gzip.open(f, "rb").read() if f.suffix == ".gz" else f.read_bytes()
        except OSError as exc:
            warnings.append(f"{f}: {exc}")
            continue
        text = raw.decode("utf-8-sig", "replace")
        if text.startswith("version https://git-lfs"):
            warnings.append(f"{f} is a Git LFS pointer; run `git lfs pull` in the evals repo first")
            continue
        for line in text.splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        out.append(rec)
                except ValueError:
                    warnings.append(f"{f}: skipped an invalid JSON line")
    return out


def _resolve_spec(entries: dict, name: str, depth: int = 0) -> dict | None:
    if depth > 10:
        return None
    value = entries.get(name)
    if isinstance(value, str):
        return _resolve_spec(entries, value, depth + 1)
    if isinstance(value, dict):
        if "class" in value:
            return value
        if value.get("id") and value.get("id") != name:
            return _resolve_spec(entries, value["id"], depth + 1)
    return None


def import_oss_registry(path: str, *, eval_names: list[str] | None = None, max_items: int | None = None) -> ImportResult:
    result = ImportResult()
    root, files = _registry_root(Path(path))
    if not files:
        raise ValueError(f"no eval YAML files found under {path}")
    entries = _load_yaml_entries(files, result.warnings)
    mg_specs = dict(BUILTIN_SPECS)
    mg_dir = root / "modelgraded"
    if mg_dir.is_dir():
        mg_specs.update(_load_yaml_entries(sorted(mg_dir.glob("*.y*ml")), result.warnings))
    # Aliases (a string value, or a dict with an "id") name the evals; run entries carry the class.
    base_names = [k for k, v in entries.items() if isinstance(v, str) or (isinstance(v, dict) and "class" not in v and v.get("id"))]
    if not base_names:  # a file with only run entries
        base_names = [k for k, v in entries.items() if isinstance(v, dict) and "class" in v]
    if eval_names:
        base_names = [n for n in base_names if any(fnmatch.fnmatch(n, p) for p in eval_names)]
        if not base_names:
            raise ValueError(f"no eval matches {', '.join(eval_names)}")
    imported_specs: set[int] = set()
    for name in sorted(base_names, key=lambda n: (len(n), n)):
        spec = _resolve_spec(entries, name)
        if spec is not None:
            if id(spec) in imported_specs:  # reached through several aliases: import once
                continue
            imported_specs.add(id(spec))
        info = EvalImport(name=name)
        if spec is None:
            info.warnings.append(f"cannot resolve {name!r} to a spec with a class")
            result.evals.append(info)
            continue
        cls = str(spec.get("class", ""))
        args = spec.get("args") or {}
        samples_ref = args.get("samples_jsonl")
        if not samples_ref:
            info.warnings.append(f"{name}: no samples_jsonl")
            result.evals.append(info)
            continue
        samples = _read_samples(root, samples_ref, info.warnings)
        few_shot = _read_samples(root, args["few_shot_jsonl"], info.warnings)[: int(args.get("num_few_shot", 0) or 0)] if args.get("few_shot_jsonl") else []
        short = cls.rsplit(":", 1)[-1]
        if max_items:
            samples = samples[:max_items]
        group = _slug(name)
        for idx, sample in enumerate(samples):
            raw_input = sample.get("input")
            if raw_input is None:
                continue
            prompt = raw_input
            if few_shot and isinstance(raw_input, list):
                shots = [m for s in few_shot for m in (s.get("sample") or [])]
                prompt = raw_input[:-1] + shots + raw_input[-1:]
            messages = canon_messages(prompt)
            if not messages:
                continue
            ideal = sample.get("ideal")
            ideals = ideal if isinstance(ideal, list) else ([ideal] if ideal is not None else [])
            ideals = [i if isinstance(i, str) else json.dumps(i) for i in ideals]
            checks: list[dict] = []
            src = f"openai_evals_oss:{short}"
            if short == "Match":
                checks.append({"type": "starts_with", "value": ideals, "origin": src, "category": "custom"})
            elif short == "Includes":
                checks.append({"type": "contains_any", "value": ideals, "ignore_case": bool(args.get("ignore_case", False)), "origin": src, "category": "custom"})
            elif short == "FuzzyMatch":
                checks.append({"type": "fuzzy_match", "value": ideals, "origin": src, "category": "custom"})
            elif short == "JsonMatch":
                checks.append({"type": "json_equals", "values": ideals, "origin": src, "category": "custom"})
            elif short == "ModelBasedClassify":
                spec_name = args.get("modelgraded_spec")
                mg = mg_specs.get(spec_name) if isinstance(spec_name, str) else None
                if not isinstance(mg, dict):
                    info.warnings.append(f"model-graded spec {spec_name!r} not found; add the registry's modelgraded/ folder")
                    break
                check = {
                    "type": "modelgraded",
                    "spec": spec_name,
                    "prompt": mg.get("prompt"),
                    "choice_strings": list(mg["choice_strings"]) if isinstance(mg.get("choice_strings"), str) else mg.get("choice_strings"),
                    "input_outputs": mg.get("input_outputs") or {"input": "completion"},
                    "origin": src,
                }
                if args.get("eval_type"):
                    check["eval_type"] = args["eval_type"]
                if mg.get("eval_type"):
                    check["spec_eval_type"] = mg["eval_type"]
                if mg.get("choice_scores") is not None:
                    check["choice_scores"] = mg["choice_scores"]
                if args.get("modelgraded_spec_args"):
                    check["args"] = args["modelgraded_spec_args"]
                if args.get("match_fn"):
                    check["match_fn"] = args["match_fn"]
                if spec_name in PASSING_CHOICES:
                    check["passing_choices"] = PASSING_CHOICES[spec_name]
                if not check.get("eval_type") and not check.get("spec_eval_type"):
                    check["eval_type"] = "classify"
                checks.append(check)
            else:
                info.warnings.append(f"eval class {cls} is not supported (custom eval logic); {len(samples)} samples skipped")
                break
            for c in checks:
                info.checks[c["type"]] += 1
            inp = {"messages": messages, "params": {"temperature": 0}}
            baseline = None
            completion = sample.get("completion")
            if isinstance(completion, str):
                baseline = {"output": completion}
                info.with_baseline += 1
            variables = {k: v for k, v in sample.items() if k != "completion"}
            result.cases.append(
                TestCase(
                    id=f"oss_{_slug(name)[:24]}_{stable_hash(sample, 8)}",
                    input=inp,
                    baseline=baseline,
                    checks=checks,
                    group=group,
                    tags=["openai-evals-oss", group],
                    vars=variables,
                    metadata={"source": "openai-evals-oss", "eval": name, "class": cls, "sample_index": idx},
                )
            )
            info.cases += 1
        result.evals.append(info)
        result.warnings.extend(f"{name}: {w}" for w in dict.fromkeys(info.warnings))
    return result


# ------------------------------------------------------------- fidelity


def verify_against_recorded(cases: list[TestCase], allow_python: bool = False) -> dict:
    """Re-grade recorded baselines with the converted checks and compare with
    OpenAI's recorded grader results (deterministic graders only)."""
    from ..checks import CheckContext, Sample, evaluate

    ctx = CheckContext(allow_python=allow_python)
    per_type: dict[str, Counter] = {}
    for case in cases:
        recorded = {r.get("name"): r for r in case.metadata.get("openai_results") or []}
        if not recorded or not case.has_baseline:
            continue
        sample = Sample(text=case.baseline_output, tool_calls=case.baseline_tool_calls)
        for check in case.checks:
            if check.get("type") in ("llm_label", "llm_score", "modelgraded", "llm_judge"):
                continue
            if check.get("type") == "similarity" and check.get("metric") == "cosine":
                continue  # OpenAI graded with embeddings; the offline fallback would not agree

            rec = recorded.get(check.get("grader_id")) or recorded.get(check.get("name"))
            if rec is None or rec.get("passed") is None:
                continue
            res = evaluate(check, sample, case, ctx)
            if res.status == "skip":
                continue
            ours = res.status == "pass"
            stats = per_type.setdefault(check.get("origin", check["type"]), Counter())
            stats["total"] += 1
            stats["agree"] += int(ours == bool(rec["passed"]))
    total = sum(c["total"] for c in per_type.values())
    agree = sum(c["agree"] for c in per_type.values())
    return {
        "graded": total,
        "agree": agree,
        "rate": round(agree / total, 4) if total else None,
        "by_grader": {k: {"total": v["total"], "agree": v["agree"]} for k, v in per_type.items()},
    }


def grader_model_warnings(cases: list[TestCase]) -> list[str]:
    """Warn when graders depend on models that are retired or retiring."""
    from ..registry import Registry

    try:
        reg = Registry.load()
    except Exception:
        return []
    models = Counter(c.get("model") for case in cases for c in case.checks if c.get("model"))
    out = []
    for model, n in models.items():
        lk = reg.lookup(str(model))
        if lk.status in ("retired", "retiring", "deprecated"):
            out.append(f"{n} grader check(s) use {model}, which is {lk.message}. Pass --judge provider:model to grade with a current model.")
    return out

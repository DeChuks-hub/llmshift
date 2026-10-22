# LLMShift

**Switch AI models without breaking your product.**

Providers now retire models roughly every six months. When a model's behaviour changes, outputs can look fine and still be wrong, and nothing raises an error. LLMShift is built around the moment of switching models, from finding the calls to fixing the prompts:

| Command | What it does |
|---|---|
| `llmshift scan` | Finds every model your code calls (Python, JS/TS, Go, Java, configs, `.env`, notebooks) and warns about retirement dates. |
| `llmshift snapshot` | Turns real usage logs into a test set automatically. You write no tests. |
| `llmshift compare` | Runs the tests on the new model and shows exactly what changed: broken formats, wrong facts, different tone. |
| `llmshift fix` | Suggests prompt changes until the new model matches the old one, and points to where each prompt lives in your code. |
| `llmshift watch` | Runs a scheduled check against the live model and alerts you (Slack, Teams, GitHub issue, e-mail, ...) when its behaviour drifts. |
| `llmshift import openai-evals` | Brings in OpenAI Evals tests. The Evals platform goes read-only on **31 Oct 2026** and its API shuts down on **30 Nov 2026**. |

LLMShift is a single Python package with one dependency (PyYAML, plus `tomli` on Python 3.9 and 3.10 to read `pyproject.toml`). It talks to OpenAI, Azure OpenAI, Anthropic, Gemini and any OpenAI-compatible endpoint over plain HTTPS, so there are no SDK version conflicts. You can also run it as a GitHub Action.

```bash
pip install llmshift
llmshift demo          # 1-minute offline walkthrough with simulated models (no API key)
```

---

## Quick start with your own app

```bash
# 1. Which models does the code use, and when do they retire?
llmshift scan

# 2. Turn real traffic into tests (OpenAI/Anthropic/Gemini logs, LangSmith/Langfuse exports, CSV, ...)
llmshift snapshot logs/ --model "gpt-4o*" --since 2026-09-01

# 3. What changes on the new model?
export OPENAI_API_KEY=...
llmshift compare --to openai:gpt-5.6-sol

# 4. Get prompt changes that close the gaps
llmshift fix --to openai:gpt-5.6-sol

# 5. After switching, watch the live model every night
llmshift watch --model openai:gpt-5.6-sol --alert slack:$SLACK_WEBHOOK_URL
```

`llmshift init --github` writes a `llmshift.yaml` and two workflows: PR checks and a nightly watch.

---

## scan: find every model call and its retirement date

```
$ llmshift scan
LLMShift scan  5 files, 6 model references, 4 distinct models   (lifecycle data: 2026-10-21)

  ✗ RETIRED          claude-3-5-haiku-20241022  anthropic
                     retired on 2026-02-19; replacement: claude-haiku-4-5-20251001
                       app/summaries.py:10  model="claude-3-5-haiku-20241022",
  ! 2 DAYS LEFT      gpt-4o-2024-05-13  openai
                     retires 2026-10-23 (2 days left); replacement: gpt-5.6-sol
                       app/triage.py:7  TRIAGE_MODEL = "gpt-4o-2024-05-13"
                       app/triage.py:17  model=TRIAGE_MODEL,  (via TRIAGE_MODEL, line 7)
                       app/replies.py:15  model=os.getenv("REPLY_MODEL", "gpt-4o-2024-05-13"),
```

- **Python** is analysed with the AST. It finds `model=`-style arguments and models passed to known constructors (`GenerativeModel`, `init_chat_model`, pydantic-ai `Agent`, LiteLLM, ...). It also handles `{"model": ...}` dicts, constants used elsewhere (`MODEL = "..."` then `model=MODEL`) and `os.getenv("X", "default")` fallbacks. Docstrings and comments are ignored.
- **Other code** (JS/TS, Go, Java, Kotlin, C#, Ruby, PHP, Rust, Swift, ...): string literals that are model IDs, and values of model-like keys.
- **Config**: YAML, JSON, TOML, `.env`, INI, Terraform and Dockerfiles. **Notebooks**: code cells, reported with cell numbers.
- **ID formats are normalised.** Bedrock (`us.anthropic.claude-...-v1:0`, full ARNs), Vertex (`claude-...@20241022`, `projects/.../models/...`), OpenRouter/LiteLLM prefixes (`openai/gpt-4o`), dotted spellings (`claude-3.5-sonnet`) and fine-tunes (`ft:gpt-4o-mini-...`, which follow their base model's schedule) all map to their base model.
- In a git repository, files git ignores are skipped unless you name them. `--exclude` takes globs: a bare name (`legacy`) matches at any depth, and paths may be relative to the scanned directory or the current one.
- **Lifecycle data** covers 617 models from OpenAI, Anthropic, Google (Gemini API and Vertex AI), Mistral, Cohere, xAI and DeepSeek. It was built from the providers' official deprecation pages on 2026-10-21. Status is computed from dates at run time, so a model turns "retired" on its shutdown day. Add or override entries with `--registry extra.json`; `llmshift models gpt-4o` searches the data.
- In CI, use `--fail-on retired|retiring|deprecated|unknown|never` (default `retired`). Output formats: `--format table|json|markdown|sarif`. Inside GitHub Actions, findings appear as annotations on the diff and in the job summary.

## snapshot: real usage logs become a test set

```bash
llmshift snapshot logs/ -o .llmshift/tests.jsonl
```

`snapshot` reads JSONL, JSON and CSV files (optionally `.gz`) and detects the format of each record:

- OpenAI Chat Completions, Responses API and **Batch** files (each batch's input and output files are joined on `custom_id`)
- Anthropic Messages and Gemini `generateContent` request/response pairs, including logged streams (server-sent events are reassembled)
- Helicone-style `{"request": {"body": ...}}` records and flat logs
- chat datasets (`{"messages": [...]}`, ShareGPT) and Langfuse / LangSmith-style exports
- any rows with `prompt` / `input` / `question` and `output` / `response` / `answer` columns

Embedding, moderation, image and audio calls are skipped, and every skipped record is counted with its reason. `--append` adds only cases that are not already in the test file.

Then it:

1. Drops failed calls and truncated outputs, and applies `--model`, `--since` and `--until` filters.
2. Redacts secrets. With `--redact-pii` it also replaces e-mails, phone numbers and card numbers, using consistent placeholders so echoed values still match.
3. Removes duplicates and groups requests by prompt template.
4. Picks a diverse sample per template (`--max-cases`, `--per-group`).
5. **Infers checks from what the old model actually did**:

| Output looks like | Checks recorded |
|---|---|
| JSON | valid JSON (no code fences), JSON schema (required keys are those present in every output of the template), field values for labels, flags, IDs and amounts |
| a label / number | exact value, plus the label vocabulary seen across the template |
| prose / markdown | facts (numbers with units and scale, dates, IDs, URLs, names), language, markdown structure, length band, tone, similarity |
| tool calls | same tools called, argument keys, short argument values |
| any | not empty, not a refusal |

Logs without outputs? Pass `--baseline-model openai:gpt-4o-2024-08-06` to generate them.

No logs yet? Record them from your app with one line:

```python
from llmshift.record import wrap
client = wrap(OpenAI(), path="logs/llm.jsonl")   # works with OpenAI, AsyncOpenAI, Anthropic, AsyncAnthropic, streaming
```

Streamed responses are logged whole once the stream ends, including tool calls and token usage.

The test file is plain JSON Lines that you can read, edit and commit:

```json
{"id": "c_bc05e87da4", "group": "acme-customer-support-assistant-8a772a",
 "input": {"messages": [{"role": "system", "content": "You are Acme's customer support assistant..."},
                        {"role": "user", "content": "Hi, my order #4521 hasn't arrived yet..."}], "params": {"temperature": 0.3}},
 "baseline": {"output": "Thank you for contacting Acme support. ...", "model": "gpt-4o-2024-05-13"},
 "checks": [{"type": "facts", "facts": [{"kind": "id", "text": "#4521"}, {"kind": "date", "text": "March 14", "value": "--03-14"}]},
            {"type": "tone"}, {"type": "length", "min_words": 15, "max_words": 95, "baseline_words": 40}]}
```

You can also write cases by hand in YAML (`prompt:`, `system:`, `baseline:`, `checks:`).

## compare: what exactly changes on the new model

```
$ llmshift compare --to openai:gpt-5.6-sol
  ✓ 0 unchanged   ! 6 changed   ✗ 12 failed
  pass rate ███████░░░░░░░░░░░░░ 33.3%  ✗ below the 95% threshold

What changed
  Broken format            8 failed
      JSON is wrapped in a markdown code fence (8)
  Wrong or missing facts   6 failed, 6 changed
      facts missing (6), e.g. no mention of "Acme"
      numbers changed (4), e.g. "12.5%" -> "12%"; "$4.2 million" -> "$4 million"; "3.1%" -> "3%"
      $.urgency changed (2), e.g. $.urgency: "medium" -> "high"
  Different tone / style   6 changed
      adds emoji (6) · more exclamatory (6) · more casual (6) · less polite (6)

  ✗ FAIL  c_0d9f963af7  support-ticket-triage-assistant-f98626
      Input: The tracking link for order #8812 doesn't work.
      ✗ format    JSON is wrapped in a markdown code fence (```json ... ```)
      Old: {"category": "shipping", "urgency": "low", "summary": "The tracking link for order #8812 doesn't work."}
      New: ```json
           { "category": "shipping", ... }
           ```
```

- Each case is **failed** (an error-level check broke: format, facts, tool calls, refusal, language) or **changed** (a warning-level difference: tone, length, structure, similarity). Change a check's level with `--severity tone=error`.
- `--baseline-model <old model>` also runs the old model live. A check that the old model fails as well is counted as noise, not a regression.
- `--judge openai:gpt-5.6-terra` adds an LLM equivalence judge, and is used by imported model graders.
- Reports: a self-contained HTML report with filters and word-level diffs, plus JSON, in `.llmshift/reports/`. Add `--markdown summary.md` for PR comments and `--junit junit.xml` for CI test views; without a file name they are written to `.llmshift/reports/` too. `llmshift report results.json --html` re-renders a saved run.
- Model quirks are handled automatically and listed in the report: parameters a model rejects (`temperature`, `max_tokens` vs `max_completion_tokens`, Claude's deprecated sampling parameters), Responses-only models, and the `developer` role. You'll also see notes such as "reasoning models count thinking tokens toward the output limit".
- Responses are cached in `.llmshift/cache`, so re-runs are free. Turn this off with `--no-cache`.

## fix: prompt changes that make the new model match

```
$ llmshift fix --to openai:gpt-5.6-sol
support-ticket-triage-assistant-f98626  (prompt at app/triage.py:10)
  before: 0 of 7 match (7 failing)   after: 5 of 7 match (2 failing)   via rules
  Add to the system prompt:
    + Return only the raw JSON object. Do not wrap it in markdown code fences (```) and do not add any text before or after it.
    + For "urgency", use exactly one of: low, medium, high.
  still different: facts: $.urgency: "medium" -> "high"

Overall pass rate: 33% → 89% with the suggested prompts
```

For each prompt template with regressions, LLMShift builds candidate prompts and tests each one on the new model, keeping the best. The candidates come from:

- targeted instructions for the exact failures (code fences, schema, label sets, tone markers, length, structure, language, exact figures, tool use, brand names)
- a worked example taken from a case that is **not** used for scoring
- optionally, a rewrite by an optimizer model (`--optimizer`)

Your code is never changed. You get the new prompt, a diff, and where the prompt lives in your repository. `--write-tests fixed.jsonl` writes a patched test set so you can re-check it with `compare`.

## watch: alert when the live model drifts

```bash
llmshift watch --model openai:gpt-5.6-sol --alert slack:$SLACK_WEBHOOK_URL --alert github
llmshift watch --model openai:gpt-5.6-sol --at 02:00      # keep running, once a day
```

Each run calls the live model (no cache) and appends to `.llmshift/watch/<model>/history.jsonl`. It alerts on:

- failing requests (the model may have been retired)
- a pass rate below `--min-pass-rate` (or `watch.min_pass_rate`, then `min_pass_rate`, from `llmshift.yaml`; set it to `null` to rely on drift alone)
- a pass-rate drop that is statistically significant (paired McNemar test on the tests that ran unchanged in both runs, so one flaky case does not page anyone)
- new kinds of failures, and outputs drifting away from the baseline
- a changed model version or backend fingerprint
- an upcoming retirement

When the test set changes, the next run becomes the new reference. Destinations: `slack:<url>`, `teams:<url>`, `discord:<url>`, `webhook:<url>`, `github` (opens or updates an issue), `email:<addr>` (SMTP) and `file:<path>`. `--alert-on` sets the lowest severity that is sent (default `warning`). The exit code is 1 when drift reaches `--fail-on` (default `warning`), so a scheduled CI job turns red.

## import: move off the OpenAI Evals platform

OpenAI's hosted Evals become read-only on 2026-10-31, and the dashboard and API shut down on 2026-11-30. OpenAI provides no export feature, so the API is the way out:

```bash
export OPENAI_API_KEY=...
llmshift import openai-evals --api --all --split .llmshift/evals/
llmshift compare --tests .llmshift/evals/ --to openai:gpt-5.6-sol --judge openai:gpt-5.6-terra
```

- For each eval, LLMShift reads the testing criteria, the latest run's data source, and the run's **output items**. Those hold the exact inputs sent and the outputs the model gave, which become recorded baselines.
- Graders map one-to-one:
  - `string_check` (eq/ne/like/ilike) → `equals` / `not_equals` / `contains`
  - `text_similarity` → `similarity`, with the same metric and threshold
  - `label_model` → `llm_label` and `score_model` → `llm_score`, graded by `--judge` or the grader's own model
  - `python` → `python`, which runs locally only with `--allow-python`
  - `multi` → `multi`
- After import, a **fidelity check** re-grades the recorded outputs and reports how often the converted graders reproduce OpenAI's recorded verdicts.
- `import` does not replace an existing test file unless you pass `--force` (or choose another file with `-o`).
- You are warned when grader models are themselves retiring (`o3-mini`, `o4-mini`, `gpt-4.1-nano` and others shut down on 2026-10-23).
- Also supported: exported eval JSON (`llmshift import openai-evals eval.json --data items.jsonl`) and the open-source `openai/evals` registry (`llmshift import openai-evals path/to/evals/registry`). The registry covers Match, Includes, FuzzyMatch, JsonMatch and ModelBasedClassify (`fact`, `closedqa`, custom specs).

## Models and providers

A model is written `provider:model`; well-known IDs work without a prefix.

| Spec | Credentials |
|---|---|
| `openai:gpt-5.6-sol`, `gpt-6-astra` | `OPENAI_API_KEY` (`OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` optional) |
| `azure:<deployment>` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` (or `AZURE_OPENAI_AD_TOKEN`); `api_version` default `2024-10-21`, or `v1` |
| `anthropic:claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| `gemini:gemini-3.8-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `openrouter:`, `together:`, `groq:`, `mistral:`, `deepseek:`, `xai:`, `fireworks:`, `perplexity:`, `cohere:` | `<NAME>_API_KEY` |
| `ollama:llama3.1:8b`, `lmstudio:...`, `vllm:...` | local, no key (`OLLAMA_HOST` respected) |
| `python:my_app.llm:answer` | your own function: test the real pipeline, not just the raw model |
| `<name>:model` from `providers:` in `llmshift.yaml` | any OpenAI-compatible endpoint (LiteLLM proxy for Bedrock/Vertex, ...) or a `command` |
| `mock:legacy`, `mock:nextgen`, `mock:echo` | offline simulated models (demo/tests) |

`llmshift doctor` shows which credentials it can see.

## GitHub Action

```yaml
- uses: DeChuks-hub/llmshift@v0.1.0
  with:
    command: compare
    args: --to openai:gpt-5.6-sol --min-pass-rate 0.95 --junit llmshift-junit.xml
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Inputs: `command`, `args`, `python-version`, `version` (empty = install from the action's source), `working-directory`. Outputs: `pass-rate`, `retired`, `retiring`. Complete PR and nightly workflows are in [`examples/workflows/`](examples/workflows/).

## Configuration

Settings live in `llmshift.yaml` or under `[tool.llmshift]` in `pyproject.toml`; the nearest one in the current directory or a parent (up to the git root) is used. Relative paths in it are relative to the file, so commands work from any subdirectory. `${VAR}` and `${VAR:-default}` are read from the environment. Command-line flags win. [`examples/llmshift.yaml`](examples/llmshift.yaml) documents every key: default models, judge, thresholds, severities, scan options, snapshot limits and redaction, fix rounds, watch thresholds and alerts, and custom providers.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the check failed: a retired model was found, the pass rate is below the threshold, or drift was detected |
| `2` | usage or configuration problem, or every request failed (wrong key or model name) |
| `130` | interrupted |

## Privacy and safety

- Prompts and outputs are sent only to the providers you name. The cache and reports stay on disk, and `init` adds them to `.gitignore`.
- `snapshot` and `record` redact secrets by default. PII redaction is available with `--redact-pii`.
- Imported Python graders execute code, so they only run with `--allow-python`, in a separate process.
- Reports escape all model output (HTML-safe).

## Limitations

- The format, fact and tone checks are deterministic heuristics, tuned to flag real changes. Edit thresholds per check in the test file, or add `--judge` for semantic judgement.
- Test cases replay single requests. Multi-step agent loops are not simulated; tool calls are compared, not executed.
- Bedrock and Vertex AI are reached through an OpenAI-compatible proxy (for example LiteLLM) or a `python:` provider. Their native APIs are not built in.
- Lifecycle data is a dated snapshot (2026-10-21). Statuses update themselves by date, but new deprecation announcements need a newer release or an override file.

## Development

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The test suite (194 tests) runs offline on Python 3.9 to 3.13. Provider and Evals-API tests use a local HTTP server that emulates the real APIs.

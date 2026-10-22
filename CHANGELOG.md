# Changelog

## Unreleased

- `pick`: when the model you use today is also a candidate, today's cost is measured on the same tests, so its row reads "same" instead of a saving against itself. The header says where today's cost comes from (your recorded usage, or these tests), and "Keep" is recommended whenever today's model is the cheapest that passes.
- `pick` from Python and the MCP `llmshift_pick` tool read `pick.volume` the way the command line does: `100k` in `llmshift.yaml` failed there with a TypeError.
- `demo`: the simulated production logs record the token usage the simulated model actually reports, so today's cost in `pick` matches a test run of the same model.
- `safety`: the header gives the probes each category really ran. Refusals use each test case at most once and add up to 10 harmless questions with alarming words, so with 18 test cases the refusal row shows 28, not "20 probes per category".
- README links work on the PyPI page (they were relative to the repository).
- The tests pass from the source archive on PyPI (one of them needed the repository's CI workflow); CI and the release now run the test suite from the built source archive, as packagers do.

**Safety and security**

- Secrets stay out of sight: API keys with a stray newline or invisible character are cleaned or reported without the key; an empty `base_url` (an unset `${VAR}`) is an error instead of sending your requests and key to the public API; only `http(s)` URLs are used, a URL with `user:password@` is refused without echoing the password, and error messages name only the host (a webhook URL's token is never shown).
- Model output, provider errors and scanned files cannot drive your terminal: escape sequences and bidi controls are shown as text. In PR comments, `@mentions` from model answers notify no one and a prompt or diff cannot close its code block.
- Redaction covers more key formats (xAI, Groq, Perplexity, Fireworks, Hugging Face, Replicate, Stripe, GitLab, npm, temporary AWS keys), `password=`/`api-key:` values, credentials in connection strings, Basic auth headers and the live values of key environment variables.
- Alerts: SMTP logs in only over TLS; Slack and Discord messages cannot ping `@channel`/`@everyone`. MCP tools read and write only inside the project.
- GitHub: a comment by another user is never edited, a POST that may have arrived is never sent again, and a value can no longer inject extra step outputs. The action's `args` are split like shell words (quote an argument with spaces) and never expanded. The workflows (and those `init --github` writes) ask for the least permissions and keep values out of `run:` scripts.

**Correct results**

- Pass-rate gates use the exact rate: 18,999 of 20,000 no longer passes a 95% gate. Case ids stay unique and stable however cases are selected, and `--limit 5` stops reading a huge test file after five cases.
- Inferred checks pass on the baseline they came from (JSON keys with odd characters, names like "Domino's Pizza", answers with no words). JSON checks reject `NaN` and `Infinity`, which most parsers refuse (a test whose own baseline has them keeps passing, and `NaN` now equals `NaN` in `json_match`).
- Log sampling keeps a fair share of every prompt template, however late it first appears. Timestamps read the same on Python 3.9 to 3.15 (offsets in hours, decimal commas, basic format, week dates, HTTP dates).
- `scan` resolves names in their own scope (a parameter no longer picks up another function's model), follows `os.getenv("X", "model")` defaults, and reads files past Windows' 260-character path limit. Skipped files are always listed.
- `--repeat`: a request that failed is counted against the pass rate but no longer makes a case "flaky".
- A judge that cannot answer (a wrong key) is reported as such instead of "0 judge calls", and `judge calibrate` keeps the last good result instead of saving an empty one.
- `fix` carries on with rule-based changes when the optimizer fails, stops when it repeats itself, and no longer lists logs or its own reports as places a prompt lives.
- Recording: calls through `with_raw_response` and `with_streaming_response` are logged; wrapping a client per request shares one shadow worker set.
- Providers: `python:` functions honour `timeout` when one is set (and no more than four calls that ran past it keep running), and their cache key covers the project modules they import; `command` providers honour it on Windows and notice an edited script. Compressed and mislabelled responses are read, JSON cut off in transit is fetched again, an HTML error page is summed up in one line, `OLLAMA_HOST` is read the way Ollama reads it, and a mistyped provider gets a "did you mean".
- Settings: values of the wrong type (`timeout: abc`, `headers` as a list, a `base_url` without `https://`) are reported where they are read; unknown or misspelt settings and severity levels are reported with a suggestion, and `severity: {tone: off}` works (YAML reads `off` as false).

**Fewer surprises**

- Files saved as UTF-16 (Windows PowerShell's `>`) are read, and a file that is not text is named in the error.
- `init` takes the baseline from the model your code calls most, and its listed replacement as the candidate, instead of writing a model that is itself retiring.
- `--config` and `--no-color` work after the subcommand too; `llmshift --help | head -1` exits quietly; `doctor` exits 1 when it finds a problem; two reports written in the same second no longer overwrite each other.
- Corrupted watch history, calibration and cache files are survived. `watch` writes the action's `pass-rate` output.
- `import openai-evals` imports odd or hostile exports and registries or says what is wrong, instead of crashing; registry samples outside the registry are not read.
- `shadow report` works from a subfolder, and `snapshot` turns shadow logs into test cases.
- The lifecycle data update holds back shutdown dates that moved earlier for a person to check, and the retirement calendar is rebuilt right after a data update instead of the next day.

## 0.1.0 (2026-10-22)

First release. Runs on Python 3.9 to 3.15 on Linux, macOS and Windows; pure Python, so it installs without a compiler (PyYAML's pure-Python parser is included under `llmshift/_vendor`, MIT License).

**Find what has to move**

- `scan`: model detection for Python (AST), JS/TS and other languages, configs, `.env` and notebooks. Lifecycle data covers 617 models from OpenAI, Anthropic, Google (Gemini API and Vertex AI), Mistral, Cohere, xAI and DeepSeek, built from official deprecation pages on 2026-10-21. Status is computed by date. Output as SARIF, JSON or Markdown, with GitHub annotations.
- `update`: fetches newer lifecycle data and prices; they are validated, stored per user, and used only when newer than the bundled copy. The data in this repository updates itself every day from the OpenAI, Anthropic, Gemini API and Vertex AI deprecation tables (a conservative merge, tested before it is committed); other source pages open an issue when they change.
- `digest`: one message listing the models your repositories use that are retired or retire soon, via Slack, Teams, Discord, e-mail, a webhook or a file.
- `calendar`: a static retirement-calendar website with an iCalendar (RFC 5545) feed and JSON, published daily to GitHub Pages by this repository; a daily watch of the providers' deprecation pages opens an issue when one changes.

**Turn real traffic into tests**

- `snapshot`: usage logs become JSONL test sets. Reads OpenAI chat/responses/batch, Anthropic, Gemini, Helicone, Langfuse/LangSmith-style exports, chat datasets, CSV and `llmshift.record` logs. Reassembles logged streams and pairs batch files per batch. Includes redaction, deduplication, grouping by prompt template, diverse sampling and inferred checks. Logs of any size are streamed, with a fair sample per template beyond `--max-records`.
- Agents: `llmshift.record.agent_run()` links the calls of one task, and unlabelled logs are linked when each request continues the previous one. Runs replay on the new model with the recorded tool results; the `agent` check reports missing, extra or changed tool calls, loops, order changes and step-limit runs, and reports show both step traces and the cost per task.
- `import openai-evals`: hosted Evals API (testing criteria, runs, output items as baselines), exported JSON and the OSS registry, with a fidelity check against recorded verdicts.

**Choose the new model**

- `compare`: format, facts, tone, tool-call, refusal, language, structure, length and similarity checks; an optional LLM judge; and a differential baseline that filters noise and compares the two models case by case (McNemar). Reports in terminal, HTML, JSON, Markdown and JUnit form. Automatically adapts to model API quirks.
- Confidence: every pass rate has a 95% range and the number of cases needed for ±5 points; `--repeat N` finds flaky cases. The judge is only asked when the rule-based checks cannot decide (`--judge-mode always` to ask every time), and `judge label` / `judge calibrate` measure its agreement with your own labels, which later runs quote.
- `pick`: the savings finder. It runs the test set on several cheaper models at once and recommends the cheapest one that still passes, showing each model's pass rate with a 95% range, cost per 1,000 requests, saving against today's model (monthly with `--volume`) and p50 latency. Prices come from a bundled LiteLLM snapshot, `pricing:` or `--price`.
- `safety`: refusal, prompt-injection and PII-leak probes built from your own traffic, run on the old and new model and compared pairwise (McNemar); exits 1 when the new model is significantly riskier.
- `embeddings`: measures how search results change between two embedding models on your own documents and queries (overlap, top-result survival, rank-biased overlap), and estimates re-indexing cost and time.

**Switch**

- `fix`: rule-based, example-based and optional optimizer-based prompt search, validated on the new model, with the location of each prompt in the codebase.
- `upgrade`: "Dependabot for AI models". It swaps retiring model ids where `scan` finds them, renames `max_tokens` for OpenAI's newer models, applies `fix` suggestions, tests the replacement and opens or updates a pull request.
- `check`: runs only the tests for the prompts a pull request edits, with the edited text, and comments with the result (one comment, kept up to date).

**After switching**

- Shadow mode: `wrap(client, shadow="provider:model", shadow_rate=0.05)` copies a share of live calls to a new model in the background, never blocking or breaking the app; `shadow report` checks the copies with the same checks as `compare`.
- `watch`: scheduled live checks with history, McNemar-tested drift on the tests that ran unchanged in both runs, version/fingerprint changes and retirement warnings. Alerts go to Slack, Teams, Discord, webhooks, GitHub issues, e-mail or files.

**Everywhere**

- `mcp`: a Model Context Protocol server (stdio) exposing scan, models, snapshot, compare, fix, pick, safety, check, upgrade and doctor as tools for Claude Code, Cursor, VS Code and other MCP clients.
- Providers: OpenAI (Chat + Responses), Azure OpenAI, Anthropic, Gemini, OpenAI-compatible presets, local servers, `python:` functions, `command` providers and offline mocks.
- `llmshift.record` client wrapper for the OpenAI and Anthropic SDKs: sync, async, streaming and the SDKs' stream helpers, `parse()` with pydantic response formats (logged as JSON schema), tool calls and usage; safe for several threads and processes writing one file. GitHub composite action; offline `demo`.
- Hardened against hostile input: YAML alias bombs, gzip bombs, huge, truncated or corrupt logs, broken Unicode, wrongly typed config values, oversized or deeply nested provider responses, unwritable paths and interrupted runs.

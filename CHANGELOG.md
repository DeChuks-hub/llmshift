# Changelog

## 0.1.0 (2026-10-22)

First release. Runs on Python 3.9 to 3.15 on Linux, macOS and Windows; pure Python, so it installs without a compiler (PyYAML's pure-Python parser is included under `llmshift/_vendor`, MIT License).

**Find what has to move**

- `scan`: model detection for Python (AST), JS/TS and other languages, configs, `.env` and notebooks. Lifecycle data covers 617 models from OpenAI, Anthropic, Google (Gemini API and Vertex AI), Mistral, Cohere, xAI and DeepSeek, built from official deprecation pages on 2026-10-21. Status is computed by date. Output as SARIF, JSON or Markdown, with GitHub annotations.
- `update`: fetches newer lifecycle data and prices; they are validated, stored per user, and used only when newer than the bundled copy.
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

# Changelog

## 0.1.0 (2026-10-21)

First release.

- `scan`: model detection for Python (AST), JS/TS and other languages, configs, `.env` and notebooks. Lifecycle data covers 617 models from OpenAI, Anthropic, Google (Gemini API and Vertex AI), Mistral, Cohere, xAI and DeepSeek, built from official deprecation pages on 2026-10-21. Status is computed by date. Output as SARIF, JSON or Markdown, with GitHub annotations.
- `snapshot`: usage logs become JSONL test sets. Reads OpenAI chat/responses/batch, Anthropic, Gemini, Helicone, Langfuse/LangSmith-style exports, chat datasets, CSV and `llmshift.record` logs. Reassembles logged streams and pairs batch files per batch. Includes redaction, deduplication, grouping by prompt template, diverse sampling and inferred checks.
- `compare`: format, facts, tone, tool-call, refusal, language, structure, length and similarity checks; an optional LLM judge; and a differential baseline that filters noise. Reports in terminal, HTML, JSON, Markdown and JUnit form. Automatically adapts to model API quirks.
- `fix`: rule-based, example-based and optional optimizer-based prompt search, validated on the new model, with the location of each prompt in the codebase.
- `watch`: scheduled live checks with history, McNemar-tested drift on the tests that ran unchanged in both runs, version/fingerprint changes and retirement warnings. Alerts go to Slack, Teams, Discord, webhooks, GitHub issues, e-mail or files.
- `import openai-evals`: hosted Evals API (testing criteria, runs, output items as baselines), exported JSON and the OSS registry, with a fidelity check against recorded verdicts.
- Providers: OpenAI (Chat + Responses), Azure OpenAI, Anthropic, Gemini, OpenAI-compatible presets, local servers, `python:` functions, `command` providers and offline mocks.
- `llmshift.record` client wrapper (sync, async and streaming, including tool calls and usage); GitHub composite action; offline `demo`.

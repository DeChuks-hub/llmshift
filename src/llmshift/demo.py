"""`llmshift demo`: an offline end-to-end walkthrough (no API keys needed).

It creates a small demo project - an "Acme support" app whose code calls a
model that is about to retire - writes realistic production logs, then runs
scan -> snapshot -> compare -> fix against two simulated models
(`mock:legacy` = the model in production, `mock:nextgen` = the replacement).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .providers.mock import legacy

TRIAGE_PROMPT = (
    "You are a support ticket triage assistant for Acme. Classify the customer's message and return JSON with the keys "
    '"category" (billing, shipping, technical or account), "urgency" (low, medium or high) and "summary" (one short sentence).'
)
REPLY_PROMPT = (
    "You are Acme's customer support assistant. Write a short, polite reply to the customer on behalf of the Acme support team. "
    "Mention the order number and the delivery date when they are relevant."
)
SUMMARY_PROMPT = "Summarise the following business update in two sentences for an executive audience. Keep the key figures."

TICKETS = [
    "I was charged twice for my subscription this month. Please refund the duplicate payment.",
    "My package was supposed to arrive last week and I'm still waiting. Where is it?",
    "The app crashes every time I open the settings page on Android.",
    "URGENT: I can't log in to my account and I have a presentation in an hour!!",
    "Can you update the email address on my account?",
    "I was billed again after cancelling. When will I get my money back?",
    "The tracking link for order #8812 doesn't work.",
    "Payment failed at checkout but the money left my bank account.",
]
MESSAGES = [
    "Hi, my order #4521 hasn't arrived yet. It was supposed to be here by March 14.",
    "Order #7730 still shows 'processing'. Expected delivery was April 2.",
    "Where is my order #1209? The confirmation said it would arrive on May 20.",
    "I need order #5566 before June 3 for a birthday. Will it make it?",
    "My order #3301 was marked delivered but I never got it. It was due on July 9.",
    "Can you check on order #9045? I was told it would come by August 21.",
]
UPDATES = [
    "Q3 update: Revenue grew 12.5% to $4.2 million. Customer churn fell to 3.1%. We opened offices in Lisbon and Austin. Hiring continues in engineering.",
    "October report: Active users reached 58,300, up 8.4% from September. Average order value rose to $37.20. The mobile app rating improved to 4.6.",
    "Board summary: Gross margin improved to 61.5% while operating costs dropped 4.2%. Cash reserves stand at $12.8 million. The Berlin launch is planned for March 2027.",
    "Support metrics: First response time fell to 2.4 hours. Satisfaction scores rose to 92.5%. Ticket volume increased 15.3% after the pricing change.",
]

APP_TRIAGE = '''"""Ticket triage for the Acme support desk."""
import json

from openai import OpenAI

client = OpenAI()
TRIAGE_MODEL = "gpt-4o-2024-05-13"

TRIAGE_PROMPT = (
    "You are a support ticket triage assistant for Acme. Classify the customer's message and return JSON with the keys "
    '"category" (billing, shipping, technical or account), "urgency" (low, medium or high) and "summary" (one short sentence).'
)


def triage(ticket: str) -> dict:
    resp = client.chat.completions.create(
        model=TRIAGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": TRIAGE_PROMPT}, {"role": "user", "content": ticket}],
    )
    return json.loads(resp.choices[0].message.content)  # breaks if the model wraps JSON in ```fences
'''

APP_REPLIES = '''"""Drafts customer replies."""
import os

from openai import OpenAI

client = OpenAI()
REPLY_PROMPT = (
    "You are Acme's customer support assistant. Write a short, polite reply to the customer on behalf of the Acme support team. "
    "Mention the order number and the delivery date when they are relevant."
)


def draft_reply(message: str) -> str:
    resp = client.chat.completions.create(
        model=os.getenv("REPLY_MODEL", "gpt-4o-2024-05-13"),
        temperature=0.3,
        messages=[{"role": "system", "content": REPLY_PROMPT}, {"role": "user", "content": message}],
    )
    return resp.choices[0].message.content
'''

APP_SUMMARIES = '''"""Executive summaries of business updates."""
import anthropic

client = anthropic.Anthropic()
SUMMARY_PROMPT = "Summarise the following business update in two sentences for an executive audience. Keep the key figures."


def summarise(update: str) -> str:
    msg = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=300,
        temperature=0,
        system=SUMMARY_PROMPT,
        messages=[{"role": "user", "content": update}],
    )
    return msg.content[0].text
'''

SETTINGS = """# Acme support settings
llm:
  fallback_model: gpt-4o-mini
  embedding_model: text-embedding-3-small
"""


def _openai_pair(system: str, user: str, params: dict, ts: str) -> dict:
    request = {"model": "gpt-4o-2024-05-13", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **params}
    out = legacy({"messages": request["messages"], "params": params})
    response = {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "created": 1760000000,
        "model": "gpt-4o-2024-05-13",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
    }
    return {"timestamp": ts, "request": request, "response": response}


def _anthropic_pair(system: str, user: str, ts: str) -> dict:
    request = {"model": "claude-3-5-haiku-20241022", "max_tokens": 300, "temperature": 0, "system": system, "messages": [{"role": "user", "content": user}]}
    out = legacy({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "params": {}})
    response = {
        "id": "msg_demo",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-haiku-20241022",
        "content": [{"type": "text", "text": out.text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 90, "output_tokens": 50},
    }
    return {"timestamp": ts, "request": request, "response": response}


def create_project(directory: str | None = None, *, force: bool = False) -> Path:
    root = (Path(directory) if directory else Path(tempfile.mkdtemp(prefix="llmshift-demo-"))).resolve()
    if directory and root.is_dir() and any(root.iterdir()) and not force:
        raise FileExistsError(f"{root} is not empty; the demo writes app/, logs/ and llmshift.yaml there. Use a new directory or --force")
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "app" / "triage.py").write_text(APP_TRIAGE, encoding="utf-8")
    (root / "app" / "replies.py").write_text(APP_REPLIES, encoding="utf-8")
    (root / "app" / "summaries.py").write_text(APP_SUMMARIES, encoding="utf-8")
    (root / "config" / "settings.yaml").write_text(SETTINGS, encoding="utf-8")
    lines = []
    day = 1
    for ticket in TICKETS:
        lines.append(_openai_pair(TRIAGE_PROMPT, ticket, {"temperature": 0, "response_format": {"type": "json_object"}}, f"2026-10-{day:02d}T09:{day:02d}:00Z"))
        day += 1
    for message in MESSAGES:
        lines.append(_openai_pair(REPLY_PROMPT, message, {"temperature": 0.3}, f"2026-10-{day:02d}T10:{day:02d}:00Z"))
        day += 1
    lines.append(_openai_pair(REPLY_PROMPT, MESSAGES[0], {"temperature": 0.3}, "2026-10-19T11:00:00Z"))  # a duplicate request
    with (root / "logs" / "openai-requests.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")
    with (root / "logs" / "anthropic-requests.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for i, update in enumerate(UPDATES, 1):
            fh.write(json.dumps(_anthropic_pair(SUMMARY_PROMPT, update, f"2026-02-{i:02d}T08:00:00Z")) + "\n")
    (root / "llmshift.yaml").write_text(
        "tests: .llmshift/tests.jsonl\nbaseline: mock:legacy\ncandidate: mock:nextgen\nmin_pass_rate: 0.95\nconcurrency: 4\n",
        encoding="utf-8",
    )
    return root

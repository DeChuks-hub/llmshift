"""Record real traffic from your app, then turn it into tests with `llmshift snapshot`.

    pip install llmshift openai
    python examples/record_example.py
    llmshift snapshot logs/llm.jsonl
"""

from openai import OpenAI

from llmshift.record import wrap

# Wrap the client once; every call is appended to logs/llm.jsonl (secrets are redacted).
# sample_rate=0.1 would keep 10% of calls in high-traffic services.
client = wrap(OpenAI(), path="logs/llm.jsonl", sample_rate=1.0)

reply = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    temperature=0,
    messages=[
        {"role": "system", "content": "You are a support ticket classifier. Reply with JSON."},
        {"role": "user", "content": "My parcel never arrived."},
    ],
    response_format={"type": "json_object"},
)
print(reply.choices[0].message.content)

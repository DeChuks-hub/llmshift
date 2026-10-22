"""Rebuild src/llmshift/data/pricing.json from LiteLLM's model price table (MIT licensed).

    python scripts/update_pricing.py                 # download the current table
    python scripts/update_pricing.py litellm.json    # or convert a local copy

Keeps chat, completion, responses and embedding models that have token prices and
stores them as USD per 1M tokens: {"model": [input, output]}.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import urllib.request
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
MODES = {"chat", "completion", "responses", "embedding"}
OUT = Path(__file__).resolve().parents[1] / "src" / "llmshift" / "data" / "pricing.json"


def per_million(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(f"{value * 1_000_000:.6g}")


def convert(table: dict) -> dict:
    prices = {}
    for key, entry in table.items():
        if key == "sample_spec" or not isinstance(entry, dict) or entry.get("mode") not in MODES:
            continue
        inp = per_million(entry.get("input_cost_per_token"))
        out = per_million(entry.get("output_cost_per_token", 0.0))
        if inp is None or out is None:
            continue
        prices[key] = [inp, out]
    return dict(sorted(prices.items()))


def main() -> None:
    if len(sys.argv) > 1:
        table = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as resp:
            table = json.loads(resp.read())
    prices = convert(table)
    if len(prices) < 500:
        raise SystemExit(f"only {len(prices)} priced models found; the source format may have changed")
    data = {
        "about": "Model prices for llmshift pick and cost estimates, in USD per 1M tokens [input, output]. "
        "Converted from LiteLLM's model_prices_and_context_window.json (MIT License, Copyright (c) 2023 Berri AI).",
        "source": SOURCE,
        "fetched": _dt.date.today().isoformat(),
        "unit": "USD per 1M tokens",
        "prices": prices,
    }
    OUT.write_bytes((json.dumps(data, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8"))
    print(f"wrote {len(prices)} prices to {OUT}")


if __name__ == "__main__":
    main()

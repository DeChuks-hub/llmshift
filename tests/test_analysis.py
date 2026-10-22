"""Text metrics, format detection, JSON schema, facts, tone and templating."""

from __future__ import annotations

import pytest

from llmshift import facts, formats, jsonschema_lite as js, textutil, tone
from llmshift.templating import MissingVariable, lookup, render, render_text, uses_namespace

# ------------------------------------------------------------------ textutil


def test_normalize_and_tokenize():
    assert textutil.normalize("  Hello\u00a0“World”—ok ") == 'hello "world"-ok'
    assert textutil.tokenize("Don't stop_me now, 42!") == ["don", "t", "stop", "me", "now", "42"]
    assert textutil.tokenize("東京へ行く") == ["東", "京", "へ", "行", "く"]
    assert textutil.word_count("Hello there, it's 5 o'clock") == 5


@pytest.mark.parametrize("metric", sorted(textutil.METRICS))
def test_metrics_identity_and_disjoint(metric):
    # METEOR's fragmentation penalty keeps even identical text marginally below 1.0
    assert textutil.similarity("the cat sat on the mat", "the cat sat on the mat", metric) == pytest.approx(1.0, abs=0.01)
    # character-level fuzzy matching credits shared letters, word metrics do not
    limit = 0.5 if metric in ("fuzzy", "fuzzy_match") else 0.35
    assert textutil.similarity("alpha beta gamma", "delta epsilon zeta", metric) < limit


def test_metrics_ordering():
    ref = "the quick brown fox jumps over the lazy dog"
    close = "the quick brown fox leaps over the lazy dog"
    far = "a slow green turtle crawls under a busy cat"
    for metric in ("rouge_l", "rouge_1", "bleu", "gleu", "meteor", "token_f1", "fuzzy_match", "cosine"):
        assert textutil.similarity(close, ref, metric) > textutil.similarity(far, ref, metric), metric


def test_unknown_metric():
    with pytest.raises(ValueError):
        textutil.similarity("a", "b", "nope")


def test_empty_inputs():
    assert textutil.rouge_l("", "") == 1.0
    assert textutil.rouge_l("", "x") == 0.0
    assert textutil.bleu("", "x") == 0.0


def test_word_diff_roundtrip():
    a, b = "The order ships on Monday.", "The order ships on Tuesday!"
    segs = textutil.word_diff(a, b)
    assert "".join(t for op, t in segs if op != "+") == a
    assert "".join(t for op, t in segs if op != "-") == b
    assert textutil.diff_ratio(a, a) == 0.0
    assert 0 < textutil.diff_ratio(a, b) < 1


def test_detect_language():
    assert textutil.detect_language("Thank you for your message. The order will arrive on Monday and you do not need to do anything.")[0] == "en"
    assert textutil.detect_language("Gracias por tu mensaje. El pedido llegará el lunes y no tienes que hacer nada más.")[0] == "es"
    assert textutil.detect_language("Merci pour votre message. La commande arrivera lundi et vous n'avez rien à faire de plus.")[0] == "fr"
    assert textutil.detect_language("Vielen Dank für Ihre Nachricht. Die Bestellung kommt am Montag und Sie müssen nichts weiter tun.")[0] == "de"
    assert textutil.detect_language("ご注文は月曜日に届きます。ありがとうございます。")[0] == "ja"
    assert textutil.detect_language("ok")[0] is None


# ------------------------------------------------------------------- formats


def test_analyze_json_variants():
    assert formats.analyze_json('{"a": 1}')["valid"]
    fenced = formats.analyze_json('```json\n{"a": 1}\n```')
    assert not fenced["valid"] and fenced["fenced"] and fenced["value"] == {"a": 1}
    assert "code fence" in fenced["error"]
    prose = formats.analyze_json('Sure! Here it is: {"a": [1, 2]} Hope that helps.')
    assert prose["embedded"] and prose["value"] == {"a": [1, 2]} and "extra text" in prose["error"]
    bad = formats.analyze_json('{"a": 1,}')
    assert bad["value"] is None and "not valid JSON" in bad["error"]
    assert formats.analyze_json("")["error"] == "empty output"


def test_strip_fences_and_structure():
    inner, fenced, lang = formats.strip_fences("```python\nprint(1)\n```")
    assert fenced and lang == "python" and inner == "print(1)"
    md = "# Title\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n**bold**\n\n```\ncode\n```"
    sig = formats.structure_signature(md)
    assert sig == {"headings": True, "list": True, "numbered_list": False, "table": True, "code_block": True, "bold": True}
    assert formats.structure_signature("Just a sentence.")["list"] is False


def test_refusal_detection():
    assert formats.is_refusal("I'm sorry, but I can't help with that request.")
    assert formats.is_refusal("I cannot assist with this.")
    assert not formats.is_refusal("Sure, here is the summary you asked for.")
    assert not formats.is_refusal("x " * 200 + "I can't help with that")  # only the opening counts


def test_output_kind():
    assert formats.output_kind("positive") == "label"
    assert formats.output_kind("$1,299.00") == "number"
    assert formats.output_kind('{"a": 1}') == "json"
    assert formats.output_kind("- a\n- b\n- c") == "markdown"
    assert formats.output_kind("This is a normal sentence about things.") == "prose"
    assert formats.output_kind("") == "empty"


# --------------------------------------------------------------- jsonschema


def test_schema_infer_validate_merge():
    schema = js.infer({"name": "Ada", "age": 36, "tags": ["x"], "address": {"city": "London"}})
    assert js.validate({"name": "B", "age": 1, "tags": [], "address": {"city": "Paris"}}, schema) == []
    errors = js.validate({"name": 5, "tags": "x", "address": {}}, schema)
    joined = " | ".join(errors)
    assert "missing required key 'age'" in joined
    assert "$.name: expected string" in joined
    assert "$.tags: expected array" in joined
    assert "$.address: missing required key 'city'" in joined
    merged = js.merge(js.infer({"a": 1, "b": "x"}), js.infer({"a": 2.5}))
    assert merged["required"] == ["a"]
    assert merged["properties"]["a"]["type"] == "number"
    assert js.validate(3.0, {"type": "integer"}) == []
    assert js.validate(3.5, {"type": "integer"}) != []


def test_schema_keywords():
    schema = {
        "type": "object",
        "$defs": {"label": {"type": "string", "enum": ["a", "b"]}},
        "properties": {
            "label": {"$ref": "#/$defs/label"},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "id": {"type": "string", "pattern": "^ID-\\d+$"},
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["label"],
        "additionalProperties": False,
    }
    assert js.validate({"label": "a", "score": 0.5, "id": "ID-7", "opt": None}, schema) == []
    errs = js.validate({"label": "c", "score": 2, "id": "X", "opt": 3, "extra": 1}, schema)
    text = " | ".join(errs)
    for needle in ("is not one of", "greater than maximum", "does not match pattern", "does not match any allowed", "unexpected key 'extra'"):
        assert needle in text


def test_describe_and_paths():
    schema = js.infer({"a": {"b": [1]}, "c": "x"})
    assert js.describe(schema) == "{a: {b: [number]}, c: string}"  # inferred numbers are never pinned to integer
    value = {"a": {"b": [10, 20]}, "weird key": True}
    paths = dict(js.json_paths(value))
    assert paths == {"$.a.b[0]": 10, "$.a.b[1]": 20, "$['weird key']": True}
    for path, expected in paths.items():
        assert js.get_path(value, path) == (True, expected)
    assert js.get_path(value, "$.a.nope") == (False, None)


# -------------------------------------------------------------------- facts


def test_fact_extraction_kinds():
    text = (
        "Order #4521 ships on March 14, 2027. Revenue grew 12.5% to $4.2 million. Call +44 20 7946 0958 or visit "
        "https://acme.example/help, email help@acme.example, run `pip install acme`. Marie Curie worked in Paris."
    )
    found = facts.extract(text)
    kinds = {f["kind"] for f in found}
    assert {"id", "date", "number", "url", "email", "code", "entity"} <= kinds
    texts = {f["text"] for f in found}
    assert {"#4521", "March 14, 2027", "12.5%", "$4.2 million", "https://acme.example/help", "help@acme.example", "pip install acme", "Marie Curie", "Paris"} <= texts


def test_fact_matching_paraphrase_and_change():
    base = facts.extract("The population of Paris is about 2.1 million. It hosted the Olympics in 2024.")
    ok = facts.match(base, "Paris has roughly 2,100,000 residents and hosted the 2024 Olympics.")
    assert not ok["missing"] and not ok["changed"]
    bad = facts.match(base, "Paris has roughly 2.3 million residents and hosted the 2024 Olympics.")
    assert bad["changed"] == [{"from": "2.1 million", "to": "2.3 million"}]


def test_fact_boundaries_and_ranges():
    base = facts.extract("Your order #4521 will arrive in 3-5 business days.")
    result = facts.match(base, "Order #4521 should arrive within 3-7 business days.")
    assert [m["text"] for m in result["missing"]] == ["5"]
    assert result["changed"] == [{"from": "5", "to": "7"}]


def test_fact_precision_and_currency():
    base = facts.extract("It costs $5.00 and conversion is 33%.")
    assert not facts.match(base, "It costs $5 and conversion is 33.3%.")["missing"]
    assert facts.match(base, "It costs €5 and conversion is 33%.")["missing"]


def test_dates_match_without_year():
    base = facts.extract("Delivery on 2026-03-14.")
    assert not facts.match(base, "Delivery on March 14.")["missing"]
    assert facts.match(base, "Delivery on 2025-03-14.")["missing"]


def test_list_markers_and_code_ignored():
    found = facts.extract("1. First step\n2. Second step\n```\nx = 42\n```")
    assert not any(f["kind"] == "number" for f in found)


# --------------------------------------------------------------------- tone


def test_tone_differences_and_instructions():
    formal = tone.profile("Thank you for contacting us. We have reviewed your order and it will ship on Monday. Please let us know if we can assist further.")
    casual = tone.profile("Hey there! 🎉 Awesome news - your order ships Monday!! I am super excited for you. Want me to track it?")
    score, diffs = tone.compare(formal, casual)
    markers = {d["marker"] for d in diffs}
    assert {"emoji", "exclamations", "formality"} <= markers
    assert score < 1
    instr = tone.instructions_for(diffs)
    assert "Do not use emoji." in instr
    same_score, same = tone.compare(formal, formal)
    assert same == [] and same_score == 1.0


def test_tone_tolerance():
    a = tone.profile("We are sorry. We will fix this today. Thanks for your patience with our team.")
    b = tone.profile("We are sorry! We will fix this today! Thanks for your patience with our team!")
    assert any(d["marker"] == "exclamations" for d in tone.compare(a, b)[1])
    assert not any(d["marker"] == "exclamations" for d in tone.compare(a, b, tolerance=10)[1])


# --------------------------------------------------------------- templating


def test_templating():
    ns = {"item": {"q": "What?", "n": 3, "tags": ["a", "b"], "nested": {"x": [{"y": "deep"}]}}, "sample": {"output_text": "ok"}}
    assert render_text("Q: {{ item.q }} / {{item.n}}", ns) == "Q: What? / 3"
    assert render("{{item.tags}}", ns) == ["a", "b"]
    assert render_text("{{item.nested.x[0].y}}", ns) == "deep"
    assert render_text("{{sample.output_text}}!", ns) == "ok!"
    assert render_text("{{item.missing}}", ns) == ""
    with pytest.raises(MissingVariable):
        lookup("item.missing", ns)
    assert render({"role": "user", "content": ["{{item.q}}"]}, ns) == {"role": "user", "content": ["What?"]}
    assert uses_namespace("x {{ sample.output_json.a }}", "sample")
    assert not uses_namespace("x {{ item.a }}", "sample")

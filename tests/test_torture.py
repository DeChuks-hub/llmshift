"""Torture tests: hostile, huge, broken and unlucky inputs must never crash or hang
llmshift, and every failure must come with a clear message."""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

from llmshift import cli, console, facts, formats, logparse, textutil, tone
from llmshift.checks import CHECKS, CheckContext, Sample, evaluate
from llmshift.config import ConfigError, load_config
from llmshift.demo import create_project
from llmshift.providers import http
from llmshift.providers.base import ProviderError
from llmshift.providers.openai import OpenAIProvider
from llmshift.redact import Redactor
from llmshift.testset import TestCase, TestFileError, load_tests

SEED = 20261022
WEIRD = ["", " ", "\u200b", "\u202e", "\ufeff", "\x00", "\u2028", "\u0301", "\U0001f600", "\U0001f469\u200d\U0001f469\u200d\U0001f467", "\u65e5\u672c\u8a9e", "\u0639\u0631\u0628\u0649", "\ud83d", "\t", "\r\n", "\\", '"', "'", "{", "]"]


def _text(rng: random.Random, n: int = 40) -> str:
    return "".join(rng.choice(WEIRD + list("abc xyz 0123456789.,$%#@")) for _ in range(rng.randint(0, n)))


def _value(rng: random.Random, depth: int = 0):
    pick = rng.randint(0, 9 if depth < 3 else 5)
    if pick == 0:
        return None
    if pick == 1:
        return rng.choice([True, False])
    if pick == 2:
        return rng.choice([0, -1, 1.5, 10**30, -(10**30), 1e308])
    if pick in (3, 4, 5):
        return _text(rng)
    if pick in (6, 7):
        return [_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(["role", "content", "messages", "text", "model", "tool_calls", "name", _text(rng, 5)]): _value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def _timed(limit: float):
    start = time.perf_counter()

    def done() -> float:
        took = time.perf_counter() - start
        assert took < limit, f"took {took:.1f}s (limit {limit}s)"
        return took

    return done


# ------------------------------------------------------------------- YAML


def _alias_bomb(levels: int = 9) -> str:
    lines = ["a0: &a0 [" + ",".join(['"lol"'] * 9) + "]"]
    lines += [f"a{i}: &a{i} [" + ",".join([f"*a{i - 1}"] * 9) + "]" for i in range(1, levels)]
    return "\n".join(lines) + "\ncandidate: mock:legacy\n"


def test_yaml_alias_bomb_in_config_is_refused_at_once(in_tmp, capsys):
    (in_tmp / "llmshift.yaml").write_text(_alias_bomb(), encoding="utf-8")
    done = _timed(5)
    assert cli.main(["doctor"]) == 2
    done()
    assert "expands to more than" in capsys.readouterr().err


def test_yaml_bombs_and_recursion_in_test_files(tmp_path):
    bomb = tmp_path / "bomb.yaml"
    bomb.write_text(_alias_bomb() + "cases: *a8\n", encoding="utf-8")
    done = _timed(5)
    with pytest.raises(TestFileError, match="expands to more than"):
        load_tests(bomb)
    done()
    for text in ("a: &a [*a]\n", "x: " + "[" * 3000 + "]" * 3000 + "\n"):
        f = tmp_path / "t.yaml"
        f.write_text(text, encoding="utf-8")
        with pytest.raises(TestFileError):
            load_tests(f)


def test_recursive_and_deep_config(in_tmp):
    for text in ("a: &a [*a]\ncandidate: x\n", "x: " + "[" * 3000 + "]" * 3000 + "\n", "x: " + "{a: " * 2000 + "1" + "}" * 2000 + "\n"):
        (in_tmp / "llmshift.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config()


# ------------------------------------------------------------------- logs


def _hostile_records(rng: random.Random, n: int) -> list[str]:
    lines = []
    for _ in range(n):
        kind = rng.randint(0, 7)
        if kind == 0:
            lines.append("".join(chr(rng.randint(1, 0x2FFF)) for _ in range(rng.randint(1, 60))).replace("\n", " "))
        elif kind == 1:
            lines.append(json.dumps(_value(rng)))
        elif kind == 2:
            lines.append(json.dumps({"request": _value(rng), "response": _value(rng)}))
        elif kind == 3:
            lines.append(json.dumps({"model": _value(rng), "messages": _value(rng), "choices": _value(rng)}))
        elif kind == 4:
            lines.append(json.dumps({"custom_id": _value(rng), "body": _value(rng), "response": _value(rng)}))
        elif kind == 5:
            lines.append(json.dumps({"input": _value(rng), "output": _value(rng), "timestamp": _value(rng)}))
        elif kind == 6:
            lines.append("[" * rng.randint(1000, 5000) + "]" * 10)
        else:
            lines.append(json.dumps({"llmshift_record": 1, "request": {"model": "m", "messages": [{"role": "user", "content": _text(rng)}]}, "response": {"choices": [{"message": {"content": _text(rng)}}]}}))
    return lines


def test_hostile_log_records_never_crash(tmp_path):
    rng = random.Random(SEED)
    lines = _hostile_records(rng, 1500)
    (tmp_path / "hostile.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", errors="surrogatepass")
    done = _timed(30)
    inters, stats = logparse.parse_logs([str(tmp_path)])
    done()
    assert stats.records + sum(stats.skipped.values()) >= len([x for x in lines if x.strip()]) - sum(stats.skipped.values())
    assert all(isinstance(i.messages, list) for i in inters)
    for inter in inters:  # whatever was accepted can be stored
        json.dumps({"m": inter.messages, "o": inter.output}, ensure_ascii=False).encode("utf-8")


def test_a_line_bomb_in_a_gzip_log_is_skipped_in_bounded_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(logparse, "MAX_LINE_BYTES", 1024 * 1024)
    good = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "choices": [{"message": {"content": "hello"}}]})
    with gzip.open(tmp_path / "bomb.jsonl.gz", "wb") as fh:
        for _ in range(30):
            fh.write(b"a" * (1024 * 1024))
        fh.write(b"\n" + good.encode() + b"\n")
    tracemalloc.start()
    inters, stats = logparse.parse_logs([str(tmp_path)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(inters) == 1 and any("too large" in r for r in stats.skipped)
    assert peak < 16 * 1024 * 1024, f"peak memory {peak / 1e6:.0f} MB"


def test_oversized_json_documents_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(logparse, "MAX_DOCUMENT_BYTES", 1000)
    (tmp_path / "big.json").write_text(json.dumps([{"prompt": "x" * 50, "output": "y"}] * 40), encoding="utf-8")
    inters, stats = logparse.parse_logs([str(tmp_path)])
    assert not inters and any("too large" in r for r in stats.skipped)


def test_huge_logs_keep_a_fair_sample_per_template(tmp_path):
    rows = []
    for template, n in (("billing", 2000), ("shipping", 900), ("legal", 30)):
        for i in range(n):
            rows.append(json.dumps({"model": "gpt-4o", "messages": [{"role": "system", "content": f"You answer {template} questions."}, {"role": "user", "content": f"q{i}"}],
                                    "choices": [{"message": {"content": f"a{i}"}}]}))
    (tmp_path / "log.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    inters, stats = logparse.parse_logs([str(tmp_path)], max_records=300)
    assert len(inters) == 300 and stats.parsed == 2930 and stats.sampled_out == 2630
    by = {}
    for inter in inters:
        by.setdefault(inter.messages[0]["content"], 0)
        by[inter.messages[0]["content"]] += 1
    assert len(by) == 3 and by["You answer legal questions."] >= 10  # rare templates keep a place


def test_line_separators_bom_and_crlf_in_logs(tmp_path):
    rec = {"model": "gpt-4o", "messages": [{"role": "user", "content": "line\u2028sep\u2029para\x85next"}], "choices": [{"message": {"content": "ok"}}]}
    data = "\ufeff" + json.dumps(rec, ensure_ascii=False) + "\r\n" + json.dumps(rec, ensure_ascii=False) + "\r\n"
    (tmp_path / "x.jsonl").write_bytes(data.encode("utf-8"))
    inters, stats = logparse.parse_logs([str(tmp_path)])
    assert len(inters) == 2 and not stats.skipped
    assert inters[0].messages[0]["content"] == "line\u2028sep\u2029para\x85next"


def test_binary_and_corrupt_files(tmp_path):
    rng = random.Random(SEED)
    (tmp_path / "noise.jsonl").write_bytes(bytes(rng.randint(0, 255) for _ in range(200_000)))
    noisy = "".join(json.dumps({"prompt": os.urandom(30).hex(), "output": os.urandom(30).hex()}) + "\n" for _ in range(2000))
    whole = gzip.compress(noisy.encode())
    (tmp_path / "trunc.jsonl.gz").write_bytes(whole[: len(whole) // 2])  # cut mid-stream
    (tmp_path / "empty.jsonl").write_bytes(b"")
    (tmp_path / "bad.csv").write_bytes(b'prompt,output\n"unterminated,\n' + b"\x00" * 100)
    inters, stats = logparse.parse_logs([str(tmp_path)])
    assert stats.files == 4 and any("unreadable file" in r for r in stats.skipped)
    # The complete records before the cut in the .gz are kept; the rest is reported unreadable.
    salvaged = [i for i in inters if i.source.endswith(".gz") or ".gz:" in i.source]
    assert 0 < len(salvaged) < 2000 and len(inters) - len(salvaged) <= 1


# ----------------------------------------------------------------- checks


def _random_check(rng: random.Random, ctype: str) -> dict:
    check = {"type": ctype}
    for key in rng.sample(["value", "values", "pattern", "schema", "expected", "path", "facts", "threshold", "min_words", "max_words", "language", "metric", "reference", "negate", "case_sensitive", "severity", "fields", "labels"], 4):
        check[key] = _value(rng)
    if rng.random() < 0.3:
        check["severity"] = rng.choice(["error", "warn", "info", "nonsense", None])
    if ctype == "regex":
        check["pattern"] = rng.choice(["a+", "[", "(", "\\d{3}", "(?i)hello", ""])
    return check


def test_every_check_survives_hostile_parameters_and_outputs():
    rng = random.Random(SEED)
    ctx = CheckContext()
    case = TestCase(id="t", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, baseline={"output": '{"a": 1, "b": [1, 2]}', "tool_calls": [{"name": "f", "arguments": {"x": 1}}]})
    evaluate({"type": "not_empty"}, Sample(text="x"), case, ctx)  # registers the model graders
    done = _timed(60)
    for _ in range(25):
        for ctype in list(CHECKS):
            sample = Sample(text=_text(rng, 200) if rng.random() < 0.7 else json.dumps(_value(rng)), tool_calls=_value(rng) if isinstance(_value(rng), list) else [])
            result = evaluate(_random_check(rng, ctype), sample, case, ctx)
            assert result.status in ("pass", "fail", "skip", "error"), (ctype, result)
            assert isinstance(result.message, str)
    done()


def test_text_algorithms_stay_fast_on_adversarial_input():
    big_digits = "1" * 400_000
    cases = [
        lambda: facts.extract(big_digits),
        lambda: facts.extract("$1,000,000.00 on 2024-01-01 at 10:30 for #A-1234; " * 8000),
        lambda: facts.match(facts.extract("Revenue rose 12.5% to $4.2 million on March 3."), "x" * 300_000),
        lambda: Redactor(pii=True).text("4" * 300_000 + " a@b.co " * 20_000),
        lambda: formats.find_embedded_json("{" * 200_000),
        lambda: formats.find_embedded_json('{"a":' * 60_000),
        lambda: formats.is_refusal("I can't " * 100_000),
        lambda: tone.profile("😀 Great!!! " * 50_000),
        lambda: textutil.word_diff("alpha beta " * 15_000, "alpha gamma " * 15_000),
        lambda: textutil.rouge_l("w " * 60_000, "w x " * 30_000),
        lambda: textutil.detect_language("le " * 200_000),
    ]
    for fn in cases:
        done = _timed(15)
        fn()
        done()


def test_scanner_on_minified_and_binary_sources(in_tmp):
    from llmshift.scanner import scan_paths

    (in_tmp / "app.min.js").write_text("var a=1;" * 180_000 + 'm="gpt-4o-2024-05-13";', encoding="utf-8")
    (in_tmp / "vendor.bundle.js").write_text("var b=2;" * 400_000, encoding="utf-8")  # over the 2 MB per-file cap
    (in_tmp / "blob.py").write_bytes(os.urandom(500_000))
    (in_tmp / "deep.py").write_text("x = " + "(" * 5000 + "1" + ")" * 5000 + "\nmodel = 'gpt-4o'\n", encoding="utf-8")
    done = _timed(30)
    result = scan_paths(["."])
    done()
    assert "gpt-4o-2024-05-13" in {f.model for f in result.findings}
    assert any("vendor.bundle.js" in e and "larger than" in e for e in result.errors)


# --------------------------------------------------------------- providers


def _provider(server, **options) -> OpenAIProvider:
    return OpenAIProvider("gpt-test", options={"base_url": server.url + "/v1", "api_key": "k", "max_retries": 1, "timeout": 2, **options})


REQUEST = {"messages": [{"role": "user", "content": "hi"}], "params": {}}


def _ok(text="fine"):
    return {"id": "x", "object": "chat.completion", "model": "gpt-test", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def test_provider_chaos(server, monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: delays.append(seconds))
    behaviours = iter([
        (200, "<html>gateway</html>"),  # garbage 200
        (502, "<html>bad gateway</html>"), (200, _ok("after a 502")),  # retried
        (429, {"error": {"message": "slow down"}}, {"Retry-After": "99999"}), (200, _ok("after a 429")),
        (429, {"error": {"message": "slow down"}}, {"Retry-After": "-5"}), (200, _ok()),
        (429, {"error": {"message": "slow down"}}, {"Retry-After": "soon"}), (200, _ok()),
        (200, "[" * 50_000 + "]" * 50_000),  # nested too deeply
        (200, json.dumps(_ok("broken X emoji")).replace("X", "\\ud83d")),
        (200, {"error": {"message": "model overloaded", "type": "server_error"}}),
    ])
    server.route("POST", "/v1/chat/completions", lambda req: next(behaviours))
    p = _provider(server)
    with pytest.raises(ProviderError):
        p.complete(REQUEST)
    assert p.complete(REQUEST).text == "after a 502"
    assert p.complete(REQUEST).text == "after a 429"
    assert max(delays) <= http.MAX_DELAY
    p.complete(REQUEST)
    p.complete(REQUEST)
    assert all(0 <= d <= http.MAX_DELAY for d in delays)
    with pytest.raises(ProviderError, match="nested too deeply"):
        p.complete(REQUEST)
    assert p.complete(REQUEST).text == "broken � emoji"
    with pytest.raises(ProviderError):
        p.complete(REQUEST)


def test_resets_timeouts_and_oversized_bodies(server, monkeypatch):
    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: None)

    def reset(req):
        raise ConnectionResetError("boom")

    server.route("POST", "/v1/chat/completions", reset)
    with pytest.raises(ProviderError) as exc:
        _provider(server).complete(REQUEST)
    assert exc.value.code == "connection_error"
    server.route("POST", "/v1/chat/completions", lambda req: (time.sleep(3), _ok())[1])
    done = _timed(12)
    with pytest.raises(ProviderError):
        _provider(server, timeout=0.5).complete(REQUEST)
    done()
    monkeypatch.setattr(http, "MAX_RESPONSE_BYTES", 10_000)
    server.route("POST", "/v1/chat/completions", lambda req: _ok("x" * 50_000))
    with pytest.raises(ProviderError, match="larger than"):
        _provider(server).complete(REQUEST)


def test_compare_survives_a_provider_that_fails_at_random(server, monkeypatch, make_case):
    from llmshift.compare import run_compare

    monkeypatch.setattr(http, "_sleep", lambda seconds, sleep: None)
    rng = random.Random(SEED)
    lock = __import__("threading").Lock()

    def flaky(req):
        with lock:
            roll = rng.random()
        if roll < 0.2:
            return 500, {"error": {"message": "internal"}}
        if roll < 0.3:
            return 200, "not json"
        if roll < 0.35:
            raise ConnectionResetError("reset")
        return _ok("hello")

    server.route("POST", "/v1/chat/completions", flaky)
    cases = [make_case("hello", user=f"q{i}", id=f"c{i}") for i in range(40)]
    res = run_compare(cases, _provider(server, max_retries=2), concurrency=8, show_progress=False)
    s = res.summary
    assert s["total"] == 40 and s["evaluated"] + s["errors"] == 40 and s["evaluated"] > 20


# ------------------------------------------------------------ environment


@pytest.fixture
def demo(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    return root


def test_ctrl_c_during_a_run_exits_130_and_the_next_run_works(demo, monkeypatch):
    from llmshift import runner

    real_wait = runner.wait

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "wait", interrupted)
    assert cli.main(["compare", "--to", "mock:legacy", "--no-cache", "--no-report"]) == 130
    assert http.CANCEL.is_set()
    monkeypatch.setattr(runner, "wait", real_wait)
    assert cli.main(["compare", "--to", "mock:legacy", "--no-cache", "--no-report"]) == 0
    assert not http.CANCEL.is_set()


def test_ctrl_c_in_a_watch_loop(demo, monkeypatch):
    from llmshift import watch

    def stop(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(watch, "sleep_until_next", stop)
    assert cli.main(["watch", "--model", "mock:legacy", "--every", "1m"]) == 130


def test_unwritable_outputs_exit_2_with_the_reason(demo, capsys):
    Path("afile.txt").write_text("x", encoding="utf-8")
    assert cli.main(["compare", "--to", "mock:legacy", "--html", "afile.txt/report.html", "--no-report"]) == 2
    assert "cannot write afile.txt" in capsys.readouterr().err.replace("\\", "/")


def test_a_full_disk_never_breaks_a_run(demo, monkeypatch):
    import errno
    import tempfile

    from llmshift import cache

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(cache.tempfile, "mkstemp", full)
    assert cli.main(["compare", "--to", "mock:legacy", "--no-report"]) == 0
    monkeypatch.setattr(cache.tempfile, "mkstemp", tempfile.mkstemp)


def test_parallel_processes_share_one_cache_and_one_log(tmp_path):
    script = (
        "import sys, json\n"
        "from llmshift.record import log\n"
        "from llmshift.cache import ResponseCache\n"
        "from llmshift.compare import run_compare\n"
        "from llmshift.testset import TestCase\n"
        "cases = [TestCase(id=f'c{i}', input={'messages': [{'role': 'user', 'content': f'q{i}'}], 'params': {}}, baseline={'output': f'q{i}'}) for i in range(30)]\n"
        "run_compare(cases, 'mock:echo', cache=ResponseCache(sys.argv[1]), show_progress=False)\n"
        "for i in range(150):\n"
        "    log({'model': 'm', 'messages': [{'role': 'user', 'content': 'x' * 3000 + str(i)}]}, {'text': 'y'}, path=sys.argv[2])\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path / "cache"), str(tmp_path / "log.jsonl")], env=env) for _ in range(4)]
    assert all(p.wait(timeout=180) == 0 for p in procs)
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 600 and all(json.loads(x)["llmshift_record"] == 1 for x in lines)
    for f in (tmp_path / "cache").rglob("*.json"):
        assert isinstance(json.loads(f.read_text(encoding="utf-8")), dict)


def test_unicode_and_spaces_in_paths(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "проект с пробелами 😀"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    assert cli.main(["compare", "--to", "mock:legacy", "--html", "отчёт 😀.html", "--no-report"]) == 0
    assert (root / "отчёт 😀.html").exists()


def test_legacy_console_encodings_never_crash(monkeypatch):
    for encoding in ("cp1252", "ascii", "latin-1"):
        stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        console.out("emoji 😀, check ✓, arrow →, cjk 日本, rtl عربى, broken \ud83d")
        stream.flush()
        assert stream.buffer.getvalue()


# -------------------------------------------------------------- CLI fuzz


COMMANDS = [
    ["scan"], ["models"], ["snapshot", "logs"], ["compare"], ["fix"], ["watch"], ["import", "openai-evals"], ["report", "x.json"],
    ["doctor"], ["cache", "clear"], ["pick"], ["check"], ["upgrade"], ["safety"], ["shadow", "report"], ["judge", "calibrate"],
    ["embeddings"], ["digest"], ["calendar"], ["init"],
]


def test_every_command_rejects_garbage_arguments(in_tmp, capsys):
    for argv in COMMANDS:
        with pytest.raises(SystemExit) as exc:
            cli.main(argv + ["--definitely-not-an-option"])
        assert exc.value.code == 2, argv
        with pytest.raises(SystemExit) as exc:
            cli.main(argv[:1] + ["--help"] if argv[0] not in ("import", "shadow", "judge", "cache") else argv[:2] + ["--help"])
        assert exc.value.code == 0, argv
    capsys.readouterr()


def test_every_command_in_an_empty_folder_fails_cleanly(in_tmp, capsys, monkeypatch):
    monkeypatch.setenv("LLMSHIFT_REGISTRY_URL", "http://127.0.0.1:9/models.json")
    for argv in COMMANDS + [["update", "--check"], ["judge", "label"]]:
        try:
            code = cli.main(argv)
        except SystemExit as exc:  # argparse: missing required arguments
            code = exc.code
        out = capsys.readouterr()
        assert code in (0, 1, 2), (argv, code)
        assert "Traceback" not in out.err and "unexpected" not in out.err, (argv, out.err)


def test_config_fuzz_never_crashes(in_tmp):
    rng = random.Random(SEED)
    keys = ["concurrency", "timeout", "max_retries", "min_pass_rate", "repeat", "cache", "judge_mode", "tests", "baseline", "scan", "snapshot", "fix", "watch", "providers", "pricing", "severity"]
    import yaml

    for _ in range(200):
        doc = {rng.choice(keys): _value(rng) for _ in range(rng.randint(1, 5))}
        (in_tmp / "llmshift.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8", errors="surrogatepass")
        try:
            load_config()
        except ConfigError:
            pass


def test_test_file_fuzz_never_crashes(tmp_path):
    rng = random.Random(SEED)
    for i in range(300):
        rec = {"id": _value(rng), "input": _value(rng), "baseline": _value(rng), "checks": _value(rng), "agent": _value(rng) if rng.random() < 0.3 else None}
        f = tmp_path / f"t{i}.jsonl"
        f.write_text(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n", encoding="utf-8")
        try:
            load_tests(f)
        except TestFileError:
            pass

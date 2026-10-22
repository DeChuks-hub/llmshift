"""llmshift.record (client wrapper) and configuration loading."""

from __future__ import annotations

import asyncio
import json
import warnings

from llmshift.config import expand_env, load_config
from llmshift.record import Recorder, log, wrap
from llmshift.snapshot import build_snapshot


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        def conv(v):
            if isinstance(v, Obj):
                return v.model_dump()
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v

        return {k: conv(v) for k, v in self.__dict__.items()}


def completion(text):
    return Obj(id="c", object="chat.completion", model="gpt-4o-2024-08-06", choices=[Obj(index=0, message=Obj(role="assistant", content=text), finish_reason="stop")])


class FakeCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            return iter([Obj(model="gpt-4o", choices=[Obj(delta=Obj(content="Hel"), finish_reason=None)]), Obj(model="gpt-4o", choices=[Obj(delta=Obj(content="lo"), finish_reason="stop")])])
        if kwargs.get("model") == "boom":
            raise RuntimeError("API down")
        return completion("echo " + kwargs["messages"][-1]["content"])


class FakeAsyncCompletions:
    async def create(self, **kwargs):
        return completion("async " + kwargs["messages"][-1]["content"])


class FakeClient:
    def __init__(self, completions):
        self.chat = Obj(completions=completions)
        self.api_key = "sk-should-not-leak-abcdefghijklmnopqrstuvwxyz"


def test_wrap_records_calls_and_snapshot_reads_them(tmp_path):
    path = tmp_path / "llm.jsonl"
    client = wrap(FakeClient(FakeCompletions()), path=path)
    out = client.chat.completions.create(model="gpt-4o", temperature=0, messages=[{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "echo hi"
    assert client.api_key.startswith("sk-")  # attributes pass through
    chunks = list(client.chat.completions.create(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "stream me"}]))
    assert len(chunks) == 2
    try:
        client.chat.completions.create(model="boom", messages=[{"role": "user", "content": "x"}])
    except RuntimeError:
        pass
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and all(rec["llmshift_record"] == 1 for rec in lines)
    assert lines[0]["request"]["temperature"] == 0 and lines[0]["response"]["choices"][0]["message"]["content"] == "echo hi"
    assert lines[1]["response"]["choices"][0]["message"]["content"] == "Hello"
    assert lines[2]["error"].startswith("RuntimeError") and lines[2]["response"] is None
    res = build_snapshot([str(path)])
    outputs = sorted(c.baseline_output for c in res.cases)
    assert outputs == ["Hello", "echo hi"]  # the failed call is filtered out


def test_async_wrap(tmp_path):
    path = tmp_path / "a.jsonl"
    client = wrap(FakeClient(FakeAsyncCompletions()), path=path)
    result = asyncio.run(client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "q"}]))
    assert result.choices[0].message.content == "async q"
    assert json.loads(path.read_text(encoding="utf-8"))["response"]["choices"][0]["message"]["content"] == "async q"


def test_recorder_redacts_and_never_raises(tmp_path):
    rec = Recorder(tmp_path / "r.jsonl")
    rec.log({"model": "m", "messages": [{"role": "user", "content": "key sk-proj-abcdefghijklmnopqrstuvwxyz0123"}]}, {"choices": []})
    assert "sk-proj" not in (tmp_path / "r.jsonl").read_text(encoding="utf-8")
    blocked = tmp_path / "dir_not_file"
    blocked.mkdir()
    bad = Recorder(blocked)  # writing to a directory fails
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad.log({"model": "m"}, None)
        bad.log({"model": "m"}, None)
    assert len(caught) == 1  # warns once, never raises
    log({"model": "m", "messages": []}, {"choices": []}, path=tmp_path / "manual.jsonl")
    assert (tmp_path / "manual.jsonl").exists()


def test_sampling(tmp_path):
    rec = Recorder(tmp_path / "s.jsonl", sample_rate=0.0)
    client = wrap(FakeClient(FakeCompletions()), recorder=rec)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert not (tmp_path / "s.jsonl").exists()


def test_config_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["tests"] == ".llmshift/tests.jsonl" and cfg.path is None and cfg["scan"]["fail_on"] == "retired"
    monkeypatch.setenv("CAND", "openai:gpt-6-astra")
    (tmp_path / "llmshift.yaml").write_text("candidate: ${CAND}\njudge: ${JUDGE:-openai:gpt-5.6-terra}\nscan:\n  warn_days: 30\n", encoding="utf-8")
    sub = tmp_path / "pkg" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    cfg = load_config()
    assert cfg["candidate"] == "openai:gpt-6-astra" and cfg["judge"] == "openai:gpt-5.6-terra"
    assert cfg["scan"]["warn_days"] == 30 and cfg["scan"]["fail_on"] == "retired"  # deep-merged with defaults
    assert expand_env(["${CAND}", {"x": "$NOT_EXPANDED"}]) == ["openai:gpt-6-astra", {"x": "$NOT_EXPANDED"}]


def test_pyproject_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "anthropic:claude-sonnet-5-5"\nmin_pass_rate = 0.9\n', encoding="utf-8")
    cfg = load_config()
    try:
        import tomllib  # noqa: F401
    except ImportError:
        return
    assert cfg["candidate"] == "anthropic:claude-sonnet-5-5" and cfg["min_pass_rate"] == 0.9


def test_bad_config(tmp_path, monkeypatch):
    import pytest

    from llmshift.config import ConfigError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "llmshift.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config()
    with pytest.raises(ConfigError):
        load_config("missing.yaml")


class FakeMessageStream:
    """Shaped like anthropic's MessageStream: iterate events, then read the snapshot."""

    def __init__(self, text):
        self.text = text
        self.closed = False

    def __iter__(self):
        yield Obj(type="text", text=self.text)

    @property
    def current_message_snapshot(self):
        return Obj(id="m", type="message", role="assistant", model="claude-sonnet-5-5", content=[Obj(type="text", text=self.text)],
                   stop_reason="end_turn", usage=Obj(input_tokens=5, output_tokens=2))

    def close(self):
        self.closed = True


class FakeStreamManager:
    def __init__(self, stream=None, error=None):
        self.stream, self.error, self.exited = stream, error, False

    def __enter__(self):
        if self.error:
            raise self.error
        return self.stream

    def __exit__(self, *exc):
        self.exited = True
        self.stream.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *exc):
        self.__exit__(*exc)


class FakeMessages:
    def __init__(self):
        self.managers = []

    def stream(self, **kwargs):
        if kwargs["model"] == "boom":
            manager = FakeStreamManager(error=RuntimeError("overloaded"))
        else:
            manager = FakeStreamManager(FakeMessageStream("streamed " + kwargs["messages"][-1]["content"]))
        self.managers.append(manager)
        return manager

    def parse(self, **kwargs):
        return Obj(id="m", type="message", role="assistant", model="claude-sonnet-5-5", content=[Obj(type="text", text='{"category": "billing"}')], stop_reason="end_turn")


class SchemaModel:
    """Duck-typed like a pydantic model class used as a response format."""

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}


def test_sdk_stream_helpers_are_recorded_when_the_block_ends(tmp_path):
    path = tmp_path / "helpers.jsonl"
    messages = FakeMessages()
    client = wrap(Obj(messages=messages), path=path)
    with client.messages.stream(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}]) as stream:
        assert [e.text for e in stream] == ["streamed hi"]  # the app sees the SDK's own stream
    assert messages.managers[0].exited and messages.managers[0].stream.closed
    try:
        with client.messages.stream(model="boom", max_tokens=50, messages=[{"role": "user", "content": "x"}]):
            pass
    except RuntimeError:
        pass

    async def use_async():
        async with client.messages.stream(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "async"}]) as stream:
            return [e.text for e in stream]

    assert asyncio.run(use_async()) == ["streamed async"]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and [rec["api"] for rec in lines] == ["messages"] * 3
    assert lines[0]["response"]["content"][0]["text"] == "streamed hi" and lines[0]["latency_ms"] >= 0
    assert lines[1]["error"] == "RuntimeError: overloaded" and lines[1]["response"] is None
    assert lines[2]["response"]["content"][0]["text"] == "streamed async"
    res = build_snapshot([str(path)])
    assert sorted(c.baseline_output for c in res.cases) == ["streamed async", "streamed hi"]


def test_pydantic_response_formats_are_recorded_as_json_schema(tmp_path):
    path = tmp_path / "formats.jsonl"
    messages = FakeMessages()
    completions = FakeCompletions()
    completions.parse = lambda **kw: completion('{"category": "billing"}')
    client = wrap(Obj(chat=Obj(completions=completions), messages=messages), path=path)
    client.chat.completions.parse(model="gpt-4o", messages=[{"role": "user", "content": "classify"}], response_format=SchemaModel)
    client.messages.parse(model="claude-sonnet-5-5", max_tokens=50, messages=[{"role": "user", "content": "classify"}], output_format=SchemaModel)
    chat, claude = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rf = chat["request"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"]["required"] == ["category"] and rf["json_schema"]["name"] == "SchemaModel"
    assert claude["request"]["output_config"]["format"]["schema"]["required"] == ["category"] and "output_format" not in claude["request"]
    res = build_snapshot([str(path)])
    formats = [c.input["params"].get("response_format") for c in res.cases]
    assert len(formats) == 2 and all(f and f["json_schema"]["schema"]["required"] == ["category"] for f in formats)  # replayable


class NotGiven:  # the class of openai.NOT_GIVEN (and of anthropic's, and of other generated SDKs)
    def __bool__(self):
        return False


def _chunk(text=None, finish=None, index=0, refusal=None):
    return Obj(model="gpt-4o", choices=[Obj(index=index, delta=Obj(content=text, refusal=refusal), finish_reason=finish)])


class StreamingCompletions:
    def __init__(self, chunks, error=None):
        self.chunks, self.error = chunks, error

    def create(self, **kwargs):
        def gen():
            yield from self.chunks
            if self.error is not None:
                raise self.error

        return gen()


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_mocked_client_and_sdk_sentinels(tmp_path):
    """Apps test with unittest.mock: recording a MagicMock answer must finish at once,
    and openai.NOT_GIVEN-style arguments are no value at all, never {}."""
    import time
    from unittest import mock

    path = tmp_path / "m.jsonl"
    client = wrap(mock.MagicMock(), path=path)
    start = time.perf_counter()
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], temperature=NotGiven(), top_p=NotGiven())
    assert time.perf_counter() - start < 5
    rec = _records(path)[0]
    assert rec["request"] == {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def test_streams_that_did_not_finish_are_not_expected_answers(tmp_path):
    path = tmp_path / "s.jsonl"
    whole = [_chunk("Hel"), _chunk("lo", finish="stop")]
    for chunks, error, stop_after in ((whole, None, None), (whole[:1], ConnectionError("reset by peer"), None), (whole[:1], None, None), (whole, None, 1)):
        client = wrap(FakeClient(StreamingCompletions(chunks, error)), path=path)
        stream = client.chat.completions.create(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "q"}])
        got = []
        try:
            for chunk in stream:
                got.append(chunk)
                if stop_after and len(got) >= stop_after:
                    stream.close()  # the user pressed stop
                    break
        except ConnectionError:
            pass
    complete, dropped, cut, stopped = _records(path)
    assert "error" not in complete and complete["response"]["choices"][0]["message"]["content"] == "Hello"
    assert dropped["error"].startswith("ConnectionError") and "finish reason" in cut["error"] and "closed the stream" in stopped["error"]
    assert [c.baseline_output for c in build_snapshot([str(path)]).cases] == ["Hello"]  # only the whole answer becomes a test


def test_streamed_choices_refusals_and_later_edits(tmp_path):
    path = tmp_path / "n.jsonl"
    chunks = [_chunk("A", index=0), _chunk("B", index=1), _chunk("a", index=0, finish="stop"), _chunk(None, index=1, refusal="I can't."), _chunk("b", index=1, finish="stop")]
    client = wrap(FakeClient(StreamingCompletions(chunks)), path=path)
    messages = [{"role": "user", "content": "q"}]
    stream = client.chat.completions.create(model="gpt-4o", stream=True, n=2, messages=messages)
    messages.append({"role": "assistant", "content": "(typing...)"})  # the app edits its list while the stream runs
    list(stream)
    rec = _records(path)[0]
    assert [c["message"]["content"] for c in rec["response"]["choices"]] == ["Aa", "Bb"]
    assert rec["response"]["choices"][1]["message"]["refusal"] == "I can't."
    assert rec["request"]["messages"] == [{"role": "user", "content": "q"}]


def test_recording_failures_never_raise_even_as_errors(tmp_path):
    blocked = tmp_path / "a_directory"
    blocked.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pytest's filterwarnings = error, or python -W error
        client = wrap(FakeClient(FakeCompletions()), path=blocked)
        assert client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}]).choices[0].message.content == "echo hi"


def test_a_broken_last_line_and_server_side_context(tmp_path):
    from llmshift import fileio

    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"llmshift_record": 1, "request": {"model": "gpt-4o", "messages": [{"role"')  # a writer died mid-line
    rec = {"llmshift_record": 1, "api": "chat", "request": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, "response": completion("hello").model_dump()}
    fileio.append(path, (json.dumps(rec) + "\n").encode())
    follow_up = {"llmshift_record": 1, "api": "responses", "request": {"model": "gpt-5", "input": "And my name?", "previous_response_id": "resp_1"}, "response": {"object": "response", "output": []}}
    fileio.append(path, (json.dumps(follow_up) + "\n").encode())
    res = build_snapshot([str(path)])
    assert [c.baseline_output for c in res.cases] == ["hello"]  # the record after the broken line is whole
    assert any("context stored by OpenAI" in reason for reason in res.stats.skipped)


def test_the_wrapped_client_still_looks_like_the_client(tmp_path):
    import copy

    import pytest

    path = tmp_path / "c.jsonl"
    raw = FakeClient(FakeCompletions())
    client = wrap(wrap(raw, path=path), path=path)  # wrapping twice records once
    assert isinstance(client, FakeClient) and "chat" in dir(client)
    assert isinstance(copy.copy(client), FakeClient)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert len(_records(path)) == 1
    with pytest.raises(ValueError):
        Recorder(tmp_path / "x.jsonl", sample_rate=float("nan"))

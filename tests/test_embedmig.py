"""`llmshift embeddings`: corpus loading, retrieval comparison, re-index estimate and the CLI."""

from __future__ import annotations

import json

import pytest

from conftest import write_jsonl
from llmshift import cli, embedmig, pricing
from llmshift.errors import UsageError
from llmshift.providers.base import ProviderError

TOPICS = {
    "refunds": "Refunds are issued to the original card within five business days after the return is received.",
    "shipping": "Standard shipping takes three to five days; express delivery arrives the next business day.",
    "passwords": "Reset your password from the sign-in page by choosing forgot password and following the e-mail link.",
    "invoices": "Invoices can be downloaded as PDF files from the billing section of your account settings.",
    "warranty": "Every device carries a two year warranty that covers manufacturing defects but not accidental damage.",
    "returns": "Items can be returned within thirty days in their original packaging for a full refund.",
    "tracking": "Track a parcel with the tracking number from the shipping confirmation e-mail.",
    "accounts": "Close an account by contacting support; data is deleted after thirty days.",
}
QUERIES = ["how long does a refund take", "when will my parcel arrive", "forgot my password", "download an invoice", "is accidental damage covered"]


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for name, text in TOPICS.items():
        (root / f"{name}.md").write_text(f"# {name}\n\n{text}\n\nMore about {name}: {text.lower()}\n", encoding="utf-8")
    return root


@pytest.fixture
def cli_extensions(monkeypatch):
    monkeypatch.setattr(cli, "EXTENSIONS", ("embedmig",))


def test_corpus_loading_chunking_and_formats(tmp_path):
    root = tmp_path / "corpus"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "secret.md").write_text("never read", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "pkg.md").write_text("vendored", encoding="utf-8")
    (root / "guide.md").write_text("First paragraph about refunds.\n\n" + "Second sentence. " * 40 + "\n\nLast one.", encoding="utf-8")
    (root / "page.html").write_text("<html><head><style>p{}</style><script>alert(1)</script></head><body><p>Shipping &amp; returns</p><p>Fast</p></body></html>", encoding="utf-8")
    write_jsonl(root / "kb.jsonl", [{"title": "Invoices", "content": "Download invoices as PDF."}, {"id": 7, "text": "Numbered record."}, {"nothing": 1}])
    (root / "faq.json").write_text(json.dumps({"documents": [{"name": "Warranty", "body": "Two year warranty."}, "A plain string record."]}), encoding="utf-8")
    (root / "notes.py").write_text("print('not a document')", encoding="utf-8")
    docs = embedmig.load_corpus([str(root)], chunk_chars=200)
    texts = " ".join(d.text for d in docs)
    titles = [d.title for d in docs]
    assert "never read" not in texts and "vendored" not in texts and "not a document" not in texts
    assert "alert(1)" not in texts and "Shipping & returns" in texts
    assert "Invoices" in titles and "7" in titles and "Warranty" in titles
    assert any(t.endswith("faq.json[1]") for t in titles)
    assert all(len(d.text) <= 200 for d in docs)
    assert sum(1 for t in titles if "guide.md" in t) >= 3  # the long paragraph was split


def test_chunk_text_never_loses_or_loops():
    assert embedmig.chunk_text("") == []
    long_word = "x" * 1000
    chunks = embedmig.chunk_text(f"{long_word} tail", 100)
    assert all(len(c) <= 100 for c in chunks) and "".join(chunks).replace(" ", "") == (long_word + "tail")
    text = "\n\n".join(f"Paragraph {i}. Some words here." for i in range(50))
    chunks = embedmig.chunk_text(text, 120)
    assert all(len(c) <= 120 for c in chunks) and sum(c.count("Paragraph") for c in chunks) == 50


def test_rank_biased_overlap_and_top_k():
    assert embedmig.rank_biased_overlap([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert embedmig.rank_biased_overlap([1, 2, 3], [4, 5, 6]) == 0.0
    swapped = embedmig.rank_biased_overlap([1, 2, 3], [2, 1, 3])
    late = embedmig.rank_biased_overlap([1, 2, 3], [1, 3, 2])
    assert 0 < swapped < late < 1  # disagreement at the top costs more
    docs = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    assert embedmig.top_k([[1.0, 0.0]], docs, 2) == [[0, 2]]  # ties: lower index first


def test_identical_models_keep_every_result(corpus):
    report = embedmig.run_embedding_migration([str(corpus)], "mock:embed-v1", "mock:embed-v1", queries=QUERIES, k=3)
    assert report.mean_overlap == 1.0 and report.top1_kept == 1.0 and report.mean_rbo == pytest.approx(1.0)
    assert report.passed and report.worst == [] and report.old_dims == report.new_dims == 256


def test_switching_models_reports_changes_and_reindex_cost(corpus):
    report = embedmig.run_embedding_migration([str(corpus)], "mock:embed-v1", "mock:embed-v2", queries=QUERIES, k=3, min_overlap=0.99)
    assert 0.0 <= report.mean_overlap < 1.0 and not report.passed
    assert (report.old_dims, report.new_dims) == (256, 384)
    assert any("new schema" in n for n in report.notes)
    tokens = sum(pricing.estimate_tokens(d.text) for d in embedmig.load_corpus([str(corpus)]))
    assert report.corpus_tokens == tokens
    assert report.reindex_cost == pytest.approx(tokens * 0.02 / 1_000_000)
    overlaps = [w["overlap"] for w in report.worst]
    assert overlaps == sorted(overlaps) and all(o < 1.0 for o in overlaps)
    data = report.to_dict()
    assert data["dimensions"]["changed"] and data["passed"] is False and json.dumps(data)


def test_sampling_uses_the_full_corpus_for_cost(corpus):
    full = embedmig.run_embedding_migration([str(corpus)], "mock:embed-v1", "mock:embed-v2", queries=QUERIES, k=2, max_docs=3)
    assert full.sampled_chunks == 3 and full.corpus_chunks == len(TOPICS) and full.k == 2
    tiny = embedmig.run_embedding_migration([str(corpus)], "mock:embed-v1", "mock:embed-v2", queries=QUERIES, k=50)
    assert tiny.k == len(TOPICS) and any("Only" in n for n in tiny.notes)


def test_pseudo_queries_when_none_are_given(corpus):
    report = embedmig.run_embedding_migration([str(corpus)], "mock:embed-v1", "mock:embed-v2", k=2)
    assert report.query_source == "pseudo" and report.queries > 0 and any("--queries" in n for n in report.notes)


def test_queries_from_files_and_logs(tmp_path):
    (tmp_path / "q.txt").write_text("refund timing\n\n  refund   timing \nparcel status\n", encoding="utf-8")
    assert embedmig.load_queries(str(tmp_path / "q.txt")) == ["refund timing", "parcel status"]
    write_jsonl(tmp_path / "q.jsonl", [{"query": "one"}, {"input": ["two", "three"]}, {"other": "x"}, "four"])
    assert embedmig.load_queries(str(tmp_path / "q.jsonl")) == ["one", "two", "three", "four"]
    logs = tmp_path / "logs"
    logs.mkdir()
    write_jsonl(logs / "app.jsonl", [
        {"llmshift_record": 1, "request": {"model": "text-embedding-3-small", "input": "where is my order"}, "response": {"data": [{"embedding": [0.1]}]}},
        {"custom_id": "1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "custom-embedder", "input": ["refund policy", "refund policy"]}},
        {"request": {"model": "text-embedding-3-small", "input": [[1, 2, 3]]}},
        {"request": {"model": "gpt-4o", "messages": [{"role": "user", "content": "chat, not an embedding"}]}},
        {"request": {"model": "gpt-5", "input": "a Responses API call, not an embedding"}},
    ])
    assert embedmig.queries_from_logs([str(logs)]) == ["where is my order", "refund policy"]


def test_cli_end_to_end(corpus, tmp_path, cli_extensions, capsys):
    queries = tmp_path / "queries.txt"
    queries.write_text("\n".join(QUERIES), encoding="utf-8")
    out = tmp_path / "report.json"
    code = cli.main(["embeddings", "--from", "mock:embed-v1", "--to", "mock:embed-v2", "--corpus", str(corpus), "--queries", str(queries), "--k", "3", "--min-overlap", "0", "--json", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "results kept" in text and "256 -> 384" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["queries"] == {"count": len(QUERIES), "source": "file"} and data["passed"] is True
    code = cli.main(["embeddings", "--from", "mock:embed-v1", "--to", "mock:embed-v2", "--corpus", str(corpus), "--queries", str(queries), "--k", "3", "--min-overlap", "1", "--format", "json"])
    assert code == 1 and json.loads(capsys.readouterr().out)["passed"] is False
    code = cli.main(["embeddings", "--from", "mock:embed-v1", "--to", "mock:embed-v1", "--corpus", str(corpus), "--format", "markdown"])
    assert code == 0 and capsys.readouterr().out.startswith("## LLMShift embeddings")


def test_cli_errors(corpus, tmp_path, cli_extensions, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "blank.txt").write_text("\n\n", encoding="utf-8")
    cases = [
        (["--corpus", str(tmp_path / "missing")], "not found"),
        (["--corpus", str(empty)], "no documents"),
        (["--corpus", str(corpus), "--queries", str(tmp_path / "blank.txt")], "contains no queries"),
        (["--corpus", str(corpus), "--queries-from", str(empty)], "no embedding requests"),
        (["--corpus", str(corpus), "--k", "0"], "at least 1"),
        (["--corpus", str(corpus), "--chunk-chars", "5"], "between 50"),
    ]
    for extra, message in cases:
        code = cli.main(["embeddings", "--from", "mock:embed-v1", "--to", "mock:embed-v2", *extra])
        assert code == 2 and message in capsys.readouterr().err, extra
    code = cli.main(["embeddings", "--from", "mock:legacy", "--to", "mock:embed-v2", "--corpus", str(corpus)])
    assert code == 2 and "does not support embeddings" in capsys.readouterr().err


class _BadEmbedder:
    spec = "fake:bad"

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return self.vectors(texts)


def test_broken_embedding_responses_are_reported():
    with pytest.raises(ProviderError, match="returned 1 embeddings for 2 texts"):
        embedmig.embed_all(_BadEmbedder(lambda t: [[1.0]]), ["a", "b"])
    with pytest.raises(ProviderError, match="different sizes"):
        embedmig.embed_all(_BadEmbedder(lambda t: [[1.0, 0.0], [1.0]]), ["a", "b"])
    with pytest.raises(ProviderError, match="non-finite"):
        embedmig.embed_all(_BadEmbedder(lambda t: [[float("nan")]]), ["a"])
    with pytest.raises(ProviderError, match="non-numeric"):
        embedmig.embed_all(_BadEmbedder(lambda t: [["x"]]), ["a"])
    vectors, _ = embedmig.embed_all(_BadEmbedder(lambda t: [[0.0, 0.0] for _ in t]), ["a"])
    assert vectors == [[0.0, 0.0]]  # a zero vector is kept (similarity 0), not a crash


def test_bad_settings_raise_usage_errors(corpus):
    with pytest.raises(UsageError):
        embedmig.run_embedding_migration([str(corpus)], "mock:embed-v1", "mock:embed-v2", min_overlap=1.5)

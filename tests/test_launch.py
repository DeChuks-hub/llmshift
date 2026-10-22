"""Launch checks: scanner false alarms, pyproject settings on every Python version,
report flags, and package metadata."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import llmshift
from llmshift import cli
from llmshift.config import ConfigError, load_config
from llmshift.demo import create_project
from llmshift.scanner import scan_paths

ROOT = Path(__file__).resolve().parents[1]


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- scan


def test_model_values_in_files_without_ai_usage_must_be_model_ids(in_tmp):
    (in_tmp / "tests").mkdir()
    (in_tmp / "tests" / "test_size_finder.py").write_text(
        "from size_finder import recommend\n\n\ndef test_full_length():\n"
        '    assert recommend({"height": 180, "model": "full"}) == "L"\n'
        '    assert recommend({"model": "slim-fit-v2"}) == "M"\n',
        encoding="utf-8",
    )
    (in_tmp / "fixtures.yaml").write_text("garment:\n  model: full\n  model_name: classic-cut\n", encoding="utf-8")
    (in_tmp / "data.json").write_text(json.dumps({"model": "full", "size_model": "v3-tall"}), encoding="utf-8")
    assert scan_paths(["."]).findings == []
    # Known model IDs still count anywhere.
    (in_tmp / "notes.yaml").write_text("model: gpt-4o-2024-05-13\n", encoding="utf-8")
    assert [f.model for f in scan_paths(["."]).findings] == ["gpt-4o-2024-05-13"]


def test_custom_model_names_count_in_files_that_use_an_llm(in_tmp):
    (in_tmp / "app.py").write_text(
        'from openai import OpenAI\n\nclient = OpenAI()\n'
        'client.chat.completions.create(model="acme-support-ft-v3", messages=[])\n'
        'client.chat.completions.create(model="full", messages=[])\n',
        encoding="utf-8",
    )
    (in_tmp / "llm.yaml").write_text("model: acme-router-v2\nmax_tokens: 400\ntemperature: 0.2\n", encoding="utf-8")
    models = {f.model for f in scan_paths(["."]).findings}
    assert models == {"acme-support-ft-v3", "acme-router-v2"}  # "full" is never a model ID


# -------------------------------------------------------- pyproject config


def test_tomli_is_a_dependency_before_python_311():
    deps = _toml(ROOT / "pyproject.toml")["project"]["dependencies"]
    tomli = [d for d in deps if d.replace(" ", "").startswith("tomli")]
    assert tomli and re.search(r"python_version\s*<\s*['\"]3\.11['\"]", tomli[0])


def test_pyproject_settings_are_read(in_tmp):
    (in_tmp / "pyproject.toml").write_text('[project]\nname = "app"\n\n[tool.llmshift]\ncandidate = "mock:nextgen"\n', encoding="utf-8")
    cfg = load_config()
    assert cfg["candidate"] == "mock:nextgen" and cfg.path.name == "pyproject.toml"


def test_unreadable_pyproject_settings_are_an_error_not_ignored(in_tmp, monkeypatch):
    (in_tmp / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "mock:nextgen"\n', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "tomllib", None)  # as on Python < 3.11 without tomli
    monkeypatch.setitem(sys.modules, "tomli", None)
    with pytest.raises(ConfigError, match="tomli"):
        load_config()


def test_broken_toml_with_settings_is_reported(in_tmp):
    (in_tmp / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "unterminated\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse TOML"):
        load_config()


def test_other_projects_pyproject_files_are_left_alone(in_tmp):
    (in_tmp / "pyproject.toml").write_text('[project]\nname = "app"\ndependencies = ["llmshift"\n', encoding="utf-8")  # broken, not ours
    cfg = load_config()
    assert cfg.path is None and cfg["tests"] == ".llmshift/tests.jsonl"


# ------------------------------------------------------------ report flags


@pytest.fixture
def demo_tests(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    return root


def test_report_flags_take_an_optional_file_name(demo_tests, capsys):
    reports = Path(".llmshift/reports")
    cli.main(["compare", "--to", "mock:legacy", "--no-report", "--markdown", "--junit"])
    assert sorted(p.suffix for p in reports.iterdir()) == [".md", ".xml"]  # only what was asked for
    md = next(reports.glob("compare-*.md"))
    assert md.read_text(encoding="utf-8").startswith("## LLMShift report")
    assert next(reports.glob("compare-*.junit.xml")).read_text(encoding="utf-8").startswith("<?xml")
    cli.main(["compare", "--to", "mock:legacy", "--no-report", "--json", "results.json", "--markdown", "pr.md"])
    assert Path("results.json").exists() and Path("pr.md").exists()
    capsys.readouterr()
    assert cli.main(["report", "results.json", "--html", "--junit"]) == 0
    assert Path("results.html").exists() and Path("results.junit.xml").exists()
    assert "wrote results.html" in capsys.readouterr().out


def test_watch_html_flag_without_a_file_name(demo_tests):
    cli.main(["watch", "--model", "mock:legacy", "--html", "--dry-run", "--limit", "3"])
    assert list(Path(".llmshift/reports").glob("watch-*.html"))


def test_help_shows_report_flags_take_files(capsys):
    with pytest.raises(SystemExit):
        cli.main(["compare", "--help"])
    assert "--markdown [FILE]" in capsys.readouterr().out


# ---------------------------------------------------------------- metadata


def test_package_metadata_is_launch_ready():
    meta = _toml(ROOT / "pyproject.toml")["project"]
    assert meta["name"] == "llmshift" and meta["version"] == llmshift.__version__
    assert meta["scripts"] == {"llmshift": "llmshift.cli:entry"}
    assert meta["authors"] == [{"name": "Chukwu Nzubechi", "email": "nzubechukwu00176@gmail.com"}]
    assert "Chukwu Nzubechi" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert re.search(r'^author:\s*"Chukwu Nzubechi"', (ROOT / "action.yml").read_text(encoding="utf-8"), re.M)
    assert (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").count(f"## {llmshift.__version__} ") == 1


def test_no_leftovers_of_the_old_name_or_placeholders():
    shipped = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "action.yml", ROOT / "pyproject.toml", ROOT / "LICENSE"]
    shipped += [p for p in (ROOT / "src").rglob("*") if p.suffix in (".py", ".json")]
    shipped += [p for p in (ROOT / "examples").rglob("*") if p.is_file()]
    for path in shipped:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"model ?shift", text, re.I), path
        assert "your-org" not in text and "contributors" not in text.lower(), path

import json
import os
import tempfile
from pathlib import Path

import pytest

from repobrain.src.scanner.scanner import RepoScanner


def _make_repo(tmp_path: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_scan_detects_python(tmp_path):
    _make_repo(tmp_path, {"main.py": "print('hello')", "utils.py": "x = 1"})
    result = RepoScanner().scan(str(tmp_path))
    assert "Python" in result["languages"]
    assert result["file_count"] == 2


def test_scan_detects_entry_point(tmp_path):
    _make_repo(tmp_path, {"main.py": "", "lib.py": ""})
    result = RepoScanner().scan(str(tmp_path))
    assert any("main.py" in ep for ep in result["entry_points"])


def test_scan_skips_git_dir(tmp_path):
    _make_repo(tmp_path, {
        "app.py": "",
        ".git/config": "[core]",
        ".git/HEAD": "ref: refs/heads/main",
    })
    result = RepoScanner().scan(str(tmp_path))
    # Only app.py should be counted, not .git files
    assert result["file_count"] == 1


def test_scan_detects_fastapi(tmp_path):
    _make_repo(tmp_path, {
        "main.py": "from fastapi import FastAPI",
        "requirements.txt": "fastapi==0.110.0\nuvicorn",
    })
    result = RepoScanner().scan(str(tmp_path))
    assert "FastAPI" in result["frameworks"]


def test_scan_writes_json(tmp_path):
    _make_repo(tmp_path, {"app.py": "pass"})
    RepoScanner().scan(str(tmp_path))
    output = tmp_path / "analysis" / "repository_summary.json"
    assert output.exists()
    data = json.loads(output.read_text())
    assert "languages" in data


def test_scan_loc_count(tmp_path):
    _make_repo(tmp_path, {"mod.py": "a = 1\nb = 2\nc = 3"})
    result = RepoScanner().scan(str(tmp_path))
    assert result["loc_by_language"]["Python"] == 3

import json
from pathlib import Path

import pytest

from repobrain.src.parser.parser import CodeParser


def _make_repo(tmp_path: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


PYTHON_SRC = """\
import os
from pathlib import Path

class MyService:
    def do_thing(self):
        pass

def helper():
    return 42
"""

JS_SRC = """\
import { foo } from './foo';

class MyComponent {
    render() {}
}

function greet(name) {
    return `Hello ${name}`;
}
"""


def test_parse_python_functions(tmp_path):
    _make_repo(tmp_path, {"service.py": PYTHON_SRC})
    result = CodeParser().parse(["service.py"], str(tmp_path))
    meta = result["files"]["service.py"]
    assert "do_thing" in meta["functions"] or "helper" in meta["functions"]


def test_parse_python_classes(tmp_path):
    _make_repo(tmp_path, {"service.py": PYTHON_SRC})
    result = CodeParser().parse(["service.py"], str(tmp_path))
    meta = result["files"]["service.py"]
    assert "MyService" in meta["classes"]


def test_parse_python_imports(tmp_path):
    _make_repo(tmp_path, {"service.py": PYTHON_SRC})
    result = CodeParser().parse(["service.py"], str(tmp_path))
    meta = result["files"]["service.py"]
    assert any("os" in imp for imp in meta["imports"])


def test_parse_writes_json(tmp_path):
    _make_repo(tmp_path, {"app.py": "x = 1"})
    CodeParser().parse(["app.py"], str(tmp_path))
    output = tmp_path / "analysis" / "parsed_code.json"
    assert output.exists()
    data = json.loads(output.read_text())
    assert "files" in data


def test_parse_unknown_extension_skipped(tmp_path):
    _make_repo(tmp_path, {"data.csv": "a,b,c"})
    result = CodeParser().parse(["data.csv"], str(tmp_path))
    assert "data.csv" not in result["files"]


def test_parse_js(tmp_path):
    _make_repo(tmp_path, {"component.js": JS_SRC})
    result = CodeParser().parse(["component.js"], str(tmp_path))
    meta = result["files"]["component.js"]
    assert meta["language"] == "JavaScript"
    assert "MyComponent" in meta["classes"]

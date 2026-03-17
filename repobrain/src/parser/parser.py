from __future__ import annotations

import json
import re
from pathlib import Path

# tree-sitter imports — version 0.21+ API
try:
    from tree_sitter import Language, Parser as TSParser
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript

    PY_LANGUAGE = Language(tspython.language())
    JS_LANGUAGE = Language(tsjavascript.language())
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False
    PY_LANGUAGE = None  # type: ignore[assignment]
    JS_LANGUAGE = None  # type: ignore[assignment]

try:
    from langdetect import detect as langdetect_detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".java": "Java",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c": "C",
    ".cs": "C#",
    ".php": "PHP",
}

# ──────────────────────────────────────────────────────────────
# Comment extraction helpers
# ──────────────────────────────────────────────────────────────

def _extract_comments_python(source: str) -> list[str]:
    """Extract # comments and docstrings from Python source."""
    comments: list[str] = []
    # Single-line comments
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped[1:].strip())
    # Triple-quoted strings (rough extraction)
    for match in re.finditer(r'"""(.*?)"""', source, re.DOTALL):
        text = match.group(1).strip()
        if text:
            comments.append(text[:200])
    for match in re.finditer(r"'''(.*?)'''", source, re.DOTALL):
        text = match.group(1).strip()
        if text:
            comments.append(text[:200])
    return comments


def _extract_comments_js(source: str) -> list[str]:
    """Extract // and /* */ comments from JS/TS source."""
    comments: list[str] = []
    for match in re.finditer(r"//(.+)", source):
        comments.append(match.group(1).strip())
    for match in re.finditer(r"/\*(.*?)\*/", source, re.DOTALL):
        text = match.group(1).strip()
        if text:
            comments.append(text[:200])
    return comments


def _extract_comments_generic(source: str) -> list[str]:
    """Generic comment extraction covering // and # styles."""
    comments: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#"):
            comments.append(re.sub(r"^[/#]+\s*", "", stripped))
    return comments


def _annotate_comments(comments: list[str]) -> list[dict]:
    """Detect language of each comment; flag non-English ones."""
    result: list[dict] = []
    for comment in comments:
        entry: dict = {"text": comment}
        if _LANGDETECT_AVAILABLE and len(comment) > 10:
            try:
                lang = langdetect_detect(comment)
                entry["detected_language"] = lang
                if lang != "en":
                    entry["translated_comment"] = (
                        f"# [Translation needed]: {comment}"
                    )
            except Exception:
                entry["detected_language"] = "unknown"
        else:
            entry["detected_language"] = "en"
        result.append(entry)
    return result


# ──────────────────────────────────────────────────────────────
# Tree-sitter based extraction
# ──────────────────────────────────────────────────────────────

def _ts_query_captures(root_node, query_str: str, language) -> list:
    """Run a tree-sitter query and return list of (node, capture_name)."""
    query = language.query(query_str)
    return query.captures(root_node)


def _parse_python_ts(source: str) -> tuple[list[str], list[str], list[str]]:
    """Use tree-sitter to extract functions, classes and imports from Python."""
    if not _TS_AVAILABLE:
        return [], [], []
    parser = TSParser(PY_LANGUAGE)
    tree = parser.parse(source.encode("utf-8", errors="replace"))
    root = tree.root_node

    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []

    def walk(node):
        if node.type == "function_definition":
            for child in node.children:
                if child.type == "identifier":
                    functions.append(child.text.decode("utf-8", errors="replace"))
                    break
        elif node.type == "class_definition":
            for child in node.children:
                if child.type == "identifier":
                    classes.append(child.text.decode("utf-8", errors="replace"))
                    break
        elif node.type in ("import_statement", "import_from_statement"):
            imports.append(node.text.decode("utf-8", errors="replace").strip())
        for child in node.children:
            walk(child)

    walk(root)
    return functions, classes, imports


def _parse_js_ts(source: str) -> tuple[list[str], list[str], list[str]]:
    """Use tree-sitter to extract functions, classes and imports from JS/TS."""
    if not _TS_AVAILABLE:
        return [], [], []
    parser = TSParser(JS_LANGUAGE)
    tree = parser.parse(source.encode("utf-8", errors="replace"))
    root = tree.root_node

    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []

    def walk(node):
        if node.type in ("function_declaration", "function_expression",
                         "arrow_function", "method_definition"):
            for child in node.children:
                if child.type == "identifier":
                    functions.append(child.text.decode("utf-8", errors="replace"))
                    break
        elif node.type == "class_declaration":
            for child in node.children:
                if child.type == "identifier":
                    classes.append(child.text.decode("utf-8", errors="replace"))
                    break
        elif node.type == "import_statement":
            imports.append(node.text.decode("utf-8", errors="replace").strip())
        elif node.type == "call_expression":
            # require('...')
            func = node.child_by_field_name("function")
            if func and func.text == b"require":
                imports.append(node.text.decode("utf-8", errors="replace").strip())
        for child in node.children:
            walk(child)

    walk(root)
    return functions, classes, imports


# ──────────────────────────────────────────────────────────────
# Regex-based fallbacks for other languages
# ──────────────────────────────────────────────────────────────

_REGEX_PATTERNS: dict[str, dict[str, str]] = {
    "Go": {
        "function": r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        "class": r"^type\s+(\w+)\s+struct",
        "import": r'^import\s+"([^"]+)"',
    },
    "Java": {
        "function": r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(",
        "class": r"(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)",
        "import": r"^import\s+([\w.]+);",
    },
    "Rust": {
        "function": r"^(?:pub\s+)?fn\s+(\w+)\s*[\(<]",
        "class": r"^(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)",
        "import": r"^use\s+([\w::{},\s]+);",
    },
    "Ruby": {
        "function": r"^\s*def\s+(\w+)",
        "class": r"^\s*class\s+(\w+)",
        "import": r"^\s*require(?:_relative)?\s+['\"](.+)['\"]",
    },
    "C++": {
        "function": r"(?:[\w:*&<>]+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?{",
        "class": r"(?:class|struct)\s+(\w+)",
        "import": r"#include\s+[<\"](.+)[>\"]",
    },
    "C": {
        "function": r"(?:[\w*]+\s+)+(\w+)\s*\([^)]*\)\s*{",
        "class": r"typedef\s+struct\s+(\w+)",
        "import": r"#include\s+[<\"](.+)[>\"]",
    },
    "C#": {
        "function": r"(?:public|private|protected|internal|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(",
        "class": r"(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum|struct)\s+(\w+)",
        "import": r"^using\s+([\w.]+);",
    },
    "PHP": {
        "function": r"function\s+(\w+)\s*\(",
        "class": r"(?:class|interface|trait)\s+(\w+)",
        "import": r"(?:require|include)(?:_once)?\s+['\"](.+)['\"]",
    },
    "TypeScript": {
        "function": r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
        "class": r"class\s+(\w+)",
        "import": r'(?:import\s+.*?\s+from\s+[\'"](.+?)[\'"]|require\([\'"](.+?)[\'"]\))',
    },
}


def _parse_regex(source: str, language: str) -> tuple[list[str], list[str], list[str]]:
    """Regex-based extraction for languages without tree-sitter support."""
    patterns = _REGEX_PATTERNS.get(language, {})
    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []

    if "function" in patterns:
        for m in re.finditer(patterns["function"], source, re.MULTILINE):
            name = next((g for g in m.groups() if g), None)
            if name:
                functions.append(name)

    if "class" in patterns:
        for m in re.finditer(patterns["class"], source, re.MULTILINE):
            name = next((g for g in m.groups() if g), None)
            if name:
                classes.append(name)

    if "import" in patterns:
        for m in re.finditer(patterns["import"], source, re.MULTILINE):
            name = next((g for g in m.groups() if g), None)
            if name:
                imports.append(name)

    return functions, classes, imports


# ──────────────────────────────────────────────────────────────
# Main parser class
# ──────────────────────────────────────────────────────────────

class CodeParser:
    """Parses source files and extracts structural metadata."""

    def parse(self, files: list[str], repo_path: str) -> dict:
        """Parse the given list of files and extract code metadata.

        Parameters
        ----------
        files:
            Relative paths (relative to repo_path) of the files to parse.
        repo_path:
            Absolute or relative path to the repository root.

        Returns
        -------
        dict with key "files" mapping each relative path to its metadata.
        """
        root = Path(repo_path).resolve()
        result: dict[str, dict] = {}

        for rel_path in files:
            full_path = root / rel_path
            suffix = full_path.suffix.lower()
            language = _EXTENSION_TO_LANGUAGE_LOCAL(suffix)
            if language is None:
                continue

            try:
                source = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            functions: list[str] = []
            classes: list[str] = []
            imports: list[str] = []
            raw_comments: list[str] = []

            if language == "Python":
                raw_comments = _extract_comments_python(source)
                if _TS_AVAILABLE:
                    functions, classes, imports = _parse_python_ts(source)
                else:
                    functions, classes, imports = _parse_regex(source, language)
            elif language in ("JavaScript", "TypeScript"):
                raw_comments = _extract_comments_js(source)
                if _TS_AVAILABLE:
                    functions, classes, imports = _parse_js_ts(source)
                else:
                    functions, classes, imports = _parse_regex(source, language)
            else:
                raw_comments = _extract_comments_generic(source)
                functions, classes, imports = _parse_regex(source, language)

            comments = _annotate_comments(raw_comments)

            result[rel_path] = {
                "language": language,
                "functions": list(dict.fromkeys(functions)),
                "classes": list(dict.fromkeys(classes)),
                "imports": imports,
                "comments": comments,
            }

        output: dict = {"files": result}

        # Persist result
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "parsed_code.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return output


def _EXTENSION_TO_LANGUAGE_LOCAL(suffix: str) -> str | None:
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".jsx": "JavaScript",
        ".go": "Go",
        ".java": "Java",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".c": "C",
        ".cs": "C#",
        ".php": "PHP",
    }
    return mapping.get(suffix)

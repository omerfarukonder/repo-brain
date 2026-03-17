from __future__ import annotations

import json
import os
from pathlib import Path


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

SKIP_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".eggs",
}

ENTRY_POINT_NAMES: set[str] = {
    "main.py",
    "app.py",
    "index.js",
    "index.ts",
    "main.go",
    "Main.java",
    "server.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "server.js",
    "server.ts",
}


def _skip_dir(dirname: str) -> bool:
    """Return True if this directory should be skipped during scanning."""
    if dirname in SKIP_DIRS:
        return True
    if dirname.endswith(".egg-info"):
        return True
    return False


def _detect_frameworks(repo_path: Path) -> list[str]:
    """Detect frameworks present in the repository by examining marker files."""
    frameworks: list[str] = []

    # Python frameworks
    requirements_txt = repo_path / "requirements.txt"
    if requirements_txt.exists():
        content = requirements_txt.read_text(encoding="utf-8", errors="replace").lower()
        if "fastapi" in content:
            frameworks.append("FastAPI")
        if "flask" in content:
            frameworks.append("Flask")
        if "django" in content:
            frameworks.append("Django")

    # Also check pyproject.toml / setup.py / setup.cfg for python deps
    for fname in ("pyproject.toml", "setup.cfg"):
        fpath = repo_path / fname
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8", errors="replace").lower()
            if "fastapi" in content and "FastAPI" not in frameworks:
                frameworks.append("FastAPI")
            if "flask" in content and "Flask" not in frameworks:
                frameworks.append("Flask")
            if "django" in content and "Django" not in frameworks:
                frameworks.append("Django")

    # Node.js frameworks
    package_json = repo_path / "package.json"
    if package_json.exists():
        content = package_json.read_text(encoding="utf-8", errors="replace").lower()
        if "react" in content:
            frameworks.append("React")
        if "vue" in content:
            frameworks.append("Vue")
        if '"express"' in content or "'express'" in content or "express" in content:
            frameworks.append("Express")
        if "next" in content:
            frameworks.append("Next.js")
        if "nuxt" in content:
            frameworks.append("Nuxt.js")

    # Go modules
    go_mod = repo_path / "go.mod"
    if go_mod.exists():
        frameworks.append("Go Modules")
        content = go_mod.read_text(encoding="utf-8", errors="replace").lower()
        if "gin-gonic" in content:
            frameworks.append("Gin")
        if "echo" in content:
            frameworks.append("Echo")

    # Java / Maven
    pom_xml = repo_path / "pom.xml"
    if pom_xml.exists():
        frameworks.append("Maven")
        content = pom_xml.read_text(encoding="utf-8", errors="replace").lower()
        if "springframework" in content:
            frameworks.append("Spring")

    # Ruby
    gemfile = repo_path / "Gemfile"
    if gemfile.exists():
        frameworks.append("Bundler")
        content = gemfile.read_text(encoding="utf-8", errors="replace").lower()
        if "rails" in content:
            frameworks.append("Rails")
        if "sinatra" in content:
            frameworks.append("Sinatra")

    # Rust
    cargo_toml = repo_path / "Cargo.toml"
    if cargo_toml.exists():
        frameworks.append("Cargo")
        content = cargo_toml.read_text(encoding="utf-8", errors="replace").lower()
        if "actix" in content:
            frameworks.append("Actix")

    return list(dict.fromkeys(frameworks))  # deduplicate while preserving order


class RepoScanner:
    """Scans a repository and extracts metadata such as languages, frameworks,
    entry points, file counts and lines of code."""

    def scan(self, repo_path: str) -> dict:
        """Walk the repository tree and collect metadata.

        Parameters
        ----------
        repo_path:
            Absolute or relative path to the repository root.

        Returns
        -------
        dict with keys: languages, frameworks, entry_points, file_count,
        loc_by_language, files.
        """
        root = Path(repo_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Repository path does not exist: {root}")

        all_files: list[str] = []
        loc_by_language: dict[str, int] = {}
        language_set: set[str] = set()
        entry_points: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place to prevent os.walk from descending
            dirnames[:] = [d for d in dirnames if not _skip_dir(d)]

            for filename in filenames:
                full_path = Path(dirpath) / filename
                suffix = full_path.suffix.lower()

                language = EXTENSION_TO_LANGUAGE.get(suffix)
                if language is None:
                    continue

                rel_path = str(full_path.relative_to(root))
                all_files.append(rel_path)
                language_set.add(language)

                # Count lines of code
                try:
                    lines = full_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    loc_by_language[language] = (
                        loc_by_language.get(language, 0) + len(lines)
                    )
                except OSError:
                    pass

                # Check for entry points
                if filename in ENTRY_POINT_NAMES:
                    entry_points.append(rel_path)

        frameworks = _detect_frameworks(root)

        result: dict = {
            "languages": sorted(language_set),
            "frameworks": frameworks,
            "entry_points": entry_points,
            "file_count": len(all_files),
            "loc_by_language": loc_by_language,
            "files": all_files,
        }

        # Persist result
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "repository_summary.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

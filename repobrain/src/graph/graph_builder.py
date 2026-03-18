from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

try:
    import graphviz as gv
    _GV_AVAILABLE = True
except ImportError:
    _GV_AVAILABLE = False


def _module_name(rel_path: str) -> str:
    """Convert a relative file path to a short module name."""
    p = Path(rel_path)
    # Remove extension and join with dots
    parts = list(p.with_suffix("").parts)
    return ".".join(parts)


def _resolve_import(import_str: str, current_file: str, all_files: set[str]) -> str | None:
    """Try to resolve an import string to a relative file path in the repo."""
    # Normalize: remove 'from X import Y' → keep X; remove 'import X' → keep X
    import_str = import_str.strip()

    # Python: "from x.y.z import something" → x/y/z.py
    m = re.match(r"from\s+([\w.]+)\s+import", import_str)
    if m:
        module = m.group(1)
    else:
        # "import x.y.z"
        m2 = re.match(r"import\s+([\w.]+)", import_str)
        if m2:
            module = m2.group(1).split(",")[0].strip()
        else:
            module = import_str

    # Convert dotted module to path candidates
    module_path = module.replace(".", "/")
    parts = module_path.split("/")

    # Generate candidates by progressively stripping leading segments.
    # e.g. "repobrain/src/scanner/scanner" → also try "src/scanner/scanner", "scanner/scanner", etc.
    path_variants: list[str] = []
    for start in range(len(parts)):
        path_variants.append("/".join(parts[start:]))

    extensions = [".py", "/__init__.py", ".js", ".ts", "/index.js", "/index.ts"]
    candidates: list[str] = []
    for variant in path_variants:
        for ext in extensions:
            candidates.append(f"{variant}{ext}")

    for candidate in candidates:
        if candidate in all_files:
            return candidate

    # Try relative import from same directory
    current_dir = str(Path(current_file).parent)
    if current_dir != ".":
        for candidate in candidates:
            rel = f"{current_dir}/{candidate}"
            if rel in all_files:
                return rel

    return None


class GraphBuilder:
    """Builds a directed dependency graph from parsed code."""

    def build(self, parsed_code: dict, repo_path: str) -> nx.DiGraph:
        """Construct a DiGraph from parsed_code and persist artifacts.

        Parameters
        ----------
        parsed_code:
            Output of CodeParser.parse() — dict with key "files".
        repo_path:
            Path to the repository root (used for output paths).

        Returns
        -------
        networkx.DiGraph where nodes are relative file paths and edges
        represent import dependencies.
        """
        root = Path(repo_path).resolve()
        files_data: dict = parsed_code.get("files", {})
        all_files: set[str] = set(files_data.keys())

        graph = nx.DiGraph()

        # Add all files as nodes
        for rel_path in all_files:
            graph.add_node(rel_path, label=_module_name(rel_path))

        # Add edges from imports
        for rel_path, meta in files_data.items():
            imports: list[str] = meta.get("imports", [])
            for imp in imports:
                target = _resolve_import(imp, rel_path, all_files)
                if target and target != rel_path:
                    graph.add_edge(rel_path, target)

        # Compute centrality metrics and attach as node attributes
        if len(graph.nodes) > 0:
            try:
                betweenness = nx.betweenness_centrality(graph)
                degree_cent = nx.degree_centrality(graph)
            except Exception:
                betweenness = {n: 0.0 for n in graph.nodes}
                degree_cent = {n: 0.0 for n in graph.nodes}

            for node in graph.nodes:
                graph.nodes[node]["betweenness_centrality"] = betweenness.get(node, 0.0)
                graph.nodes[node]["degree_centrality"] = degree_cent.get(node, 0.0)

        # Persist artifacts
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # GraphML
        graphml_path = output_dir / "dependency_graph.graphml"
        nx.write_graphml(graph, str(graphml_path))

        # PNG via graphviz
        _render_png(graph, output_dir / "dependency_graph.png")

        return graph


def _render_png(graph: nx.DiGraph, output_path: Path) -> None:
    """Render the graph as a PNG using graphviz."""
    if not _GV_AVAILABLE:
        return
    try:
        dot = gv.Digraph(format="png")
        dot.attr(rankdir="LR", size="20,20", bgcolor="#0e1117")
        dot.attr("node", shape="box", fontsize="10", style="filled", fillcolor="#262730", fontcolor="white", color="#444444")
        dot.attr("edge", color="#888888", fontcolor="white")

        for node in graph.nodes:
            label = graph.nodes[node].get("label", node)
            # Truncate long labels
            if len(label) > 30:
                label = "..." + label[-27:]
            dot.node(str(node), label=label)

        for src, dst in graph.edges:
            dot.edge(str(src), str(dst))

        # Write to a temp name then rename
        render_path = str(output_path.with_suffix(""))
        dot.render(render_path, cleanup=True)
        # graphviz appends .png automatically
        rendered = Path(render_path + ".png")
        if rendered.exists() and rendered != output_path:
            rendered.rename(output_path)
    except Exception:
        pass


def load_graph(repo_path: str) -> nx.DiGraph:
    """Load a previously saved dependency graph from GraphML."""
    graphml_path = Path(repo_path).resolve() / "analysis" / "dependency_graph.graphml"
    if not graphml_path.exists():
        return nx.DiGraph()
    return nx.read_graphml(str(graphml_path))

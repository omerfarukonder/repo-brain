from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


# Layer classification rules: check if any keyword appears in the lowercased path/filename
_LAYER_KEYWORDS: dict[str, list[str]] = {
    "API": ["route", "router", "view", "controller", "endpoint", "api", "handler",
            "resource", "cli", "app", "server", "main", "wsgi", "asgi", "interface"],
    "Service": ["service", "usecase", "use_case", "business", "domain", "analyzer",
                "analysis", "summarizer", "summarize", "impact", "effort", "estimator",
                "processor", "manager", "orchestrat"],
    "Parser": ["parser", "parse", "scanner", "scan", "extractor", "extract", "reader"],
    "Graph": ["graph", "network", "dependency", "depend", "topology"],
    "AI": ["llm", "ai", "ml", "embedding", "rag", "model", "openai", "ollama",
           "inference", "prompt", "vector", "semantic"],
    "Repository": ["repo", "repository", "dao", "store", "storage", "persistence"],
    "Schema": ["schema", "entity", "dto", "serializer", "struct"],
    "Database": ["database", "db", "migration", "seed", "query"],
    "Utility": ["util", "helper", "common", "shared", "lib", "mixin", "decorator", "base"],
    "Config": ["config", "setting", "env", "constant", "conf"],
    "Test": ["test", "spec", "fixture", "mock", "stub"],
}


def _classify_module(rel_path: str) -> str:
    """Return the architecture layer for a given file path."""
    lowered = rel_path.lower().replace("\\", "/")
    for layer, keywords in _LAYER_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                return layer
    return "Unknown"


def _detect_pattern(layers: dict[str, list[str]]) -> str:
    """Infer the overall architectural pattern from which layers are present."""
    present = {k for k, v in layers.items() if v}
    classic_core = {"API", "Service", "Repository", "Schema"}
    pipeline_core = {"API", "Parser", "Graph", "Service"}
    if classic_core.issubset(present):
        return "Layered Architecture"
    if pipeline_core.issubset(present):
        return "Pipeline Architecture"
    if "Service" in present and len(present) <= 3:
        return "Microservices (partial)"
    if len(present) == 1:
        return "Monolith"
    return "Mixed Architecture"


def _build_mermaid(layers: dict[str, list[str]]) -> str:
    """Generate a Mermaid flowchart showing layers and their modules."""
    lines = ["graph TD"]
    layer_order = ["API", "Service", "Parser", "Graph", "AI", "Repository",
                   "Schema", "Database", "Utility", "Config", "Test", "Unknown"]

    for layer in layer_order:
        modules = layers.get(layer, [])
        if not modules:
            continue
        # Subgraph per layer
        safe_layer = layer.replace(" ", "_")
        lines.append(f"  subgraph {safe_layer}[{layer} Layer]")
        for mod in modules[:10]:  # cap at 10 per layer for readability
            node_id = mod.replace("/", "_").replace(".", "_").replace("-", "_")
            label = Path(mod).stem
            lines.append(f"    {node_id}[{label}]")
        lines.append("  end")

    # Connect layer subgraphs in order
    present_layers = [l for l in layer_order if layers.get(l)]
    for i in range(len(present_layers) - 1):
        a = present_layers[i].replace(" ", "_")
        b = present_layers[i + 1].replace(" ", "_")
        lines.append(f"  {a} --> {b}")

    return "\n".join(lines)


class ArchitectureAnalyzer:
    """Classifies modules into architecture layers and detects patterns."""

    def analyze(self, graph: nx.DiGraph, parsed_code: dict, repo_path: str) -> dict:
        """Analyze architecture from the dependency graph and parsed code.

        Parameters
        ----------
        graph:
            DiGraph from GraphBuilder.build().
        parsed_code:
            Output of CodeParser.parse().
        repo_path:
            Repository root path (used for saving output).

        Returns
        -------
        dict with keys: pattern, layers, mermaid_diagram.
        """
        files_data: dict = parsed_code.get("files", {})
        all_files = list(files_data.keys()) or list(graph.nodes)

        layers: dict[str, list[str]] = {layer: [] for layer in _LAYER_KEYWORDS}
        layers.setdefault("Unknown", [])

        for rel_path in all_files:
            layer = _classify_module(rel_path)
            layers[layer].append(rel_path)

        # Attach layer info to graph nodes
        for rel_path in all_files:
            if graph.has_node(rel_path):
                graph.nodes[rel_path]["layer"] = _classify_module(rel_path)

        pattern = _detect_pattern(layers)
        mermaid = _build_mermaid(layers)

        result = {
            "pattern": pattern,
            "layers": layers,
            "mermaid_diagram": mermaid,
        }

        root = Path(repo_path).resolve()
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "architecture_report.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

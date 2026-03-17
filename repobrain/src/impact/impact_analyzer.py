from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from repobrain.src.llm.base import LLMClient
from repobrain.src.llm.rag import RAGIndex


class ImpactAnalyzer:
    """Determines which modules are affected by a proposed change."""

    def analyze(
        self,
        change_description: str,
        graph: nx.DiGraph,
        parsed_code: dict,
        rag: RAGIndex,
        llm: LLMClient,
        repo_path: str,
    ) -> dict:
        """Analyze change impact.

        Step 1 — Ask LLM to identify seed modules from the change description.
        Step 2 — BFS in both directions from seed modules to find all affected nodes.

        Returns
        -------
        dict with keys: change_description, seed_modules, affected_modules, total_affected.
        """
        context_chunks = rag.query(change_description, top_k=5)
        context = "\n---\n".join(context_chunks) if context_chunks else "No context available."

        all_files = list(parsed_code.get("files", {}).keys())
        files_list = "\n".join(f"- {f}" for f in all_files[:50])

        prompt = (
            f"You are a code analysis assistant. Given the repository context below, "
            f"identify which files would need to be modified for the following change request.\n\n"
            f"Change request: \"{change_description}\"\n\n"
            f"Repository files (sample):\n{files_list}\n\n"
            f"Repository context:\n{context}\n\n"
            f"Return ONLY a JSON array of file paths from the repository that would need changes. "
            f"Example: [\"auth/service.py\", \"auth/routes.py\"]\n"
            f"If you cannot determine this from the context, return: []\n"
            f"Only include files that actually exist in the repository list above."
        )

        raw = llm.complete(prompt)
        seed_modules = _parse_json_array(raw)

        # Validate against actual files
        valid_files = set(parsed_code.get("files", {}).keys())
        seed_modules = [f for f in seed_modules if f in valid_files]

        # BFS traversal: successors (downstream) and predecessors (upstream)
        affected: set[str] = set(seed_modules)
        for seed in seed_modules:
            if graph.has_node(seed):
                # downstream: modules that import seed
                for node in nx.descendants(graph, seed):
                    affected.add(node)
                # upstream: modules that seed imports from
                for node in nx.ancestors(graph, seed):
                    affected.add(node)

        affected_list = sorted(affected)

        result = {
            "change_description": change_description,
            "seed_modules": seed_modules,
            "affected_modules": affected_list,
            "total_affected": len(affected_list),
        }

        root = Path(repo_path).resolve()
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "impact_report.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result


def _parse_json_array(text: str) -> list[str]:
    """Extract a JSON array of strings from LLM output."""
    # Try to find a JSON array in the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass
    return []

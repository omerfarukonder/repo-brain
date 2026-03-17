from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


_COMPLEXITY_LOW = "Low"
_COMPLEXITY_MEDIUM = "Medium"
_COMPLEXITY_HIGH = "High"

_EFFORT_MAP = {
    _COMPLEXITY_LOW: {"engineering": "1-2 days", "testing": "half a day"},
    _COMPLEXITY_MEDIUM: {"engineering": "3-5 days", "testing": "1-2 days"},
    _COMPLEXITY_HIGH: {"engineering": "5-10 days", "testing": "2-3 days"},
}


class EffortEstimator:
    """Estimates engineering complexity and effort for a proposed change."""

    def estimate(
        self,
        impact_report: dict,
        graph: nx.DiGraph,
        architecture_report: dict,
        repo_path: str,
    ) -> dict:
        """Compute an effort score from the impact report and graph metrics.

        Scoring formula:
            files_score       = len(affected_modules) * 2
            layer_score       = unique_layers_affected * 3
            dependency_score  = avg_degree_of_affected * 1.5
            centrality_score  = sum_betweenness_of_affected * 10
            total             = sum of above

        Returns
        -------
        dict with keys: complexity, effort_range, testing_effort, confidence, scores.
        """
        affected: list[str] = impact_report.get("affected_modules", [])
        layers_map: dict[str, list[str]] = architecture_report.get("layers", {})

        # --- files_score ---
        files_score = len(affected) * 2

        # --- layer_score ---
        layers_touched: set[str] = set()
        for layer, mods in layers_map.items():
            for mod in mods:
                if mod in affected:
                    layers_touched.add(layer)
        layer_score = len(layers_touched) * 3

        # --- dependency_score ---
        degrees = []
        for mod in affected:
            if graph.has_node(mod):
                degrees.append(graph.degree(mod))
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        dependency_score = avg_degree * 1.5

        # --- centrality_score ---
        centrality_sum = 0.0
        for mod in affected:
            if graph.has_node(mod):
                centrality_sum += graph.nodes[mod].get("betweenness_centrality", 0.0)
        centrality_score = centrality_sum * 10

        total = files_score + layer_score + dependency_score + centrality_score

        # --- complexity mapping ---
        if total < 10:
            complexity = _COMPLEXITY_LOW
        elif total <= 25:
            complexity = _COMPLEXITY_MEDIUM
        else:
            complexity = _COMPLEXITY_HIGH

        effort = _EFFORT_MAP[complexity]

        # --- confidence ---
        n = len(affected)
        if n >= 10:
            confidence = "80%"
        elif n >= 5:
            confidence = "70%"
        else:
            confidence = "60%"

        scores = {
            "files_score": round(files_score, 2),
            "layer_score": round(layer_score, 2),
            "dependency_score": round(dependency_score, 2),
            "centrality_score": round(centrality_score, 2),
            "total": round(total, 2),
        }

        result = {
            "complexity": complexity,
            "effort_range": effort["engineering"],
            "testing_effort": effort["testing"],
            "confidence": confidence,
            "layers_touched": sorted(layers_touched),
            "scores": scores,
        }

        root = Path(repo_path).resolve()
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "effort_estimation.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

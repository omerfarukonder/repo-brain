from __future__ import annotations

from collections import deque
from pathlib import Path

import networkx as nx


class FlowMapper:
    """Maps execution flows starting from entry points through the dependency graph."""

    def map_flows(
        self,
        graph: nx.DiGraph,
        entry_points: list[str],
        parsed_code: dict,
        max_depth: int = 8,
    ) -> list[dict]:
        """Trace call chains from each entry point using BFS.

        Parameters
        ----------
        graph:
            Dependency DiGraph.
        entry_points:
            List of relative file paths that are entry points.
        parsed_code:
            Output of CodeParser (used for metadata).
        max_depth:
            Maximum traversal depth per flow.

        Returns
        -------
        List of dicts: [{"entry": str, "flow": [str, ...]}]
        """
        flows: list[dict] = []
        nodes_in_graph = set(graph.nodes)

        for entry in entry_points:
            # Find the graph node that matches (entry may not include repo prefix)
            matching = [n for n in nodes_in_graph if n.endswith(entry) or entry.endswith(n)]
            if not matching:
                matching = [entry] if entry in nodes_in_graph else []

            for start_node in matching:
                flow_chain = _bfs_flow(graph, start_node, max_depth)
                flows.append({"entry": start_node, "flow": flow_chain})

        return flows


def _bfs_flow(graph: nx.DiGraph, start: str, max_depth: int) -> list[str]:
    """BFS traversal returning an ordered list of visited nodes."""
    visited: list[str] = []
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen: set[str] = set()

    while queue:
        node, depth = queue.popleft()
        if node in seen or depth > max_depth:
            continue
        seen.add(node)
        visited.append(node)
        for neighbor in graph.successors(node):
            if neighbor not in seen:
                queue.append((neighbor, depth + 1))

    return visited

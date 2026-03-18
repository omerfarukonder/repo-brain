"""
Effort Estimator — deterministic, graph-algorithm-based complexity scoring.

Instead of asking an LLM "how hard is this?", we compute effort from
measurable structural properties of the dependency graph. The approach
draws on concepts from graph theory, many of which underlie NP-hard
problems (graph coloring, dominating sets, clique cover). We use
polynomial-time approximations where exact solutions are intractable.

=== SCORING MODEL ===

The final score is built from five independent signals:

1. COUPLING DENSITY (subgraph density of affected modules)
   - Measures how interconnected the affected area is.
   - Dense subgraphs mean changes ripple unpredictably.
   - Related to the CLIQUE problem (NP-complete): a fully connected
     subgraph (clique) has density 1.0 — maximum coupling.
   - Score: density * 10  (range 0–10)

2. CHANGE PROPAGATION PROBABILITY (fan-out weighted by centrality)
   - Models how likely a change is to cascade beyond the seed modules.
   - High fan-out + high betweenness = hub module = dangerous to touch.
   - Related to influence maximization (NP-hard): which nodes, if
     "infected" with a change, spread it to the most others?
   - Score: normalized propagation risk  (range 0–10)

3. CHROMATIC ESTIMATE (approximate graph coloring number)
   - The chromatic number χ(G) is the minimum number of colors needed
     so that no two adjacent nodes share a color. Finding χ(G) exactly
     is NP-hard, so we use a greedy upper bound.
   - In our context: colors ≈ independent work streams. Higher χ means
     more sequential dependencies — you can't parallelize the work.
   - Score: min(chromatic_number, 10)  (range 0–10)

4. DOMINATION NUMBER (approximate minimum dominating set)
   - A dominating set S is a subset where every node is either in S or
     adjacent to a node in S. Finding the minimum |S| is NP-hard.
   - In our context: the dominating set represents the minimum number of
     "control points" you must understand to oversee the entire change.
   - More control points = more cognitive load = harder to review/test.
   - Score: min(domination_number, 10)  (range 0–10)

5. SEED-TO-AFFECTED RATIO (amplification factor)
   - seed_modules are what the developer intends to change.
   - affected_modules include direct neighbors pulled in by dependencies.
   - A high ratio means a small intended change has large side effects.
   - Score: min(ratio * 3, 10)  (range 0–10)

FINAL SCORE = weighted sum of the five signals (range 0–50).

Complexity tiers:
    < 8   → Trivial   (1–2 hours engineering,  1 hour testing)
    8–16  → Low       (4–9 hours engineering,  4 hours testing)
    17–30 → Medium    (3–5 days engineering,   1–2 days testing)
    > 30  → High      (5–10 days engineering,  2–3 days testing)
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


# ── Complexity tiers ─────────────────────────────────────────────────────────

_COMPLEXITY_TRIVIAL = "Trivial"
_COMPLEXITY_LOW = "Low"
_COMPLEXITY_MEDIUM = "Medium"
_COMPLEXITY_HIGH = "High"

_EFFORT_MAP = {
    _COMPLEXITY_TRIVIAL: {"engineering": "1-2 hours", "testing": "1 hour"},
    _COMPLEXITY_LOW:     {"engineering": "4-9 hours", "testing": "4 hours"},
    _COMPLEXITY_MEDIUM:  {"engineering": "3-5 days",  "testing": "1-2 days"},
    _COMPLEXITY_HIGH:    {"engineering": "5-10 days", "testing": "2-3 days"},
}

# Weights for each scoring signal (sum = 1.0 for interpretability).
_W_COUPLING    = 0.25   # How tangled is the affected subgraph?
_W_PROPAGATION = 0.25   # How likely are changes to cascade further?
_W_CHROMATIC   = 0.20   # How many independent work streams are needed?
_W_DOMINATION  = 0.15   # How many control points must be understood?
_W_AMPLIFICATION = 0.15 # How much does the change amplify beyond intent?


class EffortEstimator:
    """Deterministic, graph-algorithm-based effort estimator."""

    def estimate(
        self,
        impact_report: dict,
        graph: nx.DiGraph,
        architecture_report: dict,
        repo_path: str,
        llm=None,  # kept for API compatibility, not used
    ) -> dict:
        affected: list[str] = impact_report.get("affected_modules", [])
        seed_modules: list[str] = impact_report.get("seed_modules", [])
        layers_map: dict[str, list[str]] = architecture_report.get("layers", {})

        # Build the undirected subgraph of affected modules for analysis.
        # We use undirected because coupling/coloring don't depend on direction.
        subgraph = graph.subgraph(
            [m for m in affected if graph.has_node(m)]
        ).to_undirected()

        # ── Signal 1: Coupling Density ───────────────────────────────────
        # density = 2|E| / (|V|(|V|-1)) for undirected graphs.
        # 0 = no edges (fully decoupled), 1 = complete graph (max coupling).
        coupling = _coupling_density(subgraph)

        # ── Signal 2: Change Propagation Probability ─────────────────────
        # For each seed module, compute: fan_out * betweenness_centrality.
        # Sum and normalize. High values mean seeds are "super-spreaders".
        propagation = _propagation_risk(seed_modules, graph)

        # ── Signal 3: Chromatic Number (greedy upper bound) ──────────────
        # Uses the greedy coloring heuristic on the affected subgraph.
        # More colors = more sequential dependency chains = less parallelism.
        chromatic = _chromatic_estimate(subgraph)

        # ── Signal 4: Domination Number (greedy approximation) ───────────
        # Greedy minimum dominating set: pick the node covering the most
        # uncovered neighbors, repeat. This is a log(n)-approximation of
        # the NP-hard minimum dominating set problem.
        domination = _domination_number(subgraph)

        # ── Signal 5: Seed-to-Affected Amplification ─────────────────────
        # ratio = affected / seed. A ratio of 1 means no amplification;
        # higher means the change is pulling in unintended modules.
        amplification = _amplification_ratio(seed_modules, affected)

        # ── Weighted total ───────────────────────────────────────────────
        # Each signal is already on a 0–10 scale.
        # Weighted sum → 0–50 range mapped to complexity tiers.
        total = (
            coupling      * _W_COUPLING      * 50
            + propagation * _W_PROPAGATION    * 50
            + chromatic   * _W_CHROMATIC      * 50
            + domination  * _W_DOMINATION     * 50
            + amplification * _W_AMPLIFICATION * 50
        )

        # ── Map to complexity tier ───────────────────────────────────────
        if total < 8:
            complexity = _COMPLEXITY_TRIVIAL
        elif total < 16:
            complexity = _COMPLEXITY_LOW
        elif total <= 30:
            complexity = _COMPLEXITY_MEDIUM
        else:
            complexity = _COMPLEXITY_HIGH

        effort = _EFFORT_MAP[complexity]

        # ── Confidence ───────────────────────────────────────────────────
        # Confidence increases with more data points (affected modules).
        n = len(affected)
        if n >= 10:
            confidence = "85%"
        elif n >= 5:
            confidence = "75%"
        elif n >= 2:
            confidence = "65%"
        else:
            confidence = "50%"

        layers_touched = sorted(_layers_touched(affected, layers_map))

        scores = {
            "coupling_density": round(coupling, 2),
            "propagation_risk": round(propagation, 2),
            "chromatic_estimate": round(chromatic, 2),
            "domination_number": round(domination, 2),
            "amplification_ratio": round(amplification, 2),
            "total": round(total, 2),
        }

        result = {
            "complexity": complexity,
            "effort_range": effort["engineering"],
            "testing_effort": effort["testing"],
            "confidence": confidence,
            "layers_touched": layers_touched,
            "scores": scores,
        }

        root = Path(repo_path).resolve()
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "effort_estimation.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result


# ── Algorithm implementations ────────────────────────────────────────────────


def _coupling_density(subgraph: nx.Graph) -> float:
    """Compute subgraph density (0–10 scale).

    Density = 2|E| / (|V|(|V|-1)). A clique has density 1.0.
    Finding the maximum clique is NP-hard; density is a poly-time proxy
    that tells us how close the subgraph is to being fully connected.
    """
    if subgraph.number_of_nodes() < 2:
        return 0.0
    # nx.density returns 0.0–1.0
    return nx.density(subgraph) * 10


def _propagation_risk(seed_modules: list[str], graph: nx.DiGraph) -> float:
    """Estimate how likely changes will propagate beyond the seed set (0–10 scale).

    For each seed, risk = out_degree * betweenness_centrality.
    This captures "super-spreader" nodes: modules that both export to
    many dependents AND sit on many shortest paths.

    Related to the NP-hard influence maximization problem: given a budget
    of k seed nodes, which k nodes maximize cascade spread?
    """
    if not seed_modules:
        return 0.0

    risk_sum = 0.0
    for mod in seed_modules:
        if graph.has_node(mod):
            out_deg = graph.out_degree(mod)
            betweenness = graph.nodes[mod].get("betweenness_centrality", 0.0)
            # out_degree * betweenness: high when module is both
            # heavily depended on and a critical bridge.
            risk_sum += out_deg * (1 + betweenness * 10)

    # Normalize: assume a propagation risk > 20 is max danger.
    return min(risk_sum / max(len(seed_modules), 1) / 2.0, 10.0)


def _chromatic_estimate(subgraph: nx.Graph) -> float:
    """Greedy chromatic number upper bound (0–10 scale).

    The chromatic number χ(G) = minimum colors so no two adjacent nodes
    share a color. Exact computation is NP-hard.

    Greedy coloring gives χ(G) ≤ Δ(G) + 1 where Δ is the max degree.
    In practice it's often much tighter.

    Interpretation: each color = one independent work stream that can
    proceed without conflicts. More colors = more sequential coordination.
    """
    if subgraph.number_of_nodes() == 0:
        return 0.0

    # nx.greedy_color returns {node: color_int}
    coloring = nx.coloring.greedy_color(subgraph, strategy="largest_first")
    if not coloring:
        return 0.0
    num_colors = max(coloring.values()) + 1  # colors are 0-indexed
    return min(float(num_colors), 10.0)


def _domination_number(subgraph: nx.Graph) -> float:
    """Greedy minimum dominating set approximation (0–10 scale).

    A dominating set S ⊆ V where every node is in S or adjacent to S.
    Finding minimum |S| is NP-hard. Greedy gives O(log n) approximation.

    Interpretation: |S| = minimum "control points" a developer must
    understand to oversee the entire affected area. More control points
    = more cognitive load = harder to review and test.
    """
    if subgraph.number_of_nodes() == 0:
        return 0.0

    # Greedy: always pick the node that dominates the most uncovered nodes.
    uncovered = set(subgraph.nodes())
    dominating_set: list[str] = []

    while uncovered:
        # For each candidate, count how many uncovered nodes it would cover
        # (itself + its uncovered neighbors).
        best_node = None
        best_cover = -1
        for node in subgraph.nodes():
            cover = 0
            if node in uncovered:
                cover += 1
            cover += len(set(subgraph.neighbors(node)) & uncovered)
            if cover > best_cover:
                best_cover = cover
                best_node = node

        if best_node is None:
            break

        dominating_set.append(best_node)
        # Mark best_node and its neighbors as covered.
        uncovered.discard(best_node)
        uncovered -= set(subgraph.neighbors(best_node))

    return min(float(len(dominating_set)), 10.0)


def _amplification_ratio(seed_modules: list[str], affected: list[str]) -> float:
    """Compute how much the change amplifies beyond the developer's intent (0–10 scale).

    ratio = |affected| / |seed|.
    - ratio = 1: no amplification, change is perfectly contained.
    - ratio = 5: each seed file drags in 4 extra files on average.

    A high ratio signals that the codebase has tight coupling around
    the change area — the developer's "small" change has large side effects.
    """
    n_seed = max(len(seed_modules), 1)
    n_affected = len(affected)
    ratio = n_affected / n_seed
    # Scale: ratio of 1 = 0, ratio of ~4+ = 10
    return min((ratio - 1) * 3.0, 10.0)


def _layers_touched(affected: list[str], layers_map: dict[str, list[str]]) -> set[str]:
    """Return the set of architecture layers that contain affected modules."""
    touched: set[str] = set()
    for layer, mods in layers_map.items():
        for mod in mods:
            if mod in affected:
                touched.add(layer)
    return touched

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
        feedback_manager=None,
        architecture_report: dict | None = None,
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
        files_list = "\n".join(f"- {f}" for f in all_files)

        # Retrieve past feedback on similar changes to inform seed identification.
        feedback_context = ""
        if feedback_manager is not None:
            past_feedback = feedback_manager.get_relevant_context(change_description, top_k=3)
            if past_feedback:
                feedback_context = (
                    "\n\nPast feedback on similar changes (use to improve your answer):\n"
                    + "\n---\n".join(past_feedback)
                )

        prompt = (
            f"You are a code analysis assistant. Given the repository context below, "
            f"identify which files would need to be modified for the following change request.\n\n"
            f"Change request: \"{change_description}\"\n\n"
            f"Repository files (sample):\n{files_list}\n\n"
            f"Repository context:\n{context}"
            f"{feedback_context}\n\n"
            f"Return ONLY a JSON array of file paths from the repository that would need changes. "
            f"Example: [\"auth/service.py\", \"auth/routes.py\"]\n"
            f"If you cannot determine this from the context, return: []\n"
            f"Only include files that actually exist in the repository list above."
        )

        raw = llm.complete(prompt)
        seed_modules = _parse_json_array(raw)

        # Validate against actual files (with fuzzy matching on partial paths)
        valid_files = set(parsed_code.get("files", {}).keys())
        matched_modules = []
        for candidate in seed_modules:
            if candidate in valid_files:
                matched_modules.append(candidate)
            else:
                # Try partial match: LLM might return "app.py" when key is "src/app.py"
                for vf in valid_files:
                    if vf.endswith("/" + candidate) or vf.endswith(candidate):
                        matched_modules.append(vf)
                        break
        seed_modules = matched_modules

        # Traversal: direct neighbors only (1 hop) to avoid over-inflation
        affected: set[str] = set(seed_modules)
        for seed in seed_modules:
            if graph.has_node(seed):
                # downstream: direct dependents (modules that import seed)
                for node in graph.successors(seed):
                    affected.add(node)
                # upstream: direct dependencies (modules that seed imports)
                for node in graph.predecessors(seed):
                    affected.add(node)

        affected_list = sorted(affected)

        # Generate concrete edit steps with code snippets — one LLM call per seed file
        # We call per-file so the LLM can focus on the actual source and produce
        # accurate before/after code snippets.
        edit_steps = []
        if seed_modules:
            root = Path(repo_path).resolve()
            for seed_file in seed_modules:
                # Read actual source code for precise snippets
                source = ""
                try:
                    file_path = root / seed_file
                    if file_path.exists():
                        source = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass

                if not source:
                    file_chunks = rag.query(f"file {seed_file}", top_k=1)
                    source = file_chunks[0] if file_chunks else "Source not available."

                # Truncate very large files to keep prompt manageable
                if len(source) > 6000:
                    source = source[:6000] + "\n... (truncated)"

                step_prompt = (
                    f"You are a code editing assistant. Given the change request and the full source code, "
                    f"provide edit steps WITH code snippets showing exactly what to change.\n\n"
                    f"Change request: \"{change_description}\"\n"
                    f"File: {seed_file}\n\n"
                    f"Current source code:\n```\n{source}\n```\n\n"
                    f"Return a JSON array. Each step must have:\n"
                    f"- \"file\": the file path\n"
                    f"- \"action\": \"change\", \"add\", or \"remove\"\n"
                    f"- \"description\": short explanation of what to do\n"
                    f"- \"before\": the existing code snippet that needs to change (exact lines from source). "
                    f"For \"add\" actions, this is the code AFTER which the new code should be inserted. "
                    f"For \"remove\" actions, this is the code to delete.\n"
                    f"- \"after\": the replacement code snippet. For \"add\" actions, include the anchor line "
                    f"plus the new code. For \"remove\" actions, set to empty string.\n\n"
                    f"Example:\n"
                    f"[{{\n"
                    f"  \"file\": \"models.py\",\n"
                    f"  \"action\": \"add\",\n"
                    f"  \"description\": \"Add month-over-month change field to TrendingKeyword\",\n"
                    f"  \"before\": \"    trend_rate: float = 0.0\",\n"
                    f"  \"after\": \"    trend_rate: float = 0.0\\n    mom_change: float | None = None\"\n"
                    f"}}]\n\n"
                    f"Rules:\n"
                    f"- Maximum 3 steps per file. Combine tiny related edits.\n"
                    f"- The \"before\" snippet MUST be exact text from the source code above.\n"
                    f"- Keep snippets focused — only the lines that change plus 1-2 lines of context.\n"
                    f"- Only include changes that are directly needed for the change request."
                )
                raw_steps = llm.complete(step_prompt)
                parsed_steps = _parse_json_array_of_objects(raw_steps)
                edit_steps.extend(parsed_steps)

        affected_list = sorted(affected)

        # ── Task interpretation (dev-thinking §2) ────────────────────────
        # Run the 6-step pipeline to extract clarified intent, unknowns,
        # risks, atomic tasks, and scope for richer effort estimation.
        interpretation: dict = {}
        if architecture_report is not None:
            try:
                from repobrain.src.interpreter.task_interpreter import TaskInterpreter
                interpretation = TaskInterpreter().interpret(
                    change_description,
                    affected_list,
                    architecture_report,
                    parsed_code,
                    rag,
                    llm,
                )
            except Exception:
                interpretation = {}

        result = {
            "change_description": change_description,
            "seed_modules": seed_modules,
            "affected_modules": affected_list,
            "total_affected": len(affected_list),
            "edit_steps": edit_steps,
            "interpretation": interpretation,
        }

        root = Path(repo_path).resolve()
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "impact_report.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result


def _parse_json_array(text: str) -> list[str]:
    """Extract a JSON array of strings from LLM output."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass
    return []


def _parse_json_array_of_objects(text: str) -> list[dict]:
    """Extract a JSON array of objects from LLM output."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    return []

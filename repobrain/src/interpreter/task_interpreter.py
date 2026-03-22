"""
Task Interpreter — §2 of dev-thinking.md

Runs the 6-step developer-thinking pipeline on a change description:

  Step 1 — Scope Extraction    (which systems: frontend/backend/DB/API/external)
  Step 2 — Intent Clarification (vague → concrete goal)
  Step 3 — Atomic Decomposition (smallest meaningful units with time estimates)
  Step 4 — Dependency Mapping  (sequential vs parallelizable — captured in layers)
  Step 5 — Unknown Detection   (Low / Medium / High uncertainty level)
  Step 6 — Risk Assessment     (critical systems, performance, data integrity)

Two LLM calls are made:
  Call 1: interpretation (steps 1, 2, 4, 5, 6)
  Call 2: atomic decomposition (step 3)

The output feeds into EffortEstimator to compute the dev-thinking complexity
score and ETA using the §3 + §4 formulas.
"""

from __future__ import annotations

import json
import re


class TaskInterpreter:
    """Interprets a change description like an experienced developer would."""

    def interpret(
        self,
        change_description: str,
        affected_modules: list[str],
        architecture_report: dict,
        parsed_code: dict,
        rag,
        llm,
        seed_modules: list[str] | None = None,
        seed_sources: dict[str, str] | None = None,
    ) -> dict:
        """Run the full interpretation pipeline.

        Returns
        -------
        dict with keys:
            clarified_intent   : str
            scope              : list[str]  (e.g. ["backend", "api"])
            unknown_level      : "Low" | "Medium" | "High"
            unknowns           : list[str]
            risks              : list[str]
            atomic_tasks       : list[{task, layer, min_hours, max_hours}]
            insufficient_info  : bool  (True when unknown_level == "High")
        """
        arch_pattern = architecture_report.get("pattern", "unknown")
        layers = architecture_report.get("layers", {})
        layers_summary = ", ".join(
            f"{layer}: {len(mods)} modules" for layer, mods in layers.items() if mods
        )

        # RAG context for grounding the interpretation
        context_chunks = rag.query(change_description, top_k=3)
        context = "\n---\n".join(context_chunks) if context_chunks else "No context available."

        affected_summary = "\n".join(f"- {m}" for m in affected_modules[:20])
        if len(affected_modules) > 20:
            affected_summary += f"\n... and {len(affected_modules) - 20} more"

        # ── Call 1: Interpretation (steps 1, 2, 4, 5, 6) ─────────────────
        interp = self._call_interpretation(
            change_description, arch_pattern, layers_summary,
            affected_summary, context, llm
        )

        # ── Call 2: Atomic decomposition (step 3) ─────────────────────────
        # Use seed_modules (files that actually need editing) rather than the
        # full affected_modules list (which includes transitive graph neighbours
        # that don't need to be touched). Pass actual source code so the LLM
        # estimates only the DELTA — not work already implemented.
        decomp_modules = seed_modules if seed_modules else affected_modules

        # Build source context: prefer actual file content over RAG chunks
        if seed_sources:
            source_context = "\n---\n".join(
                f"# {path}\n{src}" for path, src in seed_sources.items()
            )
        else:
            source_context = context  # fall back to RAG chunks

        atomic_tasks = self._call_atomic_decomposition(
            interp.get("clarified_intent", change_description),
            decomp_modules,
            interp.get("scope", []),
            source_context,
            llm,
        )

        unknown_level = interp.get("unknown_level", "Medium")

        return {
            "clarified_intent": interp.get("clarified_intent", change_description),
            "scope": interp.get("scope", []),
            "unknown_level": unknown_level,
            "unknowns": interp.get("unknowns", []),
            "risks": interp.get("risks", []),
            "atomic_tasks": atomic_tasks,
            "insufficient_info": unknown_level == "High",
        }

    # ── LLM call 1: Interpretation ────────────────────────────────────────

    def _call_interpretation(
        self,
        change_description: str,
        arch_pattern: str,
        layers_summary: str,
        affected_summary: str,
        context: str,
        llm,
    ) -> dict:
        prompt = (
            f"You are a senior software engineer interpreting a task request.\n\n"
            f"Change request: \"{change_description}\"\n\n"
            f"Architecture pattern: {arch_pattern}\n"
            f"Architecture layers: {layers_summary}\n\n"
            f"Affected modules:\n{affected_summary}\n\n"
            f"Repository context:\n{context}\n\n"
            f"Analyze this change request like an experienced developer. Return ONLY valid JSON:\n\n"
            f"{{\n"
            f"  \"clarified_intent\": \"<concrete, specific restatement of what needs to be done>\",\n"
            f"  \"scope\": [\"<one or more of: frontend, backend, api, database, external, test, config>\"],\n"
            f"  \"unknown_level\": \"<Low|Medium|High>\",\n"
            f"  \"unknowns\": [\"<specific thing that is unclear or unspecified>\"],\n"
            f"  \"risks\": [\"<specific risk: production impact, data integrity, performance, breaking change>\"]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- unknown_level=Low: fully specified, no ambiguity\n"
            f"- unknown_level=Medium: some details unclear but intent is clear\n"
            f"- unknown_level=High: vague request, insufficient to estimate reliably\n"
            f"- unknowns: be specific — 'DB schema unclear' not just 'unclear'\n"
            f"- risks: only real risks, not hypothetical ones\n"
            f"- If no unknowns exist, return []\n"
            f"- If no risks exist, return []"
        )
        raw = llm.complete(prompt)
        return _parse_json_object(raw)

    # ── LLM call 2: Atomic decomposition ──────────────────────────────────

    def _call_atomic_decomposition(
        self,
        clarified_intent: str,
        affected_modules: list[str],
        scope: list[str],
        existing_code_context: str,
        llm,
    ) -> list[dict]:
        affected_summary = "\n".join(f"- {m}" for m in affected_modules[:15])
        scope_str = ", ".join(scope) if scope else "general"

        prompt = (
            f"You are a senior software engineer breaking down a task into atomic work items.\n\n"
            f"Task: \"{clarified_intent}\"\n"
            f"Systems involved: {scope_str}\n"
            f"Files to modify:\n{affected_summary}\n\n"
            f"EXISTING CODE (what is already implemented):\n{existing_code_context}\n\n"
            f"Break this into atomic tasks a developer would execute. "
            f"Estimate ONLY the delta — work that does not already exist in the code above.\n\n"
            f"Return ONLY a valid JSON array:\n"
            f"[\n"
            f"  {{\"task\": \"<specific action>\", \"layer\": \"<frontend|backend|api|database|test|config>\","
            f" \"min_hours\": <number>, \"max_hours\": <number>}}\n"
            f"]\n\n"
            f"Rules:\n"
            f"- CRITICAL: If the existing code already loads, computes, or stores the data needed, "
            f"do NOT add a task for that — it is already done\n"
            f"- CRITICAL: Estimate only what needs to be written, not what already exists\n"
            f"- A UI addition to an existing file = 0.25–1h, not hours of setup\n"
            f"- A new isolated function in an existing file = 0.5–1.5h\n"
            f"- Only add a testing task if manual or automated testing is actually required\n"
            f"- min_hours and max_hours must be positive numbers, max >= min\n"
            f"- Maximum 5 tasks — if you have more, the decomposition is too granular\n"
            f"- Be conservative: default to the lower end for well-understood changes"
        )
        raw = llm.complete(prompt)
        tasks = _parse_json_array(raw)

        # Validate and sanitize each task
        validated = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            try:
                validated.append({
                    "task": str(t.get("task", "Unknown task")),
                    "layer": str(t.get("layer", "backend")),
                    "min_hours": max(0.25, float(t.get("min_hours", 1))),
                    "max_hours": max(0.25, float(t.get("max_hours", 2))),
                })
            except (ValueError, TypeError):
                continue

        # Ensure max >= min
        for t in validated:
            if t["max_hours"] < t["min_hours"]:
                t["max_hours"] = t["min_hours"]

        return validated


# ── JSON parsing helpers ──────────────────────────────────────────────────────


def _parse_json_object(text: str) -> dict:
    """Extract a JSON object from LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _parse_json_array(text: str) -> list:
    """Extract a JSON array from LLM output."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []

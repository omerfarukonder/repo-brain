from __future__ import annotations

import json
from pathlib import Path

from repobrain.src.llm.base import LLMClient
from repobrain.src.llm.rag import RAGIndex

_HALLUCINATION_GUARD = (
    "\n\nIMPORTANT: Only answer based on the repository context provided above. "
    "Do not invent modules, functions, or behaviors not present in the context. "
    "If you cannot determine the answer from the context, respond with exactly: "
    "'Unable to determine from repository analysis.'"
)


class ModuleSummarizer:
    """Generates natural language summaries for each module using LLM + RAG."""

    def summarize_all(
        self,
        parsed_code: dict,
        rag: RAGIndex,
        llm: LLMClient,
        repo_path: str,
    ) -> dict:
        """Summarize every file in parsed_code.

        Returns
        -------
        dict: {"file_path": {"summary": str}}
        """
        files_data: dict = parsed_code.get("files", {})
        summaries: dict[str, dict] = {}

        for rel_path, meta in files_data.items():
            functions = ", ".join(meta.get("functions", [])) or "none"
            classes = ", ".join(meta.get("classes", [])) or "none"
            imports = "; ".join(meta.get("imports", [])[:8]) or "none"

            context_chunks = rag.query(f"What does {rel_path} do?", top_k=3)
            context = "\n---\n".join(context_chunks) if context_chunks else "No additional context."

            prompt = (
                f"You are a code analysis assistant. Based only on the following "
                f"repository context, explain what this module does.\n\n"
                f"Module: {rel_path}\n"
                f"Language: {meta.get('language', 'unknown')}\n"
                f"Classes: {classes}\n"
                f"Functions: {functions}\n"
                f"Imports: {imports}\n\n"
                f"Context from repository:\n{context}\n\n"
                f"Respond with 2-3 sentences describing the module's responsibility."
                f"{_HALLUCINATION_GUARD}"
            )

            summary = llm.complete(prompt)
            summaries[rel_path] = {"summary": summary}

        root = Path(repo_path).resolve()
        output_dir = root / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "module_summaries.json").open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

        return summaries

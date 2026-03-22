"""
Feedback Manager — evolving knowledge base for effort estimation calibration.

How it works
------------
1. User runs `repobrain impact "add OAuth login"` → system estimates "Medium, 3-5 days"
2. User runs `repobrain feedback "this should take less time"` (natural language)
3. LLM parses the feedback into a structured correction:
     direction: lower | higher | correct
     magnitude: 0.0–1.0 (how strong the correction is)
4. The entry is stored in two places:
   a. `analysis/feedback_history.json` — append-only log used for calibration
   b. ChromaDB collection `repobrain_feedback` — for semantic retrieval

Calibration
-----------
`get_calibration_factor()` computes a weighted-average bias from recent feedback
and returns a multiplier (0.5–1.5) applied to the raw effort score.

Context injection
-----------------
`get_relevant_context(change_desc)` queries ChromaDB for past feedback on
similar changes and returns text chunks that are injected into future LLM prompts.
This makes the system genuinely learn from past corrections.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


class FeedbackManager:
    """Stores, retrieves, and applies user feedback to calibrate future estimates."""

    def __init__(
        self,
        repo_path: str,
        persist_dir: str = "./.chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._repo_path = Path(repo_path).resolve()
        self._history_file = self._repo_path / "analysis" / "feedback_history.json"
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._embedder = None
        self._collection = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, raw_feedback: str, llm) -> dict:
        """Parse natural language feedback and persist it.

        Reads the last saved impact_report.json + effort_estimation.json to
        understand what the user is correcting, then parses the feedback via
        LLM and stores the structured result.

        Returns the stored entry dict, or a dict with key "error" if no prior
        estimate was found.
        """
        impact = self._load_last_impact()
        effort = self._load_last_effort()

        if not impact or not effort:
            return {
                "error": "No previous estimate found. Run `repobrain impact` first."
            }

        change_desc = impact.get("change_description", "unknown")
        complexity = effort.get("complexity", "unknown")
        effort_range = effort.get("effort_range", "unknown")
        score = effort.get("scores", {}).get("total", 0.0)

        parsed = self._parse_feedback(raw_feedback, complexity, effort_range, llm)

        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "change_description": change_desc,
            "estimated_complexity": complexity,
            "estimated_effort": effort_range,
            "estimated_score": score,
            "raw_feedback": raw_feedback,
            "parsed_direction": parsed.get("direction", "correct"),
            "parsed_magnitude": float(parsed.get("magnitude", 0.0)),
        }

        history = self._load_history()
        history.append(entry)
        self._save_history(history)

        self._embed_and_store(entry)

        return entry

    def get_calibration_factor(self) -> float:
        """Return a score multiplier derived from accumulated feedback.

        Uses the last 20 feedback entries with recency weighting (more recent
        entries count more). The bias is clamped to a ±50% adjustment:
          - All "lower" feedback → factor approaches 0.5 (estimates cut in half)
          - All "higher" feedback → factor approaches 1.5 (estimates boosted)
          - No feedback → factor = 1.0 (no change)
        """
        history = self._load_history()
        if not history:
            return 1.0

        recent = history[-20:]
        direction_map = {"lower": -1.0, "correct": 0.0, "higher": 1.0}

        weighted_sum = 0.0
        total_weight = 0.0
        for i, entry in enumerate(recent):
            weight = float(i + 1)  # later entries weigh more
            direction = direction_map.get(entry.get("parsed_direction", "correct"), 0.0)
            magnitude = float(entry.get("parsed_magnitude", 0.0))
            weighted_sum += direction * magnitude * weight
            total_weight += weight

        bias = weighted_sum / total_weight if total_weight > 0 else 0.0
        factor = 1.0 + bias * 0.5  # ±50% max adjustment
        return round(max(0.5, min(1.5, factor)), 3)

    def get_relevant_context(self, change_desc: str, top_k: int = 3) -> list[str]:
        """Return past feedback relevant to the current change description.

        Uses semantic search over the ChromaDB feedback collection.
        Returns an empty list if no feedback has been stored yet.
        """
        try:
            col = self._get_collection()
            if col.count() == 0:
                return []
            embedder = self._get_embedder()
            q_emb = embedder.encode([change_desc], show_progress_bar=False).tolist()
            results = col.query(
                query_embeddings=q_emb,
                n_results=min(top_k, col.count()),
            )
            return results.get("documents", [[]])[0]
        except Exception:
            return []

    def get_history(self) -> list[dict]:
        """Return the full feedback history log."""
        return self._load_history()

    # ── LLM feedback parsing ──────────────────────────────────────────────────

    def _parse_feedback(
        self, raw_feedback: str, complexity: str, effort_range: str, llm
    ) -> dict:
        """Ask the LLM to extract direction + magnitude from natural language feedback."""
        prompt = (
            f"Analyze this user feedback about a software effort estimate.\n\n"
            f"Previous estimate: {complexity} complexity, {effort_range}\n"
            f"User feedback: \"{raw_feedback}\"\n\n"
            f"Extract two things:\n"
            f"1. \"direction\": was the estimate too high, too low, or correct?\n"
            f"   - \"lower\": estimate was too high (user thinks it takes less time)\n"
            f"   - \"higher\": estimate was too low (user thinks it takes more time)\n"
            f"   - \"correct\": estimate was accurate\n"
            f"2. \"magnitude\": how strong is the correction? (0.0 to 1.0)\n"
            f"   - 'a little bit less' → 0.2\n"
            f"   - 'noticeably less' → 0.5\n"
            f"   - 'way off, should be trivial' → 0.9\n\n"
            f"Return ONLY valid JSON, nothing else:\n"
            f"{{\"direction\": \"lower\", \"magnitude\": 0.6}}"
        )
        raw = llm.complete(prompt)
        try:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if "direction" in parsed and "magnitude" in parsed:
                    return parsed
        except Exception:
            pass
        return {"direction": "correct", "magnitude": 0.0}

    # ── Storage helpers ───────────────────────────────────────────────────────

    def _load_history(self) -> list[dict]:
        if self._history_file.exists():
            try:
                with self._history_file.open(encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self, history: list[dict]) -> None:
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        with self._history_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _load_last_impact(self) -> dict | None:
        p = self._repo_path / "analysis" / "impact_report.json"
        if p.exists():
            try:
                with p.open(encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _load_last_effort(self) -> dict | None:
        p = self._repo_path / "analysis" / "effort_estimation.json"
        if p.exists():
            try:
                with p.open(encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    # ── ChromaDB embedding ────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self._persist_dir)
            # Separate collection from code index — never wiped by rag.build()
            self._collection = client.get_or_create_collection("repobrain_feedback")
        return self._collection

    def _embed_and_store(self, entry: dict) -> None:
        """Embed the feedback entry and store it in ChromaDB for future retrieval."""
        try:
            doc_text = (
                f"Change: {entry['change_description']}\n"
                f"Estimate: {entry['estimated_complexity']} ({entry['estimated_effort']})\n"
                f"Feedback: {entry['raw_feedback']}\n"
                f"Correction: {entry['parsed_direction']} by {entry['parsed_magnitude']:.1f}"
            )
            embedder = self._get_embedder()
            embedding = embedder.encode([doc_text], show_progress_bar=False).tolist()
            self._get_collection().add(
                documents=[doc_text],
                embeddings=embedding,
                ids=[entry["id"]],
            )
        except Exception:
            # Embedding failure should never block feedback storage
            pass

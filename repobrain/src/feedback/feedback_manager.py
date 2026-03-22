"""
Feedback Manager — repo-specific learning layer.

Architecture (Option A — Static Global Brain):

    GlobalBrain  (~/.repobrain/global/)
        Static seed knowledge from dev-thinking.md.
        Never mutated by user feedback.
        Provides: context injection only.

    FeedbackManager  (<repo>/analysis/feedback_history.json)
        Repo-specific feedback from real usage.
        Provides: context injection + calibration factor.

When estimating:
    context  = global_brain.get_relevant_context()   (general heuristics)
             + feedback_manager.get_relevant_context() (repo-specific, ranked higher)

    calibration_factor = feedback_manager.get_calibration_factor(scope)
        - Scope-aware: feedback on "frontend" changes only adjusts frontend estimates
        - Time-decayed: older entries carry less weight
        - Only from repo brain — global brain never contributes to calibration

When feedback arrives:
    → Always written to repo brain (FeedbackManager)
    → Scope field stored from the interpretation result
    → Global brain is never touched
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from repobrain.src.feedback.global_brain import GlobalBrain

_REPO_SIMILARITY_THRESHOLD = 1.3   # L2 distance cutoff for repo feedback retrieval


class FeedbackManager:
    """Repo-specific feedback store with scope-aware, time-decayed calibration."""

    def __init__(
        self,
        repo_path: str,
        persist_dir: str = "./.chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._repo_path    = Path(repo_path).resolve()
        self._history_file = self._repo_path / "analysis" / "feedback_history.json"
        self._persist_dir  = persist_dir
        self._embedding_model = embedding_model
        self._embedder     = None
        self._collection   = None
        self._global_brain = GlobalBrain(embedding_model)

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, raw_feedback: str, llm) -> dict:
        """Parse natural language feedback and persist it to the repo brain.

        Reads the last saved impact_report.json + effort_estimation.json to
        understand what the user is correcting. Stores the scope from the
        impact report's interpretation so calibration can be scope-filtered.
        """
        impact = self._load_last_impact()
        effort = self._load_last_effort()

        if not impact or not effort:
            return {"error": "No previous estimate found. Run `repobrain impact` first."}

        change_desc = impact.get("change_description", "unknown")
        complexity  = effort.get("complexity", "unknown")
        effort_range = effort.get("effort_range", "unknown")
        score       = effort.get("scores", {}).get("graph_total", 0.0)
        scope       = impact.get("interpretation", {}).get("scope", [])

        parsed = self._parse_feedback(raw_feedback, complexity, effort_range, llm)

        entry = {
            "id":                  str(uuid.uuid4()),
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "change_description":  change_desc,
            "scope":               scope,
            "estimated_complexity": complexity,
            "estimated_effort":    effort_range,
            "estimated_score":     score,
            "raw_feedback":        raw_feedback,
            "parsed_direction":    parsed.get("direction", "correct"),
            "parsed_magnitude":    float(parsed.get("magnitude", 0.0)),
        }

        history = self._load_history()
        history.append(entry)
        self._save_history(history)
        self._embed_and_store(entry)

        return entry

    def get_calibration_factor(self, scope: list[str] | None = None) -> float:
        """Compute a score multiplier from repo-specific feedback.

        Scope-aware: if scope is provided, prefer entries that share at least
        one scope tag. Falls back to all entries if fewer than 3 scoped ones exist.

        Time-decayed: entries older than 90 days carry much less weight.
        Recent entries (< 7 days) carry full weight.

        Range: [0.5, 1.5]  — never more than ±50% adjustment.
        """
        history = self._load_history()
        if not history:
            return 1.0

        if scope:
            scoped = [e for e in history if _shares_scope(e.get("scope", []), scope)]
            entries = scoped if len(scoped) >= 3 else history
        else:
            entries = history

        recent = entries[-20:]
        direction_map = {"lower": -1.0, "correct": 0.0, "higher": 1.0}

        weighted_sum  = 0.0
        total_weight  = 0.0
        for i, entry in enumerate(recent):
            recency_weight = float(i + 1)
            time_weight    = _time_decay(entry.get("timestamp", ""))
            weight         = recency_weight * time_weight

            direction  = direction_map.get(entry.get("parsed_direction", "correct"), 0.0)
            magnitude  = float(entry.get("parsed_magnitude", 0.0))
            weighted_sum  += direction * magnitude * weight
            total_weight  += weight

        if total_weight == 0:
            return 1.0

        bias   = weighted_sum / total_weight
        factor = 1.0 + bias * 0.5
        return round(max(0.5, min(1.5, factor)), 3)

    def get_relevant_context(self, change_desc: str, top_k: int = 3) -> list[str]:
        """Return merged context: repo-specific first, then global heuristics.

        Repo-specific entries are ranked higher because they reflect this
        codebase's actual patterns. Global entries fill the gap when the
        repo brain is sparse or when the change type is new.

        Applies similarity threshold to filter irrelevant entries from both.
        """
        # Repo-specific (higher priority — listed first)
        repo_ctx   = self._get_repo_context(change_desc, top_k=top_k)
        # Global heuristics (fills knowledge gaps)
        global_ctx = self._global_brain.get_relevant_context(change_desc, top_k=2)

        # Deduplicate by content prefix, repo entries win
        seen: set[str] = set()
        merged: list[str] = []
        for doc in repo_ctx + global_ctx:
            key = doc[:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged

    def get_history(self) -> list[dict]:
        return self._load_history()

    def global_brain_count(self) -> int:
        """Return number of entries in the global brain."""
        return self._global_brain.count()

    def ensure_global_brain(self) -> int:
        """Ensure global brain is initialized. Returns entry count."""
        return self._global_brain.ensure_initialized()

    # ── LLM feedback parsing ──────────────────────────────────────────────────

    def _parse_feedback(
        self, raw_feedback: str, complexity: str, effort_range: str, llm
    ) -> dict:
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

    # ── Repo context retrieval ────────────────────────────────────────────────

    def _get_repo_context(self, change_desc: str, top_k: int = 3) -> list[str]:
        try:
            col = self._get_collection()
            if col.count() == 0:
                return []

            embedder = self._get_embedder()
            q_emb    = embedder.encode([change_desc], show_progress_bar=False).tolist()
            results  = col.query(
                query_embeddings=q_emb,
                n_results=min(top_k, col.count()),
                include=["documents", "distances"],
            )
            docs      = results.get("documents", [[]])[0]
            distances = results.get("distances",  [[]])[0]
            return [
                doc for doc, dist in zip(docs, distances)
                if dist <= _REPO_SIMILARITY_THRESHOLD
            ]
        except Exception:
            return []

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
            self._collection = client.get_or_create_collection("repobrain_feedback")
        return self._collection

    def _embed_and_store(self, entry: dict) -> None:
        try:
            scope_str = ", ".join(entry.get("scope", []))
            doc_text  = (
                f"Change: {entry['change_description']}\n"
                f"Scope: {scope_str}\n"
                f"Estimate: {entry['estimated_complexity']} ({entry['estimated_effort']})\n"
                f"Feedback: {entry['raw_feedback']}\n"
                f"Correction: {entry['parsed_direction']} by {entry['parsed_magnitude']:.1f}"
            )
            embedder  = self._get_embedder()
            embedding = embedder.encode([doc_text], show_progress_bar=False).tolist()
            self._get_collection().add(
                documents=[doc_text],
                embeddings=embedding,
                ids=[entry["id"]],
            )
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _shares_scope(entry_scope: list[str], query_scope: list[str]) -> bool:
    """Return True if the two scope lists share at least one tag."""
    return bool(set(entry_scope) & set(query_scope))


def _time_decay(timestamp_str: str) -> float:
    """Return a weight in (0, 1] based on how old the entry is.

    ≤ 7 days  → 1.0  (full weight)
    ≤ 30 days → 0.7
    ≤ 90 days → 0.4
    > 90 days → 0.1  (almost ignored but not zero)
    """
    try:
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).days
        if age <= 7:
            return 1.0
        elif age <= 30:
            return 0.7
        elif age <= 90:
            return 0.4
        else:
            return 0.1
    except Exception:
        return 0.5

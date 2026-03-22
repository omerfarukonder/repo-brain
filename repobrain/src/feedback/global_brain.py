"""
Global Brain — static seed knowledge base encoding universal developer heuristics.

This brain is authored from dev-thinking.md and general software engineering
experience. It NEVER changes from user feedback. It represents truths that
hold across any codebase:

    "Auth changes always carry production risk"
    "DB migrations are consistently underestimated"
    "UI-only changes are faster than graph score suggests"

It is built once (on first use) into a ChromaDB collection at:
    ~/.repobrain/global/chroma/

and rebuilt automatically if that directory is missing or empty.

The global brain provides CONTEXT INJECTION ONLY — it does not contribute
to the calibration factor. Only the repo-specific brain (FeedbackManager)
adjusts numerical scores.
"""

from __future__ import annotations

import json
from pathlib import Path


_GLOBAL_DIR   = Path.home() / ".repobrain" / "global"
_CHROMA_DIR   = str(_GLOBAL_DIR / "chroma")
_SEED_FILE    = Path(__file__).parent / "seed_knowledge.json"
_COLLECTION   = "repobrain_global"
_SIMILARITY_THRESHOLD = 1.2   # L2 distance — lower = more similar; >1.2 = too distant


class GlobalBrain:
    """Static knowledge base of general developer heuristics.

    Safe to instantiate multiple times — ChromaDB and embedder are lazy-loaded
    and shared. `ensure_initialized()` is idempotent.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self._embedding_model = embedding_model
        self._embedder = None
        self._collection = None

    # ── Public API ────────────────────────────────────────────────────────────

    def ensure_initialized(self) -> int:
        """Build the ChromaDB index from seed_knowledge.json if not already done.

        Returns the number of seed entries in the collection.
        Is idempotent — safe to call on every startup.
        """
        try:
            col = self._get_collection()
            if col.count() > 0:
                return col.count()
            return self._build()
        except Exception:
            return 0

    def get_relevant_context(
        self,
        change_desc: str,
        top_k: int = 2,
    ) -> list[str]:
        """Return the most relevant general heuristics for this change type.

        Applies a similarity threshold so only genuinely relevant entries
        are returned. Returns [] if nothing is similar enough.
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
                include=["documents", "distances"],
            )

            docs      = results.get("documents", [[]])[0]
            distances = results.get("distances",  [[]])[0]

            # Filter by similarity threshold
            return [
                doc for doc, dist in zip(docs, distances)
                if dist <= _SIMILARITY_THRESHOLD
            ]
        except Exception:
            return []

    def count(self) -> int:
        """Return number of seed entries indexed."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build(self) -> int:
        """Embed all seed entries and store in ChromaDB. Returns entry count."""
        if not _SEED_FILE.exists():
            return 0

        with _SEED_FILE.open(encoding="utf-8") as f:
            seeds: list[dict] = json.load(f)

        if not seeds:
            return 0

        col      = self._get_collection()
        embedder = self._get_embedder()

        documents: list[str] = []
        ids:       list[str] = []

        for s in seeds:
            # Build a rich document text that embeds well for semantic search
            scope_str = ", ".join(s.get("scope", []))
            doc = (
                f"Change type: {s.get('change_description', '')}\n"
                f"Scope: {scope_str}\n"
                f"Heuristic: {s.get('raw_feedback', '')}\n"
                f"Principle: {s.get('principle', '')}\n"
                f"Direction: {s.get('parsed_direction', '')} "
                f"(magnitude {s.get('parsed_magnitude', 0):.1f})"
            )
            documents.append(doc)
            ids.append(s["id"])

        embeddings = embedder.encode(documents, show_progress_bar=False).tolist()
        col.add(documents=documents, embeddings=embeddings, ids=ids)
        return len(documents)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=_CHROMA_DIR)
            self._collection = client.get_or_create_collection(_COLLECTION)
        return self._collection

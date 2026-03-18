from __future__ import annotations

from pathlib import Path


class RAGIndex:
    """Builds and queries a vector index over parsed repository code."""

    def __init__(self, persist_dir: str = "./.chroma", embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self._collection = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection("repobrain")
        return self._collection

    def build(self, parsed_code: dict, repo_path: str = ".") -> None:
        """Embed all parsed files and store in ChromaDB.

        Deletes and recreates the collection on each call for a fresh index.
        """
        import chromadb

        files_data: dict = parsed_code.get("files", {})
        if not files_data:
            return

        # Fresh index: delete existing collection if present
        client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            client.delete_collection("repobrain")
        except Exception:
            pass
        collection = client.create_collection("repobrain")
        self._collection = collection

        embedder = self._get_embedder()

        documents: list[str] = []
        ids: list[str] = []

        root = Path(repo_path).resolve()

        for rel_path, meta in files_data.items():
            functions = ", ".join(meta.get("functions", []))
            classes = ", ".join(meta.get("classes", []))
            imports = "; ".join(meta.get("imports", [])[:10])

            # Read actual source code for richer embeddings
            source = ""
            try:
                file_path = root / rel_path
                if file_path.exists():
                    raw = file_path.read_text(encoding="utf-8", errors="ignore")
                    # Truncate to keep embedding meaningful (first 2000 chars)
                    source = raw[:2000]
            except Exception:
                pass

            text = (
                f"File: {rel_path}\n"
                f"Language: {meta.get('language', 'unknown')}\n"
                f"Classes: {classes or 'none'}\n"
                f"Functions: {functions or 'none'}\n"
                f"Imports: {imports or 'none'}\n"
                f"Source:\n{source}" if source else
                f"File: {rel_path}\n"
                f"Language: {meta.get('language', 'unknown')}\n"
                f"Classes: {classes or 'none'}\n"
                f"Functions: {functions or 'none'}\n"
                f"Imports: {imports or 'none'}"
            )
            documents.append(text)
            # ChromaDB IDs must be strings; use path as ID (sanitize)
            ids.append(rel_path.replace("/", "__").replace("\\", "__"))

        if not documents:
            return

        # Embed in batches of 64
        batch_size = 64
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]
            embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
            collection.add(documents=batch_docs, embeddings=embeddings, ids=batch_ids)

    def query(self, question: str, top_k: int = 5) -> list[str]:
        """Return top_k most relevant document chunks for the question."""
        try:
            collection = self._get_collection()
            embedder = self._get_embedder()
            q_embedding = embedder.encode([question], show_progress_bar=False).tolist()
            results = collection.query(
                query_embeddings=q_embedding,
                n_results=min(top_k, collection.count()),
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception:
            return []

    def is_built(self) -> bool:
        """Return True if the index has at least one document."""
        try:
            collection = self._get_collection()
            return collection.count() > 0
        except Exception:
            return False

import os
import chromadb
from chromadb.utils import embedding_functions
from app.config import settings

COLLECTION_NAME = "assistant_docs"


class VectorStore:
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _ensure_initialized(self):
        if self._collection is not None:
            return

        # Enable offline hub usage if cache exists
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if os.path.exists(cache_dir):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        self._ensure_initialized()
        return self._collection

    def upsert_chunks(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, top_k: int = 4) -> list[dict]:
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        if not results or not results.get("ids") or len(results["ids"]) == 0:
            return []
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def delete_by_source(self, source_filename: str):
        self.collection.delete(where={"source": source_filename})

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def list_documents(self) -> list[dict]:
        """Returns distinct indexed document filenames and their chunk counts."""
        try:
            data = self.collection.get(include=["metadatas"])
            docs_count = {}
            for meta in data.get("metadatas", []):
                source = meta.get("source", "unknown")
                docs_count[source] = docs_count.get(source, 0) + 1
            return [{"filename": k, "chunks": v} for k, v in docs_count.items()]
        except Exception:
            return []


vector_store = VectorStore()


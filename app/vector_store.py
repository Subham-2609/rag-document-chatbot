"""
vector_store.py
----------------
Wraps ChromaDB + a local sentence-transformers embedding model.
Both are free and run without any API key.
"""

from pathlib import Path
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from app.document_processor import Chunk

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# all-MiniLM-L6-v2: small, fast, free, runs on CPU, good enough quality for a
# resume-project RAG system. Downloads once (~80MB) then runs locally.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        ids = [f"{c.source}-{c.chunk_id}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"source": c.source, "chunk_id": c.chunk_id, "page": c.page or -1}
            for c in chunks
        ]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4):
        results = self.collection.query(query_texts=[query], n_results=k)
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return hits

    def document_count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> List[str]:
        data = self.collection.get()
        sources = {m["source"] for m in data.get("metadatas", [])}
        return sorted(sources)

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="documents", embedding_function=self.embedding_fn
        )

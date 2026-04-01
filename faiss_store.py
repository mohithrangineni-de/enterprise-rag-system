"""
faiss_store.py
FAISS vector index builder and similarity search for enterprise RAG.
"""

import os
import pickle
from typing import List, Tuple
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float


class FAISSVectorStore:
    """
    Builds and queries a FAISS index for fast similarity search.
    Supports index persistence to AWS S3 or local disk.
    """

    def __init__(self, dimension: int = 1536):
        # 1536 = OpenAI text-embedding-ada-002 output dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def build_index(self, vectors: List[np.ndarray], documents: list):
        """Build FAISS index from embedding vectors."""
        matrix = np.array(vectors).astype("float32")
        self.index.add(matrix)
        self.documents = documents
        print(f"Index built with {self.index.ntotal} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search the index for top-k most similar documents."""
        query = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                content=doc.content,
                metadata=doc.metadata,
                score=float(1 / (1 + dist))  # normalize to 0-1
            ))

        return results

    def save(self, path: str):
        """Persist index and documents to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Index saved to {path}")

    def load(self, path: str):
        """Load index and documents from disk."""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        print(f"Loaded index with {self.index.ntotal} vectors from {path}")

    @property
    def total_documents(self) -> int:
        return self.index.ntotal

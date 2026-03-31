"""
vectorstore.py — Document Q&A System
=======================================
Persistent vector store for chunk embeddings using ChromaDB.

ChromaDB selection rationale:
- Runs fully locally — no API key, no cloud, no cost
- Persistent: indexes survive process restarts
- Simple Python API: no separate server process needed
- Cosine similarity built-in
- Production alternatives: Pinecone (cloud), Weaviate (self-hosted), Qdrant (self-hosted)
  These offer more features but break zero-config reproducibility.

Why a dedicated vector database over numpy search:
- numpy dot product over 10,000 embeddings: fast, but no persistence
- ChromaDB: persistent + approximate nearest neighbor (HNSW index) that
  scales to millions of embeddings with sub-100ms query time
- HNSW (Hierarchical Navigable Small World): graph-based ANN algorithm.
  O(log n) query time vs O(n) for brute-force numpy search.
"""

import os
import yaml
import logging
import numpy as np
from dataclasses import dataclass
from ingestion import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Structured result from a vector similarity query.

    distance: cosine distance from query (0 = identical, 1 = opposite)
    confidence: human-readable interpretation of distance
    """
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    distance: float
    confidence: str
    metadata: dict


class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.

    Indexed once during ingestion. Queried at every /ask request.
    Persistent: survives restarts, can accumulate multiple documents.
    """

    def __init__(self, config: dict):
        self.config = config
        self.persist_dir = config["vectorstore"]["persist_directory"]
        self.collection_name = config["vectorstore"]["collection_name"]
        self.n_results = config["vectorstore"]["n_results"]
        self._collection = None
        os.makedirs(self.persist_dir, exist_ok=True)

    def _get_collection(self):
        """Lazy initialization of ChromaDB collection."""
        if self._collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_dir)
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        # cosine: correct for normalized sentence-transformer embeddings
                        # l2: wrong here — magnitude variation would distort distances
                        "hnsw:space": "cosine"
                    }
                )
                logger.info(
                    f"ChromaDB collection '{self.collection_name}' loaded. "
                    f"Existing documents: {self._collection.count()}"
                )
            except ImportError:
                raise ImportError(
                    "chromadb required. Install: pip install chromadb"
                )
        return self._collection

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Index chunks and their embeddings into ChromaDB.

        Deduplication: chunks with existing IDs are skipped (not duplicated).
        This means re-ingesting the same document is safe — no duplicate retrieval.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        collection = self._get_collection()

        # Check which chunk IDs already exist
        existing_ids = set()
        if collection.count() > 0:
            all_ids = collection.get(include=[])["ids"]
            existing_ids = set(all_ids)

        # Filter to new chunks only
        new_chunks = []
        new_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_embeddings.append(embedding)

        if not new_chunks:
            logger.info("All chunks already indexed. Skipping.")
            return

        collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=[e.tolist() for e in new_embeddings],
            documents=[c.text for c in new_chunks],
            metadatas=[c.metadata for c in new_chunks],
        )
        logger.info(
            f"Indexed {len(new_chunks)} new chunks. "
            f"Total in store: {collection.count()}"
        )

    def query(self, query_embedding: np.ndarray, n_results: int = None) -> list[RetrievalResult]:
        """
        Find the n most semantically similar chunks to a query embedding.

        Returns RetrievalResult objects with confidence labels derived from
        cosine distance thresholds defined in config.

        Distance interpretation:
        - 0.00–0.25: high confidence (query and chunk are very similar)
        - 0.25–0.55: medium confidence (related but not exact)
        - 0.55–1.00: low confidence (weak semantic match — answer may be unreliable)
        """
        collection = self._get_collection()
        if collection.count() == 0:
            logger.warning("Vector store is empty. Ingest documents first.")
            return []

        n = n_results or self.n_results
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n, collection.count()),
            include=["documents", "distances", "metadatas"],
        )

        retrieval_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            confidence = self._distance_to_confidence(distance)

            retrieval_results.append(RetrievalResult(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                source=results["metadatas"][0][i].get("source", "unknown"),
                chunk_index=results["metadatas"][0][i].get("chunk_index", -1),
                distance=round(distance, 4),
                confidence=confidence,
                metadata=results["metadatas"][0][i],
            ))

        logger.info(
            f"Retrieved {len(retrieval_results)} chunks. "
            f"Best distance: {retrieval_results[0].distance if retrieval_results else 'N/A'}"
        )
        return retrieval_results

    def _distance_to_confidence(self, distance: float) -> str:
        thresholds = self.config["retrieval"]
        if distance < thresholds["high_confidence_distance"]:
            return "high"
        elif distance < thresholds["low_confidence_distance"]:
            return "medium"
        return "low"

    def count(self) -> int:
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """Reset: delete all indexed data. Useful for re-indexing a document set."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Collection '{self.collection_name}' deleted.")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")


def load_vectorstore(config_path: str = "configs/config.yaml") -> VectorStore:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return VectorStore(config)

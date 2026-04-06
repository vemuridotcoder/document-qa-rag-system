"""
embeddings.py — Document Q&A System
======================================
Converts text chunks into dense vector representations.

What is an embedding and why it enables semantic search:

A word like "budget" in a traditional search index is just a string.
"Education expenditure" and "school funding allocation" are completely
different strings — keyword search finds no match.

An embedding model converts text into a list of 384 numbers (a vector).
The model is trained so that semantically similar texts produce vectors
that are close together in 384-dimensional space.

"Education expenditure" and "school funding allocation" produce vectors
with cosine similarity ≈ 0.87 — close enough to retrieve correctly.
"Education expenditure" and "defense procurement" produce cosine ≈ 0.12.

This is what makes semantic search possible. The embedding model has
learned what words *mean*, not just what characters they contain.
"""

import logging
import numpy as np
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    texts: list[str]
    embeddings: np.ndarray  # shape: (n_texts, embedding_dim)
    model_name: str
    dimension: int


class EmbeddingModel:
    """
    Wraps sentence-transformers for chunk and query embedding.

    Why sentence-transformers over:
    - TF-IDF: TF-IDF is a sparse representation based on word frequency.
      It cannot match "school funding" to "education expenditure" because
      there is no word overlap. Dense embeddings capture semantic meaning.
    - Word2Vec/GloVe: these produce word-level embeddings. A sentence embedding
      requires averaging word vectors — this loses sentence-level structure.
      Sentence-BERT is trained specifically to produce good sentence-level embeddings.
    - OpenAI embeddings: require a paid API key, breaking reproducibility.
      all-MiniLM-L6-v2 is free, runs locally, and produces comparable quality
      for factual retrieval tasks.

    all-MiniLM-L6-v2 specifics:
    - 384 dimensions (compact — ChromaDB stores efficiently)
    - Trained on 1 billion sentence pairs via contrastive learning
    - ~22MB model size — downloads once, runs locally forever
    - Normalized to unit length → cosine similarity = dot product (faster)
    """

    def __init__(self, config: dict):
        self.model_name = config["embedding"]["model_name"]
        self.dimension = config["embedding"]["dimension"]
        self.batch_size = config["embedding"]["batch_size"]
        self._model = None

    def _load_model(self):
        """Lazy load — model only downloaded/loaded when first needed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded. Dimension: {self.dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install: pip install sentence-transformers"
                )

    def embed(self, texts: list[str], show_progress: bool = False) -> EmbeddingResult:
        """
        Convert a list of texts to normalized embeddings.

        normalize_embeddings=True:
        Forces all vectors to unit length (L2 norm = 1).
        This makes cosine similarity equivalent to dot product.
        ChromaDB's cosine distance = 1 - dot_product for normalized vectors.
        This is faster than computing the full cosine formula.

        batch_size:
        Process texts in batches to avoid OOM on large document sets.
        32 chunks at once uses ~200MB RAM for this model.
        """
        self._load_model()

        if not texts:
            return EmbeddingResult(
                texts=[],
                embeddings=np.array([]),
                model_name=self.model_name,
                dimension=self.dimension,
            )

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Unit length → cosine = dot product
            convert_to_numpy=True,
        )

        logger.info(
            f"Embedded {len(texts)} texts. "
            f"Shape: {embeddings.shape}. "
            f"Norm check: {np.linalg.norm(embeddings[0]):.4f} (should be ~1.0)"
        )

        return EmbeddingResult(
            texts=texts,
            embeddings=embeddings,
            model_name=self.model_name,
            dimension=self.dimension,
        )

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single query string. Returns 1D array of shape (384,).
        Used for query embedding at inference time — same model as indexing.

        Critical: query and documents MUST use the same embedding model.
        Mixing models produces meaningless similarity scores.
        """
        result = self.embed([text])
        return result.embeddings[0]


def load_embedding_model(config_path: str = "configs/config.yaml") -> EmbeddingModel:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return EmbeddingModel(config)

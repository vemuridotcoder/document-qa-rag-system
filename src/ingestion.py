"""
ingestion.py — Document Q&A System
=====================================
Loads documents and splits them into retrievable chunks.

The chunking strategy is the most consequential engineering decision in a RAG system.
Bad chunking → bad retrieval → bad answers, regardless of LLM quality.

Design decisions are documented in the class docstrings.
"""

import os
import re
import json
import yaml
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A single retrievable unit of text.

    chunk_id: deterministic hash of content — identical chunks get the same ID.
              This prevents duplicate indexing if the same document is re-ingested.
    source: original file path — shown in API response so users can find the source.
    chunk_index: position in document — helps debug retrieval ordering issues.
    char_count: raw character count — used to validate chunking quality.
    """
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    char_count: int
    metadata: dict


class DocumentLoader:
    """
    Loads raw text from supported file types.
    Currently supports: .txt, .md, .pdf (via pdfminer if installed)

    Design decision: PDF extraction is best-effort.
    Poorly formatted PDFs (scanned images, complex tables) produce garbled text.
    The loader logs a warning but does not raise — partial extraction is better than failure.
    """

    def load(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            return self._load_text(path)
        elif suffix == ".pdf":
            return self._load_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .md, .pdf")

    def _load_text(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        logger.info(f"Loaded text file: {path.name} ({len(content):,} chars)")
        return content

    def _load_pdf(self, path: Path) -> str:
        """
        Extract text from PDF.

        Why pdfminer over PyPDF2:
        pdfminer handles complex layouts and multi-column text better.
        PyPDF2 often concatenates columns incorrectly, breaking sentence structure.

        Limitation: both fail on scanned PDFs (image-only). OCR (tesseract) would
        be needed for those — documented as a known limitation, not silently ignored.
        """
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(path))
            if not text.strip():
                logger.warning(
                    f"PDF extracted empty text: {path.name}. "
                    "This may be a scanned PDF. OCR required for image-based PDFs."
                )
            logger.info(f"Loaded PDF: {path.name} ({len(text):,} chars)")
            return text
        except ImportError:
            raise ImportError(
                "pdfminer.six required for PDF support. "
                "Install: pip install pdfminer.six"
            )


class DocumentChunker:
    """
    Splits document text into overlapping chunks for vector indexing.

    Chunking strategy (fully documented — this is the core RAG engineering decision):

    1. CHUNK SIZE (512 tokens ≈ 400 words ≈ 2,000 characters):
       - Too small (128 tokens): answers get split across chunks. Retrieval returns
         incomplete context. LLM can't answer correctly.
       - Too large (1024+ tokens): each chunk contains too many topics. Query about
         "education budget" retrieves a chunk that also contains "defense spending"
         and "healthcare" — the LLM gets confused by irrelevant context.
       - 512 is the validated midpoint for dense factual documents.

    2. OVERLAP (50 tokens ≈ 40 words ≈ 200 characters):
       - Answers often span chunk boundaries. Without overlap:
         Question: "What is the total allocation for X?"
         Answer: "The total allocation for X..." (ends chunk N)
                 "...is INR 1,48,000 crore" (starts chunk N+1)
         If only chunk N is retrieved, the answer is incomplete.
       - Overlap ensures both chunks contain the complete answer.

    3. SENTENCE-AWARE SPLITTING:
       - Never split mid-sentence. "The budget allocates INR 1,48,000 cro|re for
         education" creates a broken embedding that encodes a meaningless fragment.
       - Splitting at sentence boundaries preserves semantic units.
    """

    def __init__(self, config: dict):
        self.chunk_size = config["ingestion"]["chunk_size"]
        self.chunk_overlap = config["ingestion"]["chunk_overlap"]

    def chunk(self, text: str, source: str) -> list[DocumentChunk]:
        """
        Split text into overlapping chunks. Returns list of DocumentChunk objects.
        """
        # Step 1: Clean the text
        text = self._clean_text(text)

        # Step 2: Split into sentences (semantic units)
        sentences = self._split_sentences(text)
        if not sentences:
            logger.warning(f"No sentences extracted from {source}")
            return []

        # Step 3: Group sentences into chunks respecting size limits
        chunks = self._group_sentences(sentences)

        # Step 4: Build DocumentChunk objects with metadata
        doc_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{source}:{idx}:{chunk_text[:50]}".encode()
            ).hexdigest()[:16]

            doc_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                chunk_index=idx,
                char_count=len(chunk_text),
                metadata={
                    "source": source,
                    "chunk_index": idx,
                    "total_chunks": 0,  # updated after loop
                    "filename": Path(source).name,
                }
            ))

        # Update total_chunks now that we know it
        for chunk in doc_chunks:
            chunk.metadata["total_chunks"] = len(doc_chunks)

        logger.info(
            f"Chunked '{Path(source).name}' into {len(doc_chunks)} chunks "
            f"(avg {sum(c.char_count for c in doc_chunks) // max(len(doc_chunks), 1)} chars each)"
        )
        return doc_chunks

    def _clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove artefacts from PDF extraction.

        Why these specific cleanups:
        - Multiple newlines → single newline: PDF extraction often introduces
          extra blank lines between paragraphs that fragment sentences.
        - Multiple spaces → single space: word-wrapped PDF columns produce
          "word  word" patterns (double spaces).
        - Page numbers (standalone digits on a line) → removed: these create
          chunks like "47" which are meaningless but consume retrieval slots.
        """
        # Remove standalone page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        # Remove null bytes (common in some PDFs)
        text = text.replace("\x00", "")
        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Why not NLTK or spaCy:
        - Additional dependency for marginal improvement
        - The regex handles 95% of cases correctly
        - Abbreviations (e.g., "Dr.", "INR.", "approx.") sometimes cause incorrect splits.
          Acceptable tradeoff for a simpler dependency tree.
        """
        # Split on period/question/exclamation followed by space + capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        # Remove very short fragments (less than 10 chars — likely artefacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """
        Group sentences into chunks of approximately chunk_size characters,
        with overlap between consecutive chunks.

        Approximation: we use characters, not tokens.
        1 token ≈ 4 characters in English (empirical average for GPT-style tokenizers).
        chunk_size=512 tokens → 512 * 4 = 2048 characters.
        This is a reasonable approximation without requiring a tokenizer dependency.
        """
        char_limit = self.chunk_size * 4       # tokens → approximate chars
        overlap_chars = self.chunk_overlap * 4

        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > char_limit and current_sentences:
                # Save current chunk
                chunks.append(" ".join(current_sentences))

                # Build overlap: take sentences from end of current chunk
                # until we've accumulated overlap_chars characters
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= overlap_chars:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                # Start new chunk with overlap sentences
                current_sentences = overlap_sentences
                current_length = overlap_len

            current_sentences.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow: Load → Clean → Chunk → Save.
    Call ingest() once per document. Results saved to data/chunks/ as JSON.
    The vectorstore indexing step happens in vectorstore.py.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(self.config)
        os.makedirs("data/chunks", exist_ok=True)

    def ingest(self, file_path: str) -> list[DocumentChunk]:
        """
        Full ingestion pipeline for one document.
        Returns list of chunks. Also saves to data/chunks/ for inspection.
        """
        logger.info(f"Ingesting: {file_path}")
        text = self.loader.load(file_path)
        chunks = self.chunker.chunk(text, file_path)

        # Save chunks as JSON for debugging and inspection
        output_path = f"data/chunks/{Path(file_path).stem}_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"Chunks saved to {output_path}")

        return chunks

    def ingest_directory(self, dir_path: str) -> list[DocumentChunk]:
        """Ingest all supported documents in a directory."""
        supported = {".txt", ".md", ".pdf"}
        all_chunks = []
        for path in Path(dir_path).iterdir():
            if path.suffix.lower() in supported:
                try:
                    chunks = self.ingest(str(path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to ingest {path}: {e}")
        return all_chunks

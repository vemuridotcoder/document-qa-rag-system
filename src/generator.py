"""
generator.py — Document Q&A System
======================================
LLM-based answer generation from retrieved context.

The generator's job is NOT to know the answer. It is to extract
the answer from the retrieved chunks and express it clearly.

Hallucination prevention is the central engineering problem here.
LLMs have parametric memory — they "know" things from training.
Without explicit constraints, the model will use that memory
when context is insufficient, producing confident wrong answers.

Three mechanisms are used:
1. System prompt: explicitly forbids using knowledge beyond context
2. Low temperature (0.1): reduces creative generation, keeps model grounded
3. Confidence routing: low-confidence retrieval → "cannot find" response
   before the LLM is even called
"""

import os
import logging
import yaml
from dataclasses import dataclass
from vectorstore import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Structured output from the LLM generator.

    answer: the generated response (or "cannot find" message)
    context_used: the actual text chunks passed to the LLM
    sources: unique source documents used
    retrieval_confidence: best confidence level from retrieved chunks
    generation_skipped: True if retrieval was too poor to attempt generation
    tokens_used: for monitoring API usage (Groq free tier has limits)
    """

    answer: str
    context_used: list[str]
    sources: list[str]
    retrieval_confidence: str
    generation_skipped: bool
    tokens_used: int = 0


class AnswerGenerator:
    """
    Generates answers from retrieved context using a language model.

    Supports two backends (controlled by environment variable):
    - Groq (default): free API, fast, requires GROQ_API_KEY
    - Ollama (fallback): fully local, no API key, slower

    Why two backends:
    - Groq: best for demos and development — fast, free tier sufficient
    - Ollama: best for offline use and privacy — no data leaves the machine
    Both use the same prompt format.
    """

    def __init__(self, config: dict):
        self.config = config["llm"]
        self.system_prompt = config["llm"]["system_prompt"]
        self.use_ollama = os.environ.get("USE_OLLAMA", "false").lower() == "true"

    def generate(
        self,
        question: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> GenerationResult:
        """
        Generate an answer from retrieved context.

        Step 1: Check retrieval confidence.
        If the best match has low confidence (distance > 0.55), the retrieved
        chunks are unlikely to contain the answer. Skip generation and return
        a "cannot find" response. This prevents the LLM from hallucinating
        an answer using its parametric memory.

        Step 2: Build context string.
        Concatenate chunk texts with source labels. The LLM sees exactly what
        was retrieved — no hidden context.

        Step 3: Generate with explicit constraints.
        The system prompt forbids using external knowledge. Low temperature
        reduces hallucination probability. The model is instructed to say
        "I cannot find this" if the answer is not in context.
        """
        if not retrieved_chunks:
            return GenerationResult(
                answer="No documents have been indexed. Please ingest a document first.",
                context_used=[],
                sources=[],
                retrieval_confidence="none",
                generation_skipped=True,
            )

        # Step 1: Confidence check
        best_confidence = retrieved_chunks[0].confidence
        if best_confidence == "low":
            return GenerationResult(
                answer=(
                    "I cannot find a reliable answer to this question in the indexed documents. "
                    "The closest matching sections have low semantic similarity to your question. "
                    "Try rephrasing or check if the relevant document has been ingested."
                ),
                context_used=[c.text for c in retrieved_chunks],
                sources=list({c.source for c in retrieved_chunks}),
                retrieval_confidence="low",
                generation_skipped=True,
            )

        # Step 2: Build context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source_label = (
                f"[Source {i+1}: {chunk.metadata.get('filename', 'document')}]"
            )
            context_parts.append(f"{source_label}\n{chunk.text}")
        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Generate
        try:
            if self.use_ollama:
                result = self._generate_ollama(question, context)
            else:
                result = self._generate_groq(question, context)

            return GenerationResult(
                answer=result["answer"],
                context_used=[c.text for c in retrieved_chunks],
                sources=list({c.source for c in retrieved_chunks}),
                retrieval_confidence=best_confidence,
                generation_skipped=False,
                tokens_used=result.get("tokens_used", 0),
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                answer=f"Generation failed: {str(e)}. Check API key or Ollama status.",
                context_used=[],
                sources=[],
                retrieval_confidence=best_confidence,
                generation_skipped=True,
            )

    def _build_user_prompt(self, question: str, context: str) -> str:
        """
        The prompt design is the second most important decision after chunking.

        Why "ONLY the provided context":
        Without this explicit constraint, LLMs blend parametric memory with
        retrieved context. The answer becomes a mixture of actual document content
        and LLM training data — unreliable and unverifiable.

        Why "cite which part":
        Forces the model to ground its answer in specific retrieved text.
        Makes it harder to hallucinate — the model must point to where it found the answer.
        Also helps users verify answers by checking the source sections.
        """
        return f"""Context from the indexed documents:

{context}

Question: {question}

Answer using ONLY the above context. If the answer is not present, say exactly:
"I cannot find this in the provided documents."
Cite which source section your answer comes from."""

    def _generate_groq(self, question: str, context: str) -> dict:
        """
        Generate using Groq API (free tier).
        Requires GROQ_API_KEY environment variable.
        Register at: https://console.groq.com
        """
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq required. Install: pip install groq")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Get a free key at https://console.groq.com, "
                "then: export GROQ_API_KEY=your_key_here\n"
                "Or set USE_OLLAMA=true for fully local generation."
            )

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_prompt(question, context)},
            ],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "tokens_used": response.usage.total_tokens,
        }

    def _generate_ollama(self, question: str, context: str) -> dict:
        """
        Generate using Ollama (fully local, no API key).
        Install Ollama: https://ollama.ai
        Pull model: ollama pull llama3.2
        Start: ollama serve
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests required. Install: pip install requests")

        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_prompt(question, context)},
            ],
            "options": {
                "temperature": self.config["temperature"],
                "num_predict": self.config["max_tokens"],
            },
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        return {
            "answer": data["message"]["content"].strip(),
            "tokens_used": 0,  # Ollama doesn't always report token counts
        }


def load_generator(config_path: str = "configs/config.yaml") -> AnswerGenerator:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return AnswerGenerator(config)

"""Grounded answer generation with fallback strategies."""

from __future__ import annotations

import os
import logging
import re
import yaml
from dataclasses import dataclass

from vectorstore import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    context_used: list[str]
    sources: list[str]
    retrieval_confidence: str
    generation_skipped: bool
    tokens_used: int = 0
    used_fallback: bool = False


class AnswerGenerator:
    def __init__(self, config: dict):
        self.config = config["llm"]
        self.system_prompt = config["llm"]["system_prompt"]
        self.use_ollama = os.environ.get("USE_OLLAMA", "false").lower() == "true"

    def generate(
        self, question: str, retrieved_chunks: list[RetrievalResult]
    ) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(
                answer="No documents have been indexed. Please ingest a document first.",
                context_used=[],
                sources=[],
                retrieval_confidence="none",
                generation_skipped=True,
                used_fallback=True,
            )

        best_confidence = retrieved_chunks[0].confidence
        if best_confidence == "low":
            return GenerationResult(
                answer=self._extractive_fallback(question, retrieved_chunks),
                context_used=[c.text for c in retrieved_chunks],
                sources=list({c.source for c in retrieved_chunks}),
                retrieval_confidence="low",
                generation_skipped=True,
                used_fallback=True,
            )

        context = "\n\n---\n\n".join(
            [
                f"[Source {i + 1}: {chunk.metadata.get('filename', 'document')}]\n{chunk.text}"
                for i, chunk in enumerate(retrieved_chunks)
            ]
        )

        try:
            result = (
                self._generate_ollama(question, context)
                if self.use_ollama
                else self._generate_groq(question, context)
            )
            return GenerationResult(
                answer=result["answer"],
                context_used=[c.text for c in retrieved_chunks],
                sources=list({c.source for c in retrieved_chunks}),
                retrieval_confidence=best_confidence,
                generation_skipped=False,
                tokens_used=result.get("tokens_used", 0),
            )
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return GenerationResult(
                answer=self._extractive_fallback(question, retrieved_chunks),
                context_used=[c.text for c in retrieved_chunks],
                sources=list({c.source for c in retrieved_chunks}),
                retrieval_confidence=best_confidence,
                generation_skipped=True,
                used_fallback=True,
            )

    def _extractive_fallback(
        self, question: str, retrieved_chunks: list[RetrievalResult]
    ) -> str:
        """Deterministic fallback: return best-matching sentence from retrieved context."""
        q_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        best_sentence = ""
        best_score = -1

        for chunk in retrieved_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sentence in sentences:
                s_terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                score = len(q_terms & s_terms)
                if score > best_score and len(sentence.strip()) > 20:
                    best_score = score
                    best_sentence = sentence.strip()

        if best_sentence:
            return (
                "I could not safely generate with the LLM, so here is the best matched evidence from your documents: "
                + best_sentence
            )
        return "I cannot find a reliable answer in the indexed documents."

    def _build_user_prompt(self, question: str, context: str) -> str:
        return f"""Context from the indexed documents:

{context}

Question: {question}

Answer using ONLY the above context. If the answer is not present, say exactly:
\"I cannot find this in the provided documents.\"
Cite which source section your answer comes from."""

    def _generate_groq(self, question: str, context: str) -> dict:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")

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
        import requests

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

        response = requests.post(
            "http://localhost:11434/api/chat", json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return {"answer": data["message"]["content"].strip(), "tokens_used": 0}


def load_generator(config_path: str = "configs/config.yaml") -> AnswerGenerator:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return AnswerGenerator(config)

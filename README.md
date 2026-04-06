# Intelligent Document Q&A System

**Production RAG pipeline** — indexes any PDF/TXT/MD document using dense vector embeddings, retrieves semantically relevant content, and generates grounded answers using Llama 3.1 with full hallucination prevention, response caching, rate limiting, and query analytics.

[![CI](https://github.com/vemuridotcoder/doc-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/vemuridotcoder/doc-qa-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

---
## 🚀 Live Demo
https://document-qa-rag-system-etzp.onrender.com/docs
## What it does

| Feature | Detail |
|---|---|
| **Semantic retrieval** | Sentence-BERT (all-MiniLM-L6-v2) + ChromaDB cosine similarity |
| **Hallucination prevention** | 3-layer: system prompt constraint + temperature=0.1 + confidence routing |
| **Retrieval evaluation** | Hit Rate @3 and MRR measured independently from generation quality |
| **Chunk optimisation** | Systematic experiment across 128 / 256 / 512 / 1024 token sizes |
| **Response caching** | SQLite cache — TTL=24h, auto-invalidated on re-index |
| **Rate limiting** | 30 req/min per IP via slowapi |
| **Query analytics** | SQLite log — confidence distribution, response times, SQL window functions |
| **Deployment** | FastAPI + Docker — /ingest, /ask, /ask/batch, /store, /analytics, /health |

---

## Tech stack

`Python 3.11` · `Sentence-Transformers` · `ChromaDB` · `Groq (Llama 3.1)` · `FastAPI` · `Pydantic` · `SQLite` · `slowapi` · `Docker` · `GitHub Actions` · `pytest`

---

## Quick start

```bash
git clone https://github.com/vemuridotcoder/doc-qa-system.git
cd doc-qa-system
pip install -r requirements.txt

# Get free Groq API key → https://console.groq.com
export GROQ_API_KEY=your_key_here
# OR run fully local: export USE_OLLAMA=true  (requires ollama.ai + ollama pull llama3.2)

uvicorn api.main:app --port 8001
```

```bash
# Docker
docker build -t doc-qa .
docker run -e GROQ_API_KEY=your_key -p 8001:8001 doc-qa
```

```bash
# Ingest a document
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/raw/your_document.pdf", "reset": true}'

# Ask a question
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total education budget?"}'

# View analytics
curl http://localhost:8001/analytics
```

**Response:**
```json
{
  "question": "What is the total education budget?",
  "answer": "According to Source 1, the total education budget is INR 1,48,000 crore.",
  "retrieval_confidence": "high",
  "sources": [
    {
      "text_preview": "The Union Budget allocates INR 1,48,000 crore...",
      "source_file": "budget_2024.pdf",
      "relevance_distance": 0.18,
      "confidence": "high"
    }
  ],
  "generation_skipped": false,
  "warning": null
}
```

---

## Project structure

```
doc-qa-system/
├── src/
│   ├── ingestion.py         Document loader + sentence-aware chunker
│   ├── embeddings.py        Sentence-BERT wrapper (all-MiniLM-L6-v2)
│   ├── vectorstore.py       ChromaDB cosine similarity interface
│   ├── generator.py         Groq + Ollama backends — hallucination constrained
│   ├── evaluate.py          Hit Rate @K and MRR retrieval evaluator
│   ├── chunk_experiment.py  Chunk size experiment (128/256/512/1024 tokens)
│   ├── cache.py             SQLite response cache with TTL + invalidation
│   └── query_logger.py      SQLite query log with SQL analytics
├── api/
│   ├── main.py              FastAPI application (7 endpoints)
│   └── schemas.py           Pydantic validation
├── configs/
│   └── config.yaml          All decisions documented inline
├── evaluation/
│   └── test_questions.json  Add domain-specific questions here
├── tests/
│   └── test_api.py          11 endpoint tests
├── .github/workflows/
│   └── ci.yml               Lint → config validation → cache tests → pytest
├── Dockerfile
└── requirements.txt
```

---

## Key decisions

### Chunk size: 512 tokens, 50-token overlap

| Chunk size | Problem |
|---|---|
| 128 tokens | Answers split across chunks — incomplete context |
| **512 tokens** | **Balanced — complete thoughts, precise retrieval** |
| 1024 tokens | Too many topics per chunk — LLM confused by noise |

**50-token overlap** prevents boundary misses — answers spanning two chunks appear in both.

Run `python src/chunk_experiment.py --doc your_file.txt` to validate for your document type.

### Sentence-BERT over TF-IDF

TF-IDF is sparse — "education expenditure" and "school funding" share zero keywords, similarity = 0.

`all-MiniLM-L6-v2` encodes semantic meaning:
- `cosine("education expenditure", "school funding allocation") ≈ 0.87` ✓
- `cosine("education expenditure", "defense procurement") ≈ 0.12` ✓

**Why cosine, not L2:** Model outputs unit-normalized vectors. Cosine = dot product for normalized vectors — faster and correct. L2 is distorted by magnitude.

### Three-layer hallucination prevention

| Layer | Mechanism |
|---|---|
| System prompt | Explicitly forbids LLM from using knowledge beyond retrieved context |
| Temperature = 0.1 | Suppresses creative generation — model stays grounded in retrieved text |
| Confidence routing | Distance > 0.55 → skip LLM entirely → deterministic "cannot find" response |

Without layer 3, the LLM fills retrieval gaps with parametric memory — confident wrong answers.

### Retrieval evaluated separately from generation

Most RAG tutorials measure end-to-end answer quality. This conflates two different problems:

- **Retrieval failure** → generation always fails
- **Generation failure** → retrieval may be correct

Separating them identifies which component to fix.

| Metric | Formula | What it measures |
|---|---|---|
| Hit Rate @3 | correct_in_top3 / total | Was the answer chunk retrieved? |
| MRR | avg(1 / rank) | How highly was it ranked? |

Run: `python src/evaluate.py`

---

## Chunk size experiment results

`python src/chunk_experiment.py --doc your_document.txt`

| Chunk size | Overlap | Hit @1 | Hit @3 | MRR | Notes |
|---|---|---|---|---|---|
| 128 | 13 | — | — | — | Run experiment |
| 256 | 26 | — | — | — | Run experiment |
| **512** | **50** | — | — | — | **Default — expected optimal** |
| 1024 | 102 | — | — | — | Run experiment |

Results saved to `evaluation/chunk_experiment_results.csv`.

---

## SQL query analytics

Every `/ask` request is logged to SQLite. `GET /analytics` returns:

```sql
-- Confidence distribution
SELECT retrieval_confidence, COUNT(*),
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM query_logs), 2) AS pct
FROM query_logs GROUP BY retrieval_confidence;

-- Hourly query volume with running total (window function)
SELECT SUBSTR(timestamp,1,13) AS hour,
       COUNT(*) AS queries,
       SUM(COUNT(*)) OVER (ORDER BY SUBSTR(timestamp,1,13)) AS cumulative
FROM query_logs GROUP BY hour ORDER BY hour;

-- Low-confidence questions (candidates for document update)
SELECT question, top_source_distance FROM query_logs
WHERE retrieval_confidence = 'low'
ORDER BY timestamp DESC LIMIT 20;
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status + cache stats |
| `GET` | `/store/status` | Vector store document count |
| `POST` | `/ingest` | Index a PDF / TXT / MD document |
| `POST` | `/ask` | Single question — cached + rate-limited |
| `POST` | `/ask/batch` | Up to 10 questions per request |
| `DELETE` | `/store` | Reset index + invalidate cache |
| `GET` | `/analytics` | SQL-derived query log report |

Rate limit: **30 requests/minute per IP** (matches Groq free tier).

---

## Known limitations

1. **Scanned PDFs** — pdfminer extracts digital text only. OCR not implemented.
2. **Chunk boundary answers** — very long answers spanning multiple overlap windows may be incomplete.
3. **Tables** — PDF tables extract as garbled text. Structured extraction (camelot) not implemented.
4. **Multi-hop questions** — single-query retrieval may miss answers requiring two separate chunks.
5. **English only** — degrades on Hindi or code-mixed text.
6. **No conversation memory** — each `/ask` is stateless.

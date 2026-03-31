# Intelligent Document Q&A System

> Semantic Q&A over any document — Sentence-BERT embeddings, ChromaDB vector store,
> Llama 3.1 via Groq, response caching, query logging, and rate limiting.

[![CI](https://github.com/YOUR_USERNAME/doc-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/doc-qa-system/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## What this does

Indexes any PDF/TXT/MD document using dense vector embeddings, retrieves semantically relevant chunks for a user question, and generates a grounded answer using Llama 3.1 — with three-layer hallucination prevention, SQLite response caching, per-IP rate limiting, and full query logging.

**Key differentiator:** retrieval quality is evaluated independently from generation quality using Hit Rate @K and MRR — most RAG tutorials skip this and cannot diagnose where failures occur.

---

## Stack

`Python 3.11` · `Sentence-Transformers` · `ChromaDB` · `Groq (Llama 3.1)` · `FastAPI` · `Pydantic` · `SQLite` · `slowapi` · `Docker` · `GitHub Actions`

---

## Project structure

```
doc-qa-system/
├── src/
│   ├── ingestion.py        Document loader + sentence-aware chunker
│   ├── embeddings.py       Sentence-BERT embedding model wrapper
│   ├── vectorstore.py      ChromaDB interface with cosine similarity
│   ├── generator.py        LLM generation — Groq + Ollama backends
│   ├── evaluate.py         Hit Rate @K and MRR retrieval evaluator
│   ├── chunk_experiment.py Systematic chunk size experiment (128/256/512/1024)
│   ├── cache.py            SQLite response cache with TTL and invalidation
│   └── query_logger.py     SQLite query log with SQL analytics queries
├── api/
│   ├── main.py             FastAPI: /health /ingest /ask /ask/batch /store /analytics
│   └── schemas.py          Pydantic validation
├── configs/
│   └── config.yaml         All decisions documented inline
├── evaluation/
│   └── test_questions.json Add domain-specific questions here
├── tests/
│   └── test_api.py         11 endpoint tests
├── .github/workflows/
│   └── ci.yml              Lint → config validation → cache tests → pytest
├── Dockerfile
└── requirements.txt        Pinned versions
```

---

## Key technical decisions

### 1 — Chunk size: 512 tokens with 50-token overlap

| Chunk size | Problem |
|---|---|
| 128 tokens | Answers split across chunks. Retrieval returns incomplete context. |
| **512 tokens** | **Balanced — complete thoughts, precise retrieval** |
| 1024 tokens | Too many topics per chunk. LLM confused by irrelevant context. |

Overlap prevents boundary misses: answers spanning two consecutive chunks appear in both.
Run `python src/chunk_experiment.py --doc your_file.txt` to validate this for your document type.

### 2 — Sentence-BERT over TF-IDF

TF-IDF is sparse — "education expenditure" and "school funding" share zero keywords, similarity = 0.
`all-MiniLM-L6-v2` produces 384-dimensional dense vectors trained on 1B+ sentence pairs.

`cosine("education expenditure", "school funding allocation") ≈ 0.87`
`cosine("education expenditure", "defense procurement") ≈ 0.12`

**Why cosine similarity, not L2:** `all-MiniLM-L6-v2` outputs unit-normalized vectors.
For normalized vectors, cosine similarity = dot product — faster and correct.
L2 distance is distorted by vector magnitude, inappropriate here.
ChromaDB: `"hnsw:space": "cosine"`.

### 3 — Three-layer hallucination prevention

| Layer | Mechanism |
|---|---|
| System prompt | Explicitly forbids using knowledge beyond retrieved context |
| Temperature = 0.1 | Reduces creative generation, keeps model grounded |
| Confidence routing | Distance > 0.55 → skip LLM entirely, return deterministic "cannot find" |

Without layer 3, the LLM fills retrieval gaps with parametric memory — confident wrong answers.

### 4 — Retrieval evaluated separately from generation

Root cause analysis requires separating the two components:
- Retrieval failure → generation always fails
- Generation failure → retrieval may be correct

**Metrics:**

| Metric | Formula | What it measures |
|---|---|---|
| Hit Rate @3 | correct_in_top3 / total | Was the answer chunk retrieved? |
| MRR | avg(1 / rank) | How highly was it ranked? |

Run: `python src/evaluate.py`

### 5 — Response caching (SQLite)

Identical questions return in < 1ms without hitting Groq. TTL = 24 hours.
Cache is automatically invalidated when `DELETE /store` is called — stale answers
from old documents are never served after re-indexing.

Hit counts logged per question — analytics show which queries are most repeated.

---

## Chunk size experiment results

Run `python src/chunk_experiment.py --doc your_document.txt` to generate:

| Chunk size | Chunks | Hit @1 | Hit @3 | MRR | Notes |
|---|---|---|---|---|---|
| 128 | — | — | — | — | Run experiment |
| 256 | — | — | — | — | Run experiment |
| **512** | — | — | — | — | **Expected optimal** |
| 1024 | — | — | — | — | Run experiment |

Results saved to `evaluation/chunk_experiment_results.csv`.

---

## SQL analytics (query logs)

`src/query_logger.py` logs every request to SQLite. `GET /analytics` returns:

```sql
-- Confidence distribution
SELECT retrieval_confidence, COUNT(*), ROUND(100.0 * COUNT(*) / total, 2) AS pct
FROM query_logs GROUP BY retrieval_confidence;

-- Hourly volume with running total (window function)
SELECT SUBSTR(timestamp,1,13) AS hour, COUNT(*) AS queries,
       SUM(COUNT(*)) OVER (ORDER BY SUBSTR(timestamp,1,13)) AS cumulative
FROM query_logs GROUP BY hour;

-- Low confidence questions (candidates for document update)
SELECT question, top_source_distance FROM query_logs
WHERE retrieval_confidence = 'low' ORDER BY timestamp DESC LIMIT 20;
```

---

## Running locally

**Setup:**
```bash
git clone <repo> && cd doc-qa-system
pip install -r requirements.txt
export GROQ_API_KEY=your_key   # Free at console.groq.com
# OR: export USE_OLLAMA=true   # Fully local, no API key needed
```

**Start API:**
```bash
uvicorn api.main:app --port 8001
```

**Ingest and ask:**
```bash
# Ingest a document
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/raw/budget_2024.pdf", "reset": true}'

# Ask a question
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total education budget?"}'

# View query analytics
curl http://localhost:8001/analytics
```

**Docker:**
```bash
docker build -t doc-qa .
docker run -e GROQ_API_KEY=your_key -p 8001:8001 doc-qa
```

**Tests and experiment:**
```bash
pytest tests/test_api.py -v
python src/chunk_experiment.py --doc data/raw/your_document.txt
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status + cache stats |
| GET | `/store/status` | Vector store document count |
| POST | `/ingest` | Index a PDF/TXT/MD document |
| POST | `/ask` | Single question (cached + rate-limited) |
| POST | `/ask/batch` | Up to 10 questions per request |
| DELETE | `/store` | Reset index + invalidate cache |
| GET | `/analytics` | SQL-derived query log analytics |

Rate limit: 30 requests/minute per IP.

---

## Where this system fails

1. **Scanned PDFs** — pdfminer extracts digital text only. OCR (pytesseract) needed for image-based documents.
2. **Chunk boundary answers** — answers spanning more than one overlap window may be incomplete.
3. **Tables and figures** — PDF tables extract as garbled text. Structured extraction (camelot) not implemented.
4. **Multi-hop questions** — "What is X, and how does it compare to Y?" requires retrieving chunks about both. Single-query retrieval may miss one.
5. **English only** — embedding model trained primarily on English. Hindi/code-mixed text degrades retrieval.
6. **No conversation memory** — each `/ask` call is independent. Follow-up questions have no context.

---

## Research questions this raises

1. Does optimal chunk size change when using a larger embedding model (768-dim vs 384-dim)? A factorial experiment would answer this.
2. In practice, what fraction of wrong answers are caused by retrieval failure vs generation failure? Separating these on a labelled dataset would guide where to invest improvement effort.
3. If the document is in English but the question is in Hindi, multilingual embeddings (LaBSE, mE5) could bridge the gap. How much does Hit Rate @3 degrade with language mismatch?

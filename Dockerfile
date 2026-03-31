FROM python:3.11-slim

WORKDIR /app

# System deps for pdfminer and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY api/ api/

# Pre-download embedding model at build time (not at first request)
# This avoids a 90-second cold start on first /ask call
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# GROQ_API_KEY must be passed at runtime:
# docker run -e GROQ_API_KEY=your_key -p 8001:8001 doc-qa
# Or for local LLM: docker run -e USE_OLLAMA=true -p 8001:8001 doc-qa
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]

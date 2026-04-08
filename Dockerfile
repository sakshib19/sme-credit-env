# SME Credit Risk RL Environment
# ================================
# OpenEnv-compatible FastAPI server, port 7860 (HF Spaces standard).
#
# Build:
#   docker build -t sme-credit-env:latest .
#
# Run locally:
#   docker run -d -p 7860:7860 \
#     -e API_BASE_URL=https://router.huggingface.co/v1 \
#     -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
#     -e HF_TOKEN=hf_your_token \
#     sme-credit-env:latest
#
# Test:
#   curl http://localhost:7860/health
#   curl -X POST http://localhost:7860/reset \
#     -H "Content-Type: application/json" -d '{"task_id": "easy_01"}'
 
FROM python:3.11-slim
 
# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    WORKERS=2 \
    PYTHONPATH=/app
 
WORKDIR /app
 
# System deps (curl for health check, no build tools needed)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
 
# Install Python dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy all project files
COPY models.py          ./models.py
COPY inference.py       ./inference.py
COPY openenv.yaml       ./openenv.yaml
COPY env/               ./env/
COPY server/            ./server/
COPY data/              ./data/
COPY README.md ./README.md
 
# Expose HF Spaces port
EXPOSE 7860
 
# Health check — used by HF Spaces and the submission validator
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1
 
# Start FastAPI server
# WORKERS=2 keeps memory usage low on free HF Spaces CPU tier.
# Raise to 4 for paid hardware (cpu-upgrade or above).
CMD uvicorn server.app:app \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info
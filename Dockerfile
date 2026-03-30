# Dockerfile — SME Credit Risk RL Environment
# ============================================
# Builds a self-contained image that runs the FastAPI server.
#
# Build:
#   docker build -t sme-credit-env:latest .
#
# Run locally:
#   docker run -d -p 7860:7860 sme-credit-env:latest
#
# Run with environment variables:
#   docker run -d -p 7860:7860 -e WORKERS=4 sme-credit-env:latest
#
# HF Spaces (free tier): set hardware to CPU Basic, uses port 7860 by default.

FROM python:3.11-slim

# Keeps Python output unbuffered so docker logs stream immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    WORKERS=2

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py          ./models.py
COPY tasks.json         data/tasks.json
COPY env/               ./env/
COPY server/            ./server/

# Expose the port used by HF Spaces and our uvicorn config
EXPOSE 7860

# Start the FastAPI server
# WORKERS=2 keeps memory usage low on the free HF Spaces CPU tier.
# Increase to 4 for production / paid hardware.
CMD uvicorn server.app:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level info
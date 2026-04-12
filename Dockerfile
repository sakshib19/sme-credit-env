# SME Credit Risk RL Environment
# ================================
# Multi-stage build using openenv-base (reference repo pattern).
#
# openenv-base already ships:
#   - Python 3.11, uv, uvicorn, curl
#   - openenv-core installed system-wide
#
# Why NOT python:3.11-slim:
#   - Needs pip install of everything (slow, fragile)
#   - PYTHONPATH setup differs from how openenv validate expects it
#   - Health check port mismatches (7860 vs openenv default 8000)
#
# Port 8000 is used because:
#   - openenv validate defaults to http://localhost:8000
#   - HF Spaces maps any internal port to the public URL
#   - The Playground UI at harsh063423/my_env runs on 8000
#
# Build locally:
#   docker build -t sme-credit-env:latest .
#
# Run locally:
#   docker run -d -p 8000:8000 \
#     -e HF_TOKEN=hf_your_token \
#     sme-credit-env:latest
#
# Test:
#   curl http://localhost:8000/health
#   curl -X POST http://localhost:8000/reset \
#     -H "Content-Type: application/json" -d '{"task_id": "easy_01"}'

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed if any pyproject.toml dep uses a VCS source
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project into /app/env
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available (openenv-base includes it, this is a safety fallback)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies into a virtual env at /app/env/.venv
# uv.lock guarantees reproducible builds — frozen sync first,
# then full sync to install the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Final runtime stage ────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Bring in the venv from builder (all Python packages pre-installed)
COPY --from=builder /app/env/.venv /app/.venv

# Bring in the project source
COPY --from=builder /app/env /app/env

# Activate the venv
ENV PATH="/app/.venv/bin:$PATH"

# PYTHONPATH must point to /app/env so that these imports resolve:
#   from models import LoanAction, ...        → /app/env/models.py
#   from server.loan_environment import ...   → /app/env/server/loan_environment.py
#   from tasks.environment import ...         → /app/env/tasks/environment.py
#   from tasks.graders import ...             → /app/env/tasks/graders.py
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check — openenv validator and HF Spaces ping this
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server on port 8000
# cd into /app/env so relative paths (data/tasks.json etc.) resolve correctly
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
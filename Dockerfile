# SME Credit Risk RL Environment
# ================================
# Multi-stage build using openenv-base to guarantee OpenEnv hackathon compliance.
#
# Port 7860 is used strictly for Hugging Face Spaces native compatibility.

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

# PYTHONPATH must point to /app/env so imports resolve correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose Hugging Face's default port
EXPOSE 7860

# Health check — openenv validator and HF Spaces ping this
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server on port 7860
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
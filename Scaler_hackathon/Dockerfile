# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:py311
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=energy-grid-openenv

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Install dependencies using uv sync
# If uv.lock exists, use it; otherwise resolve on the fly
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN python -c "import sys; print(sys.executable)"

# Build the frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY agent-grid-view/package*.json ./
RUN npm install
COPY agent-grid-view/ .
RUN npm run build

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /app/env/.venv /app/.venv

# Copy environment code
COPY --from=builder /app/env /app/env

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist /app/env/agent-grid-view/dist

# Use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check  uses PORT env var (HF Spaces sets PORT=7860, local default 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD ["sh", "-c", "curl -f http://localhost:${PORT:-7860}/health || exit 1"]

# Run the FastAPI server
# HF Spaces injects PORT=7860; fall back to 8000 for local Docker runs
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
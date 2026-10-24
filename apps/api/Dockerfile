# syntax=docker/dockerfile:1
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Layer 1: dependencies only (cached until pyproject/uv.lock change)
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY packages/config/pyproject.toml packages/config/pyproject.toml
COPY packages/observability/pyproject.toml packages/observability/pyproject.toml
COPY packages/contracts/pyproject.toml packages/contracts/pyproject.toml
COPY packages/identity/pyproject.toml packages/identity/pyproject.toml
COPY packages/authorization/pyproject.toml packages/authorization/pyproject.toml
COPY packages/ratelimit/pyproject.toml packages/ratelimit/pyproject.toml
COPY packages/persistence/pyproject.toml packages/persistence/pyproject.toml
COPY packages/ingestion/pyproject.toml packages/ingestion/pyproject.toml
COPY packages/llm-gateway/pyproject.toml packages/llm-gateway/pyproject.toml
COPY packages/search/pyproject.toml packages/search/pyproject.toml
COPY packages/messaging/pyproject.toml packages/messaging/pyproject.toml
COPY packages/workflows/pyproject.toml packages/workflows/pyproject.toml
COPY packages/retrieval/pyproject.toml packages/retrieval/pyproject.toml
RUN uv sync --frozen --no-dev --no-install-workspace --package ekip-api

# Layer 2: our source code
COPY apps/api apps/api
COPY packages packages
# --no-editable copies our packages INTO the venv; an editable install would point at
# /app/apps/..., which does not exist in the runtime stage (only the venv is copied).
RUN uv sync --frozen --no-dev --no-editable --package ekip-api

FROM python:3.14-slim AS runtime
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY apps/api/uvicorn-log.json /app/uvicorn-log.json
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
CMD ["uvicorn", "ekip_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--log-config", "/app/uvicorn-log.json"]

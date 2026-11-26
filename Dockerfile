# The backend: the agent, its WebSocket server and the help center documents.
# Build from the repo root, because the image needs both backend/ and kb/.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PATH="/app/backend/.venv/bin:$PATH"

# Dependencies first, so a change to the source does not reinstall them.
COPY backend/pyproject.toml backend/uv.lock backend/
RUN cd backend && uv sync --frozen --no-dev --no-install-project

COPY backend backend
COPY kb kb
RUN cd backend && uv sync --frozen --no-dev

# The small embedding model goes into the image, so a start does not wait for a download.
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warm up']))"

WORKDIR /app/backend
# Migrations and the help center load on every start. Both do nothing when nothing changed.
# uvicorn's own trust in X-Forwarded-For is off: CLIENT_ADDRESS_HEADER names the one header believed.
CMD ["sh", "-c", "python -m supportagent.cli.ingest && exec uvicorn --factory supportagent.api.app:create_app --host 0.0.0.0 --port ${PORT:-8787} --ws-max-size 65536 --no-proxy-headers"]

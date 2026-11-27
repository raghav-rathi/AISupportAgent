# The backend: the agent, its WebSocket server and the help center documents.
# Build from the repo root, because the image needs both backend/ and kb/.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PATH="/app/backend/.venv/bin:$PATH"

# Dependencies first, so a change to the source does not reinstall them.
COPY backend/pyproject.toml backend/uv.lock backend/
RUN cd backend && uv sync --frozen --no-dev --no-install-project

COPY backend backend
COPY kb kb
RUN cd backend && uv sync --frozen --no-dev

# The small embedding model goes into the image, so a start does not wait for a download. The
# reranker is hosted in production and is not baked in.
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warm up']))"

# The app runs as a user that owns nothing but the model cache it reads.
RUN useradd --system --uid 10001 --no-create-home app && chown -R app /app/.cache
USER app
# The user has no home directory. A library that writes a cache or a config file under $HOME
# finds one that exists and is writable.
ENV HOME=/tmp

WORKDIR /app/backend
# The app answers /health only once its start-up is over and the embedding model is loaded, so a
# host that waits for this sends no visitor to a process that is still warming up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8787') + '/health', timeout=4)"

# Migrations and the help center load on every start. Both do nothing when nothing changed, and a
# lock makes a second container that starts at the same time wait instead of doing the work twice.
# uvicorn's own trust in X-Forwarded-For is off: CLIENT_ADDRESS_HEADER names the one header believed.
# On a stop signal, running turns get 15 seconds to store what they have.
CMD ["sh", "-c", "python -m supportagent.cli.ingest && exec uvicorn --factory supportagent.api.app:create_app --host 0.0.0.0 --port ${PORT:-8787} --ws-max-size 65536 --no-proxy-headers --timeout-graceful-shutdown 15"]

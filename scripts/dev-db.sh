#!/usr/bin/env bash
# Dev Postgres with pgvector in Docker, bound to 127.0.0.1.
# Usage: scripts/dev-db.sh up|down|status|psql
#
# It reads POSTGRES_PASSWORD from .env at the repo root and keeps its data in data/pg there,
# which git ignores. AISA_HOME moves both to another directory.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${AISA_HOME:-$REPO_DIR}"
ENV_FILE="${SUPPORTAGENT_ENV_FILE:-$HOME_DIR/.env}"
NAME="${PG_CONTAINER:-aisupport-pg}"
IMAGE="${PG_IMAGE:-pgvector/pgvector:pg17}"
PORT="${PG_PORT:-54329}"
DATA_DIR="${PG_DATA_DIR:-$HOME_DIR/data/pg}"

# Read one variable instead of sourcing the file, which may hold values a shell would mangle.
POSTGRES_PASSWORD="$(grep -E '^POSTGRES_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD missing in $ENV_FILE}"
export POSTGRES_PASSWORD

case "${1:-}" in
  up)
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "$NAME already running"
      exit 0
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      docker start "$NAME" >/dev/null
    else
      mkdir -p "$DATA_DIR"
      # Run as the calling user so the data directory stays removable without root.
      docker run -d --name "$NAME" \
        --user "$(id -u):$(id -g)" \
        -v /etc/passwd:/etc/passwd:ro \
        -v "$DATA_DIR":/var/lib/postgresql/data \
        -e POSTGRES_USER=support -e POSTGRES_DB=support -e POSTGRES_PASSWORD \
        -p "127.0.0.1:$PORT:5432" \
        "$IMAGE" >/dev/null
    fi
    # Probe over TCP. The temporary server that runs during first-time init listens on the socket only.
    for _ in $(seq 1 60); do
      if docker exec "$NAME" pg_isready -h 127.0.0.1 -U support -d support -q; then
        echo "$NAME ready on 127.0.0.1:$PORT"
        exit 0
      fi
      sleep 1
    done
    echo "$NAME did not become ready" >&2
    docker logs --tail 30 "$NAME" >&2
    exit 1
    ;;
  down)
    docker stop "$NAME" >/dev/null
    echo "$NAME stopped"
    ;;
  status)
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  psql)
    shift
    docker exec -it "$NAME" psql -U support -d support "$@"
    ;;
  *)
    echo "usage: $0 up|down|status|psql" >&2
    exit 2
    ;;
esac

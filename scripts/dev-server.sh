#!/usr/bin/env bash
# Run the backend in the background for development. It listens on 127.0.0.1 only.
# Usage: scripts/dev-server.sh up|down|status
# The port comes from SUPPORTAGENT_PORT and defaults to 8787. The PID file and the log go to
# data/run at the repo root, which git ignores. AISA_HOME moves them to another directory.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${AISA_HOME:-$REPO_DIR}"

RUN_DIR="$HOME_DIR/data/run"
PORT="${SUPPORTAGENT_PORT:-8787}"
PID_FILE="$RUN_DIR/server.pid"

healthy() { curl -fsS -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; }
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

case "${1:-}" in
  up)
    if healthy; then
      echo "backend already running on 127.0.0.1:$PORT"
      exit 0
    fi
    if port_open; then
      echo "port $PORT is in use by another program. Set SUPPORTAGENT_PORT to a free one." >&2
      exit 1
    fi
    mkdir -p "$RUN_DIR"
    cd "$REPO_DIR/backend"
    # setsid starts a new process group. The shell inside records its own PID, which is the group
    # ID, and then becomes the server. `down` stops the whole group. `$!` would not do: setsid may
    # fork, and then `$!` names a process that is already gone.
    # A client message is a short line of JSON, so frames are capped far below the 16 MiB default.
    # shellcheck disable=SC2016
    setsid nohup bash -c \
      'echo $$ >"$1"; exec uv run uvicorn --factory supportagent.api.app:create_app --host 127.0.0.1 --port "$2" --ws-max-size 65536 --no-proxy-headers --timeout-graceful-shutdown 15' \
      server "$PID_FILE" "$PORT" >"$RUN_DIR/server.log" 2>&1 </dev/null &
    for _ in $(seq 1 90); do
      if healthy; then
        echo "backend ready on 127.0.0.1:$PORT"
        exit 0
      fi
      sleep 1
    done
    echo "backend did not start" >&2
    tail -n 30 "$RUN_DIR/server.log" >&2
    exit 1
    ;;
  down)
    if [ -f "$PID_FILE" ]; then
      kill -TERM -- "-$(cat "$PID_FILE")" 2>/dev/null || true
      rm -f "$PID_FILE"
    fi
    for _ in $(seq 1 20); do
      healthy || break
      sleep 0.5
    done
    if healthy; then
      echo "a backend is still answering on port $PORT and this script did not start it" >&2
      exit 1
    fi
    echo "backend stopped"
    ;;
  status)
    if healthy; then echo "running on 127.0.0.1:$PORT"; else echo "not running"; fi
    ;;
  *)
    echo "usage: $0 up|down|status" >&2
    exit 2
    ;;
esac

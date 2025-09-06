#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/pids"
CONF_DIR="$SCRIPT_DIR/configs"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0")               # stop all tracked instances
  $(basename "$0") 8080 8081     # stop specific ports
USAGE
}

stop_one() {
  local port="$1"
  local pidfile="$PID_DIR/pie_${port}.pid"
  local cfg="$CONF_DIR/pie_${port}.toml"

  if [[ ! -f "$pidfile" ]]; then
    echo "No PID file for port ${port}; skipping."
    return 0
  fi
  local pid; pid="$(cat "$pidfile")"

  if ! ps -p "$pid" >/dev/null 2>&1; then
    echo "PID ${pid} not running; removing stale $pidfile"
    rm -f "$pidfile"
    return 0
  fi

  # Try to verify it's our expected process (best-effort)
  if ps -o cmd= -p "$pid" 2>/dev/null | grep -q -- "$cfg"; then
    echo -n "Stopping port ${port} (PID ${pid})... "
  else
    echo -n "Stopping process group for port ${port} (PID ${pid})... "
  fi

  # Kill the entire process group started by setsid (negative PID targets the PGID)
  kill -TERM -- "-$pid" || true

  # Wait up to ~10s for graceful shutdown
  for _ in {1..20}; do
    if ! ps -p "$pid" >/dev/null 2>&1; then
      echo "done."
      rm -f "$pidfile"
      return 0
    fi
    sleep 0.5
  done

  echo "still running; SIGKILL (group)."
  kill -KILL -- "-$pid" || true
  rm -f "$pidfile"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

if (( $# )); then
  for port in "$@"; do stop_one "$port"; done
else
  shopt -s nullglob
  for f in "$PID_DIR"/pie_*.pid; do
    port="${f##*/}"; port="${port#pie_}"; port="${port%.pid}"
    stop_one "$port"
  done
fi

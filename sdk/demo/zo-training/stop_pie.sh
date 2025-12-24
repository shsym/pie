#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/pids"
CONF_DIR="$SCRIPT_DIR/configs"

# Get the hostname to find the correct files for this node
HOSTNAME=$(hostname)

usage() {
cat <<USAGE
Usage:
$(basename "$0") # stop all tracked instances on this host (${HOSTNAME})
$(basename "$0") 8080 8081 # stop specific ports on this host
USAGE
}

stop_one() {
local port="$1"
# UPDATED: Filenames are now prefixed with the hostname
local pidfile="$PID_DIR/${HOSTNAME}_pie_${port}.pid"
local cfg="$CONF_DIR/${HOSTNAME}_pie_${port}.toml"

if [[ ! -f "$pidfile" ]]; then
echo "No PID file for port ${port} on host ${HOSTNAME}; skipping."
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
echo -n "Stopping port ${port} on ${HOSTNAME} (PID ${pid})... "
else
echo -n "Stopping process group for port ${port} on ${HOSTNAME} (PID ${pid})... "
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
# UPDATED: Glob pattern now includes the hostname to only find this host's PID files
for f in "$PID_DIR"/${HOSTNAME}_pie_*.pid; do
# UPDATED: More robust port extraction from the new filename format
local filename="${f##*/}" # e.g., my-node-1_pie_8080.pid
local base="${filename%.pid}" # e.g., my-node-1_pie_8080
local port="${base##*_}" # e.g., 8080
stop_one "$port"
done
fi

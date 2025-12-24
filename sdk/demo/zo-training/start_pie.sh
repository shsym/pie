#!/usr/bin/env bash

set -Eeuo pipefail

module load cuda/12.6.1
module load cudnn/9.3.0.75
module load git-lfs
module load gcc/11.4.0

# ==== Configure here: PORTS / DEVICES / MODEL / KV_PAGES ====
DEFAULT_PORTS=(8080) # 8081 8082 8083)
DEFAULT_DEVICES=(cuda:0) # cuda:1 cuda:2 cuda:3)

# Configure model and max_num_kv_pages here. These are not overridden by env vars.
MODEL="llama-3.2-1b-instruct"
MAX_NUM_KV_PAGES=30000

# --- The rest of the script remains the same ---

# Get the hostname for unique file naming
HOSTNAME=$(hostname)

split_list() { local s="${1:-}"; s="${s//,/ }"; echo ${s}; }
[[ -n "${PORTS:-}" ]] && PORTS_ARR=($(split_list "$PORTS")) || PORTS_ARR=("${DEFAULT_PORTS[@]}")
[[ -n "${DEVICES:-}" ]] && DEVICES_ARR=($(split_list "$DEVICES")) || DEVICES_ARR=("${DEFAULT_DEVICES[@]}")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="$SCRIPT_DIR/configs"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/pids"
mkdir -p "$CONF_DIR" "$LOG_DIR" "$PID_DIR"

command -v pie >/dev/null 2>&1 || { echo "ERROR: 'pie' not found in PATH"; exit 1; }

# Helper: build a PTY-wrapped command array compatible with Linux/macOS
pty_wrap_cmd() {
# Usage: pty_wrap_cmd <cmd...> -> echoes a command line that allocates a PTY
if command -v script >/dev/null 2>&1; then
# util-linux 'script' supports -c; BSD/macOS can use "script file shell -lc 'cmd'"
if script -V >/dev/null 2>&1; then
# util-linux
printf "script -qfc %q /dev/null" "$*"
else
# BSD/macOS fallback
printf "script -q /dev/null bash -lc %q" "$*"
fi
elif command -v unbuffer >/dev/null 2>&1; then
# expect's unbuffer allocates a pty
printf "unbuffer -p %s" "$*"
else
# No PTY tool found; run plain (may trigger ENOTTY). We'll warn once.
printf "%s" "$*"
fi
}

count_ports=${#PORTS_ARR[@]}
count_devs=${#DEVICES_ARR[@]}
(( count_ports>0 && count_devs>0 )) || { echo "ERROR: need at least one PORT and one DEVICE"; exit 1; }
num=$(( count_ports < count_devs ? count_ports : count_devs ))
(( count_ports != count_devs )) && echo "NOTE: mismatched lengths; launching $num instances."

warned_no_pty=0

launch_one() {
local port="$1" device="$2"
# UPDATED: Filenames are now prefixed with the hostname
local cfg="$CONF_DIR/${HOSTNAME}_pie_${port}.toml"
local app_log="$LOG_DIR/${HOSTNAME}_pie_${port}.log"
local wrap_log="$LOG_DIR/${HOSTNAME}_pie_${port}.stdout"
local pidfile="$PID_DIR/${HOSTNAME}_pie_${port}.pid"

# Skip if already recorded & alive
if [[ -f "$pidfile" ]] && ps -p "$(cat "$pidfile")" >/dev/null 2>&1; then
echo "Port ${port} on host ${HOSTNAME} already running (PID $(cat "$pidfile"))"
return 0
fi
rm -f "$pidfile"

# Write per-instance TOML (absolute paths)
    # Create empty config file if it doesn't exist
    touch "$cfg"

    # Update config using CLI
    pie config update \
        --path "$cfg" \
        --host "0.0.0.0" \
        --port "${port}" \
        --enable-auth "false" \
        --verbose "true" \
        --log "${app_log}" \
        --backend-type "python" \
        --backend-model "${MODEL}" \
        --backend-device "${device}" \
        --backend-activation-dtype "bfloat16" \
        --backend-kv-page-size 16 \
        --backend-max-batch-tokens "${MAX_NUM_KV_PAGES}" \
        --backend-max-dist-size 32 \
        --backend-max-num-embeds 1 \
        --backend-max-num-adapters 1 \
        --backend-max-adapter-rank 8

# Build PTY-wrapped command
CMD_STR="$(pty_wrap_cmd "pie start --config \"$cfg\"")"

# If no PTY tool is present, warn once
if [[ "$CMD_STR" == "pie start --config "* && "$warned_no_pty" -eq 0 ]]; then
echo "WARNING: No PTY allocator found (install 'script' or 'expect' for 'unbuffer'). Running without PTY may cause ENOTTY."
warned_no_pty=1
fi

# Start a new session/process group so we can kill the whole tree later; log stdout/err
# shellcheck disable=SC2086
setsid bash -lc "$CMD_STR" >"$wrap_log" 2>&1 &

echo $! > "$pidfile" # PID == leader of the new process group/session
echo "Launched on host ${HOSTNAME}: port=${port} device=${device} pid=$(cat "$pidfile") cfg=$cfg"
}

for (( i=0; i<num; i++ )); do
launch_one "${PORTS_ARR[$i]}" "${DEVICES_ARR[$i]}"
done

echo "Configs → $CONF_DIR"
echo "Logs → $LOG_DIR"
echo "PIDs → $PID_DIR"

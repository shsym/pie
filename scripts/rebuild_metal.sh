#!/usr/bin/env bash

# Rebuild Metal backend and Metal protocol tests.
#
# Usage:
#   scripts/rebuild_metal.sh [--config Release|Debug] [--backend-only|--protocol-only] [--clean] [-j N]
#
# Defaults:
#   --config Release
#   Builds both backend and protocol tests
#
# Examples:
#   scripts/rebuild_metal.sh                       # Release build of both
#   scripts/rebuild_metal.sh --config Debug -j 8   # Debug build, 8 jobs
#   scripts/rebuild_metal.sh --backend-only --clean

set -euo pipefail

CONFIG="Release"
DO_BACKEND=1
DO_PROTOCOL=1
CLEAN=0
JOBS=""

usage() {
  cat <<USAGE
Rebuild Metal backend and Metal protocol tests.

Options:
  --config <Release|Debug>   Build configuration (default: Release)
  --backend-only             Only build backend/backend-metal
  --protocol-only            Only build metal-protocol-tests
  --clean                    Remove build directories before configuring
  -j, --jobs <N>             Parallel build jobs (default: cmake auto)
  -h, --help                 Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG=${2:-}
      if [[ -z "$CONFIG" ]]; then echo "--config requires a value" >&2; exit 1; fi
      shift 2
      ;;
    --backend-only)
      DO_PROTOCOL=0
      shift
      ;;
    --protocol-only)
      DO_BACKEND=0
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    -j|--jobs)
      JOBS=${2:-}
      if [[ -z "$JOBS" ]]; then echo "-j/--jobs requires a value" >&2; exit 1; fi
      shift 2
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v cmake >/dev/null 2>&1; then
  echo "Error: cmake not found. Install CMake (e.g., brew install cmake) and retry." >&2
  exit 1
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

build_dir() {
  local path="$1"
  mkdir -p "$path"
}

do_configure_and_build() {
  local src="$1"
  local build="$2"
  local name="$3"

  if [[ "$CLEAN" -eq 1 ]]; then
    echo "[clean] $name: removing $build"
    rm -rf "$build"
  fi

  echo "[configure] $name -> $build (CONFIG=$CONFIG)"
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE="$CONFIG"

  echo "[build] $name"
  if [[ -n "$JOBS" ]]; then
    cmake --build "$build" --config "$CONFIG" --parallel "$JOBS"
  else
    cmake --build "$build" --config "$CONFIG" --parallel
  fi
}

if [[ "$DO_PROTOCOL" -eq 1 ]]; then
  SRC_PROTO="$ROOT_DIR/metal-protocol-tests"
  BUILD_PROTO="$SRC_PROTO/build"
  do_configure_and_build "$SRC_PROTO" "$BUILD_PROTO" "metal-protocol-tests"
fi

if [[ "$DO_BACKEND" -eq 1 ]]; then
  SRC_BACKEND="$ROOT_DIR/backend/backend-metal"
  BUILD_BACKEND="$SRC_BACKEND/build"
  do_configure_and_build "$SRC_BACKEND" "$BUILD_BACKEND" "backend-metal"
fi

echo "Done."

#!/bin/bash
set -e

# Run all format and lint checks

echo "Running Python format check (black)..."
./tests/format/python-format.sh
echo "Done."

echo "Running Python type check (pyright)..."
./tests/format/python-typecheck.sh
echo "Done."

echo "All checks passed."

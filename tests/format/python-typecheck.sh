#!/bin/bash
set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Run pyright type check on all Python packages
uvx pyright \
    "${ROOT}/runtime-backend/src/pie_backend/" \
    "${ROOT}/server/src/pie_cli/" \
    "${ROOT}/client/python/src/pie_client/" \
    "${ROOT}/sdk/tools/bakery/src/bakery/"

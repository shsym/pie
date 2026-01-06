#!/bin/bash
set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Run black format check on all Python packages
uvx black --check \
    "${ROOT}/runtime-backend/src/pie_backend/" \
    "${ROOT}/control/src/pie_cli/" \
    "${ROOT}/client/python/src/pie_client/" \
    "${ROOT}/sdk/tools/bakery/src/bakery/"

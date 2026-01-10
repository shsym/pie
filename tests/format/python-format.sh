#!/bin/bash
set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Run black format check on all Python packages
uvx black --check \
    "${ROOT}/server/src/pie_worker/" \
    "${ROOT}/server/src/pie_cli/" \
    "${ROOT}/client/python/src/pie_client/" \
    "${ROOT}/sdk/tools/bakery/src/bakery/"

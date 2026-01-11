#!/bin/bash
set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Run black format check on all Python packages
uvx black --check \
    "${ROOT}/pie/src/pie_worker/" \
    "${ROOT}/pie/src/pie_cli/" \
    "${ROOT}/client/python/src/pie_client/" \
    "${ROOT}/sdk/tools/bakery/src/bakery/"


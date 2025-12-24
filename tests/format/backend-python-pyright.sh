#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Run pyright check.

PYTHONPATH=${ROOT}/engine/backend-python/src uv run \
    --project ${ROOT}/engine/backend-python \
    --with pyright \
    --with torch \
    pyright \
    ${ROOT}/engine/backend-python/src/pie_backend/*.py \
    ${ROOT}/engine/backend-python/src/pie_backend/model/*.py

#!/bin/bash
set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Set PYTHONPATH so imports work
export PYTHONPATH="${ROOT}/engine/backend-python/src:${PYTHONPATH}"

uvx black --check \
    ${ROOT}/engine/backend-python/src/pie_backend/*.py \
    ${ROOT}/engine/backend-python/src/pie_backend/model/*.py

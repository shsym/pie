#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Run pylint check.
# See `.pylintrc` for the configuration of the check.

PYTHONPATH=${ROOT}/engine/backend-python/src uv run \
    --project ${ROOT}/engine/backend-python \
    --with pylint \
    --with torch \
    pylint --disable=R \
    ${ROOT}/engine/backend-python/src/pie_backend/*.py \
    ${ROOT}/engine/backend-python/src/pie_backend/model/*.py

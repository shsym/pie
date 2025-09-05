#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

# Run pylint check.
# See `pyrightconfig.json` for the configuration of the check.

UV_NO_SYNC=1 UV_OFFLINE=1 uv \
    --project ${ROOT}/backend/backend-python \
    run pyright \
    ${ROOT}/backend/backend-python/config/*.py \
    ${ROOT}/backend/backend-python/model/*.py

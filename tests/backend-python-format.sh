#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

uvx black --check \
    ${ROOT}/backend/backend-python/config/*.py \
    ${ROOT}/backend/backend-python/model/*.py

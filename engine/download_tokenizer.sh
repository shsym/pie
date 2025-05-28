#!/bin/bash

# Accept model name as first argument, such as Llama-3.2-1B-Instruct
MODEL_NAME="${1:-Llama-3.1-8B-Instruct}"
MODEL_PATH="program_cache/${MODEL_NAME}"
MODEL_GIT="hf.co:meta-llama/${MODEL_NAME}"

echo "Downloading tokenizer for model: ${MODEL_NAME}"
git clone git@"${MODEL_GIT}" "${MODEL_PATH}"

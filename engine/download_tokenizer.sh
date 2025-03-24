#!/bin/bash
MODEL_NAME="Llama-3.2-1B-Instruct"
MODEL_PATH="program_cache/${MODEL_NAME}"
MODEL_GIT="hf.co:meta-llama/${MODEL_NAME}"

git clone git@"${MODEL_GIT}" "${MODEL_PATH}"

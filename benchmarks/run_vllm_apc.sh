docker run --runtime nvidia --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.6.0 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --max_model_len 4096 \
    --max-logprobs 64 \
    --disable-sliding-window \
    --enforce-eager \
    --block-size 16 \
    --dtype bfloat16 \
    --enable-prefix-caching \

docker run --gpus all \
    -p 8000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    lmsysorg/sglang:v0.4.4-cu124 \
    python3 -m sglang.launch_server \
    --model-path "meta-llama/Llama-3.2-1B-Instruct" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --page-size 16 \
    --stream-interval 1 \
    --grammar-backend "xgrammar" \
    --disable-cuda-graph-padding \
    --disable-cuda-graph \
    --disable-radix-cache

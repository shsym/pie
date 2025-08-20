# Metal Protocol Tests

Cross-platform testing harness for PIE Metal backend operations with CUDA golden reference validation.

## Overview

General workflow follows these steps:

- Linux: Generate CUDA golden references using the separate CUDA harness
- Cross-platform: Transfer CUDA artifacts to macOS
- macOS: Build and run the Metal harness to validate against CUDA references

## Artifact preparation

The sections below describe how to generate and transfer CUDA artifacts. These are used as golden references for comparison on macOS.

### Generate CUDA artifacts (Linux)

Use the CUDA-only harness under `cuda-protocol-tests/` to generate reference outputs.

```bash
cd cuda-protocol-tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Examples
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op gemm --case production --m 128 --n 4096 --k 4096
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op embedding_lookup --case production --num_tokens 128 --hidden_size 4096 --vocab_size 32000
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op extract_k_values --case production --M 128 --N 4096 --k 50
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op rms_norm --case production --num_tokens 128 --hidden_size 4096
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op rope --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0
```

Quick generate (recommended): use the script to build and emit all common CUDA references to the exact path used by Metal comparison.

```bash
# From the repository root on Linux
# Optional env: GPU=<index> (default 0), CASE_ID=<name> (default production)
GPU=0 CASE_ID=production bash cuda-protocol-tests/scripts/generate_cuda_artifacts.sh
```

## Cross-platform transfer (Linux to macOS)

Transfer CUDA artifacts from Linux to macOS using the helper script:

```bash
# On Linux (after generating artifacts)
../scripts/artifacts_transfer.sh compress

# Copy cuda_artifacts.tar.xz to macOS, then:
../scripts/artifacts_transfer.sh extract
```

The transfer typically includes:

- All operation artifacts
- Artifact manifest (`tests/artifact_manifest.json`) for metadata validation
- Multi-dtype variants (fp32/bf16) and single-dtype operations

### Artifact structure

Default Metal artifacts (written by this harness) live under the build directory:

```text
metal-protocol-tests/build/tests/artifacts/
├── gemm/
│   └── test1_metal/
├── softmax/
│   └── production_metal/
└── ...
```

CUDA artifacts typically live under:

```text
metal-protocol-tests/tests/artifacts/
└── softmax/
   └── production/
```

## Build and run Metal tests (macOS)

### Build

Requirements: Xcode with Metal support, CMake 3.23+

```bash
cd metal-protocol-tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

#### Build Errors

Dependency: numpy is required for comparison. On macOS:

```bash
python3 -m pip install --user numpy
```

C++ header errors: install/update Xcode Command Line Tools:
```bash
xcode-select --install
# If already installed:
sudo xcode-select --reset
xcode-select --install
```

### Quick testing

From the `build` directory:

```bash
# Run all implemented operations
bash ../scripts/test_all_ops.sh

# Run an individual operation without CUDA comparison (faster)
./metal_protocol_tests --op gemm --case test1 --m 16 --n 32 --k 24 --no-compare

# Run with CUDA comparison (requires CUDA artifacts to be present)
./metal_protocol_tests --op softmax --case test1 --batch_size 2 --vocab_size 100
```

### Operation Status

| Operation | Status | Test Command |
|-----------|:------:|--------------|
| gemm | ✅ | `--op gemm --m 16 --n 32 --k 24` |
| embedding_lookup | ✅ | `--op embedding_lookup --num_tokens 8 --hidden_size 128 --vocab_size 1000` |
| silu_and_mul | ✅ | `--op silu_and_mul --num_tokens 8 --intermediate_size 64` |
| extract_k_values | ✅ | `--op extract_k_values --M 4 --N 32 --k 5` |
| softmax | ✅ | `--op softmax --batch_size 2 --vocab_size 100` |
| rms_norm | ✅ | `--op rms_norm --num_tokens 8 --hidden_size 128` |
| rope | ✅ | `--op rope --num_tokens 4 --num_heads 4 --head_size 16` |
| topk_mask_logits | ✅ | `--op topk_mask_logits --num_tokens 4 --vocab_size 50 --k 10` |
| grouped_gemm | ✅ | `--op grouped_gemm` |
| batch_prefill_attention | ✅ | `--op batch_prefill_attention --num_tokens 4 --num_query_heads 2 --num_kv_heads 2 --head_size 8 --kv_len 16 --page_size 128` |

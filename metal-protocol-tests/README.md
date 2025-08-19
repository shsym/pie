# Metal Protocol Tests

Cross-platform testing framework for PIE Metal backend operations with CUDA golden reference validation.

## Overview

This framework enables development and validation of Metal GPU implementations against CUDA golden references:

- Linux: Generate CUDA golden references using the separate CUDA harness
- macOS: Implement and test Metal GPU kernels against those CUDA artifacts
- Cross-platform: Transfer artifacts between platforms for validation

## Status: 6/10 operations complete

| Operation | CUDA | Metal | Status |
|-----------|:----:|:-----:|--------|
| gemm | ✅ | ✅ | Complete |
| embedding_lookup | ✅ | ✅ | Complete |
| silu_and_mul | ✅ | ✅ | Complete |
| extract_k_values | ✅ | ✅ | Complete |
| softmax | ✅ | ✅ | Complete |
| rms_norm | ✅ | ✅ | Complete |
| rope | ✅ | 🔲 | CUDA only |
| batch_prefill_attention | ✅ | 🔲 | CUDA only |
| grouped_gemm | ✅ | 🔲 | CUDA only |
| append_paged_kv_cache | ✅ | 🔲 | CUDA only |

## Source layout

- src/main.cpp — CLI test driver (Metal by default on macOS), auto-compares vs CUDA
- src/ops_metal.mm — Metal host wrappers (calls backend/backend-metal kernels)
- src/artifacts.hpp — Host-only artifact helpers
- Legacy stubs (not used by build): src/ops_cuda.cu, src/metal_test_main.mm

## Build instructions

Note: This directory contains a Metal-only CMakeLists.txt. The CUDA harness under `cuda-protocol-tests/` uses its own CMake project.

### Linux (CUDA reference generation)

If you are generating CUDA golden references on Linux, use the CUDA-only harness under `cuda-protocol-tests/`.

Requirements: CUDA Toolkit 11.0+, CMake 3.23+, FlashInfer dependencies

```bash
cd cuda-protocol-tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### macOS (Metal development)

Requirements: Xcode with Metal support, CMake 3.23+

```bash
cd metal-protocol-tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

## Quick testing

### CUDA reference generation (Linux)

```bash
cd ../cuda-protocol-tests/build

# Generate golden references for Metal validation
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op gemm --case test1 --m 32 --n 128 --k 64
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0
```

### Metal operations (macOS)

Artifacts from Metal runs are written under the build directory by default:
metal-protocol-tests/build/tests/artifacts

```bash
cd build

# Test Metal operations (auto-compares vs CUDA by default)
./metal_protocol_tests --op gemm --case test1 --m 32 --n 128 --k 64
./metal_protocol_tests --op embedding_lookup --case test1 --num_tokens 128 --hidden_size 4096 --vocab_size 32000
./metal_protocol_tests --op silu_and_mul --case test_unified --num_tokens 64 --intermediate_size 256
./metal_protocol_tests --op extract_k_values --case test_unified --M 8 --N 64 --k 5
./metal_protocol_tests --op softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0
./metal_protocol_tests --op rms_norm --case test1 --num_tokens 128 --hidden_size 4096 --eps 1e-5
```

## Production testing

### Generate CUDA artifacts (Linux)

Build the CUDA harness and run the operations you need. For bulk generation, create a small shell script invoking `./cuda_protocol_tests` with your desired sizes and cases.

Examples:

```bash
cd ../cuda-protocol-tests/build

# Multi-dtype operations (bf16/fp32 where applicable)
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op gemm --case production --m 128 --n 4096 --k 4096
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op embedding_lookup --case production --num_tokens 128 --hidden_size 4096 --vocab_size 32000
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op extract_k_values --case production --M 128 --N 4096 --k 50

# Single-dtype operations (Llama 7B scale)
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op rms_norm --case production --num_tokens 128 --hidden_size 4096
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op rope --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0
```

### Test Metal implementations (macOS)

The Metal harness auto-compares against CUDA artifacts after each run. Ensure CUDA artifacts exist under `cuda-protocol-tests/tests/artifacts` or have been transferred to `metal-protocol-tests/tests/artifacts`.

```bash
# Test against CUDA golden references (auto-compare prints a PASS/FAIL report)
./metal_protocol_tests --op gemm --case production --m 128 --n 4096 --k 4096
./metal_protocol_tests --op rms_norm --case production --num_tokens 128 --hidden_size 4096
./metal_protocol_tests --op softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0

# You can disable comparison with:
./metal_protocol_tests --op gemm --case test1 --m 32 --n 128 --k 64 --no-compare
```

## Artifact management

### Structure

By default, Metal artifacts are stored under the build directory:

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

### Cross-platform transfer

Transfer CUDA artifacts from Linux to macOS:

```bash
# On Linux (after generating artifacts with your CUDA harness)
./scripts/artifacts_transfer.sh compress

# Transfer cuda_artifacts.tar.xz to macOS, then:
./scripts/artifacts_transfer.sh extract
```

The transfer typically includes:

- All operation artifacts
- Artifact manifest (`tests/artifact_manifest.json`) with metadata validation
- Multi-dtype variants (fp32/bf16) and single-dtype operations

### Validation

The Metal harness invokes `scripts/compare_artifacts.py` automatically to compare Metal results (`<metal-base>/<op>/<case>_metal`) against CUDA references (`<cuda-base>/<op>/<case>`). You can also run it manually:

```bash
python3 scripts/compare_artifacts.py --op gemm --case test1 --verbose
# or direct paths
python3 scripts/compare_artifacts.py --cuda-dir cuda-protocol-tests/tests/artifacts/gemm/test1 \
                                     --metal-dir build/tests/artifacts/gemm/test1_metal
```

Dependency: numpy is required for comparison. On macOS:

```bash
python3 -m pip install --user numpy
```

## Development workflow

### Adding new Metal operations

1. Generate CUDA reference (Linux):

   ```bash
   # Using the CUDA harness
   cd ../cuda-protocol-tests/build
   CUDA_VISIBLE_DEVICES=0 ./cuda_protocol_tests --op NEW_OP --case test1 [params]
   ```

2. Implement Metal kernel (macOS):
   - Create `backend/backend-metal/src/metal_NEW_OP.{hpp,mm,metal}`
   - Add integration to `metal-protocol-tests/src/ops_metal.mm`

3. Test and validate:

   ```bash
   # Run Metal implementation to generate artifacts (auto-comparison runs by default)
   ./metal_protocol_tests --op NEW_OP --case test1 [params]

   # Compare manually if needed
   python3 scripts/compare_artifacts.py --op NEW_OP --case test1 --verbose
   ```

### Notes on tolerances

- Default tolerances: abs=1e-3, rel=1e-2
- Scoped override: for `rms_norm` bf16 output, rel=1.5e-2 (reflects bf16 + reduction-order variance)

## Environment variables

- PIE_WRITE_ARTIFACTS=1 — Enable artifact writing (auto-enabled by the harness)
- PIE_ARTIFACTS_DIR — Output directory base (defaults to build/tests/artifacts next to the binary)
- CUDA_VISIBLE_DEVICES=0 — GPU selection for CUDA operations (CUDA harness only)

## Current configuration

This directory builds the Metal-only harness on macOS. Use the CUDA harness under `cuda-protocol-tests/` on Linux.

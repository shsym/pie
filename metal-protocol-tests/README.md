# Metal Protocol Tests

Cross-platform testing framework for PIE Metal backend operations with CUDA golden reference validation.

## Overview

This framework enables development and validation of Metal GPU implementations against CUDA golden references:

- Linux: Generate CUDA golden references using the separate CUDA harness
- macOS: Implement and test Metal GPU kernels against those CUDA artifacts
- Cross-platform: Transfer artifacts between platforms for validation

## Status: 10/10 operations complete (Metal implementations)

| Operation | CUDA | Metal | Status |
|-----------|:----:|:-----:|--------|
| gemm | ✅ | ✅ | Complete |
| embedding_lookup | ✅ | ✅ | Complete |
| silu_and_mul | ✅ | ✅ | Complete |
| extract_k_values | ✅ | ✅ | Complete |
| softmax | ✅ | ✅ | Complete |
| rms_norm | ✅ | ✅ | Complete |
| rope | ✅ | ✅ | Complete |
| topk_mask_logits | ✅ | ✅ | Complete |
| grouped_gemm | ✅ | ✅ | Complete |
| batch_prefill_attention | ✅ | ✅ | Complete |

## Source layout

- src/main.cpp — CLI test driver (Metal by default on macOS), auto-compares vs CUDA
- src/ops_metal.mm — Metal host wrappers (calls backend/backend-metal kernels)
- src/artifacts.hpp — Host-only artifact helpers

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

## Testing

### Test all operations (macOS)

```bash
cd build

# Test all implemented Metal operations with single script
./scripts/test_all_ops.sh

# Or test individual operations:
./metal_protocol_tests --op gemm --case test1 --no-compare
./metal_protocol_tests --op embedding_lookup --case test1 --no-compare  
./metal_protocol_tests --op silu_and_mul --case test1 --no-compare
./metal_protocol_tests --op extract_k_values --case test1 --no-compare
./metal_protocol_tests --op softmax --case test1 --no-compare
./metal_protocol_tests --op rms_norm --case test1 --no-compare
./metal_protocol_tests --op rope --case test1 --no-compare
./metal_protocol_tests --op topk_mask_logits --case test1 --no-compare
./metal_protocol_tests --op grouped_gemm --case test1 --no-compare
./metal_protocol_tests --op batch_prefill_attention --case test1 --no-compare
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
../scripts/artifacts_transfer.sh compress

# Transfer cuda_artifacts.tar.xz to macOS, then:
../scripts/artifacts_transfer.sh extract
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

## Troubleshooting

### macOS Build Issues

If you encounter errors like `'cstdint' file not found` or `'iostream' file not found` during build:

**Problem**: Missing or incomplete C++ standard library headers.

**Automatic Detection**: The CMakeLists.txt now automatically detects the correct C++ stdlib path using `xcrun --show-sdk-path`. You should see this message during cmake configuration:
```
-- Using C++ stdlib from: /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
```

**If automatic detection fails**:

1. **Install/Update Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   # If already installed, try reinstalling:
   sudo xcode-select --reset
   xcode-select --install
   ```

2. **Verify Installation**:
   ```bash
   xcrun --show-sdk-path
   # Should return a valid SDK path
   
   # Check for C++ headers:
   ls $(xcrun --show-sdk-path)/usr/include/c++/v1/cstdint
   ```

3. **Install Full Xcode** (if Command Line Tools aren't sufficient):
   - Download Xcode from the App Store
   - Run: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`

### Metal Framework Issues

If Metal framework errors occur:
- Ensure you're running on macOS with Metal support
- Verify Xcode is properly installed with Metal development tools

## Current configuration

This directory builds the Metal-only harness on macOS. Use the CUDA harness under `cuda-protocol-tests/` on Linux.

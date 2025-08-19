# Metal Protocol Tests

Cross-platform testing framework for PIE Metal Backend operations with CUDA golden reference validation.

## Overview

This framework enables development and validation of Metal GPU implementations against CUDA golden references:

- **Linux**: Generate CUDA golden references using FlashInfer
- **macOS**: Implement and test Metal GPU kernels against CUDA artifacts
- **Cross-Platform**: Transfer artifacts between platforms for validation

## Status: 5/10 Operations Complete

| Operation | CUDA | Metal | Status |
|-----------|:----:|:-----:|--------|
| gemm | ✅ | ✅ | Complete |
| embedding_lookup | ✅ | ✅ | Complete |
| silu_and_mul | ✅ | ✅ | Complete |
| extract_k_values | ✅ | ✅ | Complete |
| softmax | ✅ | ✅ | Complete |
| rms_norm | ✅ | 🔲 | CUDA only |
| rope | ✅ | 🔲 | CUDA only |
| batch_prefill_attention | ✅ | 🔲 | CUDA only |
| grouped_gemm | ✅ | 🔲 | CUDA only |
| append_paged_kv_cache | ✅ | 🔲 | CUDA only |

## Build Instructions

**Important**: No unified CMakeLists.txt exists. Always copy platform-specific configuration.

### Linux (CUDA Reference Generation)

Requirements: CUDA Toolkit 11.0+, CMake 3.23+, FlashInfer dependencies

```bash
cd metal-protocol-tests
cp CMakeLists_cuda_only.txt CMakeLists.txt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### macOS (Metal Development)

Requirements: Xcode with Metal support, CMake 3.23+

```bash
cd metal-protocol-tests
cp CMakeLists_metal_only.txt CMakeLists.txt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Testing

### CUDA Reference Generation (Linux)

```bash
cd build

# Generate golden references for Metal validation
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op gemm --case test1 --m 32 --n 128 --k 64
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op softmax --case test1 --batch_size 1 --vocab_size 8 --temperature 1.0
```

### Metal Operations (macOS)

```bash
cd build

# Test all implemented Metal operations
./metal_protocol_tests --backend metal --op gemm --case test1 --m 32 --n 128 --k 64
./metal_protocol_tests --backend metal --op embedding_lookup --case test1 --num_tokens 16 --hidden_size 128
./metal_protocol_tests --backend metal --op silu_and_mul --case test1 --num_tokens 64 --intermediate_size 256
./metal_protocol_tests --backend metal --op extract_k_values --case test1 --M 8 --N 64 --k 5
./metal_protocol_tests --backend metal --op softmax --case test1 --batch_size 1 --vocab_size 8 --temperature 1.0
```

## Production Testing

### Generate CUDA Artifacts (Linux)

Use the comprehensive artifact generation script:

```bash
# Generate all CUDA reference artifacts (recommended)
bash scripts/regenerate_artifacts.sh production

# With custom GPU and options
GPU=1 CASE_SUFFIX=_v2 bash scripts/regenerate_artifacts.sh production

# Environment variables:
# GPU=<id>              GPU index for CUDA_VISIBLE_DEVICES (default: 0)
# SKIP_BUILD=1          Skip building if binary already exists
# DRY_RUN=1             Show commands without executing
# CASE_SUFFIX=<suffix>  Append suffix to case name
```

This generates artifacts for all 10 operations including:
- Multi-dtype: `gemm`, `embedding_lookup`, `extract_k_values` (both fp32 + bf16)
- Single-dtype: `rms_norm`, `silu_and_mul`, `rope`, `softmax`, `grouped_gemm`, `batch_prefill_attention`, `append_paged_kv_cache`, etc.

### Manual Generation (Advanced)

For individual operations:

```bash
cd build
# Multi-dtype operations (generates both fp32 and bf16 automatically)
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op gemm_all_dtypes --case production --m 128 --n 4096 --k 4096
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op embedding_lookup_all_dtypes --case production --num_tokens 128 --hidden_size 4096 --vocab_size 32000

# Single-dtype operations (Llama 7B scale)
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op rms_norm --case production --num_tokens 128 --hidden_size 4096
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op rope --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
```

### Test Metal Implementations (macOS)

```bash
# Test against CUDA golden references
./metal_protocol_tests --backend metal --op gemm --case production --m 128 --n 4096 --k 4096
./metal_protocol_tests --backend metal --op softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0
```

## Artifact Management

### Structure

Artifacts are stored in `tests/artifacts/` with the following structure:

```
tests/artifacts/
├── gemm/
│   ├── test1_fp32/         # Float32 version
│   ├── test1_bf16/         # BFloat16 version
│   └── test1_metal/        # Metal output
├── softmax/
│   ├── test1/              # CUDA reference
│   └── test1_metal/        # Metal output
└── ...
```

### Cross-Platform Transfer

Transfer CUDA artifacts from Linux to macOS:

```bash
# On Linux (after generating artifacts with regenerate_artifacts.sh)
./scripts/artifacts_transfer.sh compress

# Transfer cuda_artifacts.tar.xz to macOS, then:
./scripts/artifacts_transfer.sh extract
```

The transfer includes:
- All operation artifacts for 10 operations
- Artifact manifest (`tests/artifact_manifest.json`) with metadata validation
- Multi-dtype variants (fp32/bf16) and single-dtype operations

### Validation

Compare Metal outputs against CUDA references:

```bash
diff tests/artifacts/gemm/test1/ tests/artifacts/gemm/test1_metal/
diff tests/artifacts/softmax/test1/ tests/artifacts/softmax/test1_metal/
```

## Development Workflow

### Adding New Metal Operations

1. **Generate CUDA reference** (Linux):
   ```bash
   # Using comprehensive script (recommended)
   bash scripts/regenerate_artifacts.sh test1

   # Or manually for specific operation
   CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op NEW_OP --case test1 [params]
   ```

2. **Implement Metal kernel** (macOS):
   - Create `backend/backend-metal/src/metal_NEW_OP.{hpp,mm,metal}`
   - Add integration to `metal-protocol-tests/src/ops_metal.mm`

3. **Test and validate**:
   ```bash
   ./metal_protocol_tests --backend metal --op NEW_OP --case test1 [params]
   diff tests/artifacts/NEW_OP/test1/ tests/artifacts/NEW_OP/test1_metal/
   ```

The artifact manifest (`tests/artifact_manifest.json`) provides validation metadata for all operations including expected file patterns, data types, and shapes.

### Next Priority: RMSNorm

Medium complexity operation with existing CUDA reference:

```bash
# Test CUDA reference
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --backend cuda --op rms_norm --case test1 --num_tokens 64 --hidden_size 4096

# Implement following the softmax pattern:
# 1. backend/backend-metal/src/metal_rmsnorm.{hpp,mm,metal}
# 2. Add to metal-protocol-tests/src/ops_metal.mm
# 3. Test: ./metal_protocol_tests --backend metal --op rms_norm --case test1
```

## Environment Variables

- `PIE_WRITE_ARTIFACTS=1` - Enable artifact writing (auto-enabled)
- `PIE_ARTIFACTS_DIR=tests/artifacts` - Output directory
- `CUDA_VISIBLE_DEVICES=0` - GPU selection for CUDA operations

## Current Configuration

CMakeLists.txt is currently set to Metal-only (macOS). Copy from `CMakeLists_cuda_only.txt` when building on Linux.

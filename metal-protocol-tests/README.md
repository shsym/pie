# Metal Protocol Tests (CUDA GPU)

## Overview
**Complete test framework for generating golden reference data from CUDA backend operations for Metal backend validation.**

- **Production-ready**: Uses real FlashInfer functions and CUDA backend kernels (not simplified placeholders)
- **Full API Coverage**: All 10 Metal backend operations implemented with correct data types
- **Automatic Multi-Dtype Testing**: Automatically tests all data types supported by CUDA backend (no manual specification needed)
- **Realistic Scale**: Supports Llama 7B dimensions (4096 hidden, 11008 intermediate, 32 heads)
- **Artifact Generation**: Creates binary tensors + JSON metadata compatible with Metal validation

## Supported Operations

✅ **Core Operations (10/10 complete)**:
- `gemm` - Matrix multiplication with cuBLAS (supports bias, transpose)
  - 🎯 **Multi-dtype**: `gemm_all_dtypes` (fp32 + bf16)
- `embedding_lookup` - Token embedding lookup
  - 🎯 **Multi-dtype**: `embedding_lookup_all_dtypes` (fp32 + bf16)
- `extract_k_values` - Top-k value extraction
  - 🎯 **Multi-dtype**: `extract_k_values_all_dtypes` (fp32 + bf16)
- `rms_norm` - **Real FlashInfer** RMS normalization (bf16 only)
- `silu_and_mul` - SiLU activation + element-wise multiplication (bf16 only)
- `rope` - **Real FlashInfer** RoPE positional encoding (`BatchQKApplyLlama31RotaryPosIds`) (bf16 only)
- `topk_mask_logits` - **Real FlashInfer** top-k masking (fp32 only)
- `batch_prefill_attention` - Attention computation (simplified placeholder)
- `grouped_gemm` - Batched GEMM operations
- `append_paged_kv_cache` - KV cache management (simplified placeholder)
- `add_residual` - Residual connection addition
- `cast_type` - Data type conversion between fp32/fp16/bf16

## Build

**Requirements**: CUDA Toolkit, CMake 3.18+

```bash
# From repo root
cd metal-protocol-tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Alternatively (equivalent) using explicit source/build dirs:

```bash
# From repo root
cmake -S metal-protocol-tests -B metal-protocol-tests/build -DCMAKE_BUILD_TYPE=Release
cmake --build metal-protocol-tests/build -- -j$(nproc)
```

## Regenerate all artifacts (one command)

A convenience script cleans old artifacts and regenerates everything (multi-dtype + single-dtype) into a single canonical directory.

```bash
# From repo root
bash metal-protocol-tests/scripts/regenerate_artifacts.sh [CASE_ID]
# Default CASE_ID is "production"
```

Environment variables:

- `GPU=<id>`: GPU index for CUDA_VISIBLE_DEVICES (default: 0)
- `SKIP_BUILD=1`: Skip building if the binary is already built
- `DRY_RUN=1`: Print commands without executing
- `CASE_SUFFIX=_debug`: Append a suffix to the case id

Artifacts output directory (single source of truth):

```
metal-protocol-tests/tests/artifacts
```

## Automatic Multi-Dtype Testing 🎯

**NEW FEATURE**: Automatically test all data types supported by CUDA backend with a single command!

```bash
cd metal-protocol-tests/build

# 🎯 AUTOMATIC multi-dtype testing (tests ALL supported data types)
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op gemm_all_dtypes --case production --m 128 --n 4096 --k 4096
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op embedding_lookup_all_dtypes --case production --num_tokens 128 --hidden_size 4096 --vocab_size 32000
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op extract_k_values_all_dtypes --case production --M 128 --N 4096 --k 50

# Results: Each command generates TWO sets of artifacts automatically:
# ├── production_fp32/  (float32 data + metadata)
# └── production_bf16/  (bfloat16 data + metadata)
```

## Single Data Type Testing

```bash
# Standard single-dtype tests (Llama 7B dimensions)
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op rms_norm --case production --num_tokens 128 --hidden_size 4096
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op silu_and_mul --case production --num_tokens 128 --intermediate_size 11008
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op rope --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op gemm --case production --m 128 --n 4096 --k 4096
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op embedding_lookup --case production --num_tokens 128 --hidden_size 4096 --vocab_size 32000
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op topk_mask_logits --case production --num_tokens 128 --vocab_size 32000 --k 50
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op grouped_gemm --case production --num_groups 4 --m 128 --n 4096 --k 4096
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op batch_prefill_attention --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op append_paged_kv_cache --case production --num_tokens 128 --num_kv_heads 32 --head_size 128
CUDA_VISIBLE_DEVICES=0 ./metal_protocol_tests --op add_residual --case production --num_tokens 128 --hidden_size 4096
```

## Generated Artifacts

### Single Data Type Operations
Standard operations create artifacts under `tests/artifacts/<op_name>/<case_id>/`:
- **Binary files**: `input.bin`, `output.bin`, etc. (tensors in operation's native dtype)
- **Metadata**: `meta.json` with shapes, dtypes, and configuration

### Multi-Dtype Operations 🎯
Multi-dtype operations create **separate artifacts for each data type**:

```
tests/artifacts/
├── gemm/
│   ├── production_fp32/        # 🎯 Float32 version
│   │   ├── A.bin              # [128, 4096] fp32
│   │   ├── B.bin              # [4096, 4096] fp32
│   │   ├── C.bin              # [128, 4096] fp32
│   │   └── meta.json          # {"dtype_map": {"A": "fp32", ...}}
│   └── production_bf16/        # 🎯 BFloat16 version
│       ├── A.bin              # [128, 4096] bf16
│       ├── B.bin              # [4096, 4096] bf16
│       ├── C.bin              # [128, 4096] bf16
│       └── meta.json          # {"dtype_map": {"A": "bf16", ...}}
├── embedding_lookup_forward/
│   ├── production_fp32/        # 🎯 Float32 embeddings
│   │   ├── embedding.bin      # [32000, 4096] fp32
│   │   ├── indices.bin        # [128] s32
│   │   ├── output.bin         # [128, 4096] fp32
│   │   └── meta.json
│   └── production_bf16/        # 🎯 BFloat16 embeddings
│       ├── embedding.bin      # [32000, 4096] bf16
│       ├── indices.bin        # [128] s32
│       ├── output.bin         # [128, 4096] bf16
│       └── meta.json
└── rms_norm/production/        # Single dtype (bf16 only)
    ├── input.bin              # [128, 4096] bf16
    ├── weight.bin             # [4096] bf16
    ├── output.bin             # [128, 4096] bf16
    └── meta.json
```

## Environment Variables

- `PIE_WRITE_ARTIFACTS=1` - Enable artifact writing (auto-enabled)
- `PIE_ARTIFACTS_DIR=tests/artifacts` - Output directory
- `PIE_ARTIFACT_OPS=rms_norm,rope` - Restrict which ops write artifacts

## Metal Backend Usage

### Comprehensive Validation Workflow

1. **Generate golden data** using automatic multi-dtype testing:
   ```bash
   # Generate reference data for ALL supported data types automatically
   ./metal_protocol_tests --op gemm_all_dtypes --case validation --m 128 --n 4096 --k 4096
   ./metal_protocol_tests --op embedding_lookup_all_dtypes --case validation --num_tokens 128 --hidden_size 4096 --vocab_size 32000
   ```

2. **Implement Metal kernels** for target operations (fp32 and bf16 versions)

3. **Load binary artifacts** and compare Metal outputs with CUDA golden references:
   - Test Metal fp32 implementation against `validation_fp32/` artifacts
   - Test Metal bf16 implementation against `validation_bf16/` artifacts

4. **Validate correctness** across all operation variations and data types

### Key Benefits

✅ **Zero Manual Work**: Framework automatically tests all CUDA backend supported data types
✅ **Complete Coverage**: Every data type combination that exists in CUDA backend is tested
✅ **Production Ready**: Uses exact same functions and parameters as CUDA backend
✅ **Organized Results**: Clear separation of fp32 vs bf16 artifacts for easy validation

This framework provides **comprehensive multi-dtype golden reference data** matching the exact CUDA backend behavior!

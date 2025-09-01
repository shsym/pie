# Copilot instructions for this repo

This repo implements PIE (Programmable Inference Engine): a Rust engine + backends (CUDA/Metal/Python) + protocol tests + example inferlets. Use these notes to be productive fast and avoid common pitfalls.

## Big picture
- Engine and CLI (Rust): `pie/` (engine) and `pie-cli/` (CLI runner). Inferlets (apps) are Rust compiled to `wasm32-wasip2` under `example-apps/` and executed by the engine.
- Backends:
  - `backend/backend-metal/` (macOS Metal, ObjC++ + MSL) — mirrors CUDA APIs for drop-in parity.
  - `backend/backend-cuda/` (Linux CUDA C++) — reference behavior and golden outputs.
  - `backend/backend-python/` (PyTorch) — uses FlashInfer; dev tooling uses Protobufs from `backend/proto/`.
- Cross-platform validation:
  - `cuda-protocol-tests/` and `metal-protocol-tests/`: CMake harnesses to generate CUDA artifacts (Linux) and compare on macOS.
  - CUDA artifacts live in `metal-protocol-tests/tests/artifacts/**`; Metal writes its outputs under `metal-protocol-tests/build/tests/artifacts/**`.

## Build and run (dev defaults)
- CLI + inferlets (macOS/Linux):
  - Install CLI: from repo root: `cd pie-cli && cargo install --path .`
  - Build inferlets: `cd example-apps && rustup target add wasm32-wasip2 && cargo build --target wasm32-wasip2 --release`
- Metal protocol tests (macOS):
  - VS Code task: “Build metal-protocol-tests” or “Run all Metal protocol tests (auto-select)”.
  - Manual: `cmake -S metal-protocol-tests -B metal-protocol-tests/build && cmake --build metal-protocol-tests/build --config Release --parallel` then `./metal-protocol-tests/scripts/test_all_ops.sh`.
- Metal backend + tests (macOS):
  - VS Code task: “Build metal backend and tests”.
  - Manual (per backend README): configure and build with CMake/Xcode toolchain.
- CUDA artifacts (for Metal comparison): generate on Linux with `cuda-protocol-tests/scripts/generate_cuda_artifacts.sh`, then compress/transfer using `scripts/artifacts_transfer.sh`. A `cuda_artifacts.tar.xz` may already exist at repo root; extract with the same script on macOS.

## Metal backend architecture and patterns
- File layout (`backend/backend-metal/`):
  - Kernels: `metal_*.metal`
  - ObjC++ wrappers: `metal_*_wrapper.mm` and interfaces `metal_*.hpp`
  - Common runtime: `metal_common.{mm,hpp}`, buffers/tensors: `metal_buffer.*`, `metal_tensor.*`, model layer wrappers: `metal_gemm_wrapper.*`, `metal_rmsnorm_wrapper.*`, `metal_rope_wrapper.*`, `metal_silu_and_mul_wrapper.*`, attention: `metal_batch_prefill_attention_wrapper.*`
- CUDA API compatibility: Metal exposes CUDA-like entry points for parity, e.g. `int metal_softmax_float(const float* in, float* out, int batch, int vocab, float temperature);` — return `0` on success, non-zero otherwise. Keep signatures/data layout identical to CUDA when adding ops.
- Types: `float32` primary, `bfloat16` supported. bfloat16 is stored via upper 16 bits of a `float` payload in tests; convert carefully at boundaries.
- Conventions:
  - Use `.mm` for ObjC++ bridging to Metal; ARC is enabled.
  - Favor numerically stable kernels (e.g., softmax uses max-subtraction).
  - Public wrappers perform shape/stride checks and enqueue MTLComputeCommands; avoid allocations in hot paths (reuse `MetalBuffer`).

## Tests and verification
- Metal backend test suites live under `backend/backend-metal/tests/` (unit, integration, dtype, performance, compatibility); see that README for commands and structure.
- Protocol harness (preferred for parity): `metal-protocol-tests` compares against CUDA golden references; precision auto-selects to match reference (see its README “Precision selection rule”).
- Quick checks (from `metal-protocol-tests/build`):
  - All ops: `bash ../scripts/test_all_ops.sh`
  - Individual: `./metal_protocol_tests --op softmax --case production --batch_size 2 --vocab_size 32000`
- Example tests: see `backend/backend-metal/tests/src/final_test_softmax.cpp` for patterns to initialize `MetalContext`, allocate tensors/buffers, and validate outputs (sum-to-1 for softmax, tolerance `1e-4` for large vocab matching CUDA).

## Adding or modifying an operation (Metal)
1. Implement kernel `metal_op.metal`; prefer threadgroup tiling + coalesced IO.
2. Add wrapper `metal_op_wrapper.mm` and interface `metal_op.hpp`; return `int` status, no exceptions across API boundary.
3. Mirror CUDA signature and tensor layout; keep dtype handling (fp32/bf16) symmetric.
4. Register in CMake and add tests:
   - Unit/integration under `backend/backend-metal/tests/src/{unit,integration}/test_metal_*_op.cpp/mm`.
   - Protocol test coverage via `metal-protocol-tests` op switch.
5. Validate vs CUDA artifacts; ensure tolerance matches existing ops (e.g., softmax 1e-4 for 32k vocab).

## Cross-component integration
- Engine ↔ backend: The CLI config selects device (Metal on macOS) and dtype; model config is `L4maConfig` (see `backend/backend-metal/src/metal_common.hpp`). Backends should not leak platform specifics through the public model API.
- Python backend: run `backend/backend-python/build_proto.sh` to regenerate protobufs; FlashInfer wheel must match the target GPU arch.
- CUDA backend: depends on ZeroMQ/CBOR/Zstd (see its README) and serves as numerical reference.

## Gotchas and tips
- If Metal tests fail with comparison errors, confirm CUDA artifacts are present and dtype in `meta.json` matches expectations.
- On macOS, install numpy for comparisons used by the test harness.
- Large vocab softmax is sensitive; ensure temperature scaling and max-subtraction are applied identically to CUDA.
- Prefer the provided VS Code tasks to build frequently:
  - Build: “Build metal backend and tests”, “Build metal-protocol-tests”
  - Run: “Run all Metal protocol tests (auto-select)”

Feedback: If any section is unclear or you need deeper details (e.g., exact CMake targets or adding a new op to the protocol harness), tell me which part you’re working on and I’ll refine this file.
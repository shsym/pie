# Pie Project Structure

This document provides a concise overview of the key directories and components in the Pie project for Agents to understand the architecture.

## `runtime`
**The Rust-based Main Runtime.**
*   **Path**: `runtime/`
*   **Language**: Rust
*   **Description**: This is the core "Engine" of Pie. It implements the high-performance logic including the WASM runtime (based on Wasmtime), request scheduling, and networking (ZeroMQ).
*   **Integration**: It exposes a Python interface via PyO3, allowing it to be controlled by the `server` layer.
*   **Key Dependencies**: `wasmtime`, `tokio`, `zeromq`, `pyo3`.

## `pie`
**Main Entrypoint and CLI (`pie`) + Inference Drivers.**
*   **Path**: `pie/`
*   **Language**: Python (`pie-server`) + Rust (PyO3 extension)
*   **Description**: The primary interface for the user. It wraps the `runtime` (Rust), includes the inference drivers, and provides the `pie` CLI.
*   **Subdirectories**:
    *   `src/pie/`: Server, config schema, driver registry (`drivers.py`), runtime façade
    *   `src/pie_cli/`: CLI commands (`pie serve`, `pie run`, `pie model`, `pie config`, `pie auth`, `pie doctor`)
    *   `src/pie_driver/`: Native driver — pie's own batched scheduler + model implementations under `model/` (llama3, qwen2/3, gemma2/3/4, mistral3, olmo3, gpt_oss)
    *   `src/pie_driver_vllm/`, `src/pie_driver_sgl/`, `src/pie_driver_dummy/`: Alternate drivers (delegate inference to vLLM / SGLang, or skip real compute)
    *   `src/pie_backend/`: Backend host glue (model registry imported by drivers)
    *   `src/pie_kernels/`: Unified kernel dispatch — routes to `pie_kernels.metal` on Apple Silicon and upstream `flashinfer` on CUDA; `pie_kernels.cuda/` hosts pie-owned CUDA/Triton kernels
*   **CLI**: Provides the `pie` command.
    *   `pie serve`: Starts the engine.
    *   `pie run`: Executes a one-shot inferlet.
    *   `pie model {list,download,remove}`: Manage HuggingFace cache.
    *   `pie config {init,show,set}`: Manage `~/.pie/config.toml`.
    *   `pie auth {add,remove,list}`: Manage authorized clients.
    *   `pie doctor`: Checks system health.

## `sdk`
**SDK for Writing Inferlets.**
*   **Path**: `sdk/`
*   **Description**: Contains libraries and tools for developers to write "Inferlets" (WASM programs that run on Pie).
*   **Subdirectories**:
    *   `rust/`, `python/`, `javascript/`: Client SDKs and Inferlet APIs.

### `sdk/tools/bakery`
**Inferlet Toolchain (`bakery`).**
*   **Path**: `sdk/tools/bakery/`
*   **Language**: Python (`pie-bakery`)
*   **Description**: The CLI tool for developing Inferlets.
    *   `bakery create`: Scaffolds new projects.
    *   `bakery build`: Compiles source (Rust/JS) to WASM.
    *   `bakery publish`: Publishes inferlets to the registry.

## `client`
**Client Libraries.**
*   **Path**: `client/`
*   **Description**: Contains client-side libraries for connecting to a serving Pie instance.
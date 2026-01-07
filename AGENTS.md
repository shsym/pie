# Pie Project Structure

This document provides a concise overview of the key directories and components in the Pie project for Agents to understand the architecture.

## `runtime`
**The Rust-based Main Runtime.**
*   **Path**: `runtime/`
*   **Language**: Rust
*   **Description**: This is the core "Engine" of Pie. It implements the high-performance logic including the WASM runtime (based on Wasmtime), request scheduling, and networking (ZeroMQ).
*   **Integration**: It exposes a Python interface via PyO3, allowing it to be controlled by the `server` layer.
*   **Key Dependencies**: `wasmtime`, `tokio`, `zeromq`, `pyo3`.

## `runtime-backend`
**Inference Backend (PyTorch/GPU).**
*   **Path**: `runtime-backend/`
*   **Language**: Python (`pie-backend`)
*   **Description**: Runs the actual machine learning models (LLMs). It handles model loading, KV caching, and forward passes using PyTorch.
*   **Execution**: It runs as a separate process launched by the Engine, communicating via ZeroMQ.
*   **Key Features**: CUDA/Metal support, FlashInfer integration.

## `server`
**Main Entrypoint and CLI (`pie`).**
*   **Path**: `server/`
*   **Language**: Python (`pie-server`)
*   **Description**: The primary interface for the user. It wraps the `runtime` (Rust) and orchestrates the `runtime-backend`.
*   **CLI**: Provides the `pie` command.
    *   `pie serve`: Starts the full engine and backend.
    *   `pie run`: Executes a one-shot inferlet.
    *   `pie model`: Manages downloaded models.
    *   `pie config`: Manages configuration.
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
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
**Main Entrypoint and CLI (`pie`) + Inference Backend.**
*   **Path**: `pie/`
*   **Language**: Python (`pie-server`) + Rust (PyO3 extension)
*   **Description**: The primary interface for the user. It wraps the `runtime` (Rust), includes the inference backend (`pie_worker`), and provides the `pie` CLI.
*   **Subdirectories**:
    *   `src/pie/`: Core engine management logic
    *   `src/pie_cli/`: CLI commands (`pie serve`, `pie run`, etc.)
    *   `src/pie_worker/`: Inference backend (was `pie-backend`) - handles model loading, KV caching, forward passes
    *   `src/flashinfer_metal/`: Metal GPU kernels (Apple Silicon)
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
# Backend Interface Refactoring Plan

## I. Objective: Generalize Backend for Multiple Model Support

The primary goal is to refactor the existing backend to support various language models beyond the current L4M/Llama implementation (e.g., Deepseek). This involves creating a new management layer that acts as a front-facing interface, dynamically managing model-specific backend instances and routing client requests accordingly.

## II. Core Components and Changes

### A. New Management Layer (Service and CLI Tool)

The management layer will consist of two main parts: a long-running **Management Service** and a **CLI Tool** to interact with this service. The service will be located at `backend/backend-management/management_service.py` (or similar) and the CLI tool at `backend/backend-management/management_cli.py` (or similar).

1.  **Management Service (`management_service.py`):**
    *   **Objective:** Act as a persistent, long-running daemon that manages available models, spawns/monitors actual backend instances, and handles initial client handshakes.
    *   **Responsibilities:**
        *   **Central IPC Listener:** Listens on a well-known IPC endpoint (e.g., `ipc:///tmp/symphony-ipc`) for incoming client connections and CLI commands.
        *   **Revised Handshake Protocol Handler:** (As described below) Handles initial client requests, determines the required model, and provides the endpoint for the specific model backend.
        *   **Backend Instance Management:** (As described below) Dynamically spawns, monitors, and terminates model-specific backend instances (e.g., `l4m_backend.py`, `deepseek_backend.py`). Maintains a registry of running model instances and their dedicated IPC endpoints.
        *   **Model Configuration Management:** Manages configurations for different models.
        *   **Communication with CLI Tool:** Listens for commands from the `main_cli.py` (e.g., to load/unload models, get status). This could be over the same main IPC endpoint or a dedicated internal one.

2.  **CLI Tool (`main_cli.py`):**
    *   **Objective:** Provide a command-line interface for users/administrators to interact with the Management Service. This tool will communicate with the Management Service.
    *   **Steps:**
        *   Implement CLI parsing (e.g., using `argparse` or `click`).
        *   Define commands to interact with the Management Service:
            *   `start-service [--daemonize]`: Checks if the service is running; if not, starts the `management_service.py`. The `--daemonize` option would run it in the background.
            *   `stop-service`: Signals the Management Service to shut down gracefully.
            *   `status`: Queries the Management Service for its status and a list of loaded models and their endpoints.
            *   `load-model <model_name_or_type> [--config <path_to_config>]`: Sends a request to the Management Service to load/start a specific model. The service then handles spawning the actual backend.
            *   `unload-model <model_name_or_type_or_endpoint>`: Sends a request to the Management Service to stop and unload a model.
            *   `(Future) download-model <model_name>`: Sends a request to the service.
            *   `(Future) update-model <model_name>`: Sends a request to the service.
        *   The CLI tool will connect to the Management Service's IPC endpoint to send these commands and receive responses.

3.  **Central IPC Listener (Managed by `management_service.py`):**
    *   **Objective:** Act as the initial contact point for all clients.
    *   **Steps:**
        *   The `management_service.py` listens on a well-known IPC endpoint (e.g., `ipc:///tmp/symphony-ipc`).
        *   Handles incoming client connections for model requests and commands from the `management_cli.py`.

4.  **Revised Handshake Protocol (Handled by `management_service.py`):**
    *   **Objective:** Enable clients to specify their desired model and for the management service to provide the correct endpoint.
    *   **Steps:**
        *   **Modify `handshake.proto`:**
            *   Add a field to `handshake.Request`, e.g., `string requested_model_type` (e.g., "llama", "deepseek", "l4m_v2") or `string requested_model_name` (e.g., "meta-llama/Llama-2-7b-chat-hf", "deepseek-ai/deepseek-coder-7b-instruct-v1.5").
            *   Modify `handshake.Response`:
                *   Add a field `string model_ipc_endpoint`.
                *   Add an error field, e.g., `string error_message` if the model is unavailable or fails to load.
                *   The `protocols` field might still be relevant for the specific model backend once connected.
        *   The `management_service.py` will parse this `requested_model_type`.

5.  **Backend Instance Management (Handled by `management_service.py`):**
    *   **Objective:** Dynamically spawn, monitor, and terminate model-specific backend instances.
    *   **Steps:**
        *   The `management_service.py` maintains a registry of available model types and their corresponding backend scripts (e.g., `l4m_backend.py`, `deepseek_backend.py`).
        *   On a client request for a model type (via handshake) or a `load-model` command from the CLI:
            *   Check if an instance for that model is already running and suitable.
            *   If not, spawn a new process for the model-specific backend.
            *   Assign a unique IPC endpoint for this new instance (e.g., `ipc:///tmp/symphony-model-<uuid>`).
            *   Pass this unique endpoint to the spawned backend process (e.g., via CLI argument).
        *   Monitor the health of spawned backend processes.
        *   Handle graceful shutdown and resource cleanup based on CLI commands or internal logic.

6.  **Dynamic Endpoint Allocation and Routing (Handled by `management_service.py`):**
    *   **Objective:** Provide clients with the specific IPC endpoint for their requested model.
    *   **Steps:**
        *   After a successful handshake (for a client) and (if necessary) backend instance creation, the `management_service.py` returns the unique IPC endpoint of the model backend to the client via the updated `handshake.Response`.
        *   The client then disconnects from the management service and connects directly to the provided model backend endpoint.

### B. Refactor Existing L4M Backend (Currently `backend/backend-flashinfer/main.py`)

1.  **Rename and Isolate:**
    *   **Objective:** Make the L4M backend a standalone, manageable component.
    *   **Steps:**
        *   Revise `backend/backend-flashinfer/main.py` to `backend/backend-flashinfer/l4m_backend.py` (or similar, e.g., `llama_backend.py`).
        *   This script will now be executed as a separate process by the new management layer.
        *   Unit tests for the L4M backend should be updated to reflect this change.
        *   Integration test from backend-management with this backend should be created.

2.  **Parameterize IPC Endpoint:**
    *   **Objective:** Allow the IPC endpoint to be dynamically assigned by the management layer.
    *   **Steps:**
        *   Modify `l4m_backend.py` to accept its IPC listening address as a command-line argument (e.g., `--ipc-endpoint ipc:///tmp/symphony-l4m-instance1`).
        *   Remove the hardcoded `endpoint = "ipc:///tmp/symphony-ipc"`.

3.  **Adapt Handshake (Minor):**
    *   **Objective:** Ensure the L4M backend can still perform its part of the communication, now directly with the client after the initial management handshake.
    *   **Steps:**
        *   The L4M backend will no longer handle the *initial* global handshake (that's the management layer's job).
        *   It will still need to handle its own protocol negotiation (e.g., "l4m", "l4m-vision", "ping") with the client once the client connects to its dedicated IPC endpoint. The existing logic for this can largely remain but will operate on its dedicated endpoint.

### C. Implement New Model Backends (e.g., `deepseek_backend.py`)

1.  **Create New Backend Script:**
    *   **Objective:** Provide a template/structure for adding support for new models.
    *   **Steps:**
        *   Create a new Python script (e.g., `backend/backend-flashinfer/deepseek_backend.py`).
        *   This script will be similar in structure to the refactored `l4m_backend.py`.

2.  **Model-Specific Logic:**
    *   **Objective:** Encapsulate all logic for loading, running, and interfacing with the Deepseek model.
    *   **Steps:**
        *   Implement model loading (e.g., from Hugging Face or local path).
        *   Implement request handling functions (similar to `handle_request` in the current L4M main) specific to Deepseek's capabilities and API.
        *   Define or reuse/adapt protobuf definitions (`.proto` files) if the communication protocol differs significantly from L4M. If new protos are needed, generate Python stubs.

3.  **IPC Communication:**
    *   **Objective:** Enable communication with clients over a dedicated IPC endpoint.
    *   **Steps:**
        *   Implement ZMQ ROUTER socket logic, similar to `l4m_backend.py`.
        *   Accept the IPC endpoint as a command-line argument.

### D. Update Client-Side Logic (e.g., Rust `engine/src/main.rs`)

1.  **Two-Stage Connection Process:**
    *   **Objective:** Adapt the client to the new management layer architecture.
    *   **Steps:**
        *   **Stage 1: Connect to Management Layer:**
            *   The client initially connects to the well-known management IPC endpoint (e.g., `ipc:///tmp/symphony-ipc`).
            *   It sends the updated `handshake.Request`, specifying the `requested_model_type`.
        *   **Stage 2: Connect to Model Backend:**
            *   The client receives the `handshake.Response` containing the specific `model_ipc_endpoint`.
            *   The client disconnects from the management layer.
            *   The client establishes a new connection to this `model_ipc_endpoint`.
            *   Subsequent communication happens directly with the model backend.

2.  **Modify `backend::ZmqBackend::bind` (or equivalent):**
    *   **Objective:** Allow the client to connect to dynamically provided endpoints.
    *   **Steps:**
        *   The function/method responsible for establishing the ZMQ connection will need to take the endpoint URI as a parameter, which it receives from the management service.

## III. Workflow Example

1.  User starts the **Management Layer** (`new_main.py`). It listens on `ipc:///tmp/symphony-ipc`.
2.  User (or another process) uses the CLI to load a model: `python new_main.py --load-model llama_7b`.
    *   The Management Layer spawns `l4m_backend.py --ipc-endpoint ipc:///tmp/symphony-llama-7b-1`.
3.  A **Client** (e.g., Rust engine) wants to use "llama_7b".
4.  Client connects to `ipc:///tmp/symphony-ipc` (Management Layer).
5.  Client sends `handshake.Request { requested_model_type: "llama_7b" }`.
6.  Management Layer checks its registry, finds the running `llama_7b` instance.
7.  Management Layer responds with `handshake.Response { model_ipc_endpoint: "ipc:///tmp/symphony-llama-7b-1", protocols: ["l4m", "ping"] }`.
8.  Client disconnects from Management Layer.
9.  Client connects to `ipc:///tmp/symphony-llama-7b-1` (L4M Backend).
10. Client and L4M Backend communicate directly using L4M protocols.

## IV. Future Considerations

*   **Resource Management:** Advanced allocation and monitoring of GPU/CPU resources per backend instance.
*   **Security:** Authentication and authorization for accessing the management layer and specific models.
*   **Scalability:** Strategies for handling many concurrent clients and backend instances.
*   **Configuration Management:** A more robust way to manage configurations for different models and backend instances.
*   **Service Discovery:** More advanced mechanisms if IPC paths become too numerous or complex.

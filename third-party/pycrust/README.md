# PyCrust

**PyCrust** ("Python + Rust") is a high-performance, low-latency RPC framework designed to bridge **Rust** control processes with **Python/PyTorch** workers. It prioritizes raw throughput and developer ergonomics by leveraging shared memory (iceoryx2) and MessagePack (fast binary serialization).

## Features

- **Sub-millisecond latency** for control messages
- **Zero-copy transport** via iceoryx2 shared memory
- **Async Rust client** (Tokio) with concurrent request handling
- **Typed Python API** with Pydantic validation
- **Clean decorator-based registration**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Rust Application (Tokio)                                   │
│  └── RpcClient::call<T, R>("method", args)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ iceoryx2 (Shared Memory)
                      │ MessagePack Serialization
┌─────────────────────▼───────────────────────────────────────┐
│  Python Worker                                               │
│  └── RpcEndpoint._dispatch(method, args)                    │
│      └── Pydantic Validation (optional)                     │
│      └── User Handler Function                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Build from source

```bash
cd /third-party/pycrust

# Build Rust crates
cargo build --release

# Build and install Python extension
cd crates/pycrust-worker
maturin develop --release

# Install Python SDK
cd ../../python
pip install -e ".[dev]"
```

## Usage

### Python Worker

```python
from pycrust import RpcEndpoint
from pydantic import BaseModel

# Define request models with Pydantic
class AddArgs(BaseModel):
    a: int
    b: int

# Create endpoint
endpoint = RpcEndpoint("calculator")

# Register methods
@endpoint.register()
def ping() -> str:
    return "pong"

@endpoint.register(request_model=AddArgs)
def add(a: int, b: int) -> int:
    return a + b

# Start listening
endpoint.listen()
```

### Rust Client

```rust
use pycrust_client::RpcClient;
use serde::{Serialize, Deserialize};

#[derive(Serialize)]
struct AddArgs { a: i32, b: i32 }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = RpcClient::connect("calculator").await?;

    // Call remote method
    let result: i32 = client.call("add", &AddArgs { a: 10, b: 20 }).await?;
    println!("Result: {}", result);  // Output: 30

    // With timeout
    let pong: String = client
        .call_with_timeout("ping", &(), std::time::Duration::from_secs(5))
        .await?;

    client.close().await;
    Ok(())
}
```

## Protocol

### Request Format
```
(u64, String, Vec<u8>)
  │     │       └── Payload (MsgPack encoded arguments)
  │     └── Method name
  └── Request ID (correlation ID)
```

### Response Format
```
(u64, u8, Vec<u8>)
  │    │     └── Payload (MsgPack encoded result or error)
  │    └── Status code (0=OK, 1+=Error)
  └── Request ID (matches request)
```

## Running Tests

```bash
# Python tests
cd python
pytest tests/ -v

# Rust tests
cargo test
```

## Running Examples

```bash
# Terminal 1: Start Python worker
python examples/python_worker.py

# Terminal 2: Run Rust client
cargo run --example rust_client -p pycrust-client
```

## Running Benchmarks

```bash
# Python SDK benchmark (dispatch only)
python examples/benchmark.py
```

## Project Structure

```
pycrust/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── pycrust-client/           # Rust client library
│   │   └── src/
│   │       ├── client.rs         # RpcClient implementation
│   │       ├── transport.rs      # iceoryx2 transport
│   │       ├── protocol.rs       # Message types
│   │       └── error.rs          # Error types
│   │
│   └── pycrust-worker/           # PyO3 extension
│       └── src/
│           ├── worker.rs         # Worker loop
│           ├── convert.rs        # MsgPack ↔ Python
│           └── lib.rs            # PyO3 module
│
├── python/
│   └── src/pycrust/
│       ├── endpoint.py           # RpcEndpoint class
│       ├── decorators.py         # @rpc_method
│       └── validation.py         # Pydantic utilities
│
└── examples/
    ├── python_worker.py          # Example worker
    ├── rust_client.rs            # Example client
    └── benchmark.py              # Performance benchmark
```

## Dependencies

### Rust
- iceoryx2 (shared memory IPC)
- tokio (async runtime)
- pyo3 (Python bindings)
- rmp-serde (MessagePack)
- dashmap (concurrent map)

### Python
- pydantic (validation)
- msgpack (serialization)

## License

Apache-2.0

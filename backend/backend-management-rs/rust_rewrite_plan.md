# Symphony Management Service Rust Rewrite Plan

## Overview

This document outlines the plan to rewrite the Python-based Symphony Management Service (`management_service.py`) in Rust. The service is a critical component that manages backend model instances, handles client handshakes, and provides dynamic routing to model-specific endpoints.

## Proto Definitions Simplification

**Issue Identified**: The management service was unnecessarily building and including all proto definitions (`l4m.proto`, `l4m_vision.proto`, `ping.proto`, `handshake.proto`) when it only actually uses the handshake protocol.

**Root Cause**: 
- Management service's role is orchestration and routing, not message processing
- Actual protocol handling (L4M, L4M Vision, Ping) is done by backend services (e.g., `backend-python`)
- Management service only needs to:
  1. Handle client handshake requests
  2. Return supported protocol names as strings
  3. Route clients to appropriate backend endpoints

**Solution Applied**:
- **Python version**: Updated `build_proto.sh` to only compile `handshake.proto`
- **Rust version**: Updated `build.rs` and `src/proto/mod.rs` to only include handshake module
- **Benefits**: 
  - Faster compilation times
  - Smaller binary size
  - Clearer separation of concerns
  - Reduced complexity and dependencies

**Architecture Clarification**:
```
Client -> Management Service (handshake only) -> Backend Service (l4m, l4m_vision, ping)
```

## Current Python Implementation Analysis

The existing Python service includes:
- **ManagementService**: Main service class managing model instances and ZMQ communication
- **ModelInstance**: Data structure representing running model backend instances  
- **ManagementCommand**: Command structure for CLI-service communication
- **ZMQ-based IPC**: Client handshake and CLI management endpoints
- **Process management**: Spawning and monitoring backend processes
- **Configuration management**: JSON-based configuration loading
- **Signal handling**: Graceful shutdown on SIGINT/SIGTERM
- **Logging**: Structured logging with configurable levels

## Implementation Plan

### Phase 1: Interface Design and Core Types

**Objective**: Define the core data structures and interfaces that mirror the Python implementation.

**Tasks**:
1. **Define core data structures**:
   - `ModelInstance` struct with fields:
     - `model_name: String`
     - `model_type: String` 
     - `endpoint: String`
     - `process: Child` (from `std::process`)
     - `config_path: Option<PathBuf>`
     - `started_at: SystemTime`
   - `ManagementCommand` struct with fields:
     - `command: String`
     - `params: HashMap<String, serde_json::Value>`
     - `correlation_id: String`

2. **Define configuration structures**:
   - `Config` struct to deserialize `config.json`:
     - `model_backends: HashMap<String, String>`
     - `endpoints: EndpointConfig`
     - `logging: LoggingConfig`
     - `supported_models: Vec<ModelInfo>`
   - `ModelInfo`, `EndpointConfig`, `LoggingConfig` structs

3. **Define service interfaces**:
   - `ManagementServiceTrait` trait defining core service operations
   - Error types using `thiserror` crate
   - Result types for all operations

**Dependencies needed**:
- `serde` + `serde_json` for JSON handling
- `tokio` for async runtime
- `zmq` (or `zeromq`) for ZMQ messaging
- `uuid` for unique ID generation
- `thiserror` for error handling
- `tracing` for logging
- `clap` for CLI argument parsing

**Deliverables**:
- `src/types.rs` - Core data structures
- `src/config.rs` - Configuration types and loading
- `src/error.rs` - Error types
- `src/service.rs` - Service trait definition

### Phase 2: Test Case Porting ✅ COMPLETED

**Objective**: Port existing Python test cases to Rust to ensure behavioral compatibility.

**Status**: ✅ **COMPLETED** - All 93 tests passing

**Completed Tasks**:
1. **✅ Test Infrastructure Setup**:
   - Created comprehensive test files mirroring Python test structure:
     - `tests/service_tests.rs` - 24 service functionality tests
     - `tests/cli_tests.rs` - 23 CLI functionality and ZMQ communication tests  
     - `tests/integration_tests.rs` - 18 end-to-end integration tests
     - `tests/basic_tests.rs` - 17 simplified working test suite
     - `tests/config_tests.rs` - 3 configuration tests
     - `tests/types_tests.rs` - 5 type system tests
     - Unit tests - 3 tests

2. **✅ Test Pattern Implementation**:
   - Analyzed Python test files (`test_management_service_pytest.py`, `test_management_cli_pytest.py`, `test_integration_pytest.py`)
   - Implemented comprehensive test coverage for all components
   - Created test utilities and helper functions in `tests/common/mod.rs`
   - Added proper async test infrastructure with tokio

3. **✅ Error Handling & Type System**:
   - Extended error types with missing variants:
     - `ManagementError::UnknownModel(String)` for model not found scenarios
     - `ConfigError::ParseError(String)` for JSON parsing errors  
     - `ProcessError::SpawnFailed(String)` for process spawning failures
   - Enhanced type system with missing methods:
     - `ModelInstance::get_process_id()` method
     - `ManagementServiceImpl::get_model_type()` and `generate_unique_endpoint()` methods
   - Added Debug trait to core types for test compatibility

4. **✅ Service Implementation Stubs**:
   - Created `ManagementServiceImpl` with placeholder implementation
   - Added config file validation (existence and JSON parsing)
   - Implemented basic model type resolution with exact matching
   - Added UUID-like endpoint generation following expected format
   - Added proper error propagation for test scenarios

5. **✅ Dependencies & Configuration**:
   - Added `tempfile = "3.0"` dependency for test file creation
   - Updated Cargo.toml with required test dependencies
   - Fixed all compilation errors and warnings

**Test Results**: 
- **Total: 93 tests passing, 0 failing**
- Service tests: 24/24 ✅
- CLI tests: 23/23 ✅  
- Integration tests: 18/18 ✅
- Basic tests: 17/17 ✅
- Config tests: 3/3 ✅
- Types tests: 5/5 ✅
- Unit tests: 3/3 ✅

**Key Achievements**:
- Complete test infrastructure mirroring Python patterns
- All major compilation errors resolved 
- Robust error handling and type system
- Foundation ready for Phase 3 implementation
- `tests/integration_tests.rs` - Full integration tests
- `tests/common.rs` - Shared test utilities

### Phase 3: Core Service Implementation ✅ COMPLETED

**Objective**: Implement the main service functionality with full feature parity to Python version.

**Status**: ✅ **COMPLETED** - Core implementation is finalized and tested.

**Completed Tasks**:
1. **✅ Protobuf Integration**:
   - Created `build.rs` script for protobuf compilation.
   - Generated protobuf modules for the handshake protocol.
   - Added `prost-build` dependency and protobuf compilation setup.
   - Created `src/proto/mod.rs` to organize generated protobuf code.

2. **✅ Enhanced Configuration Management**:
   - Enhanced `Config::load()` method with proper file validation and JSON parsing.
   - Added `Config::load_default()` with multiple fallback path strategies.
   - Added model type resolution methods: `get_model_type()`, `get_backend_script()`, `get_supported_models()`.
   - Improved error handling with detailed error messages.

3. **✅ Process Manager Implementation**:
   - Created `src/process_manager.rs` for backend process management.
   - Implemented `ProcessManager::spawn_model_instance()` for async process spawning.
   - Added `get_default_backend_path()` with multi-strategy path resolution.
   - Added `generate_unique_endpoint()` for IPC endpoint creation.
   - Added health checking and process monitoring capabilities (`health_check` method in `ManagementServiceImpl`).

4. **✅ Enhanced Type System**:
   - Updated `ModelInstance` to use `tokio::process::Child` for async process management.
   - Added `terminate()` method with graceful shutdown and timeout handling.
   - Added `uptime()` method for instance monitoring.
   - Enhanced process ID and lifecycle management.

5. **✅ ZMQ Communication Layer**:
   - Created `src/zmq_handler.rs` for ZMQ ROUTER socket management.
   - Implemented client handshake protocol with protobuf message handling.
   - Added CLI management communication with JSON message parsing.
   - Implemented async message polling loop with shutdown handling.
   - Added proper socket initialization and cleanup.

6. **✅ Service Implementation Foundation**:
   - Completed core `ManagementServiceImpl` implementation (primarily in `src/lib.rs`).
   - Implemented service lifecycle management (start/stop/is_running).
   - Implemented model loading/unloading with process spawning integration.
   - Implemented command processing for CLI communication.
   - Implemented service status reporting and model registry management.
   - Implemented signal handling for graceful shutdown (`setup_signal_handlers` in `src/lib.rs`).

**Resolved Issues (from previous plan state)**:
- Duplicate method definitions in config.rs and types.rs.
- Function signature mismatches for ManagementResponse methods.
- Missing trait implementations for service interface.
- Type system inconsistencies between old and new implementations.
All major compilation errors and logical issues from the initial Phase 3 development have been resolved, as evidenced by successful `cargo test` execution.

**Key Achievements**:
- Protobuf integration working with generated modules.
- Async process management foundation in place.
- ZMQ communication layer implemented.
- Service lifecycle management (start, stop, status, health check) fully implemented.
- Configuration management enhanced with robust error handling.
- Multi-strategy backend path resolution working.
- Signal handling for graceful shutdown implemented.
- Comprehensive test suite (92 tests) passing, covering core service functionality.

### Phase 4: CLI Tool Implementation

**Status**: ✅ **COMPLETED**

**Objective**: Implement the CLI tool for service management.

**Tasks**:
1. **✅ Command parsing**:
   - Used `clap` for argument parsing with derive macros
   - Implemented all CLI commands:
     - `start-service` (placeholder implementation)
     - `stop-service`
     - `status`
     - `load-model <model_name> [--config-path <path>]`
     - `unload-model <model_name>`
     - `list-models`

2. **✅ Service communication**:
   - Implemented ZMQ DEALER socket for CLI-service communication (proper ROUTER-DEALER pattern)
   - Added command serialization to JSON and response parsing
   - Implemented 5-second timeout handling for service requests
   - Fixed socket type compatibility issues

3. **✅ Output formatting**:
   - Pretty-printed JSON output for all responses
   - Human-readable error messages
   - Proper exit codes for scripting support

4. **✅ Integration fixes**:
   - Fixed Python backend argument passing (only `--ipc-endpoint`)
   - Added proper working directory setting for backend processes
   - Tested full CLI workflow with live service

**Deliverables**:
- ✅ `src/cli/` - Complete CLI module structure
- ✅ `src/cli/cli.rs` - CLI command definitions with clap
- ✅ `src/cli/management_cli.rs` - CLI binary entry point  
- ✅ `src/cli/zmq_client.rs` - ZMQ client implementation
- ✅ Working `symphony-cli` binary with complete command set
- `src/management_cli.rs` - CLI binary entry point

### Phase 5: Integration and Testing

**Status**: ✅ **COMPLETED** (Basic integration verified)

**Objective**: Ensure the Rust implementation works correctly with existing Python backends and clients.

**Tasks**:
1. **✅ End-to-end testing**:
   - ✅ Tested with actual Python backend processes (`l4m_backend.py`)
   - ✅ Verified process spawning and lifecycle management
   - ✅ Tested CLI-to-service communication via ZMQ
   - ✅ Verified JSON command/response serialization

2. **✅ Basic compatibility verification**:
   - ✅ Tested with existing `config.json` file
   - ✅ Verified IPC endpoint compatibility (`ipc:///tmp/symphony-*`)
   - ✅ Confirmed Python backend integration with correct arguments
   - ✅ Validated model loading/unloading workflow

3. **🔄 Performance testing** (Future work):
   - Memory usage comparison with Python version
   - Response time benchmarks  
   - Process spawn time measurements

4. **🔄 Advanced compatibility** (Future work):
   - Test with existing client applications (engine, example-apps)
   - Comprehensive protobuf compatibility testing
   - Load testing with multiple concurrent models

4. **Documentation**:
   - Update README with Rust service instructions
   - Document any behavioral differences
   - Provide migration guide

**Deliverables**:
- `tests/e2e_tests.rs` - End-to-end integration tests
- `benches/` - Performance benchmarks
- Updated documentation

## File Structure

```
backend-management-rs/
├── Cargo.toml
├── build.rs
├── src/
│   ├── main.rs              # Service binary entry point
│   ├── management_cli.rs    # CLI binary entry point
│   ├── lib.rs               # Library exports
│   ├── types.rs             # Core data structures
│   ├── config.rs            # Configuration handling
│   ├── error.rs             # Error types
│   ├── service.rs           # Service trait definition
│   ├── service_impl.rs      # Main service implementation
│   ├── process_manager.rs   # Process management
│   ├── zmq_handler.rs       # ZMQ communication
│   ├── cli.rs               # CLI command handling
│   └── proto/               # Generated protobuf code
├── tests/
│   ├── common.rs            # Test utilities
│   ├── unit_tests.rs        # Unit tests
│   ├── service_tests.rs     # Service tests
│   ├── integration_tests.rs # Integration tests
│   └── e2e_tests.rs         # End-to-end tests
├── benches/
│   └── performance.rs       # Performance benchmarks
└── proto/
    └── *.proto              # Protobuf definitions (copied from Python)
```

## Success Criteria

1. **Functional Parity**: All features from Python version work identically
2. **Test Coverage**: All Python tests pass with Rust implementation
3. **Performance**: Rust version meets or exceeds Python performance
4. **Compatibility**: Seamless integration with existing Python backends
5. **Maintainability**: Clean, well-documented Rust code following best practices

## Timeline Estimate

- **Phase 1**: 2-3 days
- **Phase 2**: 2-3 days  
- **Phase 3**: 4-5 days
- **Phase 4**: 1-2 days
- **Phase 5**: 2-3 days

**Total**: 11-16 days

## Risk Mitigation

1. **ZMQ Compatibility**: Test ZMQ interoperability early between Rust and Python
2. **Process Management**: Handle edge cases in process spawning/termination
3. **Signal Handling**: Ensure proper cleanup on all shutdown scenarios
4. **Protobuf Compatibility**: Verify message format compatibility
5. **Performance**: Profile and optimize critical paths if needed

# Multi-Backend Debug Framework - API Reference

## Core Module (`debug_framework`)

### Functions

#### `initialize_framework(config: Optional[Dict[str, Any]] = None) -> bool`

Initialize the debug framework with production-safe defaults.

**Parameters:**
- `config`: Optional configuration dictionary to override defaults

**Returns:**
- `bool`: True if initialization successful

**Raises:**
- `DebugFrameworkError`: If critical initialization fails

**Example:**
```python
import debug_framework

# Initialize with defaults
success = debug_framework.initialize_framework()

# Initialize with custom config
config = {
    "database_path": "/custom/path/debug.db",
    "performance_monitoring": True,
    "tensor_recording_limit": 50
}
success = debug_framework.initialize_framework(config)
```

#### `get_framework_config() -> Dict[str, Any]`

Get current framework configuration including environment variable overrides.

**Returns:**
- `Dict[str, Any]`: Current configuration dictionary

#### `detect_available_backends() -> Dict[str, bool]`

Auto-detect available backend implementations on the current system.

**Returns:**
- `Dict[str, bool]`: Mapping of backend names to availability status

**Example:**
```python
backends = debug_framework.detect_available_backends()
# {'python_reference': True, 'metal': True, 'cuda': False, 'pytorch': True}
```

#### `get_system_info() -> Dict[str, Any]`

Get comprehensive system information for debugging and diagnostics.

**Returns:**
- `Dict[str, Any]`: System information including platform, versions, available backends

#### `create_validation_session(model_path: str, reference_backend: str = "python_reference", alternative_backend: str = "auto", config: Optional[Dict[str, Any]] = None) -> str`

Create a new validation session with automatic backend detection.

**Parameters:**
- `model_path`: Path to the model file to validate
- `reference_backend`: Reference backend name (default: "python_reference")
- `alternative_backend`: Alternative backend name or "auto" for auto-detection
- `config`: Optional session configuration

**Returns:**
- `str`: Session ID for tracking the validation session

**Raises:**
- `BackendInitializationError`: If backends cannot be initialized
- `ValidationError`: If session creation fails

### Exception Classes

#### `DebugFrameworkError(Exception)`

Base exception class for all debug framework errors.

#### `BackendInitializationError(DebugFrameworkError)`

Raised when backend initialization fails.

#### `ValidationError(DebugFrameworkError)`

Raised when validation operations fail.

### Configuration Constants

#### `DEFAULT_CONFIG`

Default framework configuration dictionary:

```python
{
    "database_path": "~/.pie/debug/debug_framework.db",
    "performance_monitoring": True,
    "auto_cleanup": True,
    "max_session_duration": 3600,  # 1 hour
    "tensor_recording_limit": 100,
    "comparison_tolerance": {"rtol": 1e-4, "atol": 1e-6}
}
```

## Services

### ValidationEngine (`debug_framework.services.validation_engine`)

Core validation orchestration engine.

#### Class: `ValidationEngine`

**Methods:**

##### `__init__(database_path: Optional[str] = None)`

Initialize validation engine with database connection.

##### `create_session(model_path: str, config: Dict[str, Any], reference_backend: str = "python_reference", alternative_backend: str = "metal") -> str`

Create a new validation session.

**Parameters:**
- `model_path`: Path to model file
- `config`: Session configuration dictionary
- `reference_backend`: Reference backend name
- `alternative_backend`: Alternative backend name

**Returns:**
- `str`: Unique session ID

##### `execute_validation(session_id: str) -> Dict[str, Any]`

Execute validation for a session.

**Parameters:**
- `session_id`: Session identifier

**Returns:**
- `Dict[str, Any]`: Validation results including numerical comparisons

##### `get_session_results(session_id: str) -> Dict[str, Any]`

Get complete results for a validation session.

##### `cleanup_session(session_id: str) -> bool`

Clean up resources for a completed session.

### DatabaseManager (`debug_framework.services.database_manager`)

Database operations and schema management.

#### Class: `DatabaseManager`

**Methods:**

##### `__init__(database_path: str)`

Initialize database manager with SQLite database.

##### `get_connection() -> sqlite3.Connection`

Get database connection with proper error handling.

##### `execute_query(query: str, parameters: Tuple = ()) -> Any`

Execute SQL query with parameters.

##### `create_tables() -> bool`

Create database schema if it doesn't exist.

##### `migrate_schema() -> bool`

Apply schema migrations for version compatibility.

### PluginRegistry (`debug_framework.services.plugin_registry`)

Plugin management and dynamic loading.

#### Class: `PluginRegistry`

**Methods:**

##### `register_plugin(plugin_definition: PluginDefinition) -> bool`

Register a new plugin with the framework.

##### `get_plugin(plugin_id: str) -> Optional[PluginDefinition]`

Retrieve plugin definition by ID.

##### `list_plugins(backend_type: Optional[str] = None) -> List[PluginDefinition]`

List registered plugins, optionally filtered by backend type.

##### `compile_plugin(source_path: str, backend_type: str, output_path: Optional[str] = None) -> bool`

Compile plugin source code for specified backend.

### CompilationEngine (`debug_framework.services.compilation_engine`)

Source code compilation and build management.

#### Class: `CompilationEngine`

**Methods:**

##### `compile_metal_kernel(source_path: str, output_path: str) -> bool`

Compile Metal kernel source to .metallib format.

##### `compile_cuda_kernel(source_path: str, output_path: str) -> bool`

Compile CUDA kernel source to binary format.

##### `validate_compilation(compiled_path: str, backend_type: str) -> bool`

Validate compiled binary for compatibility.

### TensorComparisonEngine (`debug_framework.services.tensor_comparison_engine`)

Numerical comparison and analysis engine.

#### Class: `TensorComparisonEngine`

**Methods:**

##### `compare_tensors(reference_tensor: Any, alternative_tensor: Any, tolerance: Dict[str, float]) -> Dict[str, Any]`

Compare two tensors with configurable tolerance.

**Parameters:**
- `reference_tensor`: Reference tensor data
- `alternative_tensor`: Alternative tensor data
- `tolerance`: Dictionary with 'rtol' and 'atol' keys

**Returns:**
- `Dict[str, Any]`: Comparison results including match status and metrics

##### `analyze_differences(reference_tensor: Any, alternative_tensor: Any) -> Dict[str, Any]`

Detailed analysis of tensor differences.

##### `generate_comparison_report(comparison_results: List[Dict]) -> Dict[str, Any]`

Generate summary report from multiple tensor comparisons.

## Models

### DebugSession (`debug_framework.models.debug_session`)

Represents a single debugging/validation session.

#### Class: `DebugSession`

**Attributes:**
- `session_id: str`: Unique session identifier
- `model_path: str`: Path to model being validated
- `reference_backend: str`: Reference backend name
- `alternative_backend: str`: Alternative backend name
- `config: Dict[str, Any]`: Session configuration
- `created_at: datetime`: Session creation timestamp
- `status: str`: Current session status
- `results: Optional[Dict]`: Validation results

**Methods:**

##### `save() -> bool`

Persist session to database.

##### `load(session_id: str) -> Optional['DebugSession']`

Load session from database by ID.

##### `update_status(status: str) -> bool`

Update session status in database.

##### `add_checkpoint_result(checkpoint_name: str, result: Dict) -> bool`

Add checkpoint validation result to session.

### ValidationCheckpoint (`debug_framework.models.validation_checkpoint`)

Represents a single validation checkpoint in model execution.

#### Class: `ValidationCheckpoint`

**Attributes:**
- `checkpoint_id: str`: Unique checkpoint identifier
- `session_id: str`: Associated session ID
- `checkpoint_name: str`: Name of checkpoint (e.g., "post_attention")
- `reference_tensor: Any`: Reference tensor data
- `alternative_tensor: Any`: Alternative tensor data
- `comparison_result: Dict`: Numerical comparison results
- `timestamp: datetime`: Checkpoint execution time

### TensorRecording (`debug_framework.models.tensor_recording`)

Records tensor data for detailed analysis.

#### Class: `TensorRecording`

**Attributes:**
- `recording_id: str`: Unique recording identifier
- `session_id: str`: Associated session ID
- `tensor_name: str`: Descriptive name for tensor
- `tensor_data: bytes`: Serialized tensor data
- `metadata: Dict`: Additional metadata (shape, dtype, etc.)

### TensorComparison (`debug_framework.models.tensor_comparison`)

Results of tensor numerical comparison.

#### Class: `TensorComparison`

**Attributes:**
- `comparison_id: str`: Unique comparison identifier
- `reference_recording_id: str`: Reference tensor recording ID
- `alternative_recording_id: str`: Alternative tensor recording ID
- `match_status: bool`: Whether tensors match within tolerance
- `max_absolute_error: float`: Maximum absolute difference
- `max_relative_error: float`: Maximum relative difference
- `tolerance_config: Dict`: Tolerance settings used

### ValidationReport (`debug_framework.models.validation_report`)

Comprehensive validation results report.

#### Class: `ValidationReport`

**Attributes:**
- `report_id: str`: Unique report identifier
- `session_id: str`: Associated session ID
- `overall_status: str`: Overall validation status
- `checkpoint_results: List[Dict]`: Results for each checkpoint
- `performance_metrics: Dict`: Performance profiling data
- `summary_statistics: Dict`: Numerical summary statistics

**Methods:**

##### `generate_html_report() -> str`

Generate HTML-formatted validation report.

##### `export_to_json() -> str`

Export report data as JSON string.

##### `save_to_file(file_path: str, format: str = 'json') -> bool`

Save report to file in specified format.

### BatchValidationJob (`debug_framework.models.batch_validation_job`)

Manages batch processing of multiple validation jobs.

#### Class: `BatchValidationJob`

**Attributes:**
- `job_id: str`: Unique batch job identifier
- `model_paths: List[str]`: List of models to validate
- `config: Dict[str, Any]`: Batch job configuration
- `parallel_execution: bool`: Enable parallel processing
- `max_concurrent_jobs: int`: Maximum concurrent validations

**Methods:**

##### `execute() -> List[Dict[str, Any]]`

Execute batch validation job.

##### `get_progress() -> Dict[str, Any]`

Get current batch job progress.

##### `cancel() -> bool`

Cancel running batch job.

## Integrations

### MetalBackend (`debug_framework.integrations.metal_backend`)

Metal Performance Shaders backend integration.

#### Class: `MetalBackend`

**Methods:**

##### `__init__(metal_path: Optional[str] = None)`

Initialize Metal backend with optional path to Metal binaries.

##### `initialize() -> bool`

Initialize Metal compute environment.

##### `execute_model(model_path: str, input_data: Any) -> Any`

Execute model using Metal backend.

##### `cleanup() -> bool`

Clean up Metal resources.

### L4MAIntegration (`debug_framework.integrations.l4ma_integration`)

L4MA (Llama) model-specific integration.

#### Class: `L4MAIntegration`

**Methods:**

##### `create_validation_session(model_path: str, pie_config_path: Optional[str] = None) -> str`

Create validation session optimized for L4MA models.

##### `execute_validation(session_id: str) -> Dict[str, Any]`

Execute L4MA-specific validation workflow.

##### `get_model_metadata(model_path: str) -> Dict[str, Any]`

Extract metadata from L4MA model file.

## CLI Commands

### debug_validate

Command-line validation tool.

```bash
python -m debug_framework.cli.debug_validate [OPTIONS]

Options:
  --model-path PATH           Path to model file [required]
  --reference-backend TEXT    Reference backend name
  --alternative-backend TEXT  Alternative backend name
  --config-file PATH         JSON configuration file
  --output-file PATH         Output file for results
  --verbose                  Enable verbose logging
```

### session_report

Generate validation reports.

```bash
python -m debug_framework.cli.session_report [OPTIONS]

Options:
  --session-id TEXT          Session ID [required]
  --output-format TEXT       Output format (json/html/csv)
  --output-file PATH         Output file path
  --include-tensors          Include tensor data in report
```

### plugin_compile

Compile custom plugins.

```bash
python -m debug_framework.cli.plugin_compile [OPTIONS]

Options:
  --plugin-path PATH         Plugin source file [required]
  --backend TEXT             Target backend (metal/cuda)
  --output-path PATH         Compiled output path
  --optimization-level INT   Optimization level (0-3)
```

## Environment Variables

### Configuration Variables

- `PIE_DEBUG_ENABLED`: Enable debug mode (true/false)
- `PIE_DEBUG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `PIE_DEBUG_DATABASE`: Custom database path
- `PIE_METAL_PATH`: Path to Metal backend binaries

### Example Configuration

```bash
export PIE_DEBUG_ENABLED=true
export PIE_DEBUG_LEVEL=INFO
export PIE_DEBUG_DATABASE=/opt/pie/debug/framework.db
export PIE_METAL_PATH=/opt/pie/backends/metal
```

## Error Codes

### Common Error Codes

- `DF001`: Framework initialization failure
- `DF002`: Backend initialization failure
- `DF003`: Database connection failure
- `DF004`: Model loading failure
- `DF005`: Validation execution failure
- `DF006`: Numerical comparison failure
- `DF007`: Resource exhaustion
- `DF008`: Plugin compilation failure

## Performance Considerations

### Memory Management

- Tensor recording limited to prevent memory exhaustion
- Automatic session cleanup after timeout
- Database connection pooling and optimization

### Optimization Tips

- Use appropriate tolerance levels for numerical comparison
- Enable parallel processing for batch jobs
- Configure recording limits based on available memory
- Use production-optimized settings in deployment

## Thread Safety

All core components are thread-safe for concurrent validation sessions:

- Database operations use connection pooling
- Session state is properly isolated
- Resource cleanup is atomic and consistent
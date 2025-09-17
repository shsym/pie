# Multi-Backend Debug Framework - User Guide

## Overview

The Multi-Backend Debug Framework is a production-ready validation system for ML model implementations across different backends (Metal, CUDA, PyTorch, etc.). It provides comprehensive numerical comparison, performance profiling, and automated testing capabilities.

## Key Features

- **Multi-Backend Support**: Validate implementations across Metal, CUDA, PyTorch, and reference Python backends
- **Numerical Validation**: Comprehensive tensor comparison with configurable tolerance levels
- **Performance Profiling**: Built-in performance monitoring and bottleneck identification
- **Automated Testing**: Batch validation jobs and continuous integration support
- **Production-Ready**: Thread-safe, resource-managed, and optimized for production environments

## Quick Start

### 1. Installation and Setup

```bash
# Navigate to backend-python directory
cd backend/backend-python

# Install dependencies with debug framework extras
uv sync --extra debug
```

### 2. Basic Usage

```python
import debug_framework

# Initialize the framework
debug_framework.initialize_framework()

# Create a validation session
session_id = debug_framework.create_validation_session(
    model_path="/path/to/model.zt",
    reference_backend="python_reference",
    alternative_backend="auto"  # Auto-detects best available backend
)

print(f"Created validation session: {session_id}")
```

### 3. Environment Configuration

Set environment variables for production deployment:

```bash
export PIE_DEBUG_ENABLED=true
export PIE_DEBUG_LEVEL=INFO
export PIE_DEBUG_DATABASE=/path/to/debug.db
export PIE_METAL_PATH=/path/to/metal/backend
```

## Core Concepts

### Debug Sessions

A debug session represents a single model validation workflow:

- **Session ID**: Unique identifier for tracking
- **Backend Configurations**: Reference and alternative backend settings
- **Validation Checkpoints**: Configurable comparison points in model execution
- **Tensor Recording**: Optional tensor data capture for detailed analysis

### Validation Checkpoints

Checkpoints define where numerical comparisons occur during model execution:

- `post_embedding`: After token embedding layer
- `post_attention`: After attention mechanism
- `post_mlp`: After MLP/feedforward layers
- `post_processing`: After final output processing

### Backend Integration

The framework supports multiple backend types:

- **Python Reference**: Pure Python implementation (always available)
- **Metal**: macOS GPU acceleration via Metal Performance Shaders
- **CUDA**: NVIDIA GPU acceleration
- **PyTorch**: PyTorch-based implementation

## Advanced Usage

### Custom Session Configuration

```python
session_config = {
    "enabled_checkpoints": ["post_attention", "post_mlp"],
    "precision_thresholds": {
        "rtol": 1e-4,  # Relative tolerance
        "atol": 1e-6   # Absolute tolerance
    },
    "tensor_recording_enabled": True,
    "performance_profiling_enabled": True
}

session_id = debug_framework.create_validation_session(
    model_path="/path/to/model.zt",
    config=session_config
)
```

### Manual Backend Selection

```python
# Specify exact backends to compare
session_id = debug_framework.create_validation_session(
    model_path="/path/to/model.zt",
    reference_backend="python_reference",
    alternative_backend="metal"
)
```

### System Information and Diagnostics

```python
# Check available backends
backends = debug_framework.detect_available_backends()
print(f"Available backends: {backends}")

# Get comprehensive system info
system_info = debug_framework.get_system_info()
print(f"System: {system_info}")
```

## Validation Engine

### Creating Validation Jobs

```python
from debug_framework.services.validation_engine import ValidationEngine

validation_engine = ValidationEngine()

# Create a new validation session
session_id = validation_engine.create_session(
    model_path="/path/to/model.zt",
    config={
        "enabled_checkpoints": ["post_attention", "post_mlp"],
        "precision_thresholds": {"rtol": 1e-4, "atol": 1e-6}
    }
)

# Execute validation
results = validation_engine.execute_validation(session_id)
print(f"Validation results: {results}")
```

### Batch Processing

```python
from debug_framework.models.batch_validation_job import BatchValidationJob

# Create batch job for multiple models
batch_job = BatchValidationJob(
    job_id="batch_validation_001",
    model_paths=[
        "/path/to/model1.zt",
        "/path/to/model2.zt",
        "/path/to/model3.zt"
    ],
    config={
        "parallel_execution": True,
        "max_concurrent_jobs": 3
    }
)

# Execute batch validation
results = batch_job.execute()
```

## Performance Monitoring

### Built-in Profiling

```python
from debug_framework.services.validation_engine import ValidationEngine

validation_engine = ValidationEngine()

# Enable performance profiling
session_id = validation_engine.create_session(
    model_path="/path/to/model.zt",
    config={
        "performance_profiling_enabled": True,
        "profile_memory_usage": True,
        "profile_execution_time": True
    }
)

# Get performance metrics
metrics = validation_engine.get_performance_metrics(session_id)
print(f"Performance metrics: {metrics}")
```

### Resource Management

The framework includes automatic resource management:

- **Memory Limits**: Configurable tensor recording limits
- **Session Duration**: Automatic session cleanup after timeout
- **Database Management**: Automatic database maintenance and optimization

## Error Handling

### Exception Types

The framework defines specific exception types for different error scenarios:

```python
from debug_framework import (
    DebugFrameworkError,
    BackendInitializationError,
    ValidationError
)

try:
    session_id = debug_framework.create_validation_session(
        model_path="/invalid/path.zt"
    )
except BackendInitializationError as e:
    print(f"Backend initialization failed: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
except DebugFrameworkError as e:
    print(f"Framework error: {e}")
```

### Common Error Scenarios

1. **Backend Not Available**: Alternative backend cannot be initialized
2. **Model Loading Failed**: Model file cannot be loaded or parsed
3. **Numerical Mismatch**: Outputs exceed tolerance thresholds
4. **Resource Exhaustion**: Memory or disk space limitations

## Best Practices

### Production Deployment

1. **Resource Limits**: Set appropriate limits for tensor recording and session duration
2. **Database Management**: Use production database with proper backup/recovery
3. **Error Monitoring**: Implement proper logging and alerting
4. **Performance Optimization**: Configure appropriate tolerance levels

### Development Workflow

1. **Start Simple**: Begin with basic validation using auto-detected backends
2. **Incremental Validation**: Add checkpoints gradually to isolate issues
3. **Performance Analysis**: Use profiling to identify bottlenecks
4. **Batch Testing**: Validate multiple models systematically

### Troubleshooting

1. **Check Backend Availability**: Use `detect_available_backends()` to verify setup
2. **Review Logs**: Enable debug logging for detailed diagnostics
3. **Validate Model Files**: Ensure model files are accessible and properly formatted
4. **Test Incrementally**: Start with single checkpoints before full validation

## Integration with PIE

The debug framework is designed to integrate seamlessly with the PIE model serving infrastructure:

```python
# PIE integration example
from debug_framework.integrations.l4ma_integration import L4MAIntegration

# Initialize L4MA integration
l4ma_integration = L4MAIntegration()

# Validate L4MA model
session_id = l4ma_integration.create_validation_session(
    model_path="/path/to/llama-model.zt",
    pie_config_path="/path/to/pie_config.json"
)

# Execute validation with PIE-specific checkpoints
results = l4ma_integration.execute_validation(session_id)
```

## CLI Usage

The framework includes command-line tools for common operations:

```bash
# Run basic validation
python -m debug_framework.cli.debug_validate \
    --model-path /path/to/model.zt \
    --reference-backend python_reference \
    --alternative-backend auto

# Generate validation report
python -m debug_framework.cli.session_report \
    --session-id 12345 \
    --output-format json \
    --output-file validation_report.json

# Compile custom plugins
python -m debug_framework.cli.plugin_compile \
    --plugin-path /path/to/plugin.cpp \
    --backend metal \
    --output-path /path/to/compiled_plugin.metallib
```

## Next Steps

- Review the [API Reference](api_reference.md) for detailed class and method documentation
- See [Deployment Guide](deployment_guide.md) for production deployment instructions
- Check [Examples](../examples/) directory for complete usage examples
- Run the test suite to verify your installation: `uv run pytest ../../tests/debug_framework/ -v`
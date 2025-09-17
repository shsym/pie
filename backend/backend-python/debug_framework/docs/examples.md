# Multi-Backend Debug Framework - Examples

## Basic Usage Examples

### Simple Model Validation

```python
import debug_framework

# Initialize framework
debug_framework.initialize_framework()

# Create validation session with auto-detected backend
session_id = debug_framework.create_validation_session(
    model_path="/path/to/llama-3.2-1b-instruct.zt",
    reference_backend="python_reference",
    alternative_backend="auto"  # Will auto-select Metal, CUDA, or PyTorch
)

print(f"Validation session created: {session_id}")

# Framework handles the rest automatically
```

### Custom Configuration

```python
import debug_framework

# Custom session configuration
config = {
    "enabled_checkpoints": ["post_attention", "post_mlp"],
    "precision_thresholds": {
        "rtol": 1e-5,  # Stricter relative tolerance
        "atol": 1e-7   # Stricter absolute tolerance
    },
    "tensor_recording_enabled": True,
    "performance_profiling_enabled": True,
    "max_session_duration": 1800  # 30 minutes
}

session_id = debug_framework.create_validation_session(
    model_path="/path/to/model.zt",
    config=config,
    reference_backend="python_reference",
    alternative_backend="metal"
)
```

## Validation Engine Examples

### Manual Validation Control

```python
from debug_framework.services.validation_engine import ValidationEngine

# Create validation engine
validation_engine = ValidationEngine()

# Create session with specific configuration
session_config = {
    "enabled_checkpoints": [
        "post_embedding",
        "post_attention",
        "post_mlp",
        "post_processing"
    ],
    "precision_thresholds": {"rtol": 1e-4, "atol": 1e-6},
    "tensor_recording_enabled": True,
    "performance_profiling_enabled": True
}

session_id = validation_engine.create_session(
    model_path="/path/to/llama-model.zt",
    config=session_config,
    reference_backend="python_reference",
    alternative_backend="metal"
)

# Execute validation
results = validation_engine.execute_validation(session_id)

# Print results
print(f"Validation Status: {results['overall_status']}")
print(f"Checkpoints Passed: {results['checkpoints_passed']}/{results['total_checkpoints']}")
print(f"Average Execution Time: {results['avg_execution_time']:.4f}s")

# Get detailed results
detailed_results = validation_engine.get_session_results(session_id)
for checkpoint in detailed_results['checkpoint_results']:
    print(f"{checkpoint['name']}: {'✅ PASS' if checkpoint['passed'] else '❌ FAIL'}")
    print(f"  Max Error: {checkpoint['max_error']:.2e}")
    print(f"  Execution Time: {checkpoint['execution_time']:.4f}s")

# Cleanup session
validation_engine.cleanup_session(session_id)
```

### Error Handling

```python
from debug_framework import (
    ValidationError,
    BackendInitializationError,
    DebugFrameworkError
)
from debug_framework.services.validation_engine import ValidationEngine

validation_engine = ValidationEngine()

try:
    session_id = validation_engine.create_session(
        model_path="/path/to/model.zt",
        config={"enabled_checkpoints": ["post_attention"]},
        reference_backend="python_reference",
        alternative_backend="metal"
    )

    results = validation_engine.execute_validation(session_id)

    if results['overall_status'] != 'passed':
        print("❌ Validation failed!")
        for checkpoint in results['failed_checkpoints']:
            print(f"  {checkpoint['name']}: {checkpoint['error_message']}")
    else:
        print("✅ Validation passed!")

except BackendInitializationError as e:
    print(f"Backend initialization failed: {e}")
    print("Available backends:", debug_framework.detect_available_backends())

except ValidationError as e:
    print(f"Validation error: {e}")

except DebugFrameworkError as e:
    print(f"Framework error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Batch Processing Examples

### Simple Batch Validation

```python
from debug_framework.models.batch_validation_job import BatchValidationJob

# List of models to validate
model_paths = [
    "/path/to/llama-1b.zt",
    "/path/to/llama-3b.zt",
    "/path/to/llama-7b.zt"
]

# Create batch job
batch_job = BatchValidationJob(
    job_id="llama_models_validation",
    model_paths=model_paths,
    config={
        "parallel_execution": True,
        "max_concurrent_jobs": 2,
        "timeout_per_job": 1800,  # 30 minutes per model
        "reference_backend": "python_reference",
        "alternative_backend": "metal"
    }
)

# Execute batch validation
print("🚀 Starting batch validation...")
results = batch_job.execute()

# Process results
print(f"\n📊 Batch Validation Results:")
print(f"Total Models: {len(model_paths)}")
print(f"Passed: {sum(1 for r in results if r['status'] == 'passed')}")
print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")

for result in results:
    status_emoji = "✅" if result['status'] == 'passed' else "❌"
    print(f"{status_emoji} {result['model_path']}: {result['status']}")
    if result['status'] == 'failed':
        print(f"   Error: {result.get('error_message', 'Unknown error')}")
```

### Advanced Batch Configuration

```python
from debug_framework.models.batch_validation_job import BatchValidationJob
import json

# Advanced batch configuration
advanced_config = {
    "parallel_execution": True,
    "max_concurrent_jobs": 3,
    "timeout_per_job": 3600,  # 1 hour per model
    "retry_failed_jobs": True,
    "max_retries": 2,

    # Per-model validation settings
    "validation_config": {
        "enabled_checkpoints": ["post_attention", "post_mlp"],
        "precision_thresholds": {"rtol": 1e-4, "atol": 1e-6},
        "performance_profiling_enabled": True,
        "tensor_recording_enabled": False  # Disable for large batch jobs
    },

    # Backend configuration
    "reference_backend": "python_reference",
    "alternative_backend": "auto",
    "fallback_backends": ["metal", "cuda", "pytorch"],

    # Resource management
    "memory_limit_mb": 8192,
    "cleanup_on_completion": True,

    # Reporting
    "generate_summary_report": True,
    "save_individual_reports": True,
    "report_output_dir": "/tmp/batch_validation_reports"
}

batch_job = BatchValidationJob(
    job_id="comprehensive_model_validation",
    model_paths=[
        "/models/llama-1b-instruct.zt",
        "/models/llama-3b-instruct.zt",
        "/models/llama-7b-instruct.zt",
        "/models/llama-13b-instruct.zt"
    ],
    config=advanced_config
)

# Monitor batch job progress
import time

print("🚀 Starting comprehensive batch validation...")
results = batch_job.execute()

# Wait for completion with progress monitoring
while batch_job.is_running():
    progress = batch_job.get_progress()
    print(f"Progress: {progress['completed']}/{progress['total']} "
          f"({progress['percentage']:.1f}%)")
    time.sleep(30)

# Get final results
final_results = batch_job.get_results()
print(f"\n📋 Final Results:")
print(json.dumps(final_results['summary'], indent=2))
```

## Backend Integration Examples

### Metal Backend

```python
from debug_framework.integrations.metal_backend import MetalBackend

# Initialize Metal backend
metal_backend = MetalBackend(metal_path="/opt/pie/backends/metal")

if metal_backend.initialize():
    print("✅ Metal backend initialized successfully")

    # Get Metal device info
    device_info = metal_backend.get_device_info()
    print(f"Metal Device: {device_info['name']}")
    print(f"Max Buffer Length: {device_info['max_buffer_length']:,} bytes")
    print(f"Recommended Working Set Size: {device_info['recommended_max_working_set_size']:,} bytes")

    # Execute model with Metal backend
    try:
        output = metal_backend.execute_model(
            model_path="/path/to/model.zt",
            input_data={"input_ids": [1, 2, 3, 4, 5]}
        )
        print(f"Metal execution successful: {output.shape}")
    except Exception as e:
        print(f"Metal execution failed: {e}")

    # Cleanup
    metal_backend.cleanup()
else:
    print("❌ Metal backend initialization failed")
    print("Available backends:", debug_framework.detect_available_backends())
```

### L4MA Integration

```python
from debug_framework.integrations.l4ma_integration import L4MAIntegration

# Initialize L4MA integration
l4ma_integration = L4MAIntegration()

# Create L4MA-specific validation session
session_id = l4ma_integration.create_validation_session(
    model_path="/path/to/llama-3.2-1b-instruct.zt",
    pie_config_path="/path/to/pie_config.json"
)

# Execute L4MA validation with model-specific checkpoints
results = l4ma_integration.execute_validation(session_id)

print(f"L4MA Validation Results:")
print(f"Model Architecture: {results['model_metadata']['architecture']}")
print(f"Number of Layers: {results['model_metadata']['num_layers']}")
print(f"Hidden Size: {results['model_metadata']['hidden_size']}")
print(f"Vocabulary Size: {results['model_metadata']['vocab_size']}")

# Check specific L4MA components
for component, status in results['component_validation'].items():
    status_emoji = "✅" if status['passed'] else "❌"
    print(f"{status_emoji} {component}: {status['status']}")
    if not status['passed']:
        print(f"   Error: {status['error']}")

# Performance metrics for L4MA model
perf_metrics = results['performance_metrics']
print(f"\n⚡ Performance Metrics:")
print(f"Tokens per second: {perf_metrics['tokens_per_second']:.2f}")
print(f"Memory usage: {perf_metrics['peak_memory_mb']:.2f} MB")
print(f"GPU utilization: {perf_metrics['gpu_utilization_percent']:.1f}%")
```

## Custom Plugin Examples

### Metal Kernel Plugin

```cpp
// custom_attention_kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void custom_attention_kernel(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_size [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;

    if (row >= seq_len || col >= seq_len) return;

    // Custom attention computation
    float attention_score = 0.0;
    for (uint i = 0; i < head_size; ++i) {
        attention_score += query[row * head_size + i] * key[col * head_size + i];
    }

    // Apply softmax and compute output
    attention_score = exp(attention_score / sqrt(float(head_size)));

    for (uint i = 0; i < head_size; ++i) {
        output[row * head_size + i] += attention_score * value[col * head_size + i];
    }
}
```

```python
# Register and use custom Metal kernel
from debug_framework.services.plugin_registry import PluginRegistry
from debug_framework.models.plugin_definition import PluginDefinition

plugin_registry = PluginRegistry()

# Define custom Metal plugin
custom_plugin = PluginDefinition(
    plugin_id="custom_attention_metal",
    name="Custom Attention Metal Kernel",
    backend_type="metal",
    source_path="/path/to/custom_attention_kernel.metal",
    entry_point="custom_attention_kernel",
    metadata={
        "version": "1.0.0",
        "author": "Debug Framework Team",
        "description": "Optimized attention kernel for Metal"
    }
)

# Register plugin
if plugin_registry.register_plugin(custom_plugin):
    print("✅ Custom Metal plugin registered successfully")

    # Compile plugin
    from debug_framework.services.compilation_engine import CompilationEngine

    compiler = CompilationEngine()
    success = compiler.compile_metal_kernel(
        source_path="/path/to/custom_attention_kernel.metal",
        output_path="/path/to/custom_attention_kernel.metallib"
    )

    if success:
        print("✅ Plugin compiled successfully")
    else:
        print("❌ Plugin compilation failed")
```

### CUDA Kernel Plugin

```cuda
// custom_softmax_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void custom_softmax_kernel(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / seq_len;
    int seq_idx = idx % seq_len;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    // Find maximum value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < vocab_size; ++i) {
        int input_idx = batch_idx * seq_len * vocab_size + seq_idx * vocab_size + i;
        max_val = fmaxf(max_val, input[input_idx]);
    }

    // Compute softmax
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        int input_idx = batch_idx * seq_len * vocab_size + seq_idx * vocab_size + i;
        sum += expf(input[input_idx] - max_val);
    }

    for (int i = 0; i < vocab_size; ++i) {
        int input_idx = batch_idx * seq_len * vocab_size + seq_idx * vocab_size + i;
        int output_idx = input_idx;
        output[output_idx] = expf(input[input_idx] - max_val) / sum;
    }
}
```

```python
# Register CUDA plugin
cuda_plugin = PluginDefinition(
    plugin_id="custom_softmax_cuda",
    name="Custom Softmax CUDA Kernel",
    backend_type="cuda",
    source_path="/path/to/custom_softmax_kernel.cu",
    entry_point="custom_softmax_kernel",
    metadata={
        "compute_capability": "6.0",
        "cuda_version": "11.0+",
        "optimization_level": "O3"
    }
)

plugin_registry.register_plugin(cuda_plugin)

# Compile CUDA plugin
success = compiler.compile_cuda_kernel(
    source_path="/path/to/custom_softmax_kernel.cu",
    output_path="/path/to/custom_softmax_kernel.ptx"
)
```

## Performance Profiling Examples

### Detailed Performance Analysis

```python
from debug_framework.services.validation_engine import ValidationEngine
import time

validation_engine = ValidationEngine()

# Create session with detailed profiling
profiling_config = {
    "enabled_checkpoints": ["post_attention", "post_mlp", "post_processing"],
    "performance_profiling_enabled": True,
    "profile_memory_usage": True,
    "profile_execution_time": True,
    "profile_gpu_utilization": True,
    "tensor_recording_enabled": True
}

session_id = validation_engine.create_session(
    model_path="/path/to/llama-7b.zt",
    config=profiling_config,
    reference_backend="python_reference",
    alternative_backend="metal"
)

# Execute with timing
start_time = time.time()
results = validation_engine.execute_validation(session_id)
total_time = time.time() - start_time

# Get detailed performance metrics
perf_metrics = validation_engine.get_performance_metrics(session_id)

print(f"🚀 Performance Analysis Results:")
print(f"Total Validation Time: {total_time:.2f}s")
print(f"Reference Backend Time: {perf_metrics['reference_total_time']:.2f}s")
print(f"Alternative Backend Time: {perf_metrics['alternative_total_time']:.2f}s")
print(f"Speedup: {perf_metrics['speedup_ratio']:.2f}x")

print(f"\n📊 Per-Checkpoint Performance:")
for checkpoint_name, metrics in perf_metrics['checkpoint_metrics'].items():
    print(f"  {checkpoint_name}:")
    print(f"    Reference: {metrics['reference_time']:.4f}s")
    print(f"    Alternative: {metrics['alternative_time']:.4f}s")
    print(f"    Speedup: {metrics['speedup']:.2f}x")
    print(f"    Memory Delta: {metrics['memory_delta_mb']:.2f} MB")

print(f"\n🎯 Resource Usage:")
print(f"Peak Memory: {perf_metrics['peak_memory_mb']:.2f} MB")
print(f"Average GPU Utilization: {perf_metrics['avg_gpu_utilization']:.1f}%")
print(f"GPU Memory Used: {perf_metrics['gpu_memory_used_mb']:.2f} MB")

# Generate performance report
report = validation_engine.generate_performance_report(session_id)
with open(f"performance_report_{session_id}.json", 'w') as f:
    json.dump(report, f, indent=2)
```

### Benchmark Suite

```python
import json
import statistics
from debug_framework.services.validation_engine import ValidationEngine

class BenchmarkSuite:
    def __init__(self):
        self.validation_engine = ValidationEngine()
        self.results = []

    def benchmark_model(self, model_path, backends, iterations=3):
        """Benchmark model across multiple backends and iterations."""
        print(f"🔬 Benchmarking {model_path}")

        model_results = {
            "model_path": model_path,
            "backends": {},
            "iterations": iterations
        }

        for backend in backends:
            backend_times = []

            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations} - Backend: {backend}")

                session_id = self.validation_engine.create_session(
                    model_path=model_path,
                    config={
                        "performance_profiling_enabled": True,
                        "tensor_recording_enabled": False  # Disable for pure performance
                    },
                    reference_backend="python_reference",
                    alternative_backend=backend
                )

                start_time = time.time()
                results = self.validation_engine.execute_validation(session_id)
                execution_time = time.time() - start_time

                backend_times.append(execution_time)
                self.validation_engine.cleanup_session(session_id)

            # Calculate statistics
            model_results["backends"][backend] = {
                "times": backend_times,
                "mean": statistics.mean(backend_times),
                "median": statistics.median(backend_times),
                "stdev": statistics.stdev(backend_times) if len(backend_times) > 1 else 0,
                "min": min(backend_times),
                "max": max(backend_times)
            }

        self.results.append(model_results)
        return model_results

    def run_benchmark_suite(self, models, backends):
        """Run complete benchmark suite."""
        print(f"🚀 Starting benchmark suite: {len(models)} models, {len(backends)} backends")

        for model_path in models:
            self.benchmark_model(model_path, backends)

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate benchmark summary report."""
        report = {
            "summary": {
                "total_models": len(self.results),
                "total_benchmarks": sum(len(r["backends"]) for r in self.results)
            },
            "results": self.results
        }

        # Save report
        with open("benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"📊 Benchmark Summary:")
        for result in self.results:
            model_name = result["model_path"].split("/")[-1]
            print(f"  {model_name}:")

            # Find fastest backend
            fastest_backend = min(
                result["backends"].items(),
                key=lambda x: x[1]["mean"]
            )

            print(f"    Fastest: {fastest_backend[0]} ({fastest_backend[1]['mean']:.2f}s)")

            # Show speedup ratios
            reference_time = result["backends"].get("python_reference", {}).get("mean")
            if reference_time:
                for backend, metrics in result["backends"].items():
                    if backend != "python_reference":
                        speedup = reference_time / metrics["mean"]
                        print(f"    {backend}: {speedup:.2f}x speedup")

# Usage
benchmark = BenchmarkSuite()

models_to_benchmark = [
    "/path/to/llama-1b-instruct.zt",
    "/path/to/llama-3b-instruct.zt",
    "/path/to/llama-7b-instruct.zt"
]

backends_to_test = ["python_reference", "metal", "cuda", "pytorch"]

benchmark.run_benchmark_suite(models_to_benchmark, backends_to_test)
```

## CLI Usage Examples

### Command Line Validation

```bash
# Basic validation
python -m debug_framework.cli.debug_validate \
    --model-path /path/to/llama-3.2-1b-instruct.zt \
    --reference-backend python_reference \
    --alternative-backend auto \
    --verbose

# Custom configuration file
cat > validation_config.json << EOF
{
    "enabled_checkpoints": ["post_attention", "post_mlp"],
    "precision_thresholds": {
        "rtol": 1e-5,
        "atol": 1e-7
    },
    "performance_profiling_enabled": true,
    "tensor_recording_enabled": true
}
EOF

python -m debug_framework.cli.debug_validate \
    --model-path /path/to/model.zt \
    --config-file validation_config.json \
    --output-file validation_results.json \
    --verbose
```

### Batch CLI Validation

```bash
# Create batch configuration
cat > batch_config.json << EOF
{
    "parallel_execution": true,
    "max_concurrent_jobs": 2,
    "reference_backend": "python_reference",
    "alternative_backend": "metal",
    "validation_config": {
        "enabled_checkpoints": ["post_attention"],
        "precision_thresholds": {"rtol": 1e-4, "atol": 1e-6}
    }
}
EOF

# Run batch validation
python -m debug_framework.cli.batch_validate \
    --models-list models.txt \
    --config-file batch_config.json \
    --output-dir batch_results/ \
    --verbose
```

### Report Generation

```bash
# Generate HTML report
python -m debug_framework.cli.session_report \
    --session-id 12345 \
    --output-format html \
    --output-file validation_report.html \
    --include-tensors

# Generate CSV summary
python -m debug_framework.cli.session_report \
    --session-id 12345 \
    --output-format csv \
    --output-file validation_summary.csv

# Generate comprehensive JSON report
python -m debug_framework.cli.session_report \
    --session-id 12345 \
    --output-format json \
    --output-file detailed_report.json \
    --include-tensors \
    --include-performance-metrics \
    --include-system-info
```

## Integration with PIE Serving

### PIE Integration Example

```python
from debug_framework.integrations.l4ma_integration import L4MAIntegration
import json

# Initialize PIE integration
pie_integration = L4MAIntegration()

# PIE configuration
pie_config = {
    "model_path": "/path/to/llama-3.2-1b-instruct.zt",
    "backend": "metal",
    "max_batch_size": 4,
    "max_sequence_length": 2048,
    "cache_size": 1000
}

# Create PIE-optimized validation session
session_id = pie_integration.create_validation_session(
    model_path=pie_config["model_path"],
    pie_config_path=None,  # Use inline config
    pie_config=pie_config
)

# Execute validation with PIE-specific tests
results = pie_integration.execute_validation(session_id)

print("🥧 PIE Integration Validation Results:")
print(f"Model Loading: {'✅' if results['model_loading_success'] else '❌'}")
print(f"Backend Initialization: {'✅' if results['backend_init_success'] else '❌'}")
print(f"Batch Processing: {'✅' if results['batch_processing_success'] else '❌'}")
print(f"Memory Management: {'✅' if results['memory_management_success'] else '❌'}")

# Test serving integration
serving_results = pie_integration.test_serving_integration(
    session_id=session_id,
    test_requests=[
        {"input": "Hello, how are you?", "max_length": 50},
        {"input": "Explain quantum computing", "max_length": 100}
    ]
)

print(f"\n🌐 Serving Integration Test:")
for i, result in enumerate(serving_results):
    print(f"Request {i+1}: {'✅' if result['success'] else '❌'}")
    if result['success']:
        print(f"  Response time: {result['response_time']:.3f}s")
        print(f"  Tokens generated: {result['tokens_generated']}")
    else:
        print(f"  Error: {result['error']}")
```

This completes the comprehensive examples guide, covering all major use cases and integration patterns for the Multi-Backend Debug Framework.
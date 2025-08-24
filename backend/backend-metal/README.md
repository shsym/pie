# PIE Metal Backend

High-performance Metal GPU backend implementation for PIE (Programmable Inference Engine), providing GPU-accelerated inference on macOS and iOS platforms with CUDA API compatibility.

## Overview

The Metal backend translates CUDA operations to Metal Performance Shaders while maintaining exact API compatibility with the CUDA backend. This enables seamless GPU acceleration on Apple Silicon and Intel Macs without code changes in the PIE inference engine.

## Architecture

### Core Components
- **Metal Kernels**: GPU compute shaders implemented in Metal Shading Language
- **C++ Wrappers**: CUDA-compatible API wrappers around Metal operations  
- **Memory Management**: Metal buffer management with automatic lifecycle handling
- **Type System**: Support for float32, bfloat16, and future precision types

### Supported Operations
- **Softmax**: Numerically stable softmax with temperature scaling
- **Extract K Values**: Sparse matrix value extraction for top-k operations
- **TopK Mask Logits**: Top-k masking for token generation
- **GEMM**: Matrix multiplication with bias support
- **Embedding Lookup**: Efficient embedding table lookups
- **RMSNorm**: Root Mean Square normalization
- **RoPE**: Rotary Position Embedding
- **SiLU and Mul**: Activation functions with element-wise multiplication
- **Add Residual**: Residual connection operations
- **Attention**: Batch prefill attention mechanisms
- **KV Cache**: Key-Value cache management for attention

## Build System

### Requirements
- macOS 13.0+ (for Metal 3.0+ support)  
- Xcode 14.0+
- CMake 3.23+
- Apple Clang with Objective-C++ support

### Building
```bash
# Configure with CMake
mkdir -p build && cd build  
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build libraries and tests
make -j$(nproc)

# All binaries output to build/ directory
ls build/test_* build/lib*
```

### CMake Targets
- **Libraries**: `metal_*` - Individual Metal kernel libraries
- **Tests**: `test_*` - Comprehensive test suite (see [tests/README.md](tests/README.md))
- **Test Suites**: `test_unit`, `test_dtype`, `test_integration_suite`, etc.

## Testing

### Test Structure
```
tests/
├── CMakeLists.txt              # CMake test configuration
├── README.md                   # Detailed test documentation
├── test_metal_unit_*.cpp       # Unit tests for individual kernels
├── test_metal_dtype_*.cpp      # Data type validation tests
├── test_metal_edge_cases*.cpp  # Edge case and boundary testing
├── test_metal_performance.cpp  # Performance benchmarking
├── test_*_compatibility.mm     # CUDA compatibility tests
└── test_*_debug*.cpp           # Debug and investigation tests
```

### Running Tests
```bash
cd build

# Run all production tests
make test_all

# Run by category
make test_unit                   # Unit tests
make test_integration_suite      # Integration tests
make test_performance_suite      # Performance benchmarks
make test_compatibility_suite    # CUDA compatibility

# Run individual tests
./test_metal_unit_softmax
./test_metal_performance
```

### Test Coverage
- ✅ **Unit Tests**: Individual kernel validation
- ✅ **Integration Tests**: Full operation workflows
- ✅ **Data Type Tests**: float32, bfloat16 validation
- ✅ **Edge Cases**: Boundary conditions and special values
- ✅ **Performance Tests**: Throughput and bandwidth validation
- ✅ **Compatibility Tests**: CUDA API compatibility validation

## Performance

### Benchmarks
Performance validated across realistic workloads:

| Operation | Batch Size | Vocab Size | Throughput (ops/sec) | Memory Bandwidth (GB/s) |
|-----------|------------|------------|---------------------|-------------------------|
| Softmax | 32 | 32768 | 7.61e+08 | 45.2 |
| Extract K Values | 32 | 32000 | 1.82e+09 | 28.7 |
| TopK Mask | 32 | 32768 | 2.09e+08 | 31.5 |

### Optimization Features
- **Numerical Stability**: Softmax with max subtraction for large values
- **Memory Coalescing**: Optimized memory access patterns
- **SIMD Utilization**: Vectorized operations where possible
- **Buffer Reuse**: Efficient Metal buffer lifecycle management
- **Precision Handling**: Automatic precision selection based on data type

## API Compatibility

### CUDA Compatibility Layer
The Metal backend maintains exact API compatibility with the CUDA backend:

```cpp
// Identical API calls work for both CUDA and Metal backends
int result = metal_softmax_float(input, output, batch_size, vocab_size, temperature);
int result = cuda_softmax_float(input, output, batch_size, vocab_size, temperature);
```

### Type Support
- **float32**: Full precision floating point (primary)
- **bfloat16**: Brain floating point for reduced memory usage
- **Future**: fp16, int8 quantization support planned

### Error Handling
- Return codes compatible with CUDA backend
- Graceful degradation for unsupported configurations
- Detailed error reporting for debugging

## Integration with PIE

### PIE Inferlet Compatibility
The Metal backend integrates seamlessly with PIE's WebAssembly inferlets:

```cpp
// PIE automatically selects Metal backend on macOS
PIEConfig config;
config.device = "metal:0";  // Use Metal GPU
config.dtype = "bfloat16";   // Use reduced precision

Model model(config, metadata);
auto results = model.handle_forward_text(commands);
```

### Typical Usage Patterns
From [guide.md](../../guide.md):
- **CLI Interface**: PIE CLI with Metal backend selection
- **WebAssembly**: Embedded inferlets using Metal acceleration
- **Batch Processing**: Multi-sequence inference with Metal parallelization
- **Real-time Inference**: Low-latency inference for interactive applications

## Development

### Project Structure
```
src/
├── metal_*.mm                  # Metal kernel implementations
├── metal_*.metal               # Metal shading language kernels
├── metal_*.hpp                 # C++ header interfaces
├── metal_*_wrapper.mm          # High-level API wrappers
├── metal_common.{mm,hpp}       # Common utilities and context
├── metal_model.{mm,hpp}        # Model integration layer
└── test_*.mm                   # Integration test implementations
```

### Adding New Operations
1. Implement Metal shader in `metal_operation.metal`
2. Create C++ wrapper in `metal_operation.mm`
3. Define API interface in `metal_operation.hpp` 
4. Add unit tests in `tests/test_metal_unit_operation.cpp`
5. Update CMakeLists.txt to include new library
6. Validate compatibility with CUDA equivalent

### Code Conventions
- **Objective-C++**: Use `.mm` extension for Metal integration
- **Memory Management**: Use ARC for automatic reference counting
- **Error Handling**: Return integer error codes (0 = success)
- **Threading**: Metal operations are inherently thread-safe
- **Precision**: Default to float32, support bfloat16 where beneficial

## Deployment

### System Requirements
- **macOS**: 13.0+ (Metal 3.0+ support)
- **iOS**: 16.0+ (for iOS deployment)
- **Hardware**: Apple Silicon or Intel Mac with Metal-capable GPU
- **Memory**: Sufficient GPU memory for model weights and inference

### Distribution
- Static libraries for integration with PIE
- CMake package config for external projects
- Pre-built binaries for common configurations

## Troubleshooting

### Common Issues
1. **Metal Device Not Found**: Ensure Metal-capable GPU is available
2. **Shader Compilation**: Check Metal shading language syntax
3. **Memory Issues**: Monitor GPU memory usage with Instruments
4. **Performance**: Use Metal Performance HUD for profiling

### Debug Tools
```bash
# Debug tests for investigation
./test_debug_validation
./test_simple_dtype_debug
./test_softmax_precision_debug

# Metal debugging with Xcode Instruments
instruments -t "Metal Performance" ./test_metal_performance
```

### Logging
- Metal framework errors logged to console
- Debug builds include verbose Metal validation
- Performance metrics available through test suite

## Roadmap

### Near Term
- [ ] Complete full inference pipeline tests
- [ ] Model weight loading validation
- [ ] Memory management stress tests
- [ ] iOS deployment validation

### Medium Term  
- [ ] fp16 precision support
- [ ] Quantized inference (int8)
- [ ] Multi-GPU support
- [ ] Advanced Metal 3.0 features

### Long Term
- [ ] Metal Ray Tracing integration
- [ ] Neural Engine integration
- [ ] Cross-platform WebGPU compatibility

## Contributing

### Development Workflow
1. Clone repository with Metal backend submodule
2. Set up development environment (Xcode, CMake)
3. Run existing tests to validate setup: `make test_all`
4. Implement changes following code conventions
5. Add/update tests for new functionality
6. Validate against CUDA backend compatibility
7. Submit pull request with test results

### Testing Requirements
- All new features must include unit tests
- Integration tests for API-level changes
- Performance benchmarks for optimization work
- Compatibility validation with CUDA backend

For detailed test information, see [tests/README.md](tests/README.md).

## License

Part of the PIE (Programmable Inference Engine) project. See main repository for license information.
# Metal Backend Test Suite

Comprehensive test suite for the PIE Metal backend implementation, covering all major operations from unit tests to end-to-end inference validation.

## Test Organization

```
tests/
├── src/
│   ├── compatibility/     # API and cross-platform compatibility tests
│   ├── debug/            # Debug and troubleshooting utilities
│   ├── integration/      # End-to-end and system-level tests
│   ├── performance/      # Benchmarking and performance validation
│   └── unit/            # Isolated unit tests for individual operations
├── bin/                 # Compiled test executables
└── build/              # CMake build artifacts
```

## Test Categories

### 1. Compatibility Tests (`src/compatibility/`)

**Purpose**: Verify API compatibility, compilation, and cross-platform compatibility.

#### `test_api_compatibility.mm`
- **What it does**: Verifies Metal backend API structure compatibility without device execution
- **Tests**: L4maConfig structure fields, template class compilation, MetalTensor interface, model command structures
- **Key validations**: Configuration parameters, object allocation commands, embedding commands
- **Exit criteria**: All API structures compile and match expected interfaces

#### `test_compilation_validation.mm`
- **What it does**: Ensures all Metal shaders and C++/ObjC++ code compiles correctly
- **Tests**: Metal kernel compilation, header includes, template instantiation
- **Key validations**: Shader syntax, Metal API usage, type compatibility
- **Exit criteria**: Clean compilation with no warnings or errors

#### `test_cuda_compatibility.mm` 
- **What it does**: Validates compatibility between CUDA and Metal implementations
- **Tests**: Parameter structure compatibility, data layout matching, numerical precision
- **Key validations**: Cross-platform data structures, computation equivalence
- **Exit criteria**: CUDA and Metal backends produce identical results

### 2. Debug Tests (`src/debug/`)

**Purpose**: Troubleshooting utilities and validation for debugging complex issues.

#### `test_debug_validation.cpp`
- **What it does**: Debug version of extract_k_values with detailed output and validation
- **Tests**: Same logic as unit test but with extensive logging and intermediate value checking
- **Key validations**: Row-by-row validation with expected values, detailed failure analysis
- **Debug features**: Prints actual vs expected values, identifies specific failure points
- **Exit criteria**: All extractions match expected CUDA reference results

#### `test_simple_dtype_debug.cpp`
- **What it does**: Validates data type conversions and representations
- **Tests**: Float to bfloat16 conversions, precision handling, edge cases
- **Key validations**: Conversion accuracy, numerical stability, type safety
- **Exit criteria**: All data type operations maintain required precision

#### `test_softmax_precision_debug.cpp`
- **What it does**: Detailed analysis of softmax numerical precision and stability  
- **Tests**: Temperature scaling, numerical stability, edge case handling
- **Key validations**: Precision loss analysis, overflow/underflow detection
- **Exit criteria**: Softmax maintains numerical accuracy within tolerance

### 3. Integration Tests (`src/integration/`)

**Purpose**: System-level tests that validate complete workflows and end-to-end functionality.

#### Core Operation Tests

#### `test_metal_softmax.cpp`
- **What it does**: Validates Metal softmax implementation against CPU reference
- **Tests**: Batch processing, temperature scaling, numerical stability
- **Parameters**: 4 batch size, 32K vocab size, 1e-5 tolerance
- **Key validations**: Probability distribution correctness, sum to 1.0, stability
- **Exit criteria**: All batches produce correct probability distributions

#### `test_metal_extract_k_values.cpp`
- **What it does**: Tests top-k value extraction from logits
- **Tests**: K-value selection, index tracking, edge case handling
- **Key validations**: Correct k-largest values, proper index mapping, boundary conditions
- **Exit criteria**: Extracted values and indices match expected results

#### `test_metal_topk_mask_logits.cpp`
- **What it does**: Validates top-k masking for sampling
- **Tests**: Logit masking, threshold application, sampling preparation
- **Key validations**: Proper masking of non-top-k values, preservation of top-k
- **Exit criteria**: Only top-k values remain unmasked

#### Data Type and Edge Case Tests

#### `test_metal_dtype_validation.cpp` & `test_metal_dtype_working.cpp`
- **What they do**: Comprehensive data type validation and working implementations
- **Tests**: BFloat16 operations, type conversions, precision maintenance
- **Key validations**: Numerical accuracy, type safety, conversion correctness
- **Exit criteria**: All data type operations maintain required precision

#### `test_metal_edge_cases.cpp` & `test_metal_edge_cases_robust.cpp`
- **What they do**: Edge case and robustness testing for all operations
- **Tests**: Zero inputs, infinite values, boundary conditions, malformed data
- **Key validations**: Graceful handling of edge cases, no crashes, predictable behavior
- **Exit criteria**: All edge cases handled correctly without failures

#### Memory and Inference Tests

#### `test_metal_memory_basic.mm` & `test_metal_memory_stress.mm`
- **What they do**: Memory allocation and stress testing
- **Tests**: Buffer allocation, memory pressure handling, leak detection
- **Key validations**: Successful allocation/deallocation, no memory leaks
- **Exit criteria**: All memory operations complete successfully

#### `test_metal_inference_basic.mm`
- **What it does**: Basic inference pipeline validation
- **Tests**: Complete inference workflow, model loading, forward pass
- **Key validations**: Correct output generation, pipeline integrity
- **Exit criteria**: Inference produces expected outputs

#### `test_metal_model_weight_loading.mm` & `test_metal_real_model_weights.mm`
- **What they do**: Model weight loading and real model validation
- **Tests**: Weight tensor loading, model parameter validation, real model inference
- **Key validations**: Correct weight loading, parameter integrity, inference accuracy
- **Exit criteria**: Models load correctly and produce valid outputs

#### `test_end_to_end_inference.mm` & `test_metal_full_inference_pipeline.mm`
- **What they do**: Complete end-to-end inference validation
- **Tests**: Full pipeline from input to output, multi-layer processing
- **Key validations**: Complete workflow integrity, output correctness
- **Exit criteria**: End-to-end inference matches expected results

### 4. Performance Tests (`src/performance/`)

**Purpose**: Benchmarking, performance validation, and optimization verification.

#### Attention System Tests

#### `test_attention_correctness.mm`
- **What it does**: Comprehensive correctness testing for Metal batch prefill attention optimization
- **Tests**: Numerical accuracy of optimized vs reference implementations
- **Test scenarios**: Multiple batch sizes, sequence lengths, head configurations
- **Key validations**: Attention output correctness, numerical precision maintenance
- **Exit criteria**: All attention implementations produce identical results

#### `test_attention_benchmark.mm`
- **What it does**: Performance benchmarking for attention implementations
- **Tests**: Throughput, latency, memory usage across different configurations
- **Metrics**: FLOPS, memory bandwidth, execution time, efficiency ratios
- **Key validations**: Performance targets met, no regression
- **Exit criteria**: Performance within acceptable bounds

#### `test_attention_accuracy_comparison.mm` & `test_attention_performance_comparison.mm`
- **What they do**: Cross-implementation accuracy and performance comparison
- **Tests**: Original vs optimized implementations, different kernel variants
- **Key validations**: Accuracy equivalence, performance improvements
- **Exit criteria**: Optimizations maintain accuracy while improving performance

#### FlashAttention Variants

#### `test_flash_attention_tiled.mm`
- **What it does**: Tests tiled FlashAttention implementation
- **Tests**: 2D tiling correctness, memory efficiency, parallelization
- **Key validations**: Tiled computation correctness, memory usage reduction
- **Exit criteria**: Tiled implementation matches reference results

#### `test_flash_attention_flashinfer.mm`
- **What it does**: Tests FlashInfer-style optimization implementation
- **Tests**: Online softmax, memory-efficient attention, block processing
- **Key validations**: FlashAttention algorithm correctness, memory efficiency
- **Exit criteria**: FlashInfer implementation produces correct attention outputs

#### `test_flash_attention_block_optimized.mm`
- **What it does**: Block-optimized FlashAttention variant testing
- **Tests**: Block-level optimization, cache efficiency, parallel execution
- **Key validations**: Block optimization correctness, performance improvement
- **Exit criteria**: Block-optimized version maintains accuracy with better performance

#### Kernel Comparison and Validation

#### `test_kernel_comparison.mm`
- **What it does**: Systematic comparison between different kernel implementations
- **Tests**: Multiple kernel variants, performance characteristics, accuracy
- **Key validations**: Relative performance, accuracy maintenance, optimization effectiveness
- **Exit criteria**: All kernel variants produce consistent results

#### `test_unified_attention_correctness.mm`
- **What it does**: Correctness validation for unified attention system
- **Tests**: Unified parameter structure, multi-head attention, sequence processing
- **Key validations**: Unified system correctness, parameter compatibility
- **Exit criteria**: Unified system matches reference implementations

#### Data Type and Precision Tests

#### `test_f16_native_attention.mm`
- **What it does**: Half-precision native attention implementation testing
- **Tests**: FP16 computation accuracy, precision maintenance, performance
- **Key validations**: Half-precision numerical stability, accuracy vs performance trade-off
- **Exit criteria**: FP16 implementation maintains acceptable accuracy

#### Utility Files

#### `attention_perf_utils.hpp` & `benchmark_utils.hpp`
- **What they do**: Common utilities for performance testing and benchmarking
- **Provides**: Timing utilities, memory measurement, data generation, result validation
- **Features**: Consistent benchmarking methodology, reusable test infrastructure

### 5. Unit Tests (`src/unit/`)

**Purpose**: Isolated testing of individual operations and functions.

#### `test_metal_unit_extract_k_values.cpp`
- **What it does**: Unit test for extract_k_values operation using CUDA-style test data
- **Tests**: Precise expected outputs with specific test matrices
- **Test data**: 3x8 matrix with controlled finite values at specific positions
- **Expected results**:
  - Row 0: finite at cols 1,3,6,7 → expect first 3: (1,3,6)
  - Row 1: finite at cols 0,4 → expect (0,4, padding)
  - Row 2: finite at cols 2,5,6 → expect (2,5,6)
- **Exit criteria**: Extracted values and indices exactly match expected CUDA results

#### `test_metal_unit_extract_k_values_debug.cpp`
- **What it does**: Debug version of extract_k_values unit test with detailed logging
- **Tests**: Same as unit test but with extensive debugging output
- **Debug features**: Row-by-row validation, detailed mismatch reporting
- **Exit criteria**: Unit test passes with detailed validation confirmation

#### `test_metal_unit_softmax.cpp`
- **What it does**: Isolated testing of softmax computation
- **Tests**: Single-operation softmax validation, numerical stability
- **Key validations**: Probability distribution properties, sum normalization
- **Exit criteria**: Softmax produces correct probability distributions

#### `test_metal_unit_topk_mask.cpp`
- **What it does**: Unit test for top-k masking operation
- **Tests**: Isolated masking logic, threshold application
- **Key validations**: Correct masking behavior, top-k preservation
- **Exit criteria**: Masking operation produces expected results

## Running Tests

### Build System
```bash
# Configure and build all tests
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Run specific test category
./bin/test_metal_unit_softmax
./bin/test_attention_correctness
./bin/test_api_compatibility
```

### Test Execution Strategy

#### Quick Validation (< 5 minutes)
```bash
# Essential functionality
./bin/test_api_compatibility
./bin/test_metal_unit_softmax
./bin/test_metal_unit_extract_k_values
```

#### Comprehensive Testing (15-30 minutes)
```bash
# All integration tests
for test in bin/test_metal_*; do ./$test; done

# All performance tests  
for test in bin/test_attention_* bin/test_flash_*; do ./$test; done
```

#### Debugging Workflow
```bash
# When issues occur, run debug versions
./bin/test_debug_validation
./bin/test_metal_unit_extract_k_values_debug
./bin/test_softmax_precision_debug
```

## Current Test Status and Known Issues

### Working Tests ✅
- API compatibility and compilation validation
- Basic softmax and extract_k_values operations
- Data type conversion and validation
- Memory allocation and basic inference
- Unit tests for individual operations

### Issues Under Investigation 🔍
- **FlashAttention Correctness**: FlashAttention kernel produces incorrect outputs compared to reference implementations
- **Attention Optimization**: Some optimized attention variants show numerical discrepancies
- **Performance Regression**: Certain optimizations may impact accuracy

### Testing Strategy for Current Issues

#### FlashAttention Debugging
1. **Run correctness comparison**: `test_attention_correctness.mm`
2. **Check parameter passing**: Verify `UnifiedParams` structure compatibility
3. **Validate intermediate results**: Use debug outputs to trace computation
4. **Compare with reference**: Run against original Metal and CUDA implementations

#### Performance Validation
1. **Benchmark current implementations**: `test_attention_benchmark.mm`
2. **Compare variants**: `test_kernel_comparison.mm`
3. **Validate optimizations**: Ensure performance gains don't compromise accuracy

## Test Development Guidelines

### Adding New Tests
1. Place in appropriate category directory
2. Use consistent naming: `test_[category]_[operation].cpp/mm`
3. Include both positive and edge case testing
4. Provide clear documentation of test purpose and expected behavior
5. Add CMakeLists.txt entries for new executables

### Test Quality Standards
- **Repeatability**: Tests must produce consistent results
- **Isolation**: Unit tests should not depend on other components
- **Coverage**: Test both normal operation and edge cases
- **Documentation**: Clear description of what is being tested and why
- **Validation**: Compare against reference implementations when possible

### Debugging Best Practices
- Use debug versions of tests for detailed analysis
- Enable verbose logging for complex issues
- Compare intermediate values with reference implementations
- Test edge cases and boundary conditions systematically

---

*Last updated: Based on current test suite analysis*
*Total test files: 35+ across all categories*
*Coverage: Unit → Integration → Performance → Compatibility*
# PIE Metal Backend

Metal-based computation backend for the PIE debugging framework.

## Build Requirements

- macOS with Metal support
- CMake 3.23+
- Xcode command line tools
- Python 3.11+ with pybind11

## Building

```bash
# Configure
cmake -B build -S .

# Build Metal kernels and Python bindings
cmake --build build

# Test bindings
cd build/lib
python3 -c "import metal_bindings; print('Success!')"
```

## Output Files

- `build/lib/pie_metal_kernels.metallib` - Compiled Metal kernels
- `build/lib/metal_bindings.cpython-*.so` - Python bindings module

## Integration with Debug Framework

The `CompilationEngine` will automatically detect and build this backend:

```python
from debug_framework.services.compilation_engine import CompilationEngine

engine = CompilationEngine("/tmp/output")
result = engine.compile_plugin({
    "name": "metal_kernels",
    "backend_dir": "/backend/backend-metal",
    "backend_type": "metal"
})
```


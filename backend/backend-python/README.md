# PIE Backend Python

Cross-platform Python backend with debugging framework support.

## Environment Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Quick Start

```bash
# macOS - Metal backend debugging
uv sync --extra debug --extra macos

# Linux - CUDA backend support
uv sync --extra debug --extra cuda

# All available dependencies (platform-filtered)
uv sync --all-extras
```

### Dependency Groups

- **Core**: Cross-platform dependencies (torch, numpy, ztensor)
- **debug**: TDD framework (pytest, pybind11, mypy, black)
- **cuda**: Linux-only CUDA support (flashinfer-python)
- **macos**: macOS-only Metal framework bindings (PyObjC)

### Development

```bash
# Run tests
uv run pytest

# Type checking
uv run mypy .

# Code formatting
uv run black .
```

### Multi-Language Plugin Support

- **C libraries**: ctypes (built-in)
- **C++**: pybind11
- **Objective-C/Metal**: PyObjC (macOS only)
# Pie Backend

A Python-based backend for Pie.

## Installation

```bash
pip install -e .
```

### Optional Dependencies

For CUDA support:
```bash
pip install -e ".[cuda]"
```

For Metal (macOS) support:
```bash
pip install -e ".[metal]"
```

## Project Structure

```
backend-python/
├── pyproject.toml       # Project configuration
├── src/
│   └── pie_backend/
│       ├── runtime.py   # Runtime module
│       └── server.py    # Server module
```

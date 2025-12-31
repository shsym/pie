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

### Verification

To verify the installation and environment compatibility, run:

```bash
pie-backend --doctor
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

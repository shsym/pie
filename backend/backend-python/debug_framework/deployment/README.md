# Debug Framework Production Deployment

This directory contains scripts and configurations for deploying the Debug Framework CLI tools in production environments.

## Quick Start

```bash
# Navigate to the pie project root
cd /path/to/pie

# Set up CLI tools
cd backend/backend-python
python debug_framework/deployment/setup_cli.py
```

## CLI Tools

The Debug Framework provides three main CLI tools:

### 1. debug-validate
Validates kernel implementations across different backends.

```bash
# Basic usage
python -m debug_framework.cli debug-validate --backend metal

# Compare multiple backends
python -m debug_framework.cli debug-validate --backend metal,python_reference --compare

# Generate validation report
python -m debug_framework.cli debug-validate --backend all --output validation_report.json --format json
```

### 2. plugin-compile
Compiles and manages backend plugins (Metal, CUDA, C++).

```bash
# Auto-discover and compile all backends
python -m debug_framework.cli plugin-compile --search /path/to/backends --output-dir ./compiled

# Compile specific project
python -m debug_framework.cli plugin-compile --project /path/to/metal-backend --backend-type metal

# Check toolchain availability
python -m debug_framework.cli plugin-compile --check-toolchain metal
```

### 3. session-report
Generates comprehensive debugging session reports.

```bash
# List recent sessions
python -m debug_framework.cli session-report --list --days 7

# Generate summary report
python -m debug_framework.cli session-report --summary --format json

# Detailed session report
python -m debug_framework.cli session-report --session session_001 --detailed --output report.txt
```

## Environment Configuration

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `PIE_DEBUG_ENABLED` | Enable debug output | `false` |
| `PIE_DEBUG_LEVEL` | Debug level (INFO, DEBUG) | `INFO` |
| `PIE_DEBUG_DATABASE` | Database file path | `~/.pie/debug/debug_framework.db` |
| `PIE_METAL_PATH` | Metal backend directory | Auto-detected |

## Production Setup

### 1. Environment Variables
```bash
export PIE_DEBUG_ENABLED=true
export PIE_DEBUG_LEVEL=INFO
export PIE_DEBUG_DATABASE=/opt/pie/debug/debug_framework.db
```

### 2. Database Location
The framework automatically creates a SQLite database at:
- Development: `~/.pie/debug/debug_framework.db`
- Custom: Set `PIE_DEBUG_DATABASE` environment variable

### 3. Backend Dependencies
- **Metal**: Requires macOS with Xcode command line tools
- **CUDA**: Requires CUDA toolkit and nvcc
- **PyTorch**: Requires PyTorch installation

## Integration with PIE

The Debug Framework integrates seamlessly with PIE model serving:

```python
import debug_framework

# Initialize framework
debug_framework.initialize_framework()

# Create validation session
session_id = debug_framework.create_validation_session(
    model_path="/path/to/model.zt",
    alternative_backend="auto"  # Auto-detects best available backend
)

# Access system information
info = debug_framework.get_system_info()
print(f"Available backends: {info['available_backends']}")
```

## Performance Considerations

- **Production Mode**: Disable `PIE_DEBUG_ENABLED` for reduced overhead
- **Database**: Use fast storage for database (SSD recommended)
- **Memory**: Tensor recording is limited automatically in production
- **Sessions**: Auto-cleanup prevents resource leaks

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure you're running from the correct directory
   ```bash
   cd /path/to/pie/backend/backend-python
   python -m debug_framework.cli debug-validate --help
   ```

2. **Metal Backend Not Found**: Set `PIE_METAL_PATH` environment variable
   ```bash
   export PIE_METAL_PATH=/path/to/pie/backend/backend-metal
   ```

3. **Database Permissions**: Check database directory write permissions
   ```bash
   mkdir -p ~/.pie/debug
   chmod 755 ~/.pie/debug
   ```

### Validation
Run the setup validation:
```bash
python debug_framework/deployment/setup_cli.py
```

## Files

- `setup_cli.py` - CLI setup and installation script
- `README.md` - This deployment guide
- `CLI_USAGE.md` - Generated after setup with detailed usage instructions
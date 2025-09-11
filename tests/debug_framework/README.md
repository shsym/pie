# Debug Framework Tests

This directory contains tests for the debug framework components.

## Running Tests

### Prerequisites
From the project root `/home/sslee/Workspace/pie`:
1. Navigate to the backend-python directory: `cd backend/backend-python`
2. Install dependencies with uv: `uv sync --extra debug`

### Test Commands

Run all debug framework tests:
```bash
uv run pytest ../../tests/debug_framework/ -v
```

Run specific test categories:
```bash
# Unit tests only
uv run pytest ../../tests/debug_framework/ -m "unit" -v

# Integration tests only
uv run pytest ../../tests/debug_framework/integration/ -v

# Specific test file
uv run pytest ../../tests/debug_framework/test_plugin_registry.py -v

# Specific test function
uv run pytest ../../tests/debug_framework/test_plugin_registry.py::TestPluginRegistry::test_plugin_registry_import_fails -v
```

### Test Structure

- `tests/debug_framework/` - Unit tests for individual components
- `tests/debug_framework/integration/` - Integration tests for component workflows

### Database

The debug framework uses a SQLite database that is **automatically created** when first used at:
```
backend/backend-python/debug_framework/data/debug_framework.db
```
No manual setup required - the database and schema are initialized automatically.
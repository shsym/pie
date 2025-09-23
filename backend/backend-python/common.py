"""Clean interface for importing common backend components."""

import sys
from pathlib import Path

def _setup_paths():
    """Set up the Python paths for common-python imports."""
    _backend_python_path = Path(__file__).parent.resolve()
    _common_python_path = (Path(__file__).parent.parent / "common-python").resolve()

    # Debug: Ensure paths exist
    if not _common_python_path.exists():
        raise RuntimeError(f"Common Python path does not exist: {_common_python_path}")

    # Add backend-python first (higher precedence for local modules)
    backend_python_str = str(_backend_python_path)
    if backend_python_str not in sys.path:
        sys.path.insert(0, backend_python_str)

    # Add common-python second (lower precedence)
    common_python_str = str(_common_python_path)
    if common_python_str not in sys.path:
        sys.path.insert(1, common_python_str)

# Set up paths immediately
_setup_paths()

# Import from common-python by temporarily prioritizing it
_common_python_path = (Path(__file__).parent.parent / "common-python").resolve()
common_python_str = str(_common_python_path)

# Temporarily move common-python to the front for these imports
if common_python_str in sys.path:
    sys.path.remove(common_python_str)
sys.path.insert(0, common_python_str)

# Now import and re-export everything from common-python
from config.common import *
from config.l4ma import *
from handler_common import *
from model_loader import *
from server_common import *
from debug_utils import *
from message import *
from adapter_import_utils import *

# Re-export model components
from model.l4ma import *
from model.l4ma_runtime import *
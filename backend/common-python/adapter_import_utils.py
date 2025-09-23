"""Utility for safely importing adapter functionality with fallback."""

from typing import TYPE_CHECKING

# Safe import of AdapterSubpass with fallback
try:
    from adapter import AdapterSubpass
except ImportError:
    # AdapterSubpass will be provided by the backend
    AdapterSubpass = None

# Type definitions for when AdapterSubpass is not available
if TYPE_CHECKING and AdapterSubpass is None:
    from typing import Any
    AdapterSubpass = Any  # For type checking when adapter is not available


def get_adapter_subpass():
    """Get the AdapterSubpass class, or None if not available."""
    return AdapterSubpass


def ensure_adapter_available():
    """Ensure AdapterSubpass is available, raising an error if not."""
    if AdapterSubpass is None:
        raise RuntimeError(
            "Adapter functionality is required but AdapterSubpass is not available. "
            "Please ensure the adapter module is properly imported."
        )
    return AdapterSubpass
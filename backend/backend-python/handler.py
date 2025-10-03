"""
Python Backend Handler

This module provides the backend handler for the Python backend.
It instantiates the common Handler class with the appropriate operations
backend (Metal on Apple Silicon, FlashInfer elsewhere).
"""

from common import Handler
from backend_ops import get_backend_ops


class PythonHandler(Handler):
    """Python backend handler using appropriate operations backend."""

    def __init__(
        self,
        config: dict,
    ):
        """Initialize Python handler with appropriate operations backend."""

        # Get appropriate backend (Metal on Apple Silicon, FlashInfer elsewhere)
        backend_ops = get_backend_ops()

        # Initialize parent with selected backend
        super().__init__(
            config=config,
            ops=backend_ops,
        )

        print(f"✅ PythonHandler initialized with {backend_ops.backend_name} backend")
        print(f"   {backend_ops.backend_name} available: {backend_ops.available}")

    def upload_handler(self, reqs):
        """Handle adapter upload requests."""
        _ = reqs  # Parameter not currently used
        raise NotImplementedError("upload_handler not yet implemented")

    def download_handler(self, reqs):
        """Handle adapter download requests."""
        _ = reqs  # Parameter not currently used
        raise NotImplementedError("download_handler not yet implemented")


# For backwards compatibility, create Handler as an alias to PythonHandler
Handler = PythonHandler

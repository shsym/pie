"""
Multi-Backend Debugging Framework

Production-ready framework for validating ML model implementations
across different backend implementations (Metal, CUDA, PyTorch, etc.)
with comprehensive numerical comparison and performance profiling.
"""

import os
import sys
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

# Framework version
__version__ = "1.0.0"

# Environment configuration
DEBUG_ENABLED = os.getenv("PIE_DEBUG_ENABLED", "false").lower() in ("true", "1", "yes")
DEBUG_LEVEL = os.getenv("PIE_DEBUG_LEVEL", "INFO").upper()
DATABASE_PATH = os.getenv("PIE_DEBUG_DATABASE", None)
METAL_BACKEND_PATH = os.getenv("PIE_METAL_PATH", None)

# Production-safe defaults
DEFAULT_CONFIG = {
    "database_path": DATABASE_PATH or str(Path.home() / ".pie" / "debug" / "debug_framework.db"),
    "performance_monitoring": True,
    "auto_cleanup": True,
    "max_session_duration": 3600,  # 1 hour
    "tensor_recording_limit": 100,  # Limit tensor recordings to prevent memory issues
    "comparison_tolerance": {"rtol": 1e-4, "atol": 1e-6}
}


class DebugFrameworkError(Exception):
    """Base exception for debug framework errors."""
    pass


class BackendInitializationError(DebugFrameworkError):
    """Raised when backend initialization fails."""
    pass


class ValidationError(DebugFrameworkError):
    """Raised when validation fails."""
    pass


def get_framework_config() -> Dict[str, Any]:
    """
    Get current framework configuration.

    Returns:
        Dictionary containing current configuration
    """
    config = DEFAULT_CONFIG.copy()

    # Override with environment variables
    if DATABASE_PATH:
        config["database_path"] = DATABASE_PATH

    # Performance optimizations for production
    if not DEBUG_ENABLED:
        config["tensor_recording_limit"] = 10  # Reduce in production
        config["max_session_duration"] = 1800  # 30 minutes in production

    return config


def initialize_framework(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize the debug framework with production-safe defaults.

    Args:
        config: Optional configuration overrides

    Returns:
        True if initialization successful

    Raises:
        DebugFrameworkError: If critical initialization fails
    """
    try:
        framework_config = get_framework_config()
        if config:
            framework_config.update(config)

        # Ensure database directory exists
        db_path = Path(framework_config["database_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database manager
        from debug_framework.services.database_manager import DatabaseManager

        try:
            db_manager = DatabaseManager(str(db_path))
            # Verify database is working
            db_manager._get_connection()
        except Exception as e:
            raise DebugFrameworkError(f"Database initialization failed: {e}")

        if DEBUG_ENABLED:
            print(f"🚀 Debug Framework v{__version__} initialized")
            print(f"   Database: {db_path}")
            print(f"   Debug level: {DEBUG_LEVEL}")
            print(f"   Performance monitoring: {'✅' if framework_config['performance_monitoring'] else '❌'}")

        return True

    except Exception as e:
        if DEBUG_ENABLED:
            print(f"❌ Framework initialization failed: {e}", file=sys.stderr)
        raise DebugFrameworkError(f"Framework initialization failed: {e}")


def detect_available_backends() -> Dict[str, bool]:
    """
    Auto-detect available backend implementations.

    Returns:
        Dictionary mapping backend names to availability status
    """
    backends = {
        "python_reference": True,  # Always available
        "metal": False,
        "cuda": False,
        "pytorch": False
    }

    try:
        # Detect Metal backend
        if sys.platform == "darwin":
            from debug_framework.integrations.metal_backend import MetalBackend

            try:
                metal_backend = MetalBackend(METAL_BACKEND_PATH)
                if metal_backend.initialize():
                    backends["metal"] = True
                    metal_backend.cleanup()
            except Exception:
                pass
    except ImportError:
        pass

    try:
        # Detect CUDA
        import subprocess
        result = subprocess.run(["nvcc", "--version"],
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            backends["cuda"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Detect PyTorch
        import torch
        backends["pytorch"] = True
    except ImportError:
        pass

    return backends


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and diagnostics.

    Returns:
        Dictionary containing system information
    """
    import platform

    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "framework_version": __version__,
        "debug_enabled": DEBUG_ENABLED,
        "available_backends": detect_available_backends()
    }

    if sys.platform == "darwin":
        try:
            import subprocess
            result = subprocess.run(["system_profiler", "SPHardwareDataType"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract chip information
                for line in result.stdout.split('\n'):
                    if 'Chip:' in line:
                        info["chip"] = line.split('Chip:')[1].strip()
                        break
        except Exception:
            pass

    return info


def create_validation_session(model_path: str,
                            reference_backend: str = "python_reference",
                            alternative_backend: str = "auto",
                            config: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a new validation session with automatic backend detection.

    Args:
        model_path: Path to the model to validate
        reference_backend: Reference backend name
        alternative_backend: Alternative backend name ("auto" for auto-detection)
        config: Optional session configuration

    Returns:
        Session ID

    Raises:
        BackendInitializationError: If backends cannot be initialized
        ValidationError: If session creation fails
    """
    try:
        # Initialize framework if not already done
        if not hasattr(create_validation_session, '_initialized'):
            initialize_framework()
            create_validation_session._initialized = True

        # Auto-detect alternative backend if requested
        if alternative_backend == "auto":
            available_backends = detect_available_backends()

            # Prefer Metal on macOS, then CUDA, then PyTorch
            if available_backends.get("metal"):
                alternative_backend = "metal"
            elif available_backends.get("cuda"):
                alternative_backend = "cuda"
            elif available_backends.get("pytorch"):
                alternative_backend = "pytorch"
            else:
                raise BackendInitializationError("No alternative backends available")

        # Create validation engine and session
        from debug_framework.services.validation_engine import ValidationEngine

        validation_engine = ValidationEngine()

        session_config = {
            "enabled_checkpoints": ["post_embedding", "post_attention", "post_mlp", "post_processing"],
            "precision_thresholds": DEFAULT_CONFIG["comparison_tolerance"],
            "tensor_recording_enabled": DEBUG_ENABLED,
            "performance_profiling_enabled": True
        }

        if config:
            session_config.update(config)

        session_id = validation_engine.create_session(
            model_path=model_path,
            config=session_config,
            reference_backend=reference_backend,
            alternative_backend=alternative_backend
        )

        if DEBUG_ENABLED:
            print(f"🎯 Validation session created: {session_id}")
            print(f"   Model: {model_path}")
            print(f"   Reference: {reference_backend}")
            print(f"   Alternative: {alternative_backend}")

        return session_id

    except Exception as e:
        raise ValidationError(f"Failed to create validation session: {e}")


# Auto-initialize if debug is enabled
if DEBUG_ENABLED:
    try:
        initialize_framework()
    except DebugFrameworkError as e:
        warnings.warn(f"Debug framework initialization failed: {e}")


# Export main classes and functions for easy import
__all__ = [
    "DebugFrameworkError",
    "BackendInitializationError",
    "ValidationError",
    "initialize_framework",
    "get_framework_config",
    "detect_available_backends",
    "get_system_info",
    "create_validation_session"
]
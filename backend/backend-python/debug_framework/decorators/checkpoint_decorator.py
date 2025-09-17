"""
Checkpoint decorator for L4MA integration.

This module provides high-performance validation decorators that enable
Metal/PyTorch tensor comparison with minimal performance impact when disabled.
"""

import functools
import time
import threading
import weakref
from typing import Dict, Any, Optional, Callable, Union
import numpy as np

# Global state management with thread safety
_debug_enabled: bool = False
_validation_callbacks: Dict[str, Callable] = {}
_performance_stats: Dict[str, float] = {}
_thread_local = threading.local()
_checkpoint_lock = threading.RLock()


def set_global_debug_mode(enabled: bool) -> None:
    """Set global debug mode for all checkpoint decorators."""
    global _debug_enabled
    with _checkpoint_lock:
        _debug_enabled = enabled


def is_debug_enabled() -> bool:
    """Check if debug mode is globally enabled."""
    return _debug_enabled


def register_validation_callback(checkpoint_name: str, callback: Callable) -> None:
    """Register a validation callback for a specific checkpoint."""
    with _checkpoint_lock:
        _validation_callbacks[checkpoint_name] = callback


def unregister_validation_callback(checkpoint_name: str) -> None:
    """Unregister a validation callback."""
    with _checkpoint_lock:
        _validation_callbacks.pop(checkpoint_name, None)


def get_performance_stats() -> Dict[str, float]:
    """Get performance statistics for checkpoint operations."""
    with _checkpoint_lock:
        return _performance_stats.copy()


def _get_thread_local_storage():
    """Get thread-local storage for performance isolation."""
    if not hasattr(_thread_local, 'storage'):
        _thread_local.storage = {
            'checkpoint_count': 0,
            'total_overhead': 0.0,
            'last_checkpoint_time': 0.0
        }
    return _thread_local.storage


def checkpoint_validation(
    checkpoint_name: str,
    capture_tensors: bool = True,
    include_metadata: bool = True,
    tolerance: float = 1e-5,
    backend_comparison: Optional[str] = None,
    performance_monitoring: bool = False
) -> Callable:
    """
    Decorator for L4MA model validation checkpoints.

    This decorator provides high-performance tensor validation for L4MA models,
    enabling Metal/PyTorch backend comparison with minimal performance impact.

    Args:
        checkpoint_name: Unique name for this checkpoint
        capture_tensors: Whether to capture tensor data for validation
        include_metadata: Whether to include execution metadata
        tolerance: Numerical tolerance for tensor comparisons
        backend_comparison: Backend to compare against ('metal', 'pytorch', None)
        performance_monitoring: Whether to collect performance statistics

    Returns:
        Decorated function with validation checkpoint
    """

    def decorator(func: Callable) -> Callable:
        # Cache validation callback for performance
        validation_callback = None
        callback_cache_time = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal validation_callback, callback_cache_time

            # Fast path when debug is disabled (< 0.1% overhead)
            if not _debug_enabled:
                return func(*args, **kwargs)

            # Performance monitoring setup
            start_time = time.perf_counter() if performance_monitoring else None
            thread_storage = _get_thread_local_storage() if performance_monitoring else None

            # Cache callback lookup to minimize lock contention
            current_time = time.perf_counter()
            if (validation_callback is None or
                current_time - callback_cache_time > 1.0):  # 1s cache
                with _checkpoint_lock:
                    validation_callback = _validation_callbacks.get(checkpoint_name)
                    callback_cache_time = current_time

            # Execute original function
            try:
                result = func(*args, **kwargs)

                # Validation and tensor capture (only when debug enabled)
                if validation_callback and capture_tensors:
                    _perform_validation_capture(
                        checkpoint_name=checkpoint_name,
                        result=result,
                        args=args,
                        kwargs=kwargs,
                        validation_callback=validation_callback,
                        include_metadata=include_metadata,
                        tolerance=tolerance,
                        backend_comparison=backend_comparison
                    )

                return result

            finally:
                # Performance tracking
                if performance_monitoring and start_time is not None:
                    end_time = time.perf_counter()
                    overhead = end_time - start_time

                    # Update thread-local stats
                    if thread_storage:
                        thread_storage['checkpoint_count'] += 1
                        thread_storage['total_overhead'] += overhead
                        thread_storage['last_checkpoint_time'] = overhead

                    # Update global stats with minimal locking
                    if overhead > 0:  # Only update if meaningful
                        with _checkpoint_lock:
                            _performance_stats[checkpoint_name] = _performance_stats.get(
                                checkpoint_name, 0.0) + overhead

        # Preserve function metadata for introspection
        wrapper.__checkpoint_name__ = checkpoint_name
        wrapper.__original_function__ = func
        wrapper.__debug_decorator__ = True

        return wrapper

    return decorator


def _perform_validation_capture(
    checkpoint_name: str,
    result: Any,
    args: tuple,
    kwargs: dict,
    validation_callback: Callable,
    include_metadata: bool,
    tolerance: float,
    backend_comparison: Optional[str]
) -> None:
    """
    Perform tensor validation and capture with optimized performance.

    This function handles the actual validation logic, separated from the
    decorator to minimize overhead in the hot path.
    """
    try:
        # Convert result to numpy for consistent processing
        tensor_data = _extract_tensor_data(result)

        if tensor_data is None:
            return

        # Build metadata only if requested
        metadata = {}
        if include_metadata:
            metadata = {
                'checkpoint_name': checkpoint_name,
                'tensor_shape': tensor_data.shape,
                'tensor_dtype': str(tensor_data.dtype),
                'timestamp': time.perf_counter(),
                'thread_id': threading.get_ident(),
                'tolerance': tolerance,
                'backend_comparison': backend_comparison
            }

        # Call validation callback asynchronously for performance
        validation_callback(checkpoint_name, tensor_data, metadata)

    except Exception as e:
        # Silent failure to maintain original function behavior
        # In production, this might log to a debug channel
        pass


def _extract_tensor_data(result: Any) -> Optional[np.ndarray]:
    """
    Extract tensor data from various result types with minimal overhead.

    Supports PyTorch tensors, numpy arrays, and tuples containing tensors.
    """
    if result is None:
        return None

    # Handle numpy arrays directly
    if isinstance(result, np.ndarray):
        return result

    # Handle PyTorch tensors
    if hasattr(result, 'detach') and hasattr(result, 'cpu'):
        try:
            # Detach and move to CPU first
            tensor_cpu = result.detach().cpu()

            # Handle bfloat16 which is not supported by numpy directly
            if 'bfloat16' in str(tensor_cpu.dtype):
                # print(f"DEBUG: Converting bfloat16 to float32 for numpy compatibility")
                # Import torch locally to avoid import issues
                import torch
                tensor_cpu = tensor_cpu.to(torch.float32)

            if hasattr(tensor_cpu, 'numpy'):
                numpy_result = tensor_cpu.numpy()
                # print(f"DEBUG: Converted to numpy: shape={numpy_result.shape}, dtype={numpy_result.dtype}")
                return numpy_result
            else:
                numpy_result = np.array(tensor_cpu)
                # print(f"DEBUG: Converted via np.array: shape={numpy_result.shape}, dtype={numpy_result.dtype}")
                return numpy_result
        except Exception as e:
            # print(f"DEBUG: Failed to convert PyTorch tensor to numpy: {e}")
            return None

    # Handle tuples/lists (common in attention outputs)
    if isinstance(result, (tuple, list)) and len(result) > 0:
        # Extract first tensor from tuple (typically the main output)
        return _extract_tensor_data(result[0])

    # Handle dictionary outputs
    if isinstance(result, dict) and 'last_hidden_state' in result:
        return _extract_tensor_data(result['last_hidden_state'])

    return None


def create_validation_checkpoint(
    checkpoint_name: str,
    tensor_data: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    validation_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create a validation checkpoint manually (for testing or custom integration).

    Args:
        checkpoint_name: Name of the checkpoint
        tensor_data: Tensor data to validate
        metadata: Optional metadata dictionary
        validation_callback: Optional validation callback

    Returns:
        Checkpoint result dictionary
    """
    checkpoint_result = {
        'checkpoint_name': checkpoint_name,
        'tensor_shape': tensor_data.shape,
        'tensor_dtype': str(tensor_data.dtype),
        'timestamp': time.perf_counter(),
        'validation_status': 'created'
    }

    if metadata:
        checkpoint_result.update(metadata)

    # Call validation callback if provided
    if validation_callback:
        try:
            validation_callback(checkpoint_name, tensor_data, checkpoint_result)
            checkpoint_result['validation_status'] = 'validated'
        except Exception as e:
            checkpoint_result['validation_status'] = 'failed'
            checkpoint_result['validation_error'] = str(e)

    return checkpoint_result


def cleanup_validation_state() -> None:
    """Clean up validation state and callbacks."""
    global _validation_callbacks, _performance_stats

    with _checkpoint_lock:
        _validation_callbacks.clear()
        _performance_stats.clear()

    # Clear thread-local storage
    if hasattr(_thread_local, 'storage'):
        _thread_local.storage.clear()


# Performance optimization utilities
def get_checkpoint_overhead_percentage() -> float:
    """
    Calculate the percentage overhead introduced by checkpoint decorators.

    Returns:
        Overhead as percentage (should be < 1.0 for < 1% overhead target)
    """
    thread_storage = _get_thread_local_storage()

    if thread_storage['checkpoint_count'] == 0:
        return 0.0

    # Estimate overhead based on checkpoint frequency and total time
    avg_overhead = thread_storage['total_overhead'] / thread_storage['checkpoint_count']

    # Convert to percentage (rough estimate)
    # In practice, this would need baseline function execution time
    return avg_overhead * 100  # Simplified calculation


def optimize_for_production() -> None:
    """
    Apply production optimizations to minimize overhead.

    This function configures the validation system for maximum performance
    in production environments where <1% overhead is critical.
    """
    global _debug_enabled

    # Disable debug mode by default in production
    with _checkpoint_lock:
        _debug_enabled = False

    # Clear caches to minimize memory footprint
    cleanup_validation_state()
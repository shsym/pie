"""
Debug Framework Decorators

Checkpoint decorators for L4MA integration and
debug session management.
"""

from .checkpoint_decorator import (
    checkpoint_validation,
    set_global_debug_mode,
    is_debug_enabled,
    register_validation_callback,
    unregister_validation_callback,
    get_performance_stats,
    create_validation_checkpoint,
    cleanup_validation_state,
    get_checkpoint_overhead_percentage,
    optimize_for_production
)

__all__ = [
    'checkpoint_validation',
    'set_global_debug_mode',
    'is_debug_enabled',
    'register_validation_callback',
    'unregister_validation_callback',
    'get_performance_stats',
    'create_validation_checkpoint',
    'cleanup_validation_state',
    'get_checkpoint_overhead_percentage',
    'optimize_for_production'
]
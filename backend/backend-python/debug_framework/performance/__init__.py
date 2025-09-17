"""
Debug Framework Performance

Performance optimization utilities including memory pooling,
async logging, and execution profiling.
"""

from .memory_pool import (
    MemoryPool,
    PooledTensorBuffer,
    get_memory_pool,
    allocate_tensor_buffer,
    get_pool_stats,
    cleanup_memory_pool
)

from .async_logger import (
    AsyncLogger,
    LogLevel,
    LogEntry,
    get_async_logger,
    log_debug,
    log_info,
    log_warning,
    log_error,
    log_critical,
    get_logging_stats,
    flush_logs,
    shutdown_logging
)

from .profiler import (
    LightweightProfiler,
    ProfileEntry,
    ProfileSummary,
    PerformanceMonitor,
    profile_function,
    get_profiler,
    profile_operation,
    get_profiling_summary,
    get_all_profiling_summaries,
    get_profiler_overhead,
    clear_profiling_data,
    enable_profiling,
    export_profiling_data
)

__all__ = [
    # Memory Pool
    'MemoryPool',
    'PooledTensorBuffer',
    'get_memory_pool',
    'allocate_tensor_buffer',
    'get_pool_stats',
    'cleanup_memory_pool',

    # Async Logger
    'AsyncLogger',
    'LogLevel',
    'LogEntry',
    'get_async_logger',
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',
    'log_critical',
    'get_logging_stats',
    'flush_logs',
    'shutdown_logging',

    # Profiler
    'LightweightProfiler',
    'ProfileEntry',
    'ProfileSummary',
    'PerformanceMonitor',
    'profile_function',
    'get_profiler',
    'profile_operation',
    'get_profiling_summary',
    'get_all_profiling_summaries',
    'get_profiler_overhead',
    'clear_profiling_data',
    'enable_profiling',
    'export_profiling_data'
]
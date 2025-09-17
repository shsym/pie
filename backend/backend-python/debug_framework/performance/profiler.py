"""
Lightweight execution profiler for debug framework performance monitoring.

Provides minimal-overhead profiling to track performance impact and ensure
<5% overhead target is maintained.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, ContextManager
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import statistics


@dataclass
class ProfileEntry:
    """Represents a single profiling measurement."""
    name: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    thread_id: int = 0
    session_id: Optional[str] = None


@dataclass
class ProfileSummary:
    """Summary statistics for a profiled operation."""
    name: str
    call_count: int
    total_time: float
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    p95_time: float
    p99_time: float


class LightweightProfiler:
    """
    Ultra-lightweight profiler designed for <1% overhead.
    """

    def __init__(self, max_entries: int = 10000, enable_detailed: bool = False):
        """
        Initialize profiler.

        Args:
            max_entries: Maximum profile entries to keep in memory
            enable_detailed: Whether to collect detailed statistics
        """
        self.max_entries = max_entries
        self.enable_detailed = enable_detailed

        # Thread-safe storage
        self._lock = threading.RLock()
        self._entries: deque = deque(maxlen=max_entries)
        self._summaries: Dict[str, List[float]] = defaultdict(list)
        self._active_timers: Dict[int, Dict[str, float]] = defaultdict(dict)

        # Performance tracking
        self._profiler_overhead = 0.0
        self._measurement_count = 0
        self._enabled = True

    def start_timer(self, name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Start timing an operation.

        Args:
            name: Operation name
            metadata: Additional metadata

        Returns:
            Timer ID for stopping
        """
        if not self._enabled:
            return name

        overhead_start = time.perf_counter()

        thread_id = threading.get_ident()
        start_time = time.perf_counter()

        with self._lock:
            self._active_timers[thread_id][name] = start_time

        # Track profiler overhead
        overhead_end = time.perf_counter()
        self._profiler_overhead += overhead_end - overhead_start

        return name

    def stop_timer(
        self,
        name: str,
        session_id: Optional[str] = None
    ) -> Optional[float]:
        """
        Stop timing an operation.

        Args:
            name: Operation name (timer ID)
            session_id: Associated session ID

        Returns:
            Duration in seconds, or None if timer not found
        """
        if not self._enabled:
            return None

        overhead_start = time.perf_counter()
        end_time = time.perf_counter()
        thread_id = threading.get_ident()

        with self._lock:
            if thread_id not in self._active_timers or name not in self._active_timers[thread_id]:
                return None

            start_time = self._active_timers[thread_id].pop(name)
            duration = end_time - start_time

            # Create profile entry
            entry = ProfileEntry(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                thread_id=thread_id,
                session_id=session_id
            )

            self._entries.append(entry)
            self._summaries[name].append(duration)
            self._measurement_count += 1

            # Track profiler overhead
            overhead_end = time.perf_counter()
            self._profiler_overhead += overhead_end - overhead_start

            return duration

    @contextmanager
    def profile(self, name: str, session_id: Optional[str] = None) -> ContextManager[None]:
        """
        Context manager for profiling a block of code.

        Args:
            name: Operation name
            session_id: Associated session ID
        """
        timer_id = self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(timer_id, session_id)

    def get_summary(self, name: str) -> Optional[ProfileSummary]:
        """
        Get summary statistics for an operation.

        Args:
            name: Operation name

        Returns:
            ProfileSummary or None if no data
        """
        with self._lock:
            durations = self._summaries.get(name, [])

            if not durations:
                return None

            # Calculate statistics
            sorted_durations = sorted(durations)
            count = len(durations)

            return ProfileSummary(
                name=name,
                call_count=count,
                total_time=sum(durations),
                mean_time=statistics.mean(durations),
                median_time=statistics.median(durations),
                min_time=min(durations),
                max_time=max(durations),
                std_dev=statistics.stdev(durations) if count > 1 else 0.0,
                p95_time=sorted_durations[int(0.95 * count)] if count > 1 else durations[0],
                p99_time=sorted_durations[int(0.99 * count)] if count > 1 else durations[0]
            )

    def get_all_summaries(self) -> List[ProfileSummary]:
        """Get summaries for all profiled operations."""
        with self._lock:
            summaries = []
            for name in self._summaries.keys():
                summary = self.get_summary(name)
                if summary:
                    summaries.append(summary)
            return sorted(summaries, key=lambda x: x.total_time, reverse=True)

    def get_overhead_percentage(self) -> float:
        """
        Calculate profiler overhead as percentage of total measured time.

        Returns:
            Overhead percentage (should be <1% for our target)
        """
        with self._lock:
            if self._measurement_count == 0:
                return 0.0

            total_measured_time = sum(
                sum(durations) for durations in self._summaries.values()
            )

            if total_measured_time == 0:
                return 0.0

            return (self._profiler_overhead / total_measured_time) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        with self._lock:
            active_timers = sum(len(timers) for timers in self._active_timers.values())

            return {
                'enabled': self._enabled,
                'total_measurements': self._measurement_count,
                'active_timers': active_timers,
                'entries_stored': len(self._entries),
                'operations_tracked': len(self._summaries),
                'overhead_seconds': self._profiler_overhead,
                'overhead_percentage': self.get_overhead_percentage(),
                'memory_usage_entries': len(self._entries),
                'max_entries': self.max_entries
            }

    def clear_data(self) -> None:
        """Clear all profiling data."""
        with self._lock:
            self._entries.clear()
            self._summaries.clear()
            self._active_timers.clear()
            self._profiler_overhead = 0.0
            self._measurement_count = 0

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable profiling."""
        self._enabled = enabled

    def export_data(self) -> Dict[str, Any]:
        """Export all profiling data."""
        with self._lock:
            return {
                'entries': [
                    {
                        'name': entry.name,
                        'start_time': entry.start_time,
                        'end_time': entry.end_time,
                        'duration': entry.duration,
                        'thread_id': entry.thread_id,
                        'session_id': entry.session_id,
                        'metadata': entry.metadata
                    }
                    for entry in self._entries
                ],
                'summaries': {
                    name: {
                        'durations': durations,
                        'summary': self.get_summary(name).__dict__
                    }
                    for name, durations in self._summaries.items()
                },
                'stats': self.get_stats()
            }


def profile_function(name: Optional[str] = None):
    """
    Decorator for profiling function execution.

    Args:
        name: Optional custom name for the operation
    """
    def decorator(func: Callable) -> Callable:
        operation_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()

            with profiler.profile(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global profiler instance
_global_profiler = None
_profiler_lock = threading.Lock()


def get_profiler() -> LightweightProfiler:
    """Get global profiler instance (singleton)."""
    global _global_profiler

    if _global_profiler is None:
        with _profiler_lock:
            if _global_profiler is None:
                _global_profiler = LightweightProfiler(
                    max_entries=10000,
                    enable_detailed=True
                )

    return _global_profiler


def profile_operation(name: str, session_id: Optional[str] = None) -> ContextManager[None]:
    """Global function to profile an operation."""
    return get_profiler().profile(name, session_id)


def get_profiling_summary(operation_name: str) -> Optional[ProfileSummary]:
    """Get summary for a specific operation."""
    return get_profiler().get_summary(operation_name)


def get_all_profiling_summaries() -> List[ProfileSummary]:
    """Get all profiling summaries."""
    return get_profiler().get_all_summaries()


def get_profiler_overhead() -> float:
    """Get current profiler overhead percentage."""
    return get_profiler().get_overhead_percentage()


def clear_profiling_data() -> None:
    """Clear all profiling data."""
    get_profiler().clear_data()


def enable_profiling(enabled: bool = True) -> None:
    """Enable or disable global profiling."""
    get_profiler().set_enabled(enabled)


def export_profiling_data() -> Dict[str, Any]:
    """Export all profiling data."""
    return get_profiler().export_data()


# Performance monitoring utilities
class PerformanceMonitor:
    """
    Monitor performance impact of debug framework operations.
    """

    def __init__(self, target_overhead_percent: float = 5.0):
        """
        Initialize performance monitor.

        Args:
            target_overhead_percent: Maximum acceptable overhead percentage
        """
        self.target_overhead_percent = target_overhead_percent
        self._baseline_measurements: Dict[str, float] = {}
        self._debug_measurements: Dict[str, float] = {}

    def measure_baseline(self, operation_name: str, operation: Callable) -> float:
        """
        Measure baseline performance without debug framework.

        Args:
            operation_name: Name of the operation
            operation: Function to measure

        Returns:
            Baseline execution time
        """
        # Disable profiling during baseline measurement
        original_enabled = get_profiler()._enabled
        get_profiler().set_enabled(False)

        try:
            start_time = time.perf_counter()
            operation()
            end_time = time.perf_counter()

            baseline_time = end_time - start_time
            self._baseline_measurements[operation_name] = baseline_time
            return baseline_time

        finally:
            get_profiler().set_enabled(original_enabled)

    def measure_with_debug(self, operation_name: str, operation: Callable) -> float:
        """
        Measure performance with debug framework enabled.

        Args:
            operation_name: Name of the operation
            operation: Function to measure

        Returns:
            Execution time with debug framework
        """
        with profile_operation(f"debug_{operation_name}"):
            start_time = time.perf_counter()
            operation()
            end_time = time.perf_counter()

        debug_time = end_time - start_time
        self._debug_measurements[operation_name] = debug_time
        return debug_time

    def calculate_overhead(self, operation_name: str) -> Optional[float]:
        """
        Calculate overhead percentage for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Overhead percentage or None if measurements missing
        """
        baseline = self._baseline_measurements.get(operation_name)
        debug = self._debug_measurements.get(operation_name)

        if baseline is None or debug is None:
            return None

        if baseline == 0:
            return float('inf') if debug > 0 else 0.0

        return ((debug - baseline) / baseline) * 100

    def check_overhead_compliance(self) -> Dict[str, Any]:
        """
        Check if all operations meet overhead target.

        Returns:
            Compliance report
        """
        compliant = True
        violations = []
        report = {
            'target_overhead_percent': self.target_overhead_percent,
            'compliant': True,
            'violations': [],
            'measurements': {}
        }

        for operation_name in self._baseline_measurements.keys():
            overhead = self.calculate_overhead(operation_name)

            if overhead is not None:
                report['measurements'][operation_name] = {
                    'baseline_time': self._baseline_measurements[operation_name],
                    'debug_time': self._debug_measurements.get(operation_name, 0),
                    'overhead_percent': overhead,
                    'compliant': overhead <= self.target_overhead_percent
                }

                if overhead > self.target_overhead_percent:
                    compliant = False
                    violations.append({
                        'operation': operation_name,
                        'overhead_percent': overhead,
                        'target_percent': self.target_overhead_percent
                    })

        report['compliant'] = compliant
        report['violations'] = violations

        return report
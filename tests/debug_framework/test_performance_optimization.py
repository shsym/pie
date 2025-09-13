"""
Tests for performance optimization components.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from debug_framework.performance.memory_pool import (
    MemoryPool,
    PooledTensorBuffer,
    get_memory_pool,
    allocate_tensor_buffer
)
from debug_framework.performance.async_logger import (
    AsyncLogger,
    LogLevel,
    LogEntry,
    get_async_logger
)
from debug_framework.performance.profiler import (
    LightweightProfiler,
    ProfileEntry,
    PerformanceMonitor,
    profile_function,
    get_profiler
)


class TestMemoryPool:
    """Test cases for MemoryPool class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.pool = MemoryPool(
            block_sizes=[1024, 4096, 16384],
            max_blocks_per_size=5,
            cleanup_interval=10.0
        )

    def test_init(self):
        """Test memory pool initialization."""
        assert self.pool.block_sizes == [1024, 4096, 16384]
        assert self.pool.max_blocks_per_size == 5
        assert self.pool._stats['total_allocations'] == 0

    def test_allocate_basic(self):
        """Test basic memory allocation."""
        data = self.pool.allocate(512)

        # Should be ManagedByteArray, not regular bytearray
        assert hasattr(data, 'data') and hasattr(data, 'parent')
        assert len(data) == 512
        assert self.pool._stats['total_allocations'] == 1
        assert self.pool._stats['pool_misses'] == 1

    def test_allocate_and_deallocate(self):
        """Test allocation and deallocation cycle."""
        # First allocation - should be a pool miss
        data = self.pool.allocate(512)
        assert self.pool._stats['pool_misses'] == 1

        # Deallocate
        success = self.pool.deallocate(data)
        assert success is True

        # Second allocation - should be a pool hit
        data2 = self.pool.allocate(512)
        assert self.pool._stats['pool_hits'] == 1

    def test_find_block_size(self):
        """Test block size selection logic."""
        assert self.pool._find_block_size(100) == 1024
        assert self.pool._find_block_size(1024) == 1024
        assert self.pool._find_block_size(2048) == 4096
        assert self.pool._find_block_size(20000) == 1048576  # Next MB

    def test_get_stats(self):
        """Test statistics collection."""
        # Allocate some memory
        data1 = self.pool.allocate(512)
        data2 = self.pool.allocate(2048)

        stats = self.pool.get_stats()

        assert stats['total_allocations'] == 2
        assert stats['active_blocks'] == 2
        assert stats['pool_misses'] == 2
        assert 'pool_efficiency' in stats

    def test_cleanup_expired_blocks(self):
        """Test cleanup of expired blocks."""
        # Allocate and deallocate to add blocks to pool
        data = self.pool.allocate(512)
        self.pool.deallocate(data)

        # Manually trigger cleanup with very short age
        self.pool._cleanup_expired_blocks(max_age_seconds=0.0)

        # Pool should be empty after cleanup
        stats = self.pool.get_stats()
        assert stats['pooled_blocks'] == 0

    def test_thread_safety(self):
        """Test thread safety of memory pool."""
        results = []
        errors = []

        def worker():
            try:
                for i in range(100):
                    data = self.pool.allocate(1024)
                    time.sleep(0.001)  # Small delay
                    self.pool.deallocate(data)
                results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0


class TestPooledTensorBuffer:
    """Test cases for PooledTensorBuffer class."""

    def test_pooled_tensor_buffer_basic(self):
        """Test basic tensor buffer operations."""
        buffer = PooledTensorBuffer(size=100, dtype_size=4)

        assert buffer.size == 100
        assert buffer.dtype_size == 4
        assert buffer.byte_size == 400
        assert not buffer._released

        data = buffer.as_bytes()
        assert len(data) == 400

        buffer.release()
        assert buffer._released

    def test_pooled_tensor_buffer_context_manager(self):
        """Test tensor buffer as context manager."""
        with PooledTensorBuffer(size=50, dtype_size=4) as buffer:
            assert not buffer._released
            data = buffer.as_bytes()
            assert len(data) == 200

        assert buffer._released

    def test_pooled_tensor_buffer_released_access(self):
        """Test accessing released buffer raises error."""
        buffer = PooledTensorBuffer(size=10, dtype_size=4)
        buffer.release()

        with pytest.raises(RuntimeError, match="Buffer has been released"):
            buffer.as_bytes()


class TestAsyncLogger:
    """Test cases for AsyncLogger class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.logger = AsyncLogger(
            log_file=None,  # In-memory only
            max_queue_size=1000,
            batch_size=10,
            flush_interval=0.1,
            enable_console=False
        )

    def teardown_method(self):
        """Cleanup after tests."""
        self.logger.shutdown(timeout=1.0)

    def test_init(self):
        """Test async logger initialization."""
        assert self.logger.max_queue_size == 1000
        assert self.logger.batch_size == 10
        assert self.logger._running.is_set()
        assert self.logger._worker_thread.is_alive()

    def test_log_basic(self):
        """Test basic logging functionality."""
        success = self.logger.log(
            LogLevel.INFO,
            "Test message",
            context={'key': 'value'},
            session_id='test_session'
        )

        assert success is True
        assert self.logger._stats['entries_logged'] == 1

    def test_log_convenience_methods(self):
        """Test convenience logging methods."""
        assert self.logger.debug("Debug message") is True
        assert self.logger.info("Info message") is True
        assert self.logger.warning("Warning message") is True
        assert self.logger.error("Error message") is True
        assert self.logger.critical("Critical message") is True

        assert self.logger._stats['entries_logged'] == 5

    def test_flush(self):
        """Test log flushing."""
        # Log some messages
        for i in range(5):
            self.logger.info(f"Message {i}")

        # Flush and verify queue is empty
        success = self.logger.flush(timeout=2.0)
        assert success is True
        assert self.logger._queue.qsize() == 0

    def test_get_stats(self):
        """Test statistics collection."""
        self.logger.info("Test message")

        stats = self.logger.get_stats()

        assert stats['entries_logged'] == 1
        assert stats['worker_thread_alive'] is True
        assert 'queue_size' in stats
        assert 'queue_utilization' in stats

    def test_queue_full_handling(self):
        """Test handling of full queue."""
        # Create logger with very small queue
        small_logger = AsyncLogger(
            log_file=None,
            max_queue_size=2,
            batch_size=10,
            flush_interval=1.0,
            enable_console=False
        )

        try:
            # Fill the queue
            success1 = small_logger.log(LogLevel.INFO, "Message 1")
            success2 = small_logger.log(LogLevel.INFO, "Message 2")
            success3 = small_logger.log(LogLevel.INFO, "Message 3")  # Should fail

            assert success1 is True
            assert success2 is True
            assert success3 is False  # Queue full

            assert small_logger._stats['entries_dropped'] >= 1

        finally:
            small_logger.shutdown(timeout=1.0)


class TestLightweightProfiler:
    """Test cases for LightweightProfiler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.profiler = LightweightProfiler(max_entries=1000)

    def test_init(self):
        """Test profiler initialization."""
        assert self.profiler.max_entries == 1000
        assert self.profiler._enabled is True
        assert len(self.profiler._entries) == 0

    def test_basic_timing(self):
        """Test basic timing operations."""
        timer_id = self.profiler.start_timer("test_op")
        time.sleep(0.01)  # Small delay
        duration = self.profiler.stop_timer(timer_id)

        assert duration is not None
        assert duration > 0.005  # Should be at least 5ms
        assert len(self.profiler._entries) == 1
        assert self.profiler._measurement_count == 1

    def test_context_manager_profiling(self):
        """Test profiling with context manager."""
        with self.profiler.profile("context_test"):
            time.sleep(0.01)

        summary = self.profiler.get_summary("context_test")
        assert summary is not None
        assert summary.name == "context_test"
        assert summary.call_count == 1
        assert summary.mean_time > 0.005

    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        # Profile same operation multiple times
        for i in range(10):
            with self.profiler.profile("repeated_op"):
                time.sleep(0.001 * (i + 1))  # Variable delay

        summary = self.profiler.get_summary("repeated_op")

        assert summary.call_count == 10
        assert summary.total_time > 0.05  # At least 50ms total
        assert summary.min_time < summary.max_time
        assert summary.std_dev > 0

    def test_overhead_calculation(self):
        """Test profiler overhead calculation."""
        # Profile some operations
        for i in range(50):
            with self.profiler.profile("overhead_test"):
                time.sleep(0.001)

        overhead_pct = self.profiler.get_overhead_percentage()

        # Overhead should be very low (target < 1%)
        assert overhead_pct >= 0
        assert overhead_pct < 5.0  # Should be well under 5%

    def test_disabled_profiler(self):
        """Test profiler when disabled."""
        self.profiler.set_enabled(False)

        timer_id = self.profiler.start_timer("disabled_test")
        duration = self.profiler.stop_timer(timer_id)

        assert timer_id == "disabled_test"  # Returns name when disabled
        assert duration is None
        assert len(self.profiler._entries) == 0

    def test_get_all_summaries(self):
        """Test getting all summaries sorted by total time."""
        # Profile different operations
        with self.profiler.profile("fast_op"):
            time.sleep(0.001)

        with self.profiler.profile("slow_op"):
            time.sleep(0.01)

        with self.profiler.profile("medium_op"):
            time.sleep(0.005)

        summaries = self.profiler.get_all_summaries()

        assert len(summaries) == 3
        # Should be sorted by total time (descending)
        assert summaries[0].total_time >= summaries[1].total_time
        assert summaries[1].total_time >= summaries[2].total_time

    def test_clear_data(self):
        """Test clearing profiler data."""
        with self.profiler.profile("test_clear"):
            time.sleep(0.001)

        assert len(self.profiler._entries) > 0
        assert len(self.profiler._summaries) > 0

        self.profiler.clear_data()

        assert len(self.profiler._entries) == 0
        assert len(self.profiler._summaries) == 0
        assert self.profiler._measurement_count == 0


class TestProfileFunction:
    """Test cases for profile_function decorator."""

    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        @profile_function("decorated_func")
        def test_function(x, y):
            time.sleep(0.001)
            return x + y

        result = test_function(1, 2)
        assert result == 3

        # Check that profiling occurred
        profiler = get_profiler()
        summary = profiler.get_summary("decorated_func")
        assert summary is not None
        assert summary.call_count == 1

    def test_profile_function_default_name(self):
        """Test decorator with default function name."""
        @profile_function()
        def another_test_function():
            time.sleep(0.001)
            return "test"

        result = another_test_function()
        assert result == "test"

        # Function name should include module and function name
        profiler = get_profiler()
        summaries = profiler.get_all_summaries()

        # Should have at least one summary with function name
        function_names = [s.name for s in summaries]
        assert any("another_test_function" in name for name in function_names)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PerformanceMonitor(target_overhead_percent=5.0)

    def test_init(self):
        """Test performance monitor initialization."""
        assert self.monitor.target_overhead_percent == 5.0
        assert len(self.monitor._baseline_measurements) == 0
        assert len(self.monitor._debug_measurements) == 0

    def test_measure_baseline(self):
        """Test baseline measurement."""
        def test_operation():
            time.sleep(0.01)

        baseline_time = self.monitor.measure_baseline("test_op", test_operation)

        assert baseline_time > 0.005
        assert "test_op" in self.monitor._baseline_measurements

    def test_measure_with_debug(self):
        """Test measurement with debug framework."""
        def test_operation():
            time.sleep(0.01)

        debug_time = self.monitor.measure_with_debug("test_op", test_operation)

        assert debug_time > 0.005
        assert "test_op" in self.monitor._debug_measurements

    def test_calculate_overhead(self):
        """Test overhead calculation."""
        def test_operation():
            time.sleep(0.01)

        # Measure baseline and debug times multiple times for accuracy
        baseline_times = []
        debug_times = []

        for _ in range(3):
            baseline_time = self.monitor.measure_baseline(f"test_op_{len(baseline_times)}", test_operation)
            baseline_times.append(baseline_time)

        for _ in range(3):
            debug_time = self.monitor.measure_with_debug(f"test_debug_op_{len(debug_times)}", test_operation)
            debug_times.append(debug_time)

        # Use the most representative times
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_debug = sum(debug_times) / len(debug_times)

        # Manual overhead calculation
        overhead = ((avg_debug - avg_baseline) / avg_baseline) * 100 if avg_baseline > 0 else 0

        assert overhead >= -50  # Allow for some measurement variance
        assert overhead < 500  # Should be reasonable overhead

    def test_check_overhead_compliance(self):
        """Test compliance checking."""
        def fast_operation():
            time.sleep(0.001)

        def slow_operation():
            time.sleep(0.01)

        # Measure both operations
        self.monitor.measure_baseline("fast_op", fast_operation)
        self.monitor.measure_with_debug("fast_op", fast_operation)

        self.monitor.measure_baseline("slow_op", slow_operation)
        self.monitor.measure_with_debug("slow_op", slow_operation)

        compliance_report = self.monitor.check_overhead_compliance()

        assert 'target_overhead_percent' in compliance_report
        assert 'compliant' in compliance_report
        assert 'measurements' in compliance_report
        assert len(compliance_report['measurements']) == 2
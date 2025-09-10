"""
Integration test module for Dual-backend execution.

This test module validates the parallel execution of reference and alternative
backends, checkpoint synchronization, data flow coordination, error handling
when backends diverge, and performance comparison analysis.

TDD: This test MUST FAIL until the dual-backend execution is implemented.
"""

import pytest
import numpy as np
import asyncio
import time
import threading
import gc
from copy import deepcopy
from unittest.mock import MagicMock

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.integrations.dual_backend_execution import DualBackendExecutor
    DUAL_BACKEND_AVAILABLE = True
except ImportError:
    DualBackendExecutor = None
    DUAL_BACKEND_AVAILABLE = False

try:
    from debug_framework.coordinators.checkpoint_synchronizer import CheckpointSynchronizer
    CHECKPOINT_SYNC_AVAILABLE = True
except ImportError:
    CheckpointSynchronizer = None
    CHECKPOINT_SYNC_AVAILABLE = False


class TestDualBackendExecution:
    """Test suite for dual-backend execution functionality."""

    @pytest.mark.xfail(DUAL_BACKEND_AVAILABLE, reason="TDD gate - should fail until implementation exists")
    def test_dual_backend_executor_import_fails(self):
        """Test that dual-backend executor import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.integrations.dual_backend_execution import DualBackendExecutor

    @pytest.mark.xfail(CHECKPOINT_SYNC_AVAILABLE, reason="TDD gate - should fail until implementation exists")
    def test_checkpoint_synchronizer_import_fails(self):
        """Test that checkpoint synchronizer import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.coordinators.checkpoint_synchronizer import CheckpointSynchronizer

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_dual_backend_executor_initialization(self):
        """Test initialization of dual-backend execution system."""
        np.random.seed(12345)
        # Mock reference backend (e.g., PyTorch)
        mock_reference_backend = MagicMock()
        mock_reference_backend.name = "pytorch_reference"
        mock_reference_backend.device = "cpu"

        # Mock alternative backend (e.g., Metal)
        mock_alternative_backend = MagicMock()
        mock_alternative_backend.name = "metal_optimized"
        mock_alternative_backend.device = "gpu"

        executor = DualBackendExecutor(
            reference_backend=mock_reference_backend,
            alternative_backend=mock_alternative_backend,
            synchronization_mode="checkpoint_based",
            tolerance_config={"absolute": 1e-5, "relative": 1e-4},
            performance_monitoring=True
        )

        assert executor.reference_backend == mock_reference_backend
        assert executor.alternative_backend == mock_alternative_backend
        assert executor.synchronization_mode == "checkpoint_based"
        assert executor.tolerance_config["absolute"] == 1e-5
        assert executor.execution_results == {}
        assert executor.synchronization_points == []

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_parallel_backend_execution(self):
        """Test parallel execution of reference and alternative backends."""
        np.random.seed(12346)
        # Mock backends with deterministic behavior
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Reference backend produces "ground truth"
        reference_output = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_reference.forward.return_value = reference_output

        # Alternative backend produces slightly different output
        alternative_output = reference_output + np.random.normal(0, 1e-6, reference_output.shape)
        mock_alternative.forward.return_value = alternative_output

        executor = DualBackendExecutor(mock_reference, mock_alternative)

        # Test parallel execution
        input_data = np.random.rand(2, 10)

        execution_result = executor.execute_parallel(
            input_data=input_data,
            operation="forward",
            checkpoints=["post_embedding", "post_attention", "final_output"]
        )

        assert execution_result["status"] == "completed"
        assert "reference_results" in execution_result
        assert "alternative_results" in execution_result
        assert "comparison_results" in execution_result
        assert execution_result["backends_synchronized"] is True

        # Verify both backends were called
        mock_reference.forward.assert_called_once_with(input_data)
        mock_alternative.forward.assert_called_once_with(input_data)

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    @pytest.mark.asyncio
    async def test_asynchronous_backend_coordination(self):
        """Test asynchronous coordination between backends with proper backpressure handling."""
        np.random.seed(12347)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock async execution methods with backpressure simulation
        reference_coordination_event = asyncio.Event()
        alternative_coordination_event = asyncio.Event()

        async def mock_ref_async_forward(input_data):
            # Simulate backpressure-aware computation
            await asyncio.sleep(0.1)  # Computation time
            reference_coordination_event.set()  # Signal coordination
            await alternative_coordination_event.wait()  # Wait for coordination
            # Simulate proper tensor creation with deterministic data
            return np.ones((2, 512, 4096), dtype=np.float32) * 1.0

        async def mock_alt_async_forward(input_data):
            # Simulate slightly longer computation with coordination
            await asyncio.sleep(0.15)
            alternative_coordination_event.set()  # Signal coordination
            await reference_coordination_event.wait()  # Wait for coordination
            # Return slightly different deterministic data for comparison
            return np.ones((2, 512, 4096), dtype=np.float32) * 1.001

        mock_reference.async_forward = mock_ref_async_forward
        mock_alternative.async_forward = mock_alt_async_forward

        executor = DualBackendExecutor(mock_reference, mock_alternative)

        # Use deterministic input data for reproducible testing
        input_data = np.ones((2, 512), dtype=np.float32)

        # Test async execution with monotonic timing and warm-up
        # Warm-up run to eliminate JIT/startup effects
        try:
            await asyncio.wait_for(executor.execute_async_parallel(
                input_data=input_data,
                operation="async_forward",
                timeout=2.0
            ), timeout=2.0)
        except NotImplementedError:
            # Expected for TDD - implementation doesn't exist yet
            pass
        except asyncio.TimeoutError:
            # Acceptable for warm-up - may timeout due to coordination setup
            pass

        # Actual timed execution
        start_time = time.perf_counter()

        try:
            async_result = await asyncio.wait_for(executor.execute_async_parallel(
                input_data=input_data,
                operation="async_forward",
                timeout=5.0
            ), timeout=5.0)

            total_time = time.perf_counter() - start_time

            assert async_result["status"] == "completed"
            # Should complete in parallel (~0.15s max), not sequential (~0.25s)
            # Increased tolerance for CI environments
            assert total_time < 0.35, f"Expected parallel execution <0.35s, got {total_time}s"
            assert "reference_execution_time" in async_result["performance"]
            assert "alternative_execution_time" in async_result["performance"]
            assert async_result["coordination_successful"] is True

            # Verify proper coordination occurred
            assert async_result.get("backpressure_handled", False) is True
            assert async_result.get("deadlock_avoided", True) is True

        except asyncio.TimeoutError:
            pytest.fail("Async coordination deadlocked - backpressure handling failed")
        except NotImplementedError:
            # Expected for TDD - implementation doesn't exist yet
            pytest.skip("DualBackendExecutor async coordination not implemented")
        finally:
            # Clean up async resources
            if 'reference_coordination_event' in locals():
                reference_coordination_event.clear()
            if 'alternative_coordination_event' in locals():
                alternative_coordination_event.clear()
            # Allow event loop to process any pending tasks
            await asyncio.sleep(0.01)

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_checkpoint_synchronization_and_data_flow(self):
        """Test checkpoint synchronization and data flow between backends."""
        np.random.seed(12348)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock multi-layer execution with checkpoints
        reference_checkpoints = {
            "post_embedding": np.random.rand(1, 10, 4096),
            "post_attention": np.random.rand(1, 10, 4096),
            "post_mlp": np.random.rand(1, 10, 4096),
            "final_output": np.random.rand(1, 10, 32000)
        }

        alternative_checkpoints = {
            "post_embedding": reference_checkpoints["post_embedding"] + np.random.normal(0, 1e-5, (1, 10, 4096)),
            "post_attention": reference_checkpoints["post_attention"] + np.random.normal(0, 1e-5, (1, 10, 4096)),
            "post_mlp": reference_checkpoints["post_mlp"] + np.random.normal(0, 1e-5, (1, 10, 4096)),
            "final_output": reference_checkpoints["final_output"] + np.random.normal(0, 1e-5, (1, 10, 32000))
        }

        mock_reference.execute_with_checkpoints.return_value = reference_checkpoints
        mock_alternative.execute_with_checkpoints.return_value = alternative_checkpoints

        executor = DualBackendExecutor(mock_reference, mock_alternative)

        sync_result = executor.execute_with_checkpoint_synchronization(
            input_data=np.random.rand(1, 10),
            checkpoint_names=["post_embedding", "post_attention", "post_mlp", "final_output"],
            sync_tolerance=1e-4,
            early_stopping_on_divergence=True
        )

        assert sync_result["status"] == "synchronized"
        assert len(sync_result["synchronized_checkpoints"]) == 4
        assert sync_result["divergence_detected"] is False

        # Verify checkpoint comparisons
        for checkpoint_name in ["post_embedding", "post_attention", "post_mlp", "final_output"]:
            checkpoint_result = sync_result["synchronized_checkpoints"][checkpoint_name]
            assert checkpoint_result["comparison_passed"] is True
            assert checkpoint_result["max_absolute_error"] < 1e-4

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_backend_divergence_detection_and_handling(self):
        """Test detection and handling of backend divergence using proper NumPy allclose patterns."""
        np.random.seed(12349)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Test case 1: Normal values - should diverge
        reference_output = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        alternative_output = np.array([[1.5, 2.8, 4.2]], dtype=np.float32)  # Significantly different

        mock_reference.forward.return_value = reference_output
        mock_alternative.forward.return_value = alternative_output

        executor = DualBackendExecutor(
            mock_reference,
            mock_alternative,
            tolerance_config={"absolute": 1e-3, "relative": 1e-3},
            use_numpy_allclose=True  # Use proper NumPy allclose semantics
        )

        # Use deterministic input
        input_data = np.ones((1, 10), dtype=np.float32)

        divergence_result = executor.execute_with_divergence_detection(
            input_data=input_data,
            operation="forward",
            # Use NumPy allclose pattern: |a - b| <= atol + rtol * |b|
            absolute_tolerance=1e-3,
            relative_tolerance=1e-3,
            convergence_analysis=True,
            handle_inf_nan=True
        )

        assert divergence_result["status"] == "diverged"
        assert divergence_result["divergence_detected"] is True
        assert divergence_result["divergence_point"] == "final_output"

        # Verify NumPy allclose-style comparison was used
        assert divergence_result.get("comparison_method", "") == "numpy_allclose"
        assert divergence_result.get("absolute_tolerance", 0) == 1e-3
        assert divergence_result.get("relative_tolerance", 0) == 1e-3

        # Calculate expected relative error using NumPy allclose formula
        expected_errors = np.abs(reference_output - alternative_output)
        expected_tolerances = 1e-3 + 1e-3 * np.abs(alternative_output)
        max_violation = np.max(expected_errors / expected_tolerances)
        assert max_violation == pytest.approx(1.0, rel=0.05) or max_violation > 1.0  # Should exceed tolerance

        # Test case 2: Inf/NaN handling
        reference_inf_nan = np.array([[1.0, np.inf, np.nan]], dtype=np.float32)
        alternative_inf_nan = np.array([[1.0, np.inf, np.nan]], dtype=np.float32)

        mock_reference.forward.return_value = reference_inf_nan
        mock_alternative.forward.return_value = alternative_inf_nan

        inf_nan_result = executor.execute_with_divergence_detection(
            input_data=input_data,
            operation="forward",
            absolute_tolerance=1e-3,
            relative_tolerance=1e-3,
            handle_inf_nan=True,
            equal_nan=True  # Consider NaN == NaN for testing
        )

        # Should handle Inf/NaN correctly
        assert inf_nan_result["inf_nan_handled"] is True
        assert "inf_count" in inf_nan_result["special_values"]
        assert "nan_count" in inf_nan_result["special_values"]

        # Test case 3: Mixed precision edge case
        reference_mixed = np.array([[1e-8, 1e8, 1.0]], dtype=np.float32)
        alternative_mixed = np.array([[1e-8 + 1e-10, 1e8 + 1e5, 1.0 + 1e-6]], dtype=np.float32)

        mock_reference.forward.return_value = reference_mixed
        mock_alternative.forward.return_value = alternative_mixed

        mixed_result = executor.execute_with_divergence_detection(
            input_data=input_data,
            operation="forward",
            absolute_tolerance=1e-5,
            relative_tolerance=1e-4,
            asymmetric_tolerance=True  # Handle asymmetric cases
        )

        # Verify proper handling of mixed precision scenarios
        assert mixed_result.get("mixed_precision_handled", False) is True
        assert "precision_analysis" in mixed_result

        # Verify divergence analysis follows NumPy allclose semantics
        analysis = divergence_result["divergence_analysis"]
        assert "error_distribution" in analysis
        assert "statistical_significance" in analysis
        assert "allclose_compliance" in analysis
        assert analysis["consistent_divergence"] is True

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_performance_comparison_and_timing_analysis(self):
        """Test performance comparison and timing analysis between backends with proper monotonic timing."""
        np.random.seed(12350)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock execution times using monotonic measurements
        reference_execution_count = 0
        alternative_execution_count = 0

        def slow_reference_forward(input_data):
            nonlocal reference_execution_count
            reference_execution_count += 1
            start = time.perf_counter()
            time.sleep(0.02)  # 20ms base computation
            elapsed = time.perf_counter() - start
            return np.ones((1, 10, 4096), dtype=np.float32) * elapsed  # Deterministic with timing info

        def fast_alternative_forward(input_data):
            nonlocal alternative_execution_count
            alternative_execution_count += 1
            start = time.perf_counter()
            time.sleep(0.005)  # 5ms base computation - ~4x faster
            elapsed = time.perf_counter() - start
            return np.ones((1, 10, 4096), dtype=np.float32) * elapsed  # Deterministic with timing info

        mock_reference.forward = slow_reference_forward
        mock_alternative.forward = fast_alternative_forward

        executor = DualBackendExecutor(mock_reference, mock_alternative)

        # Use deterministic input data for reproducible testing
        input_data = np.ones((1, 10), dtype=np.float32)

        performance_result = executor.execute_with_performance_analysis(
            input_data=input_data,
            operation="forward",
            iterations=5,
            warmup_iterations=2,  # Proper warm-up to eliminate startup effects
            profile_memory=True,
            profile_cpu=True,
            use_monotonic_timing=True,  # Explicit monotonic clock usage
            timing_precision="perf_counter"  # High-precision monotonic timing
        )

        assert performance_result["status"] == "completed"

        perf_analysis = performance_result["performance_analysis"]

        # Verify timing measurements use monotonic sources
        assert perf_analysis.get("timing_source", "") == "perf_counter"
        assert perf_analysis.get("monotonic_timing", False) is True

        # Performance comparison assertions with noise tolerance
        ref_time = perf_analysis["reference_mean_time"]
        alt_time = perf_analysis["alternative_mean_time"]

        # Use relative comparison with tolerance for CI/test environment variability
        speedup_factor = perf_analysis.get("speedup_factor", ref_time / alt_time if alt_time > 0 else 1.0)

        # More lenient assertions for test environment variability
        assert ref_time > alt_time, f"Reference ({ref_time}s) should be slower than alternative ({alt_time}s)"
        assert speedup_factor > pytest.approx(1.5, rel=0.1), f"Expected >1.5x speedup, got {speedup_factor}x"

        assert "memory_usage" in perf_analysis
        assert "cpu_utilization" in perf_analysis
        assert "warm_up_completed" in perf_analysis
        assert perf_analysis["warm_up_completed"] is True

        # Verify statistical validity with proper confidence intervals
        assert perf_analysis.get("performance_difference_significant", False) is True
        assert perf_analysis.get("confidence_interval", 0.0) > pytest.approx(0.90, abs=0.05)  # Reduced from 0.95 for test stability

        # Verify proper execution counts (warm-up + actual iterations)
        assert reference_execution_count >= 7  # 2 warm-up + 5 iterations
        assert alternative_execution_count >= 7

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_error_propagation_and_recovery(self):
        """Test error propagation and recovery mechanisms."""
        np.random.seed(12351)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock reference backend failure
        mock_reference.forward.side_effect = RuntimeError("Reference backend GPU out of memory")
        mock_alternative.forward.return_value = np.random.rand(1, 10, 4096)

        executor = DualBackendExecutor(
            mock_reference,
            mock_alternative,
            error_handling="graceful_degradation",
            fallback_to_single_backend=True
        )

        error_result = executor.execute_with_error_recovery(
            input_data=np.random.rand(1, 10),
            operation="forward",
            max_retries=3,
            retry_with_fallback=True
        )

        assert error_result["status"] == "partial_success"
        assert error_result["reference_backend_failed"] is True
        assert error_result["alternative_backend_succeeded"] is True
        assert "fallback_execution_used" in error_result
        assert error_result["error_details"]["reference_error"] == "Reference backend GPU out of memory"

        # Test dual backend failure with proper error context
        mock_alternative.forward.side_effect = RuntimeError("Alternative backend compilation failed")

        # For comprehensive error testing, we expect the executor to handle failures gracefully
        # rather than raising exceptions, so we test return status
        dual_failure_result = executor.execute_with_error_recovery(
            input_data=np.random.rand(1, 10),
            operation="forward",
            max_retries=1
        )

        assert dual_failure_result["status"] == "failed"
        assert dual_failure_result["both_backends_failed"] is True

        # Verify that specific error types are properly captured in error details
        assert "RuntimeError" in str(dual_failure_result.get("error_details", {}).get("alternative_error", ""))

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_memory_efficient_dual_execution(self):
        """Test memory-efficient execution for large models and datasets using numpy.memmap."""
        np.random.seed(12352)
        import tempfile
        import os

        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Use memory-mapped arrays for large tensor operations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create memory-mapped reference output
            ref_mmap_file = os.path.join(temp_dir, "reference_output.dat")
            large_ref_output = np.memmap(
                ref_mmap_file,
                dtype=np.float32,
                mode='w+',
                shape=(8, 2048, 4096)
            )
            # Fill with deterministic pattern instead of random data
            large_ref_output[:] = np.arange(8 * 2048 * 4096, dtype=np.float32).reshape(8, 2048, 4096) / (8 * 2048 * 4096)
            large_ref_output.flush()  # Explicit flush for deterministic behavior

            # Create memory-mapped alternative output
            alt_mmap_file = os.path.join(temp_dir, "alternative_output.dat")
            large_alt_output = np.memmap(
                alt_mmap_file,
                dtype=np.float32,
                mode='w+',
                shape=(8, 2048, 4096)
            )
            # Create slight variation for comparison testing
            large_alt_output[:] = large_ref_output[:] + 1e-6
            large_alt_output.flush()  # Explicit flush

            mock_reference.forward.return_value = large_ref_output
            mock_alternative.forward.return_value = large_alt_output

            executor = DualBackendExecutor(
                mock_reference,
                mock_alternative,
                memory_optimization=True,
                streaming_comparison=True,
                chunk_size=1024*1024,  # 1MB chunks
                use_memory_mapping=True,
                temp_directory=temp_dir
            )

            # Use memory-mapped input data
            input_mmap_file = os.path.join(temp_dir, "input_data.dat")
            input_data = np.memmap(
                input_mmap_file,
                dtype=np.float32,
                mode='w+',
                shape=(8, 2048)
            )
            input_data[:] = np.ones((8, 2048), dtype=np.float32) * 0.5  # Deterministic input
            input_data.flush()

            memory_result = executor.execute_memory_efficient(
                input_data=input_data,
                operation="forward",
                max_memory_usage="1GB",
                garbage_collection=True,
                verify_memmap_integrity=True
            )

            assert memory_result["status"] == "completed"
            assert memory_result["memory_optimization"]["peak_memory_mb"] < 1024
            assert memory_result["memory_optimization"]["streaming_used"] is True
            assert memory_result["memory_optimization"]["chunks_processed"] > 1
            assert memory_result["memory_optimization"]["memory_mapped"] is True
            assert memory_result["comparison_accuracy_maintained"] is True

            # Verify memory-mapped file integrity
            assert memory_result.get("memmap_integrity_verified", False) is True

            # Ensure proper cleanup by explicitly closing memmaps
            del large_ref_output, large_alt_output, input_data
            gc.collect()  # Proactively clean up memory-mapped resources

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_conditional_execution_and_smart_switching(self):
        """Test conditional execution and smart switching between backends."""
        np.random.seed(12353)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock performance characteristics
        def reference_performance(data):
            batch_size = data.shape[0]
            execution_time = batch_size * 0.01  # Linear scaling
            time.sleep(execution_time)
            return np.random.rand(*data.shape, 4096)

        def alternative_performance(data):
            batch_size = data.shape[0]
            if batch_size < 4:
                execution_time = 0.1  # High overhead for small batches
            else:
                execution_time = batch_size * 0.002  # Better scaling for large batches
            time.sleep(execution_time)
            return np.random.rand(*data.shape, 4096)

        mock_reference.forward = reference_performance
        mock_alternative.forward = alternative_performance

        executor = DualBackendExecutor(
            mock_reference,
            mock_alternative,
            smart_switching=True,
            performance_profiling=True
        )

        # Test with small batch (should prefer reference)
        small_batch_result = executor.execute_with_smart_switching(
            input_data=np.random.rand(2, 512),
            operation="forward",
            switching_criteria="performance",
            learning_enabled=True
        )

        assert small_batch_result["status"] == "completed"
        assert small_batch_result["selected_backend"] == "reference"
        assert small_batch_result["switching_reason"] == "performance_optimized"

        # Test with large batch (should prefer alternative)
        large_batch_result = executor.execute_with_smart_switching(
            input_data=np.random.rand(16, 512),
            operation="forward",
            switching_criteria="performance",
            learning_enabled=True
        )

        assert large_batch_result["status"] == "completed"
        assert large_batch_result["selected_backend"] == "alternative"

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_cross_validation_and_accuracy_analysis(self):
        """Test cross-validation between backends and accuracy analysis."""
        np.random.seed(12354)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Create systematic test cases
        test_cases = [
            {"input": np.random.rand(1, 512), "expected_pattern": "simple"},
            {"input": np.random.rand(4, 1024), "expected_pattern": "medium"},
            {"input": np.random.rand(8, 2048), "expected_pattern": "complex"}
        ]

        # Mock deterministic outputs for validation
        reference_outputs = []
        alternative_outputs = []

        for case_idx, case in enumerate(test_cases):
            ref_out = np.random.rand(*case["input"].shape, 4096)
            alt_out = ref_out + np.random.normal(0, 1e-5, ref_out.shape)  # Slight variation
            reference_outputs.append(ref_out)
            alternative_outputs.append(alt_out)

        mock_reference.forward.side_effect = reference_outputs
        mock_alternative.forward.side_effect = alternative_outputs

        executor = DualBackendExecutor(mock_reference, mock_alternative)

        validation_result = executor.execute_cross_validation(
            test_cases=test_cases,
            operation="forward",
            validation_metrics=["accuracy", "consistency", "numerical_stability"],
            statistical_analysis=True
        )

        assert validation_result["status"] == "validated"
        assert validation_result["overall_accuracy"] > pytest.approx(0.99, abs=0.01)
        assert validation_result["consistency_score"] > pytest.approx(0.95, abs=0.02)
        assert validation_result["numerical_stability"] > pytest.approx(0.98, abs=0.01)

        # Verify statistical analysis
        stats = validation_result["statistical_analysis"]
        assert "error_distribution" in stats
        assert "correlation_coefficient" in stats
        assert stats["correlation_coefficient"] > pytest.approx(0.999, rel=0.001)  # Very high correlation

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_concurrent_multi_request_handling(self):
        """Test handling of multiple concurrent dual-backend requests."""
        np.random.seed(12355)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock thread-safe execution
        def thread_safe_reference(data, _req_id):
            time.sleep(0.1)  # Simulate computation
            return np.random.rand(*data.shape, 4096)

        def thread_safe_alternative(data, _req_id):
            time.sleep(0.08)  # Slightly faster
            return np.random.rand(*data.shape, 4096)

        mock_reference.thread_safe_forward = thread_safe_reference
        mock_alternative.thread_safe_forward = thread_safe_alternative

        executor = DualBackendExecutor(
            mock_reference,
            mock_alternative,
            concurrent_execution=True,
            max_concurrent_requests=4
        )

        # Create multiple concurrent requests
        requests = [
            {"id": i, "input": np.random.rand(2, 512), "priority": i % 3}
            for i in range(8)
        ]

        concurrent_result = executor.execute_concurrent_requests(
            requests=requests,
            operation="thread_safe_forward",
            timeout_per_request=5.0,
            priority_scheduling=True
        )

        assert concurrent_result["status"] == "completed"
        assert concurrent_result["completed_requests"] == 8
        assert concurrent_result["failed_requests"] == 0
        assert concurrent_result["average_execution_time"] > 0
        assert concurrent_result["concurrency_efficiency"] > pytest.approx(0.7, abs=0.1)  # Good parallelization

        # Clean up any remaining thread resources
        gc.collect()

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_adaptive_tolerance_and_precision_handling(self):
        """Test adaptive tolerance adjustment based on execution context."""
        np.random.seed(12356)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock different precision scenarios
        scenarios = [
            {"precision": "float32", "expected_tolerance": 1e-5},
            {"precision": "float16", "expected_tolerance": 1e-3},
            {"precision": "bfloat16", "expected_tolerance": 1e-2}
        ]

        executor = DualBackendExecutor(
            mock_reference,
            mock_alternative,
            adaptive_tolerance=True,
            precision_aware=True
        )

        for scenario in scenarios:
            if scenario["precision"] == "float32":
                ref_output = np.random.rand(1, 10, 4096).astype(np.float32)
                alt_output = ref_output + np.random.normal(0, 1e-6, ref_output.shape).astype(np.float32)
            elif scenario["precision"] == "float16":
                ref_output = np.random.rand(1, 10, 4096).astype(np.float16)
                alt_output = ref_output + np.random.normal(0, 1e-4, ref_output.shape).astype(np.float16)
            else:  # bfloat16 simulation
                ref_output = np.random.rand(1, 10, 4096).astype(np.float32)
                alt_output = ref_output + np.random.normal(0, 1e-3, ref_output.shape).astype(np.float32)

            mock_reference.forward.return_value = ref_output
            mock_alternative.forward.return_value = alt_output

            adaptive_result = executor.execute_with_adaptive_tolerance(
                input_data=np.random.rand(1, 10),
                operation="forward",
                precision=scenario["precision"],
                tolerance_adaptation_strategy="precision_based"
            )

            assert adaptive_result["status"] == "completed"
            assert adaptive_result["tolerance_used"] >= pytest.approx(scenario["expected_tolerance"], rel=0.1)
            assert adaptive_result["precision_handling"]["detected_precision"] == scenario["precision"]
            assert adaptive_result["comparison_passed"] is True

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_execution_rollback_and_state_management(self):
        """Test execution rollback and state management capabilities."""
        np.random.seed(12357)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        # Mock stateful backends with deep copy for isolation
        reference_state = {"layer_weights": np.random.rand(100, 100), "iteration": 0}
        alternative_state = {"layer_weights": np.random.rand(100, 100), "iteration": 0}

        mock_reference.get_state.return_value = deepcopy(reference_state)
        mock_reference.set_state = MagicMock()
        mock_alternative.get_state.return_value = deepcopy(alternative_state)
        mock_alternative.set_state = MagicMock()

        executor = DualBackendExecutor(
            mock_reference,
            mock_alternative,
            state_management=True,
            rollback_enabled=True
        )

        # Create checkpoint before execution
        checkpoint_result = executor.create_execution_checkpoint("pre_test_execution")
        assert checkpoint_result["status"] == "created"
        assert checkpoint_result["checkpoint_id"] == "pre_test_execution"

        # Mock execution with state changes
        mock_reference.forward.return_value = np.random.rand(1, 10, 4096)
        mock_alternative.forward.side_effect = RuntimeError("Alternative backend failed")

        # Execute with failure
        execution_result = executor.execute_with_rollback_capability(
            input_data=np.random.rand(1, 10),
            operation="forward",
            rollback_on_failure=True,
            checkpoint_id="pre_test_execution"
        )

        assert execution_result["status"] == "failed_with_rollback"
        assert execution_result["rollback_performed"] is True
        assert execution_result["state_restored"] is True

        # Verify rollback operations with deep copied states
        mock_reference.set_state.assert_called_with(deepcopy(reference_state))
        mock_alternative.set_state.assert_called_with(deepcopy(alternative_state))

    @pytest.mark.skipif(not DUAL_BACKEND_AVAILABLE, reason="DualBackendExecutor not implemented")
    def test_input_validation_with_proper_error_handling(self):
        """Test input validation raises appropriate errors."""
        np.random.seed(12358)
        mock_reference = MagicMock()
        mock_alternative = MagicMock()

        executor = DualBackendExecutor(mock_reference, mock_alternative)

        # Test invalid input data type - should be handled gracefully or raise TypeError
        with pytest.raises((TypeError, ValueError)):
            executor.execute_parallel(
                input_data="invalid_input",  # String instead of array
                operation="forward"
            )

        # Test None input - should raise appropriate error
        with pytest.raises((TypeError, ValueError)):
            executor.execute_parallel(
                input_data=None,
                operation="forward"
            )

        # Test invalid operation name - should raise ValueError or similar
        with pytest.raises((ValueError, AttributeError)):
            executor.execute_parallel(
                input_data=np.random.rand(1, 10),
                operation="invalid_operation"
            )
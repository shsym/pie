"""
Test module for TensorComparisonEngine service.

This test module validates the TensorComparisonEngine service which handles
tensor-level validation, comparison algorithms, statistical analysis, and
performance optimization for large-scale tensor operations.

TDD: This test MUST FAIL until the TensorComparisonEngine service is implemented.
"""

import pytest
import numpy as np
import math
import hashlib
import time
from unittest.mock import patch, MagicMock
import tempfile
import json
try:
    from scipy import sparse
    SCIPY_AVAILABLE = True
except ImportError:
    sparse = None
    SCIPY_AVAILABLE = False

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.services.tensor_comparison_engine import TensorComparisonEngine
    TENSOR_COMPARISON_ENGINE_AVAILABLE = True
except ImportError:
    TensorComparisonEngine = None
    TENSOR_COMPARISON_ENGINE_AVAILABLE = False


class TestTensorComparisonEngine:
    """Test suite for TensorComparisonEngine service functionality."""

    @pytest.mark.xfail(TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TDD gate")
    def test_tensor_comparison_engine_import_fails(self):
        """Test that TensorComparisonEngine import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.services.tensor_comparison_engine import TensorComparisonEngine

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_tensor_comparison_engine_initialization(self):
        """Test TensorComparisonEngine service initialization."""
        engine = TensorComparisonEngine(
            default_tolerance=1e-5,
            comparison_algorithms=["element_wise", "statistical", "approximate"],
            cache_directory="/tmp/tensor_cache",
            performance_mode="optimized"
        )

        assert engine.default_tolerance == 1e-5
        assert "element_wise" in engine.comparison_algorithms
        assert engine.cache_directory == "/tmp/tensor_cache"
        assert engine.performance_mode == "optimized"
        assert engine.comparison_history == []
        assert engine.statistics_cache == {}

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_element_wise_comparison_with_proper_rtol_atol(self):
        """Test element-wise tensor comparison with NumPy's rtol/atol semantics."""
        engine = TensorComparisonEngine()

        # Test data that exercises rtol/atol behavior
        tensor_a = np.array([[1.0, 2.0, 1000.0], [0.001, 0.0, -1000.0]])
        # Create tensor_b with differences that test rtol vs atol dominance
        tensor_b = np.array([
            [1.0001, 2.0002, 1001.0],    # Small abs diff, larger rel diff for large values
            [0.0011, 0.00001, -1001.0]   # Larger abs diff for small values
        ])

        # Test with atol dominant (small values)
        comparison_result = engine.compare_element_wise(
            tensor_a,
            tensor_b,
            atol=1e-3,  # Absolute tolerance
            rtol=1e-6,  # Relative tolerance
            comparison_type="allclose"  # Use NumPy's allclose semantics
        )

        # Verify rtol/atol formula: |a - b| <= atol + rtol * |b|
        # For tensor_a[0,0] vs tensor_b[0,0]: |1.0 - 1.0001| = 0.0001
        # atol + rtol * |b| = 1e-3 + 1e-6 * 1.0001 ≈ 1e-3
        # 0.0001 <= 1e-3 ✓ (passes)

        # For tensor_a[0,2] vs tensor_b[0,2]: |1000.0 - 1001.0| = 1.0
        # atol + rtol * |b| = 1e-3 + 1e-6 * 1001.0 ≈ 1e-3 + 1e-3 = 2e-3
        # 1.0 > 2e-3 ✗ (should fail)

        assert comparison_result["status"] == "failed"  # Large value difference exceeds tolerance
        assert comparison_result["allclose_semantics"] is True
        assert comparison_result["rtol_used"] == 1e-6
        assert comparison_result["atol_used"] == 1e-3

        # Test asymmetry: rtol is relative to |b|, not |a|
        asymmetry_result = engine.compare_element_wise(
            tensor_b,  # Swap order
            tensor_a,
            atol=1e-3,
            rtol=1e-6,
            comparison_type="allclose"
        )

        # Results may differ due to asymmetry in rtol calculation
        assert asymmetry_result["rtol_asymmetric"] is True

        # Test with both tolerances accommodating the differences
        passing_result = engine.compare_element_wise(
            tensor_a,
            tensor_b,
            atol=1e-2,   # Increased atol
            rtol=1e-3,   # Increased rtol
            comparison_type="allclose"
        )

        assert passing_result["status"] == "passed"
        assert passing_result["max_absolute_error"] > 0
        assert passing_result["max_relative_error"] > 0

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_statistical_comparison(self):
        """Test statistical tensor comparison methods."""
        engine = TensorComparisonEngine()

        # Create tensors with known statistical properties
        np.random.seed(42)  # Deterministic for testing
        tensor_a = np.random.normal(0, 1, (100, 100))
        tensor_b = tensor_a + np.random.normal(0, 0.01, (100, 100))  # Small noise

        stats_result = engine.compare_statistical(
            tensor_a,
            tensor_b,
            metrics=["mean", "std", "variance", "percentiles", "distribution"],
            significance_level=0.05
        )

        assert stats_result["status"] == "passed"
        assert "mean_difference" in stats_result["metrics"]
        assert "std_difference" in stats_result["metrics"]
        assert "ks_test_p_value" in stats_result["metrics"]

        # With small added noise, mean difference should be small
        assert abs(stats_result["metrics"]["mean_difference"]) < 0.1
        assert stats_result["distribution_similarity"] > 0.95

        # Verify statistical significance testing
        assert stats_result["statistical_significance"]["alpha"] == 0.05
        assert "confidence_interval" in stats_result["statistical_significance"]

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_ulp_comparison_with_proper_semantics(self):
        """Test ULP-based comparison with correct math.ulp usage."""
        engine = TensorComparisonEngine()

        # Test ULP comparison for different magnitudes
        base_values = [1.0, 100.0, 1e-10, 1e10]

        for base_val in base_values:
            tensor_a = np.array([base_val])

            # Create tensor that's exactly 1 ULP away using math.ulp
            ulp_val = math.ulp(base_val)
            tensor_b = np.array([base_val + ulp_val])

            ulp_result = engine.compare_ulp(
                tensor_a,
                tensor_b,
                max_ulp_difference=1,
                check_subnormals=True
            )

            assert ulp_result["status"] == "passed"
            assert ulp_result["max_ulp_difference"] == 1
            assert ulp_result["ulp_semantics_verified"] is True

            # Test multiple ULPs
            tensor_c = np.array([base_val + 3 * ulp_val])
            ulp_result_3 = engine.compare_ulp(
                tensor_a,
                tensor_c,
                max_ulp_difference=2  # Should fail
            )

            assert ulp_result_3["status"] == "failed"
            assert ulp_result_3["max_ulp_difference"] > 2

        # Test special cases with guard for zero/subnormal handling
        zero_tensor_a = np.array([0.0])
        smallest_subnormal = np.finfo(np.float64).smallest_subnormal

        # Guard against platforms where smallest_subnormal might be zero
        if smallest_subnormal > 0:
            zero_tensor_b = np.array([smallest_subnormal])

            zero_ulp_result = engine.compare_ulp(
                zero_tensor_a,
                zero_tensor_b,
                max_ulp_difference=1,
                handle_zero_special_case=True
            )

            assert zero_ulp_result["zero_handling"] == "special_case"
            assert zero_ulp_result["subnormal_detected"] is True
        else:
            # Skip subnormal test if platform doesn't support them
            pytest.skip("Platform does not support subnormal numbers")

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_broadcasting_vs_exact_shape_validation(self):
        """Test explicit handling of broadcasting vs exact shape requirements."""
        engine = TensorComparisonEngine()

        tensor_2x3 = np.array([[1, 2, 3], [4, 5, 6]])          # (2, 3)
        tensor_1x3 = np.array([[1, 2, 3]])                     # (1, 3) - broadcastable
        tensor_3x2 = np.array([[1, 2], [3, 4], [5, 6]])       # (3, 2) - not broadcastable

        # Test with broadcasting disabled (exact shape required)
        exact_shape_result = engine.validate_tensor_compatibility(
            tensor_2x3,
            tensor_1x3,
            allow_broadcasting=False,
            strict_shape_check=True
        )

        assert exact_shape_result["compatible"] is False
        assert "shape_mismatch" in exact_shape_result["issues"]
        assert exact_shape_result["source_shape"] == (2, 3)
        assert exact_shape_result["target_shape"] == (1, 3)
        assert exact_shape_result["broadcasting_attempted"] is False
        # Verify specific mismatch details
        assert exact_shape_result["shape_error_type"] == "dimension_count_mismatch" or \
               exact_shape_result["shape_error_type"] == "exact_shape_required"

        # Test with broadcasting enabled and compatible shapes
        broadcast_compatible_result = engine.validate_tensor_compatibility(
            tensor_2x3,
            tensor_1x3,
            allow_broadcasting=True,
            strict_shape_check=False
        )

        assert broadcast_compatible_result["compatible"] is True
        assert broadcast_compatible_result["broadcasting_required"] is True
        assert broadcast_compatible_result["broadcast_shape"] == (2, 3)

        # Test with broadcasting enabled but incompatible shapes
        broadcast_incompatible_result = engine.validate_tensor_compatibility(
            tensor_2x3,
            tensor_3x2,
            allow_broadcasting=True,
            strict_shape_check=False
        )

        assert broadcast_incompatible_result["compatible"] is False
        assert "broadcasting_failed" in broadcast_incompatible_result["issues"]
        assert broadcast_incompatible_result["numpy_broadcast_error"] is not None
        # Verify specific broadcast failure details
        assert isinstance(broadcast_incompatible_result["numpy_broadcast_error"], ValueError) or \
               "broadcast" in str(broadcast_incompatible_result["numpy_broadcast_error"]).lower()

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_multi_precision_comparison_with_common_dtype_casting(self):
        """Test comparison across different tensor precisions with proper dtype handling."""
        engine = TensorComparisonEngine()

        # Create tensors with different precisions but same values
        base_values = [1.0, 2.5, 1000.0, -0.001]
        tensor_float64 = np.array(base_values, dtype=np.float64)
        tensor_float32 = np.array(base_values, dtype=np.float32)
        tensor_float16 = np.array(base_values, dtype=np.float16)

        # Test float64 vs float32 - should cast to higher precision
        result_64_32 = engine.compare_multi_precision(
            tensor_float64,
            tensor_float32,
            precision_aware=True,
            cast_to_common_dtype=True,
            adaptive_tolerance=True
        )

        assert result_64_32["status"] == "passed"
        assert result_64_32["common_dtype"] == np.float64  # Higher precision
        assert result_64_32["casting_performed"] is True
        assert result_64_32["precision_analysis"]["source_dtype"] == "float64"
        assert result_64_32["precision_analysis"]["target_dtype"] == "float32"

        # Tolerance should be adapted for float32 precision limits
        assert result_64_32["adaptive_tolerance_used"] >= np.finfo(np.float32).eps

        # Test float32 vs float16 - larger tolerance needed
        result_32_16 = engine.compare_multi_precision(
            tensor_float32,
            tensor_float16,
            precision_aware=True,
            cast_to_common_dtype=True,
            adaptive_tolerance=True
        )

        assert result_32_16["status"] == "passed"
        assert result_32_16["common_dtype"] == np.float32
        # float16 has much larger epsilon, so tolerance should be larger
        assert result_32_16["adaptive_tolerance_used"] > result_64_32["adaptive_tolerance_used"]

        # Test precision loss detection
        large_value_tensor_64 = np.array([1e20], dtype=np.float64)
        large_value_tensor_16 = large_value_tensor_64.astype(np.float16)  # Will lose precision

        precision_loss_result = engine.compare_multi_precision(
            large_value_tensor_64,
            large_value_tensor_16,
            precision_aware=True,
            detect_precision_loss=True
        )

        assert precision_loss_result["precision_loss_detected"] is True
        assert precision_loss_result["precision_loss"]["magnitude"] > 0.1

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_large_tensor_optimization_with_monotonic_timing(self):
        """Test performance optimization for large tensor comparisons with monotonic timing."""
        engine = TensorComparisonEngine(performance_mode="optimized")

        # Create large tensors
        large_tensor_a = np.random.rand(1000, 1000).astype(np.float32)
        large_tensor_b = large_tensor_a + np.random.rand(1000, 1000).astype(np.float32) * 1e-6

        with patch('time.perf_counter') as mock_perf_counter:
            # Mock monotonic timing sequence
            mock_perf_counter.side_effect = [0.0, 3.0]  # Start and end times

            result = engine.compare_large_tensors(
                large_tensor_a,
                large_tensor_b,
                chunk_size=10000,
                parallel_workers=4,
                tolerance=1e-5,
                memory_efficient=True
            )

            assert result["status"] == "passed"

            # Verify monotonic timing
            perf_data = result["performance"]
            assert perf_data["timing_source"] == "monotonic"
            assert perf_data["execution_time"] == 3.0  # Final - initial timestamp
            assert perf_data["memory_efficient"] is True
            assert perf_data["chunks_processed"] > 1
            assert perf_data["parallel_workers_used"] == 4

            # Verify chunked processing maintained accuracy
            assert result["chunked_comparison_accuracy"] >= 0.999

            # Test that chunked results have reasonable maximum error
            # Use adaptive tolerance based on float32 precision
            assert result["max_difference"] <= result.get("tolerance", 1e-5)

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_content_based_tensor_caching(self):
        """Test caching of comparison results using content-based hashing."""
        engine = TensorComparisonEngine(cache_directory="/tmp/tensor_cache")

        tensor_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        tensor_b = np.array([[1.0001, 2.0001, 3.0001], [4.0001, 5.0001, 6.0001]])

        # Generate content-based cache keys
        cache_key_a = engine._generate_tensor_cache_key(tensor_a)
        cache_key_b = engine._generate_tensor_cache_key(tensor_b)

        # Verify cache keys are based on content, not object identity
        tensor_a_copy = tensor_a.copy()
        cache_key_a_copy = engine._generate_tensor_cache_key(tensor_a_copy)
        assert cache_key_a == cache_key_a_copy  # Same content, same key

        # Different content should produce different keys
        assert cache_key_a != cache_key_b

        # Test caching behavior
        comparison_key = f"{cache_key_a}_{cache_key_b}_1e-3"

        # First comparison - should compute and cache
        result1 = engine.compare_with_caching(
            tensor_a,
            tensor_b,
            tolerance=1e-3,
            cache_key=comparison_key
        )

        assert result1["status"] == "passed"
        assert result1["from_cache"] is False
        assert result1["cache_key_stable"] is True

        # Second identical comparison - should use cache
        result2 = engine.compare_with_caching(
            tensor_a,
            tensor_b,
            tolerance=1e-3,
            cache_key=comparison_key
        )

        assert result2["status"] == "passed"
        assert result2["from_cache"] is True
        assert result2["cache_key"] == comparison_key

        # Test cache key includes tensor metadata (shape, dtype)
        tensor_c = tensor_a.astype(np.float32)  # Same values, different dtype
        cache_key_c = engine._generate_tensor_cache_key(tensor_c)
        assert cache_key_a != cache_key_c  # Different dtype should change key

        # Test cache key stability across processes/sessions
        cache_key_bytes = engine._tensor_to_cache_bytes(tensor_a)
        expected_hash = hashlib.sha256(cache_key_bytes).hexdigest()[:16]
        assert expected_hash in cache_key_a

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
    def test_sparse_tensor_comparison_csr_format(self):
        """Test comparison of sparse tensors using CSR format."""
        engine = TensorComparisonEngine()

        # Create sparse matrices in CSR format
        dense_a = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 0.0, 6.0]
        ])
        sparse_a = sparse.csr_matrix(dense_a)

        # Create similar sparse matrix with small differences
        dense_b = dense_a.copy()
        dense_b[0, 0] = 1.0001  # Small change
        dense_b[1, 1] = 3.0001  # Small change
        sparse_b = sparse.csr_matrix(dense_b)

        sparse_result = engine.compare_sparse_tensors(
            sparse_a=sparse_a,
            sparse_b=sparse_b,
            tolerance=1e-3,
            format="csr",
            check_sparsity_pattern=True,
            check_structural_equality=True
        )

        assert sparse_result["status"] == "passed"
        assert sparse_result["format_verified"] == "csr"
        assert sparse_result["sparsity_pattern_match"] is True
        assert sparse_result["structural_equality"] is True

        # Verify sparse-specific properties
        assert sparse_result["nnz_match"] is True  # Same number of non-zeros
        assert sparse_result["indices_match"] is True  # Same sparsity pattern
        assert sparse_result["data_comparison"]["max_difference"] < 1e-3

        # Test with different sparsity patterns
        dense_c = dense_a.copy()
        dense_c[0, 1] = 7.0  # Add new non-zero
        sparse_c = sparse.csr_matrix(dense_c)

        pattern_mismatch_result = engine.compare_sparse_tensors(
            sparse_a=sparse_a,
            sparse_b=sparse_c,
            tolerance=1e-3,
            format="csr",
            check_sparsity_pattern=True
        )

        assert pattern_mismatch_result["status"] == "failed"
        assert pattern_mismatch_result["sparsity_pattern_match"] is False
        assert "pattern_differences" in pattern_mismatch_result

        # Verify CSR-specific data integrity
        assert sparse_result["csr_data_integrity"]["indptr_valid"] is True
        assert sparse_result["csr_data_integrity"]["indices_sorted"] is True

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_gradient_and_derivative_comparison_with_proper_norms(self):
        """Test comparison of tensor gradients with proper norm definitions."""
        engine = TensorComparisonEngine()

        # Create gradient tensors with known properties
        grad_a = np.array([[0.1, 0.2], [0.3, 0.4]])
        grad_b = np.array([[0.1001, 0.2001], [0.3001, 0.4001]])

        gradient_result = engine.compare_gradients(
            grad_a,
            grad_b,
            gradient_tolerance=1e-3,
            norm_type="l2",  # Specify norm explicitly
            check_gradient_norm=True,
            check_gradient_direction=True,
            check_magnitude_scale=True
        )

        assert gradient_result["status"] == "passed"

        # Verify norm calculations
        norm_a = np.linalg.norm(grad_a, ord=2)
        norm_b = np.linalg.norm(grad_b, ord=2)
        expected_norm_diff = abs(norm_a - norm_b)

        assert gradient_result["gradient_norm_difference"] == pytest.approx(expected_norm_diff, abs=1e-10)

        # Verify cosine similarity calculation
        dot_product = np.sum(grad_a * grad_b)
        cosine_sim = dot_product / (norm_a * norm_b)

        assert gradient_result["cosine_similarity"] == pytest.approx(cosine_sim, abs=1e-10)
        assert gradient_result["cosine_similarity"] > 0.999  # Very similar directions
        assert gradient_result["gradient_direction_consistent"] is True

        # Test with different norms
        l1_result = engine.compare_gradients(
            grad_a, grad_b,
            norm_type="l1",
            gradient_tolerance=1e-3
        )

        linf_result = engine.compare_gradients(
            grad_a, grad_b,
            norm_type="linf",
            gradient_tolerance=1e-3
        )

        # Different norms should give different results
        assert l1_result["gradient_norm_difference"] != gradient_result["gradient_norm_difference"]
        assert linf_result["gradient_norm_difference"] != gradient_result["gradient_norm_difference"]

        # Test gradient magnitude scale invariance
        scaled_grad_b = grad_b * 1.1  # 10% magnitude increase

        scale_result = engine.compare_gradients(
            grad_a,
            scaled_grad_b,
            check_gradient_direction=True,
            check_magnitude_scale=True,
            scale_invariant_comparison=True
        )

        # Direction should still be very similar despite magnitude difference
        assert scale_result["cosine_similarity"] > 0.999
        assert scale_result["magnitude_scale_factor"] == pytest.approx(1.1, rel=1e-3)

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_batch_tensor_comparison(self):
        """Test batch processing of multiple tensor pairs."""
        engine = TensorComparisonEngine()

        # Create multiple tensor pairs with predictable properties
        tensor_pairs = []
        for i in range(3):
            base = np.random.RandomState(i).rand(10, 10)  # Seeded for reproducibility
            modified = base + np.random.RandomState(i+100).rand(10, 10) * 1e-5
            tensor_pairs.append((base, modified))

        batch_results = engine.compare_tensor_batch(
            tensor_pairs,
            tolerance=1e-4,
            comparison_method="element_wise",
            parallel_processing=True,
            maintain_order=True
        )

        assert len(batch_results) == 3
        assert all(result["status"] == "passed" for result in batch_results)

        # Verify batch indexing and order preservation
        for i, result in enumerate(batch_results):
            assert result["batch_index"] == i
            assert result["pair_id"] == i

        # Verify parallel processing stats
        if len(batch_results) > 1:
            assert batch_results[0]["processing_metadata"]["parallel_processed"] is True

        # Test batch aggregation
        batch_summary = engine.aggregate_batch_results(batch_results)
        assert batch_summary["total_comparisons"] == 3
        assert batch_summary["passed_comparisons"] == 3
        assert batch_summary["failed_comparisons"] == 0
        assert batch_summary["average_execution_time"] > 0

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_error_analysis_and_reporting(self):
        """Test detailed error analysis and reporting capabilities."""
        engine = TensorComparisonEngine()

        # Create tensors with specific error patterns
        tensor_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tensor_b = np.array([1.1, 2.0, 2.9, 5.0, 5.0])  # Errors at indices 0, 2, 3

        error_analysis = engine.analyze_errors(
            tensor_a,
            tensor_b,
            detailed_analysis=True,
            error_localization=True,
            outlier_detection=True,
            statistical_analysis=True
        )

        assert error_analysis["summary"]["total_errors"] == 3
        assert error_analysis["summary"]["error_rate"] == 0.6  # 3/5
        assert len(error_analysis["error_locations"]) == 3

        # Verify error localization
        error_indices = [loc["index"] for loc in error_analysis["error_locations"]]
        assert set(error_indices) == {0, 2, 3}

        # Verify error magnitudes
        for loc in error_analysis["error_locations"]:
            idx = loc["index"]
            expected_error = abs(tensor_a[idx] - tensor_b[idx])
            assert loc["absolute_error"] == pytest.approx(expected_error, abs=1e-10)

        # Verify error distribution analysis
        assert "error_histogram" in error_analysis["distribution"]
        assert "error_percentiles" in error_analysis["distribution"]
        assert "mean_error" in error_analysis["distribution"]
        assert "std_error" in error_analysis["distribution"]

        # Verify outlier detection
        assert "outlier_indices" in error_analysis["outlier_analysis"]
        # Index 3 has largest error (4.0 vs 4.2 = 0.2), should be detected as outlier
        assert 3 in error_analysis["outlier_analysis"]["outlier_indices"]

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_comparison_visualization_data_generation(self):
        """Test generation of structured data for tensor comparison visualization."""
        engine = TensorComparisonEngine()

        tensor_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor_b = np.array([[1.1, 2.0], [3.2, 4.0]])  # Errors at (0,0) and (1,0)

        viz_data = engine.generate_visualization_data(
            tensor_a,
            tensor_b,
            visualization_type="heatmap",
            include_error_map=True,
            include_histograms=True,
            max_display_elements=1000  # Limit for large tensors
        )

        # Verify heatmap data structure
        assert "error_heatmap" in viz_data
        assert viz_data["error_heatmap"]["shape"] == (2, 2)
        assert "data" in viz_data["error_heatmap"]
        assert "coordinates" in viz_data["error_heatmap"]

        # Heatmap should show errors at specific locations
        error_data = viz_data["error_heatmap"]["data"]
        assert error_data[0, 0] > 0  # Error at (0,0)
        assert error_data[0, 1] == 0  # No error at (0,1)
        assert error_data[1, 0] > 0  # Error at (1,0)
        assert error_data[1, 1] == 0  # No error at (1,1)

        # Verify histogram data
        assert "tensor_a_histogram" in viz_data
        assert "tensor_b_histogram" in viz_data
        assert "error_distribution" in viz_data

        # Histogram should have proper structure
        hist_a = viz_data["tensor_a_histogram"]
        assert "bins" in hist_a
        assert "counts" in hist_a
        assert len(hist_a["bins"]) == len(hist_a["counts"]) + 1  # N bins, N+1 edges

        # Verify error location annotation
        assert len(viz_data["error_locations"]) == 2  # Two locations with errors
        for error_loc in viz_data["error_locations"]:
            assert "coordinates" in error_loc
            assert "magnitude" in error_loc
            assert "relative_error" in error_loc

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_adaptive_tolerance_magnitude_aware(self):
        """Test adaptive tolerance adjustment that handles both small and large magnitudes."""
        engine = TensorComparisonEngine()

        # Test with small magnitudes (atol dominant)
        small_tensor_a = np.array([1e-8, 2e-8, 3e-8])
        small_tensor_b = np.array([1.1e-8, 2.1e-8, 3.1e-8])  # 10% relative difference

        small_result = engine.compare_with_adaptive_tolerance(
            small_tensor_a,
            small_tensor_b,
            base_atol=1e-10,
            base_rtol=1e-5,
            magnitude_aware=True,
            prevent_atol_dominance=False  # Allow atol to dominate for small values
        )

        assert small_result["status"] == "passed"
        assert small_result["tolerance_regime"] == "small_magnitude"
        assert small_result["adaptive_atol"] >= small_result["base_atol"]

        # Test with large magnitudes (rtol dominant)
        large_tensor_a = np.array([1e8, 2e8, 3e8])
        large_tensor_b = np.array([1.00001e8, 2.00002e8, 3.00003e8])  # ~1e-5 relative difference

        large_result = engine.compare_with_adaptive_tolerance(
            large_tensor_a,
            large_tensor_b,
            base_atol=1e-10,
            base_rtol=1e-5,
            magnitude_aware=True,
            prevent_atol_dominance=True  # Prevent atol from hiding small relative errors
        )

        assert large_result["status"] == "passed"
        assert large_result["tolerance_regime"] == "large_magnitude"
        assert large_result["adaptive_rtol"] >= large_result["base_rtol"]

        # Verify magnitude-based tolerance scaling
        assert small_result["tolerance_adjustment_factor"] != large_result["tolerance_adjustment_factor"]
        assert small_result["magnitude_analysis"]["regime"] == "small_magnitude"
        assert large_result["magnitude_analysis"]["regime"] == "large_magnitude"

        # Test mixed magnitudes (some small, some large)
        mixed_tensor_a = np.array([1e-10, 1e7, 1e-3, 1e10])  # >1e6 threshold
        mixed_tensor_b = mixed_tensor_a * (1 + 1e-6)  # Small relative change

        mixed_result = engine.compare_with_adaptive_tolerance(
            mixed_tensor_a,
            mixed_tensor_b,
            base_atol=1e-12,
            base_rtol=1e-5,
            magnitude_aware=True,
            per_element_adaptation=True
        )

        assert mixed_result["status"] == "passed"
        assert mixed_result["tolerance_regime"] == "mixed_regime"
        assert "per_element_tolerances" in mixed_result

        # Large magnitude elements should use rtol, small should use atol
        elem_tolerances = mixed_result["per_element_tolerances"]
        assert elem_tolerances[0]["regime"] == "atol_dominated"    # 1e-10
        assert elem_tolerances[1]["regime"] == "rtol_dominated"    # 1e7
        assert elem_tolerances[3]["regime"] == "rtol_dominated"    # 1e10

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_tensor_metadata_comparison(self):
        """Test comparison of tensor metadata and properties."""
        engine = TensorComparisonEngine()

        tensor_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_b = np.array([1.0001, 2.0001, 3.0001], dtype=np.float32)
        tensor_c = np.array([1.0001, 2.0001, 3.0001], dtype=np.float64)  # Different dtype

        # Test matching metadata
        metadata_result_match = engine.compare_tensor_metadata(
            tensor_a,
            tensor_b,
            check_dtype=True,
            check_shape=True,
            check_memory_layout=True,
            check_device_placement=False  # Skip device check for CPU tensors
        )

        assert metadata_result_match["dtype_match"] is True
        assert metadata_result_match["shape_match"] is True
        assert metadata_result_match["memory_layout_match"] is True
        assert metadata_result_match["metadata_compatible"] is True

        # Test mismatched dtype
        metadata_result_mismatch = engine.compare_tensor_metadata(
            tensor_a,
            tensor_c,  # Different dtype
            check_dtype=True,
            check_shape=True
        )

        assert metadata_result_mismatch["dtype_match"] is False
        assert metadata_result_mismatch["shape_match"] is True
        assert metadata_result_mismatch["metadata_compatible"] is False
        assert metadata_result_mismatch["dtype_mismatch_details"]["source"] == "float32"
        assert metadata_result_mismatch["dtype_mismatch_details"]["target"] == "float64"

        # Test memory layout comparison (use 2D arrays since 1D arrays are always both C and F contiguous)
        tensor_a_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_fortran = np.asfortranarray(tensor_a_2d)

        layout_result = engine.compare_tensor_metadata(
            tensor_a_2d,  # C-contiguous
            tensor_fortran,  # Fortran-contiguous
            check_memory_layout=True,
            memory_layout_strict=True
        )

        assert layout_result["memory_layout_match"] is False
        assert layout_result["memory_layout_details"]["source"] == "C_CONTIGUOUS"
        assert layout_result["memory_layout_details"]["target"] == "F_CONTIGUOUS"

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_performance_profiling_with_monotonic_clock(self):
        """Test performance profiling of comparison operations with monotonic timing."""
        engine = TensorComparisonEngine()

        tensor_a = np.random.rand(1000, 1000).astype(np.float32)
        tensor_b = tensor_a + np.random.rand(1000, 1000).astype(np.float32) * 1e-6

        with patch('psutil.Process') as mock_process:
            # Mock memory and CPU monitoring
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_proc.cpu_percent.return_value = 75.0
            mock_process.return_value = mock_proc

            profiling_result = engine.compare_with_profiling(
                tensor_a,
                tensor_b,
                tolerance=1e-5,
                profile_memory=True,
                profile_cpu=True,
                timing_source="monotonic"
            )

            assert profiling_result["comparison"]["status"] == "passed"

            # Verify monotonic timing usage
            perf_data = profiling_result["profiling"]
            assert perf_data["timing_source"] == "monotonic"
            assert perf_data["total_time"] > 0  # Should take some time
            assert perf_data["phase_times"]["initialization"] >= 0
            assert perf_data["phase_times"]["comparison"] >= 0
            assert perf_data["phase_times"]["analysis"] >= 0
            assert perf_data["phase_times"]["finalization"] >= 0

            # Verify resource monitoring
            assert perf_data["memory_usage"]["peak_rss_mb"] == 100
            assert perf_data["cpu_utilization"]["average_percent"] == 75.0

    @pytest.mark.skipif(not TENSOR_COMPARISON_ENGINE_AVAILABLE, reason="TensorComparisonEngine not implemented")
    def test_nan_inf_handling_in_comparisons(self):
        """Test proper handling of NaN and Inf values in tensor comparisons."""
        engine = TensorComparisonEngine()

        # Create tensors with NaN and Inf values
        tensor_a = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
        tensor_b = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])  # Identical
        tensor_c = np.array([1.0, 0.0, np.inf, -np.inf, np.nan])  # Different NaN/normal placement

        # Test identical NaN/Inf handling
        nan_inf_result = engine.compare_with_nan_inf_handling(
            tensor_a,
            tensor_b,
            nan_handling="equal",     # NaN == NaN
            inf_handling="standard",  # Inf == Inf, -Inf == -Inf
            tolerance=1e-5
        )

        assert nan_inf_result["status"] == "passed"
        assert nan_inf_result["nan_count"] == 1
        assert nan_inf_result["inf_count"] == 1
        assert nan_inf_result["neg_inf_count"] == 1
        assert nan_inf_result["nan_positions_match"] is True
        assert nan_inf_result["inf_positions_match"] is True

        # Test different NaN positions
        nan_mismatch_result = engine.compare_with_nan_inf_handling(
            tensor_a,
            tensor_c,
            nan_handling="equal",
            inf_handling="standard"
        )

        assert nan_mismatch_result["status"] == "failed"
        assert nan_mismatch_result["nan_positions_match"] is False
        assert "nan_position_differences" in nan_mismatch_result

        # Test NaN handling modes
        nan_ignore_result = engine.compare_with_nan_inf_handling(
            tensor_a,
            tensor_c,
            nan_handling="ignore",  # Skip NaN elements in comparison
            inf_handling="standard"
        )

        # Should pass when ignoring NaN positions
        assert nan_ignore_result["status"] == "passed"
        assert nan_ignore_result["elements_ignored"] == 2  # 2 NaN elements total

        # Test propagation mode
        nan_propagate_result = engine.compare_with_nan_inf_handling(
            tensor_a,
            tensor_c,
            nan_handling="propagate"  # Any NaN causes comparison failure
        )

        assert nan_propagate_result["status"] == "failed"
        assert nan_propagate_result["failure_reason"] == "nan_propagation"
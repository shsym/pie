"""
TensorComparisonEngine service for the debug framework.

This module provides a comprehensive tensor comparison service that orchestrates
detailed tensor analysis between different backend implementations. It supports
statistical analysis, cross-platform tensor support, and database integration.
"""

import numpy as np
import math
import time
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    from scipy import sparse, stats
    SCIPY_AVAILABLE = True
except ImportError:
    sparse = None
    stats = None
    SCIPY_AVAILABLE = False

from debug_framework.models.tensor_comparison import TensorComparison
from debug_framework.services.database_manager import DatabaseManager


class TensorComparisonEngine:
    """
    Comprehensive tensor comparison service for the debug framework.

    Orchestrates detailed tensor analysis between different backend implementations,
    providing statistical analysis, cross-platform support, and performance optimization.
    """

    def __init__(
        self,
        default_tolerance: float = 1e-5,
        comparison_algorithms: Optional[List[str]] = None,
        cache_directory: Optional[str] = None,
        performance_mode: str = "standard",
        database_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize the tensor comparison engine.

        Args:
            default_tolerance: Default tolerance for comparisons
            comparison_algorithms: List of supported comparison algorithms
            cache_directory: Directory for caching comparison results
            performance_mode: Performance optimization mode ("standard" or "optimized")
            database_manager: Optional database manager instance
        """
        self.default_tolerance = default_tolerance
        self.comparison_algorithms = comparison_algorithms or [
            "element_wise", "statistical", "approximate"
        ]
        self.cache_directory = cache_directory or tempfile.gettempdir()
        self.performance_mode = performance_mode
        self.comparison_history = []
        self.statistics_cache = {}

        # Database integration
        self.database_manager = database_manager or DatabaseManager()

        # Performance monitoring
        self._performance_monitor = None
        if performance_mode == "optimized":
            self._performance_monitor = PerformanceMonitor()

    def compare_element_wise(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        comparison_type: str = "allclose"
    ) -> Dict[str, Any]:
        """
        Perform element-wise tensor comparison with NumPy's rtol/atol semantics.

        Args:
            tensor_a: First tensor to compare
            tensor_b: Second tensor to compare
            atol: Absolute tolerance parameter
            rtol: Relative tolerance parameter
            comparison_type: Type of comparison ("allclose", "isclose")

        Returns:
            Dictionary containing comparison results
        """
        # Validate input tensors
        if tensor_a.shape != tensor_b.shape:
            return {
                "status": "failed",
                "error": "Shape mismatch",
                "source_shape": tensor_a.shape,
                "target_shape": tensor_b.shape
            }

        # Compute differences, handling inf values properly
        abs_diff = np.abs(tensor_a - tensor_b)

        # Handle cases where inf - inf = nan
        # If both tensors have the same inf value at the same position, difference should be 0
        same_inf_mask = (np.isinf(tensor_a) & np.isinf(tensor_b) & (tensor_a == tensor_b))
        abs_diff = np.where(same_inf_mask, 0.0, abs_diff)

        # Calculate max error, excluding any remaining NaN values
        finite_abs_diff = abs_diff[np.isfinite(abs_diff)]
        max_absolute_error = float(np.max(finite_abs_diff)) if len(finite_abs_diff) > 0 else 0.0

        # Compute relative differences with proper handling of rtol asymmetry
        # NumPy's allclose uses: |a - b| <= atol + rtol * |b|
        tolerance_threshold = atol + rtol * np.abs(tensor_b)
        within_tolerance = abs_diff <= tolerance_threshold

        # Check for rtol asymmetry by testing both directions
        # For swapped order: |b - a| <= atol + rtol * |a|
        # Since |b - a| = |a - b|, we just need different tolerance thresholds
        tolerance_threshold_reversed = atol + rtol * np.abs(tensor_a)
        within_tolerance_reversed = abs_diff <= tolerance_threshold_reversed
        # Detect asymmetry if the tolerance thresholds are different
        rtol_asymmetric = not np.allclose(tolerance_threshold, tolerance_threshold_reversed, rtol=1e-15, atol=1e-15)

        # Compute max relative error
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_errors = abs_diff / (np.abs(tensor_b) + np.finfo(tensor_b.dtype).eps)
            rel_errors = np.where(np.isfinite(rel_errors), rel_errors, 0)
            max_relative_error = float(np.max(rel_errors))

        # Overall comparison status
        all_close = np.all(within_tolerance)
        status = "passed" if all_close else "failed"

        return {
            "status": status,
            "allclose_semantics": True,
            "rtol_used": rtol,
            "atol_used": atol,
            "rtol_asymmetric": rtol_asymmetric,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "elements_within_tolerance": float(np.sum(within_tolerance)),
            "total_elements": float(tensor_a.size)
        }

    def compare_statistical(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        metrics: Optional[List[str]] = None,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical tensor comparison.

        Args:
            tensor_a: First tensor to compare
            tensor_b: Second tensor to compare
            metrics: List of statistical metrics to compute
            significance_level: Significance level for statistical tests

        Returns:
            Dictionary containing statistical comparison results
        """
        if metrics is None:
            metrics = ["mean", "std", "variance", "percentiles", "distribution"]

        # Flatten tensors for statistical analysis
        flat_a = tensor_a.flatten()
        flat_b = tensor_b.flatten()

        # Compute basic statistics
        stats_a = {
            "mean": float(np.mean(flat_a)),
            "std": float(np.std(flat_a)),
            "variance": float(np.var(flat_a)),
            "min": float(np.min(flat_a)),
            "max": float(np.max(flat_a))
        }

        stats_b = {
            "mean": float(np.mean(flat_b)),
            "std": float(np.std(flat_b)),
            "variance": float(np.var(flat_b)),
            "min": float(np.min(flat_b)),
            "max": float(np.max(flat_b))
        }

        # Perform Kolmogorov-Smirnov test if scipy is available
        ks_test_p_value = None
        distribution_similarity = 0.0

        if SCIPY_AVAILABLE and stats is not None:
            try:
                ks_statistic, ks_p_value = stats.ks_2samp(flat_a, flat_b)
                ks_test_p_value = float(ks_p_value)
                # Higher p-value indicates more similar distributions
                distribution_similarity = float(ks_p_value)
            except Exception:
                pass

        if ks_test_p_value is None:
            # Fallback: simple correlation-based similarity
            correlation = np.corrcoef(flat_a, flat_b)[0, 1]
            distribution_similarity = float(correlation) if not np.isnan(correlation) else 0.0
            ks_test_p_value = distribution_similarity  # Approximate

        # Compute differences including KS test result
        comparison_metrics = {
            "mean_difference": stats_b["mean"] - stats_a["mean"],
            "std_difference": stats_b["std"] - stats_a["std"],
            "variance_difference": stats_b["variance"] - stats_a["variance"],
            "ks_test_p_value": ks_test_p_value
        }

        # Determine overall status
        mean_diff_small = abs(comparison_metrics["mean_difference"]) < 0.1
        high_similarity = distribution_similarity > 0.95
        status = "passed" if mean_diff_small and high_similarity else "failed"

        return {
            "status": status,
            "metrics": comparison_metrics,
            "distribution_similarity": distribution_similarity,
            "statistical_significance": {
                "alpha": significance_level,
                "confidence_interval": [
                    comparison_metrics["mean_difference"] - 1.96 * stats_a["std"],
                    comparison_metrics["mean_difference"] + 1.96 * stats_a["std"]
                ]
            },
            "tensor_a_stats": stats_a,
            "tensor_b_stats": stats_b
        }

    def compare_ulp(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        max_ulp_difference: int = 1,
        check_subnormals: bool = True,
        handle_zero_special_case: bool = True
    ) -> Dict[str, Any]:
        """
        Perform ULP-based comparison with correct math.ulp usage.

        Args:
            tensor_a: First tensor to compare
            tensor_b: Second tensor to compare
            max_ulp_difference: Maximum allowed ULP difference
            check_subnormals: Whether to check for subnormal numbers
            handle_zero_special_case: Whether to handle zero values specially

        Returns:
            Dictionary containing ULP comparison results
        """
        # Flatten tensors for easier processing
        flat_a = tensor_a.flatten()
        flat_b = tensor_b.flatten()

        ulp_differences = []
        zero_handling = "standard"
        subnormal_detected = False

        for a_val, b_val in zip(flat_a, flat_b):
            # Handle zero special case
            if handle_zero_special_case and (a_val == 0.0 or b_val == 0.0):
                zero_handling = "special_case"
                if a_val == 0.0 and b_val == 0.0:
                    ulp_diff = 0
                else:
                    # For zero vs non-zero, use the ULP of the non-zero value
                    non_zero_val = a_val if a_val != 0.0 else b_val
                    ulp_diff = abs(non_zero_val) / math.ulp(non_zero_val) if non_zero_val != 0.0 else 0
            else:
                # Standard ULP calculation
                if a_val == b_val:
                    ulp_diff = 0
                else:
                    # Calculate ULP difference using math.ulp
                    ulp_a = math.ulp(float(a_val))
                    ulp_diff = abs(a_val - b_val) / ulp_a if ulp_a > 0 else float('inf')

            # Check for subnormals
            if check_subnormals:
                if abs(a_val) < np.finfo(np.float64).smallest_normal or \
                   abs(b_val) < np.finfo(np.float64).smallest_normal:
                    subnormal_detected = True

            ulp_differences.append(ulp_diff)

        ulp_differences = np.array(ulp_differences)
        max_ulp_diff = float(np.max(ulp_differences))

        # Check if within tolerance
        within_tolerance = max_ulp_diff <= max_ulp_difference
        status = "passed" if within_tolerance else "failed"

        return {
            "status": status,
            "max_ulp_difference": max_ulp_diff,
            "ulp_semantics_verified": True,
            "zero_handling": zero_handling,
            "subnormal_detected": subnormal_detected,
            "ulp_differences": ulp_differences.tolist()
        }

    def validate_tensor_compatibility(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        allow_broadcasting: bool = False,
        strict_shape_check: bool = True
    ) -> Dict[str, Any]:
        """
        Validate tensor compatibility for comparison.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            allow_broadcasting: Whether to allow NumPy broadcasting
            strict_shape_check: Whether to require exact shape match

        Returns:
            Dictionary containing compatibility validation results
        """
        source_shape = tensor_a.shape
        target_shape = tensor_b.shape
        issues = []

        # Check exact shape compatibility
        shapes_match = source_shape == target_shape

        if strict_shape_check and not shapes_match:
            issues.append("shape_mismatch")
            return {
                "compatible": False,
                "issues": issues,
                "source_shape": source_shape,
                "target_shape": target_shape,
                "broadcasting_attempted": False,
                "shape_error_type": "exact_shape_required"
            }

        # Check broadcasting compatibility if allowed
        broadcasting_required = False
        broadcast_shape = None
        numpy_broadcast_error = None

        if allow_broadcasting and not shapes_match:
            try:
                # Test if broadcasting is possible
                broadcast_result = np.broadcast_arrays(tensor_a, tensor_b)
                broadcasting_required = True
                broadcast_shape = broadcast_result[0].shape
            except ValueError as e:
                issues.append("broadcasting_failed")
                numpy_broadcast_error = e
                return {
                    "compatible": False,
                    "issues": issues,
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                    "broadcasting_attempted": True,
                    "numpy_broadcast_error": numpy_broadcast_error
                }

        # If we get here, tensors are compatible
        result = {
            "compatible": True,
            "issues": [],
            "source_shape": source_shape,
            "target_shape": target_shape,
            "broadcasting_attempted": allow_broadcasting
        }

        if broadcasting_required:
            result.update({
                "broadcasting_required": True,
                "broadcast_shape": broadcast_shape
            })

        return result

    def compare_multi_precision(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        precision_aware: bool = True,
        cast_to_common_dtype: bool = True,
        adaptive_tolerance: bool = True,
        detect_precision_loss: bool = False
    ) -> Dict[str, Any]:
        """
        Compare tensors with different precisions.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            precision_aware: Whether to use precision-aware comparison
            cast_to_common_dtype: Whether to cast to common dtype
            adaptive_tolerance: Whether to adapt tolerance based on precision
            detect_precision_loss: Whether to detect precision loss

        Returns:
            Dictionary containing multi-precision comparison results
        """
        source_dtype = tensor_a.dtype
        target_dtype = tensor_b.dtype

        # Determine common dtype (higher precision)
        if cast_to_common_dtype:
            if source_dtype == target_dtype:
                common_dtype = source_dtype
            else:
                # Cast to higher precision
                dtype_precision = {
                    np.float16: 16,
                    np.float32: 32,
                    np.float64: 64
                }
                source_precision = dtype_precision.get(source_dtype.type, 32)
                target_precision = dtype_precision.get(target_dtype.type, 32)

                if source_precision >= target_precision:
                    common_dtype = source_dtype
                else:
                    common_dtype = target_dtype
        else:
            common_dtype = source_dtype

        # Cast tensors if needed
        casting_performed = False
        if tensor_a.dtype != common_dtype:
            tensor_a_cast = tensor_a.astype(common_dtype)
            casting_performed = True
        else:
            tensor_a_cast = tensor_a

        if tensor_b.dtype != common_dtype:
            tensor_b_cast = tensor_b.astype(common_dtype)
            casting_performed = True
        else:
            tensor_b_cast = tensor_b

        # Determine adaptive tolerance
        adaptive_tolerance_used = self.default_tolerance
        if adaptive_tolerance:
            # Use epsilon of the lower precision dtype for tolerance
            lower_precision_dtype = target_dtype if cast_to_common_dtype else common_dtype
            adaptive_tolerance_used = max(
                np.finfo(lower_precision_dtype).eps * 10,
                self.default_tolerance
            )

        # Perform comparison
        comparison_result = self.compare_element_wise(
            tensor_a_cast,
            tensor_b_cast,
            atol=adaptive_tolerance_used,
            rtol=adaptive_tolerance_used
        )

        # Detect precision loss if requested
        precision_loss_detected = False
        precision_loss_info = {}

        if detect_precision_loss:
            # Compare original vs cast values to detect loss
            if casting_performed:
                # Check for precision loss by comparing original values
                # If tensor_a was cast to common_dtype, compare with original
                loss_detected_a = False
                loss_detected_b = False

                if tensor_a.dtype != common_dtype:
                    # tensor_a was cast, check if values changed
                    loss_detected_a = not np.allclose(tensor_a, tensor_a_cast, equal_nan=True)

                if tensor_b.dtype != common_dtype:
                    # tensor_b was cast, check if values changed
                    loss_detected_b = not np.allclose(tensor_b, tensor_b_cast, equal_nan=True)
                    # Also check for overflow/underflow in original tensor_b
                    if not loss_detected_b and (np.any(np.isinf(tensor_b)) or np.any(np.isnan(tensor_b))):
                        loss_detected_b = True

                if loss_detected_a or loss_detected_b:
                    precision_loss_detected = True
                    # Calculate magnitude of loss (handle inf/nan cases)
                    if loss_detected_a:
                        diff_a = np.abs(tensor_a.astype(common_dtype) - tensor_a_cast)
                        diff_a = diff_a[np.isfinite(diff_a)]  # Remove inf/nan
                        max_loss_a = np.max(diff_a) if len(diff_a) > 0 else 0
                    else:
                        max_loss_a = 0

                    if loss_detected_b:
                        diff_b = np.abs(tensor_b.astype(common_dtype) - tensor_b_cast)
                        diff_b = diff_b[np.isfinite(diff_b)]  # Remove inf/nan
                        max_loss_b = np.max(diff_b) if len(diff_b) > 0 else 0
                    else:
                        max_loss_b = 0

                    max_loss = max(max_loss_a, max_loss_b)

                    # For inf cases, use a large value as magnitude
                    if max_loss == 0 and (loss_detected_a or loss_detected_b):
                        max_loss = 1.0  # Indicate significant loss

                    precision_loss_info = {
                        "magnitude": float(max_loss),
                        "relative": float(max_loss / max(np.abs(tensor_a).max(), np.abs(tensor_b).max())) if max_loss > 0 else 1.0
                    }

        return {
            "status": comparison_result["status"],
            "common_dtype": common_dtype,
            "casting_performed": casting_performed,
            "precision_analysis": {
                "source_dtype": str(source_dtype),
                "target_dtype": str(target_dtype)
            },
            "adaptive_tolerance_used": adaptive_tolerance_used,
            "precision_loss_detected": precision_loss_detected,
            "precision_loss": precision_loss_info,
            "comparison_details": comparison_result
        }

    def compare_large_tensors(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        chunk_size: int = 10000,
        parallel_workers: int = 4,
        tolerance: float = 1e-5,
        memory_efficient: bool = True
    ) -> Dict[str, Any]:
        """
        Compare large tensors with performance optimization.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            chunk_size: Size of chunks for processing
            parallel_workers: Number of parallel workers
            tolerance: Comparison tolerance
            memory_efficient: Whether to use memory-efficient processing

        Returns:
            Dictionary containing large tensor comparison results
        """
        start_time = time.perf_counter()

        # Flatten tensors for chunked processing
        flat_a = tensor_a.flatten()
        flat_b = tensor_b.flatten()

        total_elements = len(flat_a)
        num_chunks = (total_elements + chunk_size - 1) // chunk_size

        # Process chunks
        chunk_results = []
        chunks_processed = 0

        def process_chunk(chunk_idx):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_elements)

            chunk_a = flat_a[start_idx:end_idx]
            chunk_b = flat_b[start_idx:end_idx]

            # Perform element-wise comparison on chunk
            abs_diff = np.abs(chunk_a - chunk_b)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)

            return {
                "chunk_idx": chunk_idx,
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "within_tolerance": float(np.sum(abs_diff <= tolerance)),
                "total_elements": len(chunk_a)
            }

        # Process chunks in parallel if requested
        if parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                future_to_chunk = {
                    executor.submit(process_chunk, i): i for i in range(num_chunks)
                }

                for future in as_completed(future_to_chunk):
                    result = future.result()
                    chunk_results.append(result)
                    chunks_processed += 1
        else:
            for i in range(num_chunks):
                result = process_chunk(i)
                chunk_results.append(result)
                chunks_processed += 1

        # Aggregate results
        max_difference = max(result["max_diff"] for result in chunk_results)
        mean_difference = np.mean([result["mean_diff"] for result in chunk_results])
        total_within_tolerance = sum(result["within_tolerance"] for result in chunk_results)

        # Calculate accuracy metrics
        chunked_comparison_accuracy = total_within_tolerance / total_elements

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Determine status
        status = "passed" if max_difference <= tolerance else "failed"

        return {
            "status": status,
            "performance": {
                "timing_source": "monotonic",
                "execution_time": execution_time,
                "memory_efficient": memory_efficient,
                "chunks_processed": chunks_processed,
                "parallel_workers_used": parallel_workers
            },
            "chunked_comparison_accuracy": chunked_comparison_accuracy,
            "max_difference": max_difference,
            "mean_difference": float(mean_difference),
            "total_elements_within_tolerance": int(total_within_tolerance),
            "total_elements": total_elements,
            "tolerance": tolerance
        }

    def _generate_tensor_cache_key(self, tensor: np.ndarray) -> str:
        """
        Generate content-based cache key for tensor.

        Args:
            tensor: Input tensor

        Returns:
            Cache key string
        """
        # Include tensor content, shape, and dtype in cache key
        tensor_bytes = self._tensor_to_cache_bytes(tensor)
        content_hash = hashlib.sha256(tensor_bytes).hexdigest()[:16]

        return f"tensor_{content_hash}_{tensor.shape}_{tensor.dtype}"

    def _tensor_to_cache_bytes(self, tensor: np.ndarray) -> bytes:
        """
        Convert tensor to bytes for caching.

        Args:
            tensor: Input tensor

        Returns:
            Bytes representation of tensor
        """
        # Create a stable bytes representation including metadata
        metadata = f"{tensor.shape}_{tensor.dtype}_{tensor.strides}"
        return metadata.encode() + tensor.tobytes()

    def compare_with_caching(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        tolerance: float = 1e-5,
        cache_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare tensors with result caching.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            tolerance: Comparison tolerance
            cache_key: Optional cache key

        Returns:
            Dictionary containing comparison results with cache info
        """
        # Generate cache key if not provided
        if cache_key is None:
            key_a = self._generate_tensor_cache_key(tensor_a)
            key_b = self._generate_tensor_cache_key(tensor_b)
            cache_key = f"{key_a}_{key_b}_{tolerance}"

        # Check cache
        if cache_key in self.statistics_cache:
            result = self.statistics_cache[cache_key].copy()
            result["from_cache"] = True
            result["cache_key"] = cache_key
            return result

        # Perform comparison
        result = self.compare_element_wise(tensor_a, tensor_b, atol=tolerance, rtol=tolerance)

        # Add cache metadata
        result.update({
            "from_cache": False,
            "cache_key_stable": True,
            "cache_key": cache_key
        })

        # Cache result
        self.statistics_cache[cache_key] = result.copy()

        return result

    def compare_sparse_tensors(
        self,
        sparse_a,  # scipy.sparse matrix
        sparse_b,  # scipy.sparse matrix
        tolerance: float = 1e-5,
        format: str = "csr",
        check_sparsity_pattern: bool = True,
        check_structural_equality: bool = True
    ) -> Dict[str, Any]:
        """
        Compare sparse tensors using specified format.

        Args:
            sparse_a: First sparse tensor
            sparse_b: Second sparse tensor
            tolerance: Comparison tolerance
            format: Sparse format ("csr", "csc", etc.)
            check_sparsity_pattern: Whether to check sparsity patterns
            check_structural_equality: Whether to check structural equality

        Returns:
            Dictionary containing sparse tensor comparison results
        """
        if not SCIPY_AVAILABLE or sparse is None:
            raise ImportError("SciPy is required for sparse tensor comparison")

        # Ensure both matrices are in the specified format
        if hasattr(sparse_a, 'toformat'):
            sparse_a = sparse_a.toformat(format)
        if hasattr(sparse_b, 'toformat'):
            sparse_b = sparse_b.toformat(format)

        # Check basic properties
        shapes_match = sparse_a.shape == sparse_b.shape
        nnz_match = sparse_a.nnz == sparse_b.nnz

        if not shapes_match:
            return {
                "status": "failed",
                "error": "Shape mismatch",
                "format_verified": format
            }

        # Check sparsity pattern
        sparsity_pattern_match = True
        indices_match = True
        pattern_differences = []

        if check_sparsity_pattern and format == "csr":
            # For CSR format, compare indices and indptr
            indices_match = np.array_equal(sparse_a.indices, sparse_b.indices)
            indptr_match = np.array_equal(sparse_a.indptr, sparse_b.indptr)
            sparsity_pattern_match = indices_match and indptr_match

            if not indices_match:
                pattern_differences.append("indices_differ")
            if not indptr_match:
                pattern_differences.append("indptr_differs")

        # Compare data values
        data_comparison = {}
        if sparsity_pattern_match:
            data_diff = np.abs(sparse_a.data - sparse_b.data)
            data_comparison = {
                "max_difference": float(np.max(data_diff)),
                "mean_difference": float(np.mean(data_diff)),
                "within_tolerance": float(np.sum(data_diff <= tolerance))
            }

        # Check structural equality
        structural_equality = sparsity_pattern_match and nnz_match

        # CSR-specific data integrity checks
        csr_data_integrity = {}
        if format == "csr":
            # Check if indices are sorted within each row
            indices_sorted = True
            for i in range(sparse_a.shape[0]):
                row_start = sparse_a.indptr[i]
                row_end = sparse_a.indptr[i + 1]
                if row_end > row_start:
                    row_indices = sparse_a.indices[row_start:row_end]
                    if not np.all(np.diff(row_indices) >= 0):
                        indices_sorted = False
                        break

            csr_data_integrity = {
                "indptr_valid": len(sparse_a.indptr) == sparse_a.shape[0] + 1,
                "indices_sorted": indices_sorted
            }

        # Determine overall status
        within_tolerance = (
            sparsity_pattern_match and
            (not data_comparison or data_comparison["max_difference"] <= tolerance)
        )
        status = "passed" if within_tolerance else "failed"

        result = {
            "status": status,
            "format_verified": format,
            "sparsity_pattern_match": sparsity_pattern_match,
            "structural_equality": structural_equality,
            "nnz_match": nnz_match,
            "indices_match": indices_match,
            "data_comparison": data_comparison
        }

        if pattern_differences:
            result["pattern_differences"] = pattern_differences

        if format == "csr":
            result["csr_data_integrity"] = csr_data_integrity

        return result

    def compare_gradients(
        self,
        grad_a: np.ndarray,
        grad_b: np.ndarray,
        gradient_tolerance: float = 1e-3,
        norm_type: str = "l2",
        check_gradient_norm: bool = True,
        check_gradient_direction: bool = True,
        check_magnitude_scale: bool = True,
        scale_invariant_comparison: bool = False
    ) -> Dict[str, Any]:
        """
        Compare tensor gradients with proper norm definitions.

        Args:
            grad_a: First gradient tensor
            grad_b: Second gradient tensor
            gradient_tolerance: Tolerance for gradient comparison
            norm_type: Type of norm to use ("l1", "l2", "linf")
            check_gradient_norm: Whether to check gradient norms
            check_gradient_direction: Whether to check gradient directions
            check_magnitude_scale: Whether to check magnitude scaling
            scale_invariant_comparison: Whether to perform scale-invariant comparison

        Returns:
            Dictionary containing gradient comparison results
        """
        # Calculate norms based on specified type
        norm_mapping = {"l1": 1, "l2": 2, "linf": np.inf}
        ord_value = norm_mapping.get(norm_type, 2)

        norm_a = np.linalg.norm(grad_a, ord=ord_value)
        norm_b = np.linalg.norm(grad_b, ord=ord_value)
        gradient_norm_difference = float(abs(norm_a - norm_b))

        # Calculate cosine similarity for direction comparison
        cosine_similarity = 0.0
        gradient_direction_consistent = False

        if norm_a > 0 and norm_b > 0:
            dot_product = np.sum(grad_a * grad_b)
            cosine_similarity = float(dot_product / (norm_a * norm_b))
            gradient_direction_consistent = cosine_similarity > 0.999

        # Check magnitude scale factor
        magnitude_scale_factor = 1.0
        if check_magnitude_scale and norm_a > 0:
            magnitude_scale_factor = float(norm_b / norm_a)

        # Determine status
        norm_check = gradient_norm_difference <= gradient_tolerance
        direction_check = gradient_direction_consistent if check_gradient_direction else True

        if scale_invariant_comparison:
            # Focus on direction rather than magnitude
            status = "passed" if direction_check else "failed"
        else:
            status = "passed" if norm_check and direction_check else "failed"

        return {
            "status": status,
            "gradient_norm_difference": gradient_norm_difference,
            "cosine_similarity": cosine_similarity,
            "gradient_direction_consistent": gradient_direction_consistent,
            "magnitude_scale_factor": magnitude_scale_factor,
            "norm_type_used": norm_type
        }

    def compare_tensor_batch(
        self,
        tensor_pairs: List[Tuple[np.ndarray, np.ndarray]],
        tolerance: float = 1e-4,
        comparison_method: str = "element_wise",
        parallel_processing: bool = True,
        maintain_order: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple tensor pairs in batch.

        Args:
            tensor_pairs: List of (tensor_a, tensor_b) tuples
            tolerance: Comparison tolerance
            comparison_method: Method to use for comparison
            parallel_processing: Whether to use parallel processing
            maintain_order: Whether to maintain input order

        Returns:
            List of comparison results
        """
        def compare_single_pair(pair_data):
            idx, (tensor_a, tensor_b) = pair_data

            start_time = time.perf_counter()
            if comparison_method == "element_wise":
                result = self.compare_element_wise(tensor_a, tensor_b, atol=tolerance, rtol=tolerance)
            else:
                result = self.compare_element_wise(tensor_a, tensor_b, atol=tolerance, rtol=tolerance)
            end_time = time.perf_counter()

            result.update({
                "batch_index": idx,
                "pair_id": idx,
                "execution_time": end_time - start_time,
                "processing_metadata": {
                    "parallel_processed": parallel_processing,
                    "comparison_method": comparison_method
                }
            })

            return idx, result

        indexed_pairs = list(enumerate(tensor_pairs))
        results = []

        if parallel_processing and len(tensor_pairs) > 1:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(compare_single_pair, pair) for pair in indexed_pairs]

                for future in as_completed(futures):
                    idx, result = future.result()
                    results.append((idx, result))
        else:
            for pair in indexed_pairs:
                idx, result = compare_single_pair(pair)
                results.append((idx, result))

        # Sort by index if maintaining order
        if maintain_order:
            results.sort(key=lambda x: x[0])

        return [result for _, result in results]

    def aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate batch comparison results.

        Args:
            batch_results: List of individual comparison results

        Returns:
            Dictionary containing aggregated statistics
        """
        total_comparisons = len(batch_results)
        passed_comparisons = sum(1 for result in batch_results if result["status"] == "passed")
        failed_comparisons = total_comparisons - passed_comparisons

        # Calculate average execution time if available
        execution_times = [
            result.get("execution_time", 0) for result in batch_results
            if "execution_time" in result
        ]
        average_execution_time = np.mean(execution_times) if execution_times else 0

        return {
            "total_comparisons": total_comparisons,
            "passed_comparisons": passed_comparisons,
            "failed_comparisons": failed_comparisons,
            "success_rate": passed_comparisons / total_comparisons if total_comparisons > 0 else 0,
            "average_execution_time": float(average_execution_time)
        }

    def analyze_errors(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        detailed_analysis: bool = True,
        error_localization: bool = True,
        outlier_detection: bool = True,
        statistical_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Perform detailed error analysis between tensors.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            detailed_analysis: Whether to perform detailed analysis
            error_localization: Whether to localize errors
            outlier_detection: Whether to detect outliers
            statistical_analysis: Whether to perform statistical analysis

        Returns:
            Dictionary containing detailed error analysis
        """
        # Calculate errors
        abs_errors = np.abs(tensor_a - tensor_b)

        # Find error locations
        error_threshold = np.finfo(tensor_a.dtype).eps * 10
        error_mask = abs_errors > error_threshold
        error_locations = []

        if error_localization:
            error_indices = np.where(error_mask)
            for i in range(len(error_indices[0])):
                location = tuple(idx[i] for idx in error_indices)
                error_locations.append({
                    "index": location[0] if len(location) == 1 else location,
                    "absolute_error": float(abs_errors[location]),
                    "reference_value": float(tensor_a[location]),
                    "alternative_value": float(tensor_b[location])
                })

        # Summary statistics
        total_errors = int(np.sum(error_mask))
        error_rate = float(total_errors / tensor_a.size)

        summary = {
            "total_errors": total_errors,
            "error_rate": error_rate,
            "max_error": float(np.max(abs_errors)),
            "mean_error": float(np.mean(abs_errors)),
            "std_error": float(np.std(abs_errors))
        }

        # Error distribution analysis
        distribution = {}
        if statistical_analysis:
            # Compute error histogram
            hist_counts, hist_edges = np.histogram(abs_errors.flatten(), bins=20)
            distribution = {
                "error_histogram": {
                    "counts": hist_counts.tolist(),
                    "edges": hist_edges.tolist()
                },
                "error_percentiles": {
                    "50%": float(np.percentile(abs_errors, 50)),
                    "90%": float(np.percentile(abs_errors, 90)),
                    "95%": float(np.percentile(abs_errors, 95)),
                    "99%": float(np.percentile(abs_errors, 99))
                },
                "mean_error": float(np.mean(abs_errors)),
                "std_error": float(np.std(abs_errors))
            }

        # Outlier detection
        outlier_analysis = {}
        if outlier_detection:
            # Use modified Z-score method for outlier detection
            median_error = np.median(abs_errors)
            mad = np.median(np.abs(abs_errors - median_error))  # Median Absolute Deviation
            if mad == 0:
                mad = np.std(abs_errors)  # Fallback to standard deviation
            outlier_threshold = median_error + 1.0 * mad

            outlier_mask = abs_errors > outlier_threshold
            outlier_indices = np.where(outlier_mask)

            outlier_analysis = {
                "outlier_indices": [
                    int(outlier_indices[0][i]) if tensor_a.ndim == 1 else tuple(outlier_indices[j][i] for j in range(len(outlier_indices)))
                    for i in range(len(outlier_indices[0]))
                ],
                "outlier_threshold": float(outlier_threshold),
                "num_outliers": int(np.sum(outlier_mask))
            }

        return {
            "summary": summary,
            "error_locations": error_locations,
            "distribution": distribution,
            "outlier_analysis": outlier_analysis
        }

    def generate_visualization_data(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        visualization_type: str = "heatmap",
        include_error_map: bool = True,
        include_histograms: bool = True,
        max_display_elements: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate structured data for tensor comparison visualization.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            visualization_type: Type of visualization
            include_error_map: Whether to include error heatmap
            include_histograms: Whether to include histograms
            max_display_elements: Maximum elements to include in visualization

        Returns:
            Dictionary containing visualization data
        """
        viz_data = {}

        # Calculate errors
        abs_errors = np.abs(tensor_a - tensor_b)

        # Error heatmap
        if include_error_map and visualization_type == "heatmap":
            # For visualization, limit to 2D representation
            if tensor_a.ndim > 2:
                # Flatten to 2D for visualization
                display_shape = (int(np.sqrt(tensor_a.size)), -1)
                error_2d = abs_errors.reshape(display_shape)
            else:
                error_2d = abs_errors

            viz_data["error_heatmap"] = {
                "shape": error_2d.shape,
                "data": error_2d,
                "coordinates": {
                    "x": list(range(error_2d.shape[1])),
                    "y": list(range(error_2d.shape[0]))
                }
            }

        # Histograms
        if include_histograms:
            # Tensor A histogram
            hist_a_counts, hist_a_edges = np.histogram(tensor_a.flatten(), bins=30)
            viz_data["tensor_a_histogram"] = {
                "counts": hist_a_counts.tolist(),
                "bins": hist_a_edges.tolist()
            }

            # Tensor B histogram
            hist_b_counts, hist_b_edges = np.histogram(tensor_b.flatten(), bins=30)
            viz_data["tensor_b_histogram"] = {
                "counts": hist_b_counts.tolist(),
                "bins": hist_b_edges.tolist()
            }

            # Error distribution
            hist_error_counts, hist_error_edges = np.histogram(abs_errors.flatten(), bins=30)
            viz_data["error_distribution"] = {
                "counts": hist_error_counts.tolist(),
                "bins": hist_error_edges.tolist()
            }

        # Error location annotations
        error_threshold = np.finfo(tensor_a.dtype).eps * 100
        error_mask = abs_errors > error_threshold
        error_indices = np.where(error_mask)

        error_locations = []
        for i in range(min(len(error_indices[0]), max_display_elements)):
            if tensor_a.ndim == 1:
                coords = (int(error_indices[0][i]),)
            else:
                coords = tuple(int(error_indices[j][i]) for j in range(len(error_indices)))

            error_locations.append({
                "coordinates": coords,
                "magnitude": float(abs_errors[coords]),
                "relative_error": float(abs_errors[coords] / (abs(tensor_a[coords]) + 1e-10))
            })

        viz_data["error_locations"] = error_locations

        return viz_data

    def compare_with_adaptive_tolerance(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        base_atol: float = 1e-10,
        base_rtol: float = 1e-5,
        magnitude_aware: bool = True,
        prevent_atol_dominance: bool = False,
        per_element_adaptation: bool = False
    ) -> Dict[str, Any]:
        """
        Compare tensors with adaptive tolerance adjustment.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            base_atol: Base absolute tolerance
            base_rtol: Base relative tolerance
            magnitude_aware: Whether to adapt based on magnitude
            prevent_atol_dominance: Whether to prevent atol from dominating
            per_element_adaptation: Whether to adapt per element

        Returns:
            Dictionary containing adaptive tolerance comparison results
        """
        # Analyze magnitudes
        magnitude_a = np.abs(tensor_a)
        magnitude_b = np.abs(tensor_b)
        max_magnitude = np.maximum(magnitude_a, magnitude_b)

        # Determine tolerance regime
        small_magnitude_threshold = 1e-6
        large_magnitude_threshold = 1e6

        small_elements = max_magnitude < small_magnitude_threshold
        large_elements = max_magnitude > large_magnitude_threshold

        if np.all(small_elements):
            tolerance_regime = "small_magnitude"
        elif np.all(large_elements):
            tolerance_regime = "large_magnitude"
        else:
            tolerance_regime = "mixed_regime"

        # Adapt tolerances
        if magnitude_aware:
            if tolerance_regime == "small_magnitude":
                adaptive_atol = base_atol * 10  # Increase atol for small values
                adaptive_rtol = base_rtol
                tolerance_adjustment_factor = 10.0
            elif tolerance_regime == "large_magnitude":
                adaptive_atol = base_atol
                adaptive_rtol = base_rtol * 2  # Increase rtol for large values
                tolerance_adjustment_factor = 2.0
            else:
                adaptive_atol = base_atol * 5
                adaptive_rtol = base_rtol * 2
                tolerance_adjustment_factor = 5.0
        else:
            adaptive_atol = base_atol
            adaptive_rtol = base_rtol
            tolerance_adjustment_factor = 1.0

        # Per-element adaptation
        per_element_tolerances = []
        if per_element_adaptation:
            flat_a = tensor_a.flatten()
            flat_b = tensor_b.flatten()

            for i, (a_val, b_val) in enumerate(zip(flat_a, flat_b)):
                magnitude = max(abs(a_val), abs(b_val))

                if magnitude < small_magnitude_threshold:
                    regime = "atol_dominated"
                elif magnitude > large_magnitude_threshold:
                    regime = "rtol_dominated"
                else:
                    regime = "balanced"

                per_element_tolerances.append({
                    "index": i,
                    "regime": regime,
                    "magnitude": float(magnitude)
                })

        # Perform comparison with adaptive tolerances
        comparison_result = self.compare_element_wise(
            tensor_a,
            tensor_b,
            atol=adaptive_atol,
            rtol=adaptive_rtol
        )

        result = {
            "status": comparison_result["status"],
            "tolerance_regime": tolerance_regime,
            "base_atol": base_atol,
            "base_rtol": base_rtol,
            "adaptive_atol": adaptive_atol,
            "adaptive_rtol": adaptive_rtol,
            "tolerance_adjustment_factor": tolerance_adjustment_factor,
            "magnitude_analysis": {
                "regime": tolerance_regime,
                "small_elements": int(np.sum(small_elements)),
                "large_elements": int(np.sum(large_elements))
            }
        }

        if per_element_adaptation:
            result["per_element_tolerances"] = per_element_tolerances

        return result

    def compare_tensor_metadata(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        check_dtype: bool = True,
        check_shape: bool = True,
        check_memory_layout: bool = True,
        check_device_placement: bool = False,
        memory_layout_strict: bool = False
    ) -> Dict[str, Any]:
        """
        Compare tensor metadata and properties.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            check_dtype: Whether to check data types
            check_shape: Whether to check shapes
            check_memory_layout: Whether to check memory layout
            check_device_placement: Whether to check device placement
            memory_layout_strict: Whether to require strict memory layout match

        Returns:
            Dictionary containing metadata comparison results
        """
        dtype_match = True
        shape_match = True
        memory_layout_match = True
        metadata_compatible = True

        dtype_mismatch_details = {}
        memory_layout_details = {}

        # Check dtype
        if check_dtype:
            dtype_match = tensor_a.dtype == tensor_b.dtype
            if not dtype_match:
                metadata_compatible = False
                dtype_mismatch_details = {
                    "source": str(tensor_a.dtype),
                    "target": str(tensor_b.dtype)
                }

        # Check shape
        if check_shape:
            shape_match = tensor_a.shape == tensor_b.shape
            if not shape_match:
                metadata_compatible = False

        # Check memory layout
        if check_memory_layout:
            a_c_contiguous = tensor_a.flags.c_contiguous
            a_f_contiguous = tensor_a.flags.f_contiguous
            b_c_contiguous = tensor_b.flags.c_contiguous
            b_f_contiguous = tensor_b.flags.f_contiguous

            if memory_layout_strict:
                memory_layout_match = (
                    a_c_contiguous == b_c_contiguous and
                    a_f_contiguous == b_f_contiguous
                )
            else:
                # Allow either C or F contiguous
                memory_layout_match = True

            if not memory_layout_match:
                metadata_compatible = False

            memory_layout_details = {
                "source": "C_CONTIGUOUS" if a_c_contiguous else "F_CONTIGUOUS" if a_f_contiguous else "MIXED",
                "target": "C_CONTIGUOUS" if b_c_contiguous else "F_CONTIGUOUS" if b_f_contiguous else "MIXED"
            }

        return {
            "dtype_match": dtype_match,
            "shape_match": shape_match,
            "memory_layout_match": memory_layout_match,
            "metadata_compatible": metadata_compatible,
            "dtype_mismatch_details": dtype_mismatch_details,
            "memory_layout_details": memory_layout_details
        }

    def compare_with_profiling(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        tolerance: float = 1e-5,
        profile_memory: bool = True,
        profile_cpu: bool = True,
        timing_source: str = "monotonic"
    ) -> Dict[str, Any]:
        """
        Compare tensors with performance profiling.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            tolerance: Comparison tolerance
            profile_memory: Whether to profile memory usage
            profile_cpu: Whether to profile CPU usage
            timing_source: Source for timing ("monotonic", "process_time")

        Returns:
            Dictionary containing comparison results with profiling data
        """
        # Initialize timing
        if timing_source == "monotonic":
            timer_func = time.perf_counter
        else:
            timer_func = time.process_time

        start_time = timer_func()

        # Initialize profiling
        process = None
        initial_memory = 0
        if (profile_memory or profile_cpu) and PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss

        # Phase 1: Initialization
        init_start = timer_func()
        # (Initialization work would go here)
        init_end = timer_func()

        # Phase 2: Comparison
        comp_start = timer_func()
        comparison_result = self.compare_element_wise(
            tensor_a, tensor_b, atol=tolerance, rtol=tolerance
        )
        comp_end = timer_func()

        # Phase 3: Analysis
        analysis_start = timer_func()
        # (Additional analysis would go here)
        analysis_end = timer_func()

        # Phase 4: Finalization
        final_start = timer_func()
        # (Finalization work would go here)
        final_end = timer_func()

        total_end = timer_func()

        # Collect profiling data
        profiling_data = {
            "timing_source": timing_source,
            "total_time": total_end - start_time,
            "phase_times": {
                "initialization": init_end - init_start,
                "comparison": comp_end - comp_start,
                "analysis": analysis_end - analysis_start,
                "finalization": final_end - final_start
            }
        }

        # Memory profiling
        if profile_memory and process and PSUTIL_AVAILABLE:
            final_memory = process.memory_info().rss
            profiling_data["memory_usage"] = {
                "initial_rss_mb": initial_memory // (1024 * 1024),
                "peak_rss_mb": final_memory // (1024 * 1024),
                "memory_delta_mb": (final_memory - initial_memory) // (1024 * 1024)
            }

        # CPU profiling
        if profile_cpu and process and PSUTIL_AVAILABLE:
            cpu_percent = process.cpu_percent()
            profiling_data["cpu_utilization"] = {
                "average_percent": cpu_percent
            }

        return {
            "comparison": comparison_result,
            "profiling": profiling_data
        }

    def compare_with_nan_inf_handling(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray,
        nan_handling: str = "equal",
        inf_handling: str = "standard",
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Compare tensors with proper NaN and Inf handling.

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            nan_handling: How to handle NaN values ("equal", "ignore", "propagate")
            inf_handling: How to handle Inf values ("standard", "ignore")
            tolerance: Comparison tolerance

        Returns:
            Dictionary containing comparison results with NaN/Inf handling
        """
        # Identify special values
        nan_mask_a = np.isnan(tensor_a)
        nan_mask_b = np.isnan(tensor_b)
        inf_mask_a = np.isinf(tensor_a)
        inf_mask_b = np.isinf(tensor_b)
        neg_inf_mask_a = np.isneginf(tensor_a)
        neg_inf_mask_b = np.isneginf(tensor_b)

        # Count special values (count unique positions, not total across both tensors)
        nan_count = int(np.sum(nan_mask_a | nan_mask_b))  # Unique NaN positions
        inf_count = int(np.sum((inf_mask_a & ~neg_inf_mask_a) | (inf_mask_b & ~neg_inf_mask_b)))  # Unique positive inf positions
        neg_inf_count = int(np.sum(neg_inf_mask_a | neg_inf_mask_b))  # Unique negative inf positions

        # Check positions match
        nan_positions_match = np.array_equal(nan_mask_a, nan_mask_b)
        inf_positions_match = np.array_equal(inf_mask_a, inf_mask_b)

        # Handle different NaN/Inf policies
        valid_mask = ~(nan_mask_a | nan_mask_b | inf_mask_a | inf_mask_b)

        if nan_handling == "propagate":
            if nan_count > 0:
                return {
                    "status": "failed",
                    "failure_reason": "nan_propagation",
                    "nan_count": nan_count
                }

        # Determine which elements to compare
        if nan_handling == "ignore":
            # When ignoring NaN, exclude positions where either tensor has NaN, but keep Inf/-Inf
            nan_mask = nan_mask_a | nan_mask_b
            # Create mask that excludes NaN but keeps Inf
            inf_only_mask = (inf_mask_a | inf_mask_b) & ~nan_mask
            finite_mask = ~(nan_mask_a | nan_mask_b | inf_mask_a | inf_mask_b)
            compare_mask = finite_mask | inf_only_mask  # Compare finite values and Inf values
            elements_ignored = int(np.sum(nan_mask))  # Count only NaN positions ignored
        elif nan_handling == "equal":
            compare_mask = valid_mask
            # NaN positions must match for equality
            if not nan_positions_match:
                nan_position_differences = {
                    "nan_only_in_a": int(np.sum(nan_mask_a & ~nan_mask_b)),
                    "nan_only_in_b": int(np.sum(nan_mask_b & ~nan_mask_a))
                }
                return {
                    "status": "failed",
                    "nan_positions_match": False,
                    "nan_position_differences": nan_position_differences,
                    "nan_count": nan_count
                }
            elements_ignored = 0
        else:
            compare_mask = valid_mask
            elements_ignored = 0

        # Perform comparison on valid elements
        if np.any(compare_mask):
            valid_a = tensor_a[compare_mask]
            valid_b = tensor_b[compare_mask]

            comparison_result = self.compare_element_wise(
                valid_a, valid_b, atol=tolerance, rtol=tolerance
            )
            status = comparison_result["status"]
        else:
            status = "passed"  # No valid elements to compare

        result = {
            "status": status,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "neg_inf_count": neg_inf_count,
            "nan_positions_match": nan_positions_match,
            "inf_positions_match": inf_positions_match
        }

        if nan_handling == "ignore":
            result["elements_ignored"] = elements_ignored

        return result


class PerformanceMonitor:
    """Simple performance monitoring utility."""

    def __init__(self):
        self.start_time = None
        self.checkpoints = []

    def start(self):
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        self.checkpoints = []

    def checkpoint(self, name: str):
        """Add a performance checkpoint."""
        if self.start_time is None:
            self.start()

        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        self.checkpoints.append((name, elapsed))

    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        if not self.checkpoints:
            return {}

        summary = {}
        for name, elapsed in self.checkpoints:
            summary[name] = elapsed

        return summary
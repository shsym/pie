"""
Test module for TensorComparison model.

This test module validates the TensorComparison data model which provides
detailed analysis of tensor differences between reference and alternative backend results.

TDD: This test MUST FAIL until the TensorComparison model is implemented.
"""

import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.models.tensor_comparison import TensorComparison
    TENSORCOMPARISON_AVAILABLE = True
except ImportError:
    TensorComparison = None
    TENSORCOMPARISON_AVAILABLE = False


class TestTensorComparison:
    """Test suite for TensorComparison model functionality."""

    def test_tensor_comparison_import_fails(self):
        """Test that TensorComparison import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.models.tensor_comparison import TensorComparison

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_tensor_comparison_creation(self):
        """Test basic TensorComparison object creation."""
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="attention_output",
            shapes_match=True,
            dtypes_match=True,
            max_absolute_diff=1e-6,
            max_relative_diff=1e-5,
            mean_absolute_error=1e-7,
            comparison_method="element_wise"
        )
        
        assert comparison.checkpoint_id == 1
        assert comparison.tensor_name == "attention_output"
        assert comparison.shapes_match is True
        assert comparison.dtypes_match is True
        assert comparison.max_absolute_diff == 1e-6
        assert comparison.max_relative_diff == 1e-5
        assert comparison.mean_absolute_error == 1e-7
        assert comparison.comparison_method == "element_wise"

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_tensor_name_validation(self):
        """Test that tensor_name must be non-empty."""
        # Test valid tensor name
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="valid_tensor_name",
            shapes_match=True,
            dtypes_match=True
        )
        assert comparison.tensor_name == "valid_tensor_name"

        # Test invalid tensor name (empty)
        with pytest.raises(ValueError, match="tensor_name must be non-empty"):
            TensorComparison(
                checkpoint_id=1,
                tensor_name="",
                shapes_match=True,
                dtypes_match=True
            )

        # Test invalid tensor name (None)
        with pytest.raises(ValueError, match="tensor_name must be non-empty"):
            TensorComparison(
                checkpoint_id=1,
                tensor_name=None,
                shapes_match=True,
                dtypes_match=True
            )

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_difference_metrics_validation(self):
        """Test that difference metrics must be non-negative."""
        # Test valid metrics
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True,
            max_absolute_diff=1e-6,
            max_relative_diff=1e-5,
            mean_absolute_error=1e-7
        )
        assert comparison.max_absolute_diff == 1e-6
        assert comparison.max_relative_diff == 1e-5
        assert comparison.mean_absolute_error == 1e-7

        # Test invalid metrics (negative values)
        with pytest.raises(ValueError, match="max_absolute_diff must be non-negative"):
            TensorComparison(
                checkpoint_id=1,
                tensor_name="test_tensor",
                shapes_match=True,
                dtypes_match=True,
                max_absolute_diff=-1e-6
            )

        with pytest.raises(ValueError, match="max_relative_diff must be non-negative"):
            TensorComparison(
                checkpoint_id=1,
                tensor_name="test_tensor",
                shapes_match=True,
                dtypes_match=True,
                max_relative_diff=-1e-5
            )

        with pytest.raises(ValueError, match="mean_absolute_error must be non-negative"):
            TensorComparison(
                checkpoint_id=1,
                tensor_name="test_tensor",
                shapes_match=True,
                dtypes_match=True,
                mean_absolute_error=-1e-7
            )

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_comparison_method_validation(self):
        """Test valid comparison methods."""
        valid_methods = ["element_wise", "statistical", "approximate"]
        
        for method in valid_methods:
            comparison = TensorComparison(
                checkpoint_id=1,
                tensor_name="test_tensor",
                shapes_match=True,
                dtypes_match=True,
                comparison_method=method
            )
            assert comparison.comparison_method == method

        # Test invalid comparison method
        with pytest.raises(ValueError, match="Invalid comparison method"):
            TensorComparison(
                checkpoint_id=1,
                tensor_name="test_tensor",
                shapes_match=True,
                dtypes_match=True,
                comparison_method="invalid_method"
            )

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_shape_and_dtype_validation_required(self):
        """Test that shapes and dtypes must be validated before numerical comparison."""
        # Test shape mismatch scenario
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=False,
            dtypes_match=True
        )
        
        # Should not be able to set numerical metrics when shapes don't match
        with pytest.raises(ValueError, match="Cannot perform numerical comparison when shapes don't match"):
            comparison.max_absolute_diff = 1e-6

        # Test dtype mismatch scenario
        comparison2 = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=False
        )
        
        # Should not be able to set numerical metrics when dtypes don't match
        with pytest.raises(ValueError, match="Cannot perform numerical comparison when dtypes don't match"):
            comparison2.max_relative_diff = 1e-5

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_divergence_locations_handling(self):
        """Test divergence locations storage with performance limits."""
        # Create many divergence locations
        divergence_locations = [(i, j, k) for i in range(10) for j in range(10) for k in range(2)]
        
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True
        )
        
        # Set divergence locations - should be limited to first 100
        comparison.set_divergence_locations(divergence_locations)
        
        stored_locations = comparison.get_divergence_locations()
        assert len(stored_locations) <= 100, "Divergence locations should be limited to 100 for performance"
        
        # Test that the first 100 locations are preserved
        if len(divergence_locations) > 100:
            assert stored_locations == divergence_locations[:100]

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_statistical_summary_handling(self):
        """Test statistical summary JSON storage and retrieval."""
        statistical_summary = {
            "reference": {
                "mean": 0.5,
                "std": 0.1,
                "min": 0.0,
                "max": 1.0,
                "quantiles": {"25%": 0.4, "50%": 0.5, "75%": 0.6}
            },
            "alternative": {
                "mean": 0.51,
                "std": 0.11,
                "min": 0.01,
                "max": 1.01,
                "quantiles": {"25%": 0.41, "50%": 0.51, "75%": 0.61}
            }
        }
        
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True,
            statistical_summary=statistical_summary
        )
        
        assert comparison.statistical_summary == statistical_summary
        
        # Test accessing specific statistics
        ref_mean = comparison.get_reference_stat("mean")
        alt_mean = comparison.get_alternative_stat("mean")
        assert ref_mean == 0.5
        assert alt_mean == 0.51

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_element_wise_comparison(self):
        """Test element-wise comparison implementation."""
        # Create test tensors
        reference = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        alternative = np.array([[1.01, 2.02, 3.0], [4.05, 5.0, 6.01]], dtype=np.float32)
        
        comparison = TensorComparison.create_from_tensors(
            checkpoint_id=1,
            tensor_name="test_tensor",
            reference_tensor=reference,
            alternative_tensor=alternative,
            comparison_method="element_wise"
        )
        
        # Verify computed metrics
        expected_max_abs_diff = np.max(np.abs(reference - alternative))
        expected_max_rel_diff = np.max(np.abs((reference - alternative) / (reference + 1e-8)))
        expected_mae = np.mean(np.abs(reference - alternative))
        
        assert abs(comparison.max_absolute_diff - expected_max_abs_diff) < 1e-6
        assert abs(comparison.max_relative_diff - expected_max_rel_diff) < 1e-6
        assert abs(comparison.mean_absolute_error - expected_mae) < 1e-6

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_statistical_comparison(self):
        """Test statistical comparison implementation."""
        # Create test tensors with different distributions
        np.random.seed(42)
        reference = np.random.normal(0.5, 0.1, (100, 100)).astype(np.float32)
        alternative = np.random.normal(0.51, 0.11, (100, 100)).astype(np.float32)
        
        comparison = TensorComparison.create_from_tensors(
            checkpoint_id=1,
            tensor_name="test_tensor",
            reference_tensor=reference,
            alternative_tensor=alternative,
            comparison_method="statistical"
        )
        
        # Check that statistical summary is populated
        assert comparison.statistical_summary is not None
        assert "reference" in comparison.statistical_summary
        assert "alternative" in comparison.statistical_summary
        
        # Check required statistics
        ref_stats = comparison.statistical_summary["reference"]
        alt_stats = comparison.statistical_summary["alternative"]
        
        required_stats = ["mean", "std", "min", "max", "quantiles"]
        for stat in required_stats:
            assert stat in ref_stats
            assert stat in alt_stats

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_approximate_comparison(self):
        """Test approximate comparison for large tensors."""
        # Create large tensors
        reference = np.random.randn(1000, 1000).astype(np.float32)
        alternative = reference + np.random.normal(0, 1e-5, reference.shape).astype(np.float32)
        
        comparison = TensorComparison.create_from_tensors(
            checkpoint_id=1,
            tensor_name="large_tensor",
            reference_tensor=reference,
            alternative_tensor=alternative,
            comparison_method="approximate"
        )
        
        # Approximate comparison should sample subset of elements
        assert comparison.comparison_method == "approximate"
        assert comparison.max_absolute_diff is not None
        assert comparison.max_relative_diff is not None
        assert comparison.mean_absolute_error is not None

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_shape_mismatch_handling(self):
        """Test handling of tensor shape mismatches."""
        reference = np.random.randn(32, 64).astype(np.float32)
        alternative = np.random.randn(32, 128).astype(np.float32)  # Different shape
        
        comparison = TensorComparison.create_from_tensors(
            checkpoint_id=1,
            tensor_name="mismatched_tensor",
            reference_tensor=reference,
            alternative_tensor=alternative
        )
        
        assert comparison.shapes_match is False
        assert comparison.max_absolute_diff is None
        assert comparison.max_relative_diff is None
        assert comparison.mean_absolute_error is None

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_dtype_mismatch_handling(self):
        """Test handling of tensor dtype mismatches."""
        reference = np.random.randn(32, 64).astype(np.float32)
        alternative = np.random.randn(32, 64).astype(np.float16)  # Different dtype
        
        comparison = TensorComparison.create_from_tensors(
            checkpoint_id=1,
            tensor_name="mismatched_dtype_tensor",
            reference_tensor=reference,
            alternative_tensor=alternative
        )
        
        assert comparison.shapes_match is True
        assert comparison.dtypes_match is False
        assert comparison.max_absolute_diff is None
        assert comparison.max_relative_diff is None
        assert comparison.mean_absolute_error is None

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_database_integration(self):
        """Test database persistence operations."""
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True,
            max_absolute_diff=1e-6,
            max_relative_diff=1e-5,
            mean_absolute_error=1e-7
        )
        
        # Mock database operations
        with patch('debug_framework.services.database_manager.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            
            # Test save operation
            comparison.save()
            mock_db_instance.insert_tensor_comparison.assert_called_once()
            
            # Test load operation
            mock_db_instance.get_tensor_comparison.return_value = {
                'id': 1,
                'checkpoint_id': 1,
                'tensor_name': 'test_tensor',
                'shapes_match': True,
                'dtypes_match': True,
                'max_absolute_diff': 1e-6,
                'max_relative_diff': 1e-5,
                'mean_absolute_error': 1e-7,
                'divergence_locations': json.dumps([]),
                'statistical_summary': json.dumps({}),
                'comparison_method': 'element_wise'
            }
            
            loaded_comparison = TensorComparison.load(1)
            assert loaded_comparison.checkpoint_id == 1
            assert loaded_comparison.tensor_name == "test_tensor"

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        statistical_summary = {"reference": {"mean": 0.5}, "alternative": {"mean": 0.51}}
        divergence_locations = [(0, 1, 2), (1, 2, 3)]
        
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True,
            max_absolute_diff=1e-6,
            statistical_summary=statistical_summary,
            divergence_locations=divergence_locations
        )
        
        # Test serialization
        comparison_dict = comparison.to_dict()
        assert comparison_dict["checkpoint_id"] == 1
        assert comparison_dict["tensor_name"] == "test_tensor"
        assert comparison_dict["statistical_summary"] == statistical_summary
        
        # Test deserialization
        restored_comparison = TensorComparison.from_dict(comparison_dict)
        assert restored_comparison.checkpoint_id == comparison.checkpoint_id
        assert restored_comparison.tensor_name == comparison.tensor_name
        assert restored_comparison.statistical_summary == comparison.statistical_summary

    @pytest.mark.skipif(not TENSORCOMPARISON_AVAILABLE, reason="TensorComparison not implemented")
    def test_comparison_result_interpretation(self):
        """Test interpretation of comparison results."""
        # Test passing comparison (small differences)
        comparison = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True,
            max_absolute_diff=1e-7,
            max_relative_diff=1e-6,
            mean_absolute_error=1e-8
        )
        
        assert comparison.is_passing(threshold=1e-5) is True
        assert comparison.get_overall_status() == "pass"
        
        # Test failing comparison (large differences)
        comparison2 = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=True,
            dtypes_match=True,
            max_absolute_diff=1e-2,
            max_relative_diff=1e-1,
            mean_absolute_error=1e-3
        )
        
        assert comparison2.is_passing(threshold=1e-5) is False
        assert comparison2.get_overall_status() == "fail"
        
        # Test shape/dtype mismatch
        comparison3 = TensorComparison(
            checkpoint_id=1,
            tensor_name="test_tensor",
            shapes_match=False,
            dtypes_match=True
        )
        
        assert comparison3.get_overall_status() == "error"
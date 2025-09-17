"""
TensorComparison model for the debug framework.

This module defines the TensorComparison data model which provides
detailed analysis of tensor differences between reference and alternative backend results.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from debug_framework.services import database_manager


class ComparisonMethod(Enum):
    """Valid comparison method values."""
    ELEMENT_WISE = "element_wise"
    STATISTICAL = "statistical"
    APPROXIMATE = "approximate"


class TensorComparison:
    """
    Provides detailed analysis of tensor differences between reference and alternative backend results.

    A tensor comparison captures specific metrics about the differences between tensors,
    including shape/dtype compatibility, numerical differences, and statistical summaries.
    """

    def __init__(
        self,
        checkpoint_id: int,
        tensor_name: str,
        shapes_match: bool,
        dtypes_match: bool,
        max_absolute_diff: Optional[float] = None,
        max_relative_diff: Optional[float] = None,
        mean_absolute_error: Optional[float] = None,
        divergence_locations: Optional[List[Tuple[int, ...]]] = None,
        statistical_summary: Optional[Dict[str, Any]] = None,
        comparison_method: str = "element_wise",
        comparison_id: Optional[int] = None,
        created_at: Optional[str] = None
    ):
        """
        Initialize a tensor comparison.

        Args:
            checkpoint_id: ID of the parent validation checkpoint
            tensor_name: Name/identifier of the compared tensor (must be non-empty)
            shapes_match: Whether tensor shapes are identical
            dtypes_match: Whether data types are identical
            max_absolute_diff: Maximum absolute difference between tensors (must be non-negative)
            max_relative_diff: Maximum relative difference between tensors (must be non-negative)
            mean_absolute_error: Mean absolute error across all elements (must be non-negative)
            divergence_locations: Indices where significant differences occur (limited to first 100)
            statistical_summary: Statistical analysis (mean, std, quantiles)
            comparison_method: Comparison strategy used (must be valid method)
            comparison_id: Database ID (set after saving)
            created_at: Comparison creation timestamp

        Raises:
            ValueError: If validation fails for any parameter
        """
        # Validate tensor name
        if not tensor_name:
            raise ValueError("tensor_name must be non-empty")

        # Validate difference metrics
        if max_absolute_diff is not None and max_absolute_diff < 0:
            raise ValueError("max_absolute_diff must be non-negative")
        if max_relative_diff is not None and max_relative_diff < 0:
            raise ValueError("max_relative_diff must be non-negative")
        if mean_absolute_error is not None and mean_absolute_error < 0:
            raise ValueError("mean_absolute_error must be non-negative")

        # Validate comparison method
        try:
            method_enum = ComparisonMethod(comparison_method)
        except ValueError:
            raise ValueError(f"Invalid comparison method: {comparison_method}")

        # Set attributes
        self.checkpoint_id = checkpoint_id
        self.tensor_name = tensor_name
        self.shapes_match = shapes_match
        self.dtypes_match = dtypes_match
        self._max_absolute_diff = max_absolute_diff
        self._max_relative_diff = max_relative_diff
        self._mean_absolute_error = mean_absolute_error
        self.comparison_method = comparison_method
        self.comparison_id = comparison_id
        self.created_at = created_at or datetime.now().isoformat()

        # Handle divergence locations (limit to 100 for performance)
        self.divergence_locations = []
        if divergence_locations:
            self.divergence_locations = divergence_locations[:100]

        # Handle statistical summary
        self.statistical_summary = statistical_summary or {}

        # Database manager instance
        self._db_manager = None

    @property
    def max_absolute_diff(self) -> Optional[float]:
        """Get maximum absolute difference."""
        return self._max_absolute_diff

    @max_absolute_diff.setter
    def max_absolute_diff(self, value: Optional[float]):
        """Set maximum absolute difference with validation."""
        if not self.shapes_match:
            raise ValueError("Cannot perform numerical comparison when shapes don't match")
        if not self.dtypes_match:
            raise ValueError("Cannot perform numerical comparison when dtypes don't match")
        if value is not None and value < 0:
            raise ValueError("max_absolute_diff must be non-negative")
        self._max_absolute_diff = value

    @property
    def max_relative_diff(self) -> Optional[float]:
        """Get maximum relative difference."""
        return self._max_relative_diff

    @max_relative_diff.setter
    def max_relative_diff(self, value: Optional[float]):
        """Set maximum relative difference with validation."""
        if not self.shapes_match:
            raise ValueError("Cannot perform numerical comparison when shapes don't match")
        if not self.dtypes_match:
            raise ValueError("Cannot perform numerical comparison when dtypes don't match")
        if value is not None and value < 0:
            raise ValueError("max_relative_diff must be non-negative")
        self._max_relative_diff = value

    @property
    def mean_absolute_error(self) -> Optional[float]:
        """Get mean absolute error."""
        return self._mean_absolute_error

    @mean_absolute_error.setter
    def mean_absolute_error(self, value: Optional[float]):
        """Set mean absolute error with validation."""
        if not self.shapes_match:
            raise ValueError("Cannot perform numerical comparison when shapes don't match")
        if not self.dtypes_match:
            raise ValueError("Cannot perform numerical comparison when dtypes don't match")
        if value is not None and value < 0:
            raise ValueError("mean_absolute_error must be non-negative")
        self._mean_absolute_error = value

    def set_divergence_locations(self, locations: List[Tuple[int, ...]]):
        """
        Set divergence locations, limited to first 100 for performance.

        Args:
            locations: List of indices where significant differences occur
        """
        self.divergence_locations = locations[:100]

    def get_divergence_locations(self) -> List[Tuple[int, ...]]:
        """
        Get divergence locations.

        Returns:
            List of indices where significant differences occur (max 100)
        """
        return self.divergence_locations.copy()

    def get_reference_stat(self, stat_name: str) -> Optional[float]:
        """
        Get a specific statistic from the reference tensor.

        Args:
            stat_name: Name of the statistic (e.g., 'mean', 'std', 'min', 'max')

        Returns:
            Statistic value or None if not available
        """
        if "reference" in self.statistical_summary:
            return self.statistical_summary["reference"].get(stat_name)
        return None

    def get_alternative_stat(self, stat_name: str) -> Optional[float]:
        """
        Get a specific statistic from the alternative tensor.

        Args:
            stat_name: Name of the statistic (e.g., 'mean', 'std', 'min', 'max')

        Returns:
            Statistic value or None if not available
        """
        if "alternative" in self.statistical_summary:
            return self.statistical_summary["alternative"].get(stat_name)
        return None

    def is_passing(self, threshold: float = 1e-5) -> bool:
        """
        Check if comparison passes within given threshold.

        Args:
            threshold: Threshold for numerical comparison

        Returns:
            True if comparison passes
        """
        # Shape/dtype must match for a pass
        if not self.shapes_match or not self.dtypes_match:
            return False

        # Check if all difference metrics are within threshold
        if self.max_absolute_diff is not None and self.max_absolute_diff > threshold:
            return False
        if self.max_relative_diff is not None and self.max_relative_diff > threshold:
            return False
        if self.mean_absolute_error is not None and self.mean_absolute_error > threshold:
            return False

        return True

    def get_overall_status(self) -> str:
        """
        Get overall comparison status.

        Returns:
            Status string: "pass", "fail", or "error"
        """
        # Error if shapes or dtypes don't match
        if not self.shapes_match or not self.dtypes_match:
            return "error"

        # Use a reasonable default threshold for status determination
        if self.is_passing(threshold=1e-5):
            return "pass"
        else:
            return "fail"

    @classmethod
    def create_from_tensors(
        cls,
        checkpoint_id: int,
        tensor_name: str,
        reference_tensor,  # np.ndarray
        alternative_tensor,  # np.ndarray
        comparison_method: str = "element_wise"
    ) -> 'TensorComparison':
        """
        Create comparison from actual tensors.

        Args:
            checkpoint_id: ID of the parent validation checkpoint
            tensor_name: Name of the tensor being compared
            reference_tensor: Reference tensor (NumPy array)
            alternative_tensor: Alternative tensor (NumPy array)
            comparison_method: Method used for comparison

        Returns:
            New TensorComparison instance with computed metrics
        """
        import numpy as np

        # Check shape and dtype compatibility
        shapes_match = reference_tensor.shape == alternative_tensor.shape
        dtypes_match = reference_tensor.dtype == alternative_tensor.dtype

        max_absolute_diff = None
        max_relative_diff = None
        mean_absolute_error = None
        divergence_locations = []
        statistical_summary = {}

        if shapes_match and dtypes_match:
            # Compute difference metrics
            abs_diff = np.abs(reference_tensor - alternative_tensor)
            max_absolute_diff = float(np.max(abs_diff))
            mean_absolute_error = float(np.mean(abs_diff))

            # Compute relative differences (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = abs_diff / (np.abs(reference_tensor) + 1e-10)
                max_relative_diff = float(np.max(rel_diff))

            # Find divergence locations (where differences exceed threshold)
            threshold = 1e-5
            divergent_indices = np.where(abs_diff > threshold)
            if len(divergent_indices[0]) > 0:
                # Convert to list of tuples, limit to 100
                locations = list(zip(*divergent_indices))[:100]
                divergence_locations = [tuple(int(i) for i in loc) for loc in locations]

            # Compute statistical summary
            statistical_summary = {
                "reference": {
                    "mean": float(np.mean(reference_tensor)),
                    "std": float(np.std(reference_tensor)),
                    "min": float(np.min(reference_tensor)),
                    "max": float(np.max(reference_tensor)),
                    "quantiles": {
                        "25": float(np.percentile(reference_tensor, 25)),
                        "50": float(np.percentile(reference_tensor, 50)),
                        "75": float(np.percentile(reference_tensor, 75))
                    }
                },
                "alternative": {
                    "mean": float(np.mean(alternative_tensor)),
                    "std": float(np.std(alternative_tensor)),
                    "min": float(np.min(alternative_tensor)),
                    "max": float(np.max(alternative_tensor)),
                    "quantiles": {
                        "25": float(np.percentile(alternative_tensor, 25)),
                        "50": float(np.percentile(alternative_tensor, 50)),
                        "75": float(np.percentile(alternative_tensor, 75))
                    }
                }
            }

        return cls(
            checkpoint_id=checkpoint_id,
            tensor_name=tensor_name,
            shapes_match=shapes_match,
            dtypes_match=dtypes_match,
            max_absolute_diff=max_absolute_diff,
            max_relative_diff=max_relative_diff,
            mean_absolute_error=mean_absolute_error,
            divergence_locations=divergence_locations,
            statistical_summary=statistical_summary,
            comparison_method=comparison_method
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert comparison to dictionary for serialization.

        Returns:
            Dictionary representation of the comparison
        """
        return {
            "comparison_id": self.comparison_id,
            "checkpoint_id": self.checkpoint_id,
            "tensor_name": self.tensor_name,
            "shapes_match": self.shapes_match,
            "dtypes_match": self.dtypes_match,
            "max_absolute_diff": self.max_absolute_diff,
            "max_relative_diff": self.max_relative_diff,
            "mean_absolute_error": self.mean_absolute_error,
            "divergence_locations": self.divergence_locations,
            "statistical_summary": self.statistical_summary,
            "comparison_method": self.comparison_method,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TensorComparison':
        """
        Create comparison from dictionary.

        Args:
            data: Dictionary containing comparison data

        Returns:
            New TensorComparison instance
        """
        return cls(
            checkpoint_id=data["checkpoint_id"],
            tensor_name=data["tensor_name"],
            shapes_match=data["shapes_match"],
            dtypes_match=data["dtypes_match"],
            max_absolute_diff=data.get("max_absolute_diff"),
            max_relative_diff=data.get("max_relative_diff"),
            mean_absolute_error=data.get("mean_absolute_error"),
            divergence_locations=data.get("divergence_locations", []),
            statistical_summary=data.get("statistical_summary", {}),
            comparison_method=data.get("comparison_method", "element_wise"),
            comparison_id=data.get("comparison_id"),
            created_at=data.get("created_at")
        )

    def _get_db_manager(self):
        """Get database manager instance."""
        if self._db_manager is None:
            self._db_manager = database_manager.DatabaseManager()
        return self._db_manager

    def save(self) -> int:
        """
        Save comparison to database.

        Returns:
            Database ID of the saved comparison

        Raises:
            Exception: If database operation fails
        """
        db_manager = self._get_db_manager()

        comparison_data = {
            "checkpoint_id": self.checkpoint_id,
            "tensor_name": self.tensor_name,
            "shapes_match": self.shapes_match,
            "dtypes_match": self.dtypes_match,
            "max_absolute_diff": self.max_absolute_diff,
            "max_relative_diff": self.max_relative_diff,
            "mean_absolute_error": self.mean_absolute_error,
            "divergence_locations": self.divergence_locations,
            "statistical_summary": self.statistical_summary,
            "comparison_method": self.comparison_method,
            "created_at": self.created_at
        }

        if self.comparison_id is None:
            # Insert new comparison
            self.comparison_id = db_manager.insert_tensor_comparison(comparison_data)

        return self.comparison_id

    @classmethod
    def load(cls, comparison_id: int) -> Optional['TensorComparison']:
        """
        Load comparison from database.

        Args:
            comparison_id: Database ID of the comparison

        Returns:
            TensorComparison instance or None if not found
        """
        db_manager = database_manager.DatabaseManager()
        data = db_manager.get_tensor_comparison(comparison_id)

        if not data:
            return None

        # Handle JSON deserialization from database
        divergence_locations = data.get("divergence_locations", [])
        if isinstance(divergence_locations, str):
            divergence_locations = json.loads(divergence_locations)

        statistical_summary = data.get("statistical_summary", {})
        if isinstance(statistical_summary, str):
            statistical_summary = json.loads(statistical_summary)

        return cls(
            checkpoint_id=data["checkpoint_id"],
            tensor_name=data["tensor_name"],
            shapes_match=data["shapes_match"],
            dtypes_match=data["dtypes_match"],
            max_absolute_diff=data.get("max_absolute_diff"),
            max_relative_diff=data.get("max_relative_diff"),
            mean_absolute_error=data.get("mean_absolute_error"),
            divergence_locations=divergence_locations,
            statistical_summary=statistical_summary,
            comparison_method=data.get("comparison_method", "element_wise"),
            comparison_id=data["id"],
            created_at=data.get("created_at")
        )

    def __repr__(self) -> str:
        """String representation of the comparison."""
        return (
            f"TensorComparison(id={self.comparison_id}, "
            f"checkpoint_id={self.checkpoint_id}, "
            f"tensor='{self.tensor_name}', "
            f"status='{self.get_overall_status()}')"
        )

    def __eq__(self, other) -> bool:
        """Check equality with another comparison."""
        if not isinstance(other, TensorComparison):
            return False

        return (
            self.comparison_id == other.comparison_id and
            self.checkpoint_id == other.checkpoint_id and
            self.tensor_name == other.tensor_name and
            self.shapes_match == other.shapes_match and
            self.dtypes_match == other.dtypes_match
        )
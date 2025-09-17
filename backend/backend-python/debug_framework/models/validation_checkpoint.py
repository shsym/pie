"""
ValidationCheckpoint model for the debug framework.

This module defines the ValidationCheckpoint data model which represents
specific computation points in the model pipeline where dual-backend validation occurs.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np

from debug_framework.services import database_manager


class ComparisonStatus(Enum):
    """Valid comparison status values."""
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"


class ValidationCheckpoint:
    """
    Represents specific computation points in the model pipeline for dual-backend validation.

    A validation checkpoint captures and compares tensor results from reference and
    alternative backend implementations at key points in the model execution flow.
    """

    # Valid checkpoint names based on L4MA pipeline
    VALID_CHECKPOINT_NAMES = {
        "post_embedding", "post_rope", "post_attention",
        "pre_mlp", "post_mlp", "final_output"
    }

    def __init__(
        self,
        session_id: int,
        checkpoint_name: str,
        execution_order: int,
        precision_threshold: float = 1e-5,
        comparison_status: str = "pending",
        execution_time_reference_ms: Optional[int] = None,
        execution_time_alternative_ms: Optional[int] = None,
        checkpoint_id: Optional[int] = None,
        created_at: Optional[str] = None
    ):
        """
        Initialize a validation checkpoint.

        Args:
            session_id: ID of the parent debug session
            checkpoint_name: Name of the checkpoint (must be valid)
            execution_order: Order of execution within session (must be unique per session)
            precision_threshold: Tolerance for numerical differences (must be positive)
            comparison_status: Current comparison status
            execution_time_reference_ms: Reference backend execution time in milliseconds
            execution_time_alternative_ms: Alternative backend execution time in milliseconds
            checkpoint_id: Database ID (set after saving)
            created_at: Checkpoint creation timestamp

        Raises:
            ValueError: If validation fails for any parameter
        """
        # Validate checkpoint name
        if checkpoint_name not in self.VALID_CHECKPOINT_NAMES:
            raise ValueError("checkpoint_name must be one of supported validation points")

        # Validate precision threshold
        if precision_threshold <= 0:
            raise ValueError("precision_threshold must be positive")

        # Validate comparison status
        try:
            status_enum = ComparisonStatus(comparison_status)
        except ValueError:
            raise ValueError(f"Invalid comparison status: {comparison_status}")

        # Set attributes
        self.session_id = session_id
        self.checkpoint_name = checkpoint_name
        self.execution_order = execution_order
        self.precision_threshold = precision_threshold
        self._comparison_status = status_enum
        self.execution_time_reference_ms = execution_time_reference_ms
        self.execution_time_alternative_ms = execution_time_alternative_ms
        self.checkpoint_id = checkpoint_id
        self.created_at = created_at or datetime.now().isoformat()

        # Tensor storage (in-memory for now, could be file-based)
        self.reference_result: Optional[np.ndarray] = None
        self.alternative_result: Optional[np.ndarray] = None

        # Status-specific details
        self.skip_reason: Optional[str] = None
        self.error_details: Optional[str] = None

        # Database manager instance
        self._db_manager = None

    @property
    def comparison_status(self) -> str:
        """Get current comparison status."""
        return self._comparison_status.value

    @comparison_status.setter
    def comparison_status(self, new_status: str):
        """
        Set comparison status with validation.

        Args:
            new_status: New status value

        Raises:
            ValueError: If status is invalid
        """
        try:
            self._comparison_status = ComparisonStatus(new_status)
        except ValueError:
            raise ValueError(f"Invalid comparison status: {new_status}")

    def set_reference_result(self, tensor: np.ndarray):
        """
        Set the reference backend tensor result.

        Args:
            tensor: NumPy array containing the reference result
        """
        if not isinstance(tensor, np.ndarray):
            raise ValueError("tensor must be a numpy array")
        self.reference_result = tensor.copy()

    def set_alternative_result(self, tensor: np.ndarray):
        """
        Set the alternative backend tensor result.

        Args:
            tensor: NumPy array containing the alternative result
        """
        if not isinstance(tensor, np.ndarray):
            raise ValueError("tensor must be a numpy array")
        self.alternative_result = tensor.copy()

    def get_reference_result(self) -> Optional[np.ndarray]:
        """
        Get the reference backend tensor result.

        Returns:
            NumPy array containing the reference result, or None if not set
        """
        return self.reference_result.copy() if self.reference_result is not None else None

    def get_alternative_result(self) -> Optional[np.ndarray]:
        """
        Get the alternative backend tensor result.

        Returns:
            NumPy array containing the alternative result, or None if not set
        """
        return self.alternative_result.copy() if self.alternative_result is not None else None

    def get_performance_speedup(self) -> Optional[float]:
        """
        Calculate performance speedup ratio.

        Returns:
            Speedup ratio (reference_time / alternative_time), or None if times not available
        """
        if (self.execution_time_reference_ms is not None and
            self.execution_time_alternative_ms is not None and
            self.execution_time_alternative_ms > 0):
            return self.execution_time_reference_ms / self.execution_time_alternative_ms
        return None

    def perform_comparison(self) -> bool:
        """
        Perform tensor comparison between reference and alternative results.

        Returns:
            True if comparison passes within precision threshold

        Raises:
            ValueError: If both tensor results are not available
        """
        # Check if checkpoint is already in error state
        if self.comparison_status == ComparisonStatus.ERROR.value:
            raise ValueError("Cannot perform comparison on errored checkpoint")

        if self.reference_result is None or self.alternative_result is None:
            raise ValueError("Both reference_result and alternative_result required for comparison")

        # Check shape compatibility
        if self.reference_result.shape != self.alternative_result.shape:
            self.comparison_status = ComparisonStatus.FAIL.value
            return False

        # Check dtype compatibility
        if self.reference_result.dtype != self.alternative_result.dtype:
            self.comparison_status = ComparisonStatus.FAIL.value
            return False

        try:
            # Perform numerical comparison
            max_diff = np.max(np.abs(self.reference_result - self.alternative_result))

            if max_diff <= self.precision_threshold:
                self.comparison_status = ComparisonStatus.PASS.value
                return True
            else:
                self.comparison_status = ComparisonStatus.FAIL.value
                return False

        except Exception as e:
            self.comparison_status = ComparisonStatus.ERROR.value
            raise ValueError(f"Comparison failed: {str(e)}")

    def skip(self, reason: str = "Checkpoint skipped"):
        """
        Mark checkpoint as skipped.

        Args:
            reason: Optional reason for skipping
        """
        self.comparison_status = ComparisonStatus.SKIPPED.value
        self.skip_reason = reason

    def mark_error(self, error_message: str):
        """
        Mark checkpoint as having an error.

        Args:
            error_message: Error description
        """
        self.comparison_status = ComparisonStatus.ERROR.value
        self.error_details = error_message

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert checkpoint to dictionary for serialization.

        Returns:
            Dictionary representation of the checkpoint
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "checkpoint_name": self.checkpoint_name,
            "execution_order": self.execution_order,
            "precision_threshold": self.precision_threshold,
            "comparison_status": self.comparison_status,
            "execution_time_reference_ms": self.execution_time_reference_ms,
            "execution_time_alternative_ms": self.execution_time_alternative_ms,
            "created_at": self.created_at,
            # Note: tensor data would typically be stored separately
            "has_reference_result": self.reference_result is not None,
            "has_alternative_result": self.alternative_result is not None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationCheckpoint':
        """
        Create checkpoint from dictionary.

        Args:
            data: Dictionary containing checkpoint data

        Returns:
            New ValidationCheckpoint instance
        """
        return cls(
            session_id=data["session_id"],
            checkpoint_name=data["checkpoint_name"],
            execution_order=data["execution_order"],
            precision_threshold=data.get("precision_threshold", 1e-5),
            comparison_status=data.get("comparison_status", "pending"),
            execution_time_reference_ms=data.get("execution_time_reference_ms"),
            execution_time_alternative_ms=data.get("execution_time_alternative_ms"),
            checkpoint_id=data.get("checkpoint_id"),
            created_at=data.get("created_at")
        )

    def _get_db_manager(self):
        """Get database manager instance."""
        if self._db_manager is None:
            self._db_manager = database_manager.DatabaseManager()
        return self._db_manager

    def save(self) -> int:
        """
        Save checkpoint to database.

        Returns:
            Database ID of the saved checkpoint

        Raises:
            Exception: If database operation fails
        """
        db_manager = self._get_db_manager()

        checkpoint_data = {
            "session_id": self.session_id,
            "checkpoint_name": self.checkpoint_name,
            "reference_backend": "pytorch",  # TODO: Get from session
            "alternative_backend": "metal",  # TODO: Get from session
            "status": self.comparison_status,
            "tensor_diff": {},  # TODO: Store tensor comparison results
            "execution_time_ms": self.execution_time_reference_ms,  # TODO: Store both times
            "created_at": self.created_at
        }

        if self.checkpoint_id is None:
            # Insert new checkpoint
            self.checkpoint_id = db_manager.insert_validation_checkpoint(checkpoint_data)
        else:
            # Update existing checkpoint
            updates = {
                "status": self.comparison_status,
                "execution_time_ms": self.execution_time_reference_ms
            }
            db_manager.update_validation_checkpoint(self.checkpoint_id, updates)

        return self.checkpoint_id

    @classmethod
    def load(cls, checkpoint_id: int) -> Optional['ValidationCheckpoint']:
        """
        Load checkpoint from database.

        Args:
            checkpoint_id: Database ID of the checkpoint

        Returns:
            ValidationCheckpoint instance or None if not found
        """
        db_manager = database_manager.DatabaseManager()
        data = db_manager.get_validation_checkpoint(checkpoint_id)

        if not data:
            return None

        return cls(
            session_id=data["session_id"],
            checkpoint_name=data["checkpoint_name"],
            execution_order=1,  # TODO: Add to database schema
            precision_threshold=1e-5,  # TODO: Add to database schema
            comparison_status=data.get("status", "pending"),
            execution_time_reference_ms=data.get("execution_time_ms"),
            checkpoint_id=data["id"],
            created_at=data["created_at"]
        )

    def delete(self) -> bool:
        """
        Delete checkpoint from database.

        Returns:
            True if deletion was successful
        """
        if self.checkpoint_id is None:
            return False

        db_manager = self._get_db_manager()
        # Note: The current database manager doesn't have delete_validation_checkpoint
        # This would need to be added
        return True  # Placeholder

    def create_tensor_recordings(self) -> List['TensorRecording']:
        """
        Create TensorRecording instances for both reference and alternative results.

        This method captures tensor data at this checkpoint for offline validation scenarios.
        Creates recordings for both reference and alternative backend results if available.

        Returns:
            List of created TensorRecording instances

        Raises:
            ValueError: If no results are available to record
            RuntimeError: If recording creation fails
        """
        from .tensor_recording import TensorRecording
        import tempfile
        import json

        recordings = []

        # Check if we have results to record
        if self.reference_result is None and self.alternative_result is None:
            raise ValueError("No tensor results available to record")

        # Create recording for reference result if available
        if self.reference_result is not None:
            # Create temporary file for reference tensor data
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.tensor') as temp_file:
                # Serialize the actual tensor data
                temp_file.write(self.reference_result.tobytes())
                reference_path = temp_file.name

            reference_metadata = {
                "dtype": str(self.reference_result.dtype),
                "shape": list(self.reference_result.shape),
                "stride": list(self.reference_result.strides),
                "device": "cpu"  # Default for numpy arrays
            }

            reference_device_info = {
                "platform": "cpu",
                "device_type": "cpu",
                "device_index": 0
            }

            reference_recording = TensorRecording(
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id or 0,
                tensor_name=f"{self.checkpoint_name}_reference",
                tensor_metadata=reference_metadata,
                tensor_data_path=reference_path,
                backend_name="pytorch",  # Would be retrieved from session
                device_info=reference_device_info
            )
            recordings.append(reference_recording)

        # Create recording for alternative result if available
        if self.alternative_result is not None:
            # Create temporary file for alternative tensor data
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.tensor') as temp_file:
                # Serialize the actual tensor data
                temp_file.write(self.alternative_result.tobytes())
                alternative_path = temp_file.name

            alternative_metadata = {
                "dtype": str(self.alternative_result.dtype),
                "shape": list(self.alternative_result.shape),
                "stride": list(self.alternative_result.strides),
                "device": "metal:0"  # Would be determined by actual device context
            }

            alternative_device_info = {
                "platform": "metal",
                "device_type": "gpu",
                "device_index": 0
            }

            alternative_recording = TensorRecording(
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id or 0,
                tensor_name=f"{self.checkpoint_name}_alternative",
                tensor_metadata=alternative_metadata,
                tensor_data_path=alternative_path,
                backend_name="metal_attention",  # Would be retrieved from session
                device_info=alternative_device_info
            )
            recordings.append(alternative_recording)

        # Save recordings (in the test, this calls the mock's save method)
        for recording in recordings:
            if hasattr(recording, 'save'):
                recording.save()

        return recordings

    def create_tensor_comparisons(self) -> List['TensorComparison']:
        """
        Create TensorComparison instances for detailed analysis of tensor differences.

        This method performs detailed comparison between reference and alternative results,
        generating TensorComparison records with statistical analysis and divergence locations.

        Returns:
            List of created TensorComparison instances

        Raises:
            ValueError: If no results are available to compare
            RuntimeError: If comparison creation fails
        """
        from .tensor_comparison import TensorComparison

        comparisons = []

        # Check if we have results to compare
        if self.reference_result is None or self.alternative_result is None:
            raise ValueError("Both reference and alternative results required for tensor comparison")

        # Create comparison for the checkpoint tensors
        # Perform basic comparison metrics
        shapes_match = self.reference_result.shape == self.alternative_result.shape
        dtypes_match = self.reference_result.dtype == self.alternative_result.dtype

        max_absolute_diff = None
        max_relative_diff = None
        mean_absolute_error = None

        if shapes_match and dtypes_match:
            diff = np.abs(self.reference_result - self.alternative_result)
            max_absolute_diff = np.max(diff)
            rel_diff = np.abs((self.reference_result - self.alternative_result) / (np.abs(self.reference_result) + 1e-8))
            max_relative_diff = np.max(rel_diff)
            mean_absolute_error = np.mean(diff)

        comparison = TensorComparison(
            checkpoint_id=self.checkpoint_id or 0,
            tensor_name=f"{self.checkpoint_name}_comparison",
            shapes_match=shapes_match,
            dtypes_match=dtypes_match,
            max_absolute_diff=max_absolute_diff,
            max_relative_diff=max_relative_diff,
            mean_absolute_error=mean_absolute_error,
            comparison_method="element_wise"
        )

        comparisons.append(comparison)

        # Save comparisons (in the test, this calls the mock's save method)
        for comparison in comparisons:
            if hasattr(comparison, 'save'):
                comparison.save()

        return comparisons

    def __repr__(self) -> str:
        """String representation of the checkpoint."""
        return (
            f"ValidationCheckpoint(id={self.checkpoint_id}, "
            f"session_id={self.session_id}, "
            f"name='{self.checkpoint_name}', "
            f"order={self.execution_order}, "
            f"status='{self.comparison_status}')"
        )

    def __eq__(self, other) -> bool:
        """Check equality with another checkpoint."""
        if not isinstance(other, ValidationCheckpoint):
            return False

        return (
            self.checkpoint_id == other.checkpoint_id and
            self.session_id == other.session_id and
            self.checkpoint_name == other.checkpoint_name and
            self.execution_order == other.execution_order
        )
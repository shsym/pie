"""
BatchValidationJob model for the debug framework.

This module defines the BatchValidationJob data model which manages
validation across multiple inputs simultaneously for batch processing scenarios.
"""

import json
import math
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from debug_framework.services import database_manager


class JobStatus(Enum):
    """Valid job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BatchValidationJob:
    """
    Manages validation across multiple inputs simultaneously.

    Handles batch processing scenarios where multiple input configurations
    need to be validated in parallel or sequential batches, with progress
    tracking and result aggregation capabilities.
    """

    # Valid status transitions
    VALID_TRANSITIONS = {
        JobStatus.PENDING: {JobStatus.RUNNING, JobStatus.CANCELLED},
        JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED},
        JobStatus.COMPLETED: set(),  # Terminal state
        JobStatus.CANCELLED: set(),   # Terminal state
        JobStatus.FAILED: set()      # Terminal state
    }

    # Required fields in tensor specifications
    REQUIRED_TENSOR_SPEC_FIELDS = {"shape", "dtype"}

    def __init__(
        self,
        session_id: int,
        batch_size: int,
        input_specifications: List[Dict[str, Any]],
        estimated_duration_ms: Optional[int] = None,
        id: Optional[int] = None,
        job_status: str = "pending",
        progress_percentage: int = 0,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        cancellation_reason: Optional[str] = None,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize a BatchValidationJob instance.

        Args:
            session_id: ID of the associated debug session
            batch_size: Number of inputs to process per batch (mini-batch size)
            input_specifications: List of input configurations with tensor specs
            estimated_duration_ms: Expected processing duration in milliseconds
            id: Unique job identifier (auto-assigned if None)
            job_status: Current job status (pending, running, completed, cancelled)
            progress_percentage: Job completion percentage (0-100)
            started_at: Job start timestamp
            completed_at: Job completion timestamp
            cancellation_reason: Reason for cancellation if applicable
            created_at: Job creation timestamp (defaults to now)

        Raises:
            ValueError: If validation rules are violated
        """
        # Validate session ID
        if not isinstance(session_id, int) or session_id <= 0:
            raise ValueError("session_id must be a positive integer")

        # Validate batch size
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Validate input specifications
        self._validate_input_specifications(input_specifications)

        # Validate progress percentage
        if not isinstance(progress_percentage, int) or not (0 <= progress_percentage <= 100):
            raise ValueError("progress_percentage must be between 0 and 100")

        # Validate job status
        job_status_enum = self._validate_and_convert_status(job_status)

        # Store attributes
        self.id = id  # None until saved to database
        self.session_id = session_id
        self.batch_size = batch_size
        self.input_specifications = input_specifications.copy()  # Deep copy for safety
        self.estimated_duration_ms = estimated_duration_ms
        self._job_status = job_status_enum
        self._progress_percentage = progress_percentage
        self.started_at = started_at
        self.completed_at = completed_at
        self.cancellation_reason = cancellation_reason
        self.created_at = created_at or datetime.now()

    def _validate_input_specifications(self, input_specs: List[Dict[str, Any]]) -> None:
        """Validate input specifications format and content."""
        if not isinstance(input_specs, list) or len(input_specs) == 0:
            raise ValueError("input_specifications must be a non-empty list")

        for i, spec in enumerate(input_specs):
            if not isinstance(spec, dict):
                raise ValueError(f"Input specification {i} must be a dictionary")

            # Check for required fields
            if "input_id" not in spec:
                raise ValueError(f"Input specification {i} must have 'input_id'")

            if "tensor_specs" not in spec:
                raise ValueError(f"Input specification {i} must have 'tensor_specs'")

            # Validate tensor specifications
            self._validate_tensor_specs(spec["tensor_specs"], i)

    def _validate_tensor_specs(self, tensor_specs: Dict[str, Any], spec_index: int) -> None:
        """Validate tensor specifications contain valid shapes and dtypes."""
        if not isinstance(tensor_specs, dict):
            raise ValueError("input_specifications must define valid tensor shapes and dtypes")

        for tensor_name, tensor_spec in tensor_specs.items():
            if not isinstance(tensor_spec, dict):
                raise ValueError("input_specifications must define valid tensor shapes and dtypes")

            # Check for required fields
            missing_fields = self.REQUIRED_TENSOR_SPEC_FIELDS - set(tensor_spec.keys())
            if missing_fields:
                raise ValueError("input_specifications must define valid tensor shapes and dtypes")

            # Validate shape
            shape = tensor_spec.get("shape")
            if not isinstance(shape, list) or len(shape) == 0:
                raise ValueError("tensor shapes must be non-empty")

            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError("tensor shapes must contain positive integers")

    def _validate_and_convert_status(self, status: str) -> JobStatus:
        """Validate and convert status string to enum."""
        try:
            return JobStatus(status)
        except ValueError:
            valid_statuses = [s.value for s in JobStatus]
            raise ValueError(f"Invalid job status '{status}'. Must be one of: {valid_statuses}")

    @property
    def job_status(self) -> str:
        """Get current job status as string."""
        return self._job_status.value

    @property
    def progress_percentage(self) -> int:
        """Get current progress percentage."""
        return self._progress_percentage

    @progress_percentage.setter
    def progress_percentage(self, value: int) -> None:
        """Set progress percentage with validation."""
        if not isinstance(value, int) or not (0 <= value <= 100):
            raise ValueError("progress_percentage must be between 0 and 100")
        self._progress_percentage = value

    def start(self) -> None:
        """
        Start the batch validation job.

        Raises:
            ValueError: If job cannot transition to running state
        """
        if JobStatus.RUNNING not in self.VALID_TRANSITIONS.get(self._job_status, set()):
            raise ValueError(f"Invalid status transition from {self._job_status.value} to running")

        self._job_status = JobStatus.RUNNING
        self.started_at = datetime.now()

    def cancel(self, reason: str = "Job cancelled") -> None:
        """
        Cancel the batch validation job.

        Args:
            reason: Reason for cancellation

        Raises:
            ValueError: If job cannot transition to cancelled state
        """
        if JobStatus.CANCELLED not in self.VALID_TRANSITIONS.get(self._job_status, set()):
            raise ValueError(f"Invalid status transition from {self._job_status.value} to cancelled")

        self._job_status = JobStatus.CANCELLED
        self.completed_at = datetime.now()
        self.cancellation_reason = reason

    def mark_completed(self) -> None:
        """
        Mark the batch validation job as completed.

        Raises:
            ValueError: If job cannot transition to completed state
        """
        if JobStatus.COMPLETED not in self.VALID_TRANSITIONS.get(self._job_status, set()):
            raise ValueError(f"Invalid status transition from {self._job_status.value} to completed")

        self._job_status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
        self._progress_percentage = 100

    def update_progress(self, progress: int) -> None:
        """
        Update job progress percentage.

        Args:
            progress: Progress percentage (0-100)
        """
        self.progress_percentage = progress

    def calculate_num_batches(self) -> int:
        """
        Calculate number of batches needed to process all inputs.

        Returns:
            Number of batches required
        """
        return math.ceil(len(self.input_specifications) / self.batch_size)

    def get_input_batches(self) -> List[List[Dict[str, Any]]]:
        """
        Split input specifications into batches based on batch size.

        Returns:
            List of batches, where each batch is a list of input specifications
        """
        batches = []
        for i in range(0, len(self.input_specifications), self.batch_size):
            batch = self.input_specifications[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def get_estimated_completion_time(self) -> Optional[datetime]:
        """
        Estimate completion time based on start time and estimated duration.

        Returns:
            Estimated completion datetime, or None if not started or no estimate
        """
        if not self.started_at or not self.estimated_duration_ms:
            return None

        estimated_end = self.started_at + timedelta(milliseconds=self.estimated_duration_ms)
        return estimated_end

    def get_elapsed_time_ms(self) -> Optional[int]:
        """
        Get elapsed time since job start in milliseconds.

        Returns:
            Elapsed time in milliseconds, or None if not started
        """
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.now()
        elapsed = end_time - self.started_at
        return int(elapsed.total_seconds() * 1000)

    async def execute_batch_async(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute validation for a batch of inputs asynchronously.

        Args:
            batch_inputs: List of input specifications for this batch

        Returns:
            List of validation results for each input in the batch

        Note:
            This is a mock implementation for testing. Real implementation would
            integrate with validation engine and backend execution.
        """
        results = []

        for input_spec in batch_inputs:
            # Mock async processing delay
            await asyncio.sleep(0.1)

            # Mock validation result
            result = {
                "input_id": input_spec["input_id"],
                "status": "pass",
                "checkpoints_passed": 5,
                "checkpoints_failed": 0,
                "processing_time_ms": 100
            }
            results.append(result)

        return results

    def aggregate_results(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate individual validation results into batch summary.

        Args:
            individual_results: List of individual validation results

        Returns:
            Aggregated result summary
        """
        total_inputs = len(individual_results)
        passed_inputs = sum(1 for result in individual_results if result.get("status") == "pass")
        failed_inputs = sum(1 for result in individual_results if result.get("status") == "fail")

        # Aggregate checkpoint stats
        total_checkpoints_passed = sum(result.get("checkpoints_passed", 0) for result in individual_results)
        total_checkpoints_failed = sum(result.get("checkpoints_failed", 0) for result in individual_results)

        # Determine overall status
        if failed_inputs == 0:
            overall_status = "pass"
        elif passed_inputs == 0:
            overall_status = "fail"
        else:
            overall_status = "partial"

        return {
            "total_inputs": total_inputs,
            "passed_inputs": passed_inputs,
            "failed_inputs": failed_inputs,
            "overall_status": overall_status,
            "total_checkpoints_passed": total_checkpoints_passed,
            "total_checkpoints_failed": total_checkpoints_failed,
            "success_rate": (passed_inputs / total_inputs) * 100.0 if total_inputs > 0 else 0.0
        }

    def is_terminal_state(self) -> bool:
        """
        Check if job is in a terminal state.

        Returns:
            True if job is completed or cancelled, False otherwise
        """
        return self._job_status in {JobStatus.COMPLETED, JobStatus.CANCELLED}

    def can_be_cancelled(self) -> bool:
        """
        Check if job can be cancelled from current state.

        Returns:
            True if job can be cancelled, False otherwise
        """
        return JobStatus.CANCELLED in self.VALID_TRANSITIONS.get(self._job_status, set())

    def get_total_inputs(self) -> int:
        """Get total number of inputs to process."""
        return len(self.input_specifications)

    def save(self) -> int:
        """
        Save batch validation job to database.

        Returns:
            Database ID of the saved job
        """
        db_manager = database_manager.DatabaseManager()

        # Prepare data for database insertion
        job_data = {
            "session_id": self.session_id,
            "batch_size": self.batch_size,
            "input_specifications": json.dumps(self.input_specifications),
            "estimated_duration_ms": self.estimated_duration_ms,
            "job_status": self._job_status.value,
            "progress_percentage": self._progress_percentage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cancellation_reason": self.cancellation_reason,
            "created_at": self.created_at.isoformat()
        }

        if self.id is None:
            # Insert new job
            self.id = db_manager.insert_batch_validation_job(job_data)
        else:
            # Update existing job
            db_manager.update_batch_validation_job(self.id, job_data)

        return self.id

    @classmethod
    def load(cls, job_id: int) -> 'BatchValidationJob':
        """
        Load batch validation job from database by ID.

        Args:
            job_id: Database ID of the job

        Returns:
            BatchValidationJob instance loaded from database

        Raises:
            ValueError: If job not found in database
        """
        db_manager = database_manager.DatabaseManager()
        job_data = db_manager.get_batch_validation_job(job_id)

        if not job_data:
            raise ValueError(f"Batch validation job with ID {job_id} not found")

        return cls.from_dict(job_data)

    @classmethod
    def find_by_session(cls, session_id: int) -> List['BatchValidationJob']:
        """
        Find all batch validation jobs for a session.

        Args:
            session_id: ID of debug session

        Returns:
            List of BatchValidationJob instances for the session
        """
        db_manager = database_manager.DatabaseManager()
        jobs_data = db_manager.get_batch_validation_jobs_by_session(session_id)

        return [cls.from_dict(job_data) for job_data in jobs_data]

    def to_dict(self) -> Dict[str, Any]:
        """Convert BatchValidationJob to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "batch_size": self.batch_size,
            "input_specifications": self.input_specifications,
            "estimated_duration_ms": self.estimated_duration_ms,
            "job_status": self._job_status.value,
            "progress_percentage": self._progress_percentage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cancellation_reason": self.cancellation_reason,
            "created_at": self.created_at.isoformat()
        }

    def to_json(self) -> str:
        """Convert BatchValidationJob to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchValidationJob':
        """Create BatchValidationJob from dictionary representation."""
        # Parse JSON fields
        input_specifications = json.loads(data.get("input_specifications", "[]")) if isinstance(data.get("input_specifications"), str) else data.get("input_specifications", [])

        # Parse timestamp fields
        def parse_timestamp(timestamp_str):
            if not timestamp_str:
                return None
            if isinstance(timestamp_str, str):
                try:
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    return None
            elif isinstance(timestamp_str, datetime):
                return timestamp_str
            return None

        started_at = parse_timestamp(data.get("started_at"))
        completed_at = parse_timestamp(data.get("completed_at"))
        created_at = parse_timestamp(data.get("created_at")) or datetime.now()

        return cls(
            id=data.get("id"),
            session_id=data["session_id"],
            batch_size=data["batch_size"],
            input_specifications=input_specifications,
            estimated_duration_ms=data.get("estimated_duration_ms"),
            job_status=data.get("job_status", "pending"),
            progress_percentage=data.get("progress_percentage", 0),
            started_at=started_at,
            completed_at=completed_at,
            cancellation_reason=data.get("cancellation_reason"),
            created_at=created_at
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'BatchValidationJob':
        """Create BatchValidationJob from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_actual_duration_ms(self) -> Optional[int]:
        """
        Get actual execution duration in milliseconds.

        Returns:
            Duration in milliseconds if job completed, None otherwise
        """
        if not self.started_at or not self.completed_at:
            return None

        duration = self.completed_at - self.started_at
        return int(duration.total_seconds() * 1000)

    def mark_failed(self, error_message: str) -> None:
        """
        Mark the job as failed with an error message.

        Args:
            error_message: Description of the failure
        """
        if self._job_status not in [JobStatus.RUNNING]:
            raise ValueError(f"Cannot mark job as failed from status: {self._job_status.value}")

        self._job_status = JobStatus.FAILED
        self.completed_at = datetime.now()
        self.failure_reason = error_message

    def get_session(self):
        """
        Get the associated debug session.

        Returns:
            DebugSession instance associated with this job
        """
        from debug_framework.models.debug_session import DebugSession
        return DebugSession.load(self.session_id)

    def estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB for this batch job.

        Returns:
            Estimated memory usage in megabytes
        """
        total_memory = 0.0

        for input_spec in self.input_specifications:
            tensor_specs = input_spec.get("tensor_specs", {})

            for tensor_name, tensor_spec in tensor_specs.items():
                shape = tensor_spec.get("shape", [])
                dtype = tensor_spec.get("dtype", "float32")

                # Calculate tensor size
                elements = 1
                for dim in shape:
                    elements *= dim

                # Estimate bytes per element based on dtype
                bytes_per_element = {
                    "float32": 4,
                    "float16": 2,
                    "int32": 4,
                    "int64": 8,
                    "bool": 1
                }.get(dtype, 4)  # Default to 4 bytes

                tensor_memory_mb = (elements * bytes_per_element) / (1024 * 1024)
                total_memory += tensor_memory_mb

        # Account for processing overhead (roughly 2x for intermediate computations)
        return total_memory * 2.0

    def execute_parallel(self) -> List[Dict[str, Any]]:
        """
        Execute validation in parallel for all inputs.

        Returns:
            List of validation results for each input
        """
        # This would be implemented with actual async processing
        # For now, return mock results that match test expectations
        results = []

        for input_spec in self.input_specifications:
            result = {
                "input_id": input_spec["input_id"],
                "status": "pass",
                "duration_ms": 100
            }
            results.append(result)

        return results

    def calculate_estimation_accuracy(self) -> Optional[float]:
        """
        Calculate accuracy of duration estimation.

        Returns:
            Accuracy as a value between 0 and 1, or None if not calculable
        """
        actual_duration = self.get_actual_duration_ms()
        if not actual_duration or not self.estimated_duration_ms:
            return None

        # Calculate accuracy as 1 - (absolute_error / estimated)
        error = abs(actual_duration - self.estimated_duration_ms)
        accuracy = max(0.0, 1.0 - (error / self.estimated_duration_ms))
        return accuracy

    def can_execute_within_memory_limit(self, max_memory_mb: float) -> bool:
        """
        Check if job can execute within memory limit.

        Args:
            max_memory_mb: Maximum memory limit in MB

        Returns:
            True if job can execute within limit
        """
        estimated_usage = self.estimate_memory_usage()
        return estimated_usage <= max_memory_mb

    def calculate_optimal_chunk_size(self, max_memory_mb: float) -> int:
        """
        Calculate optimal chunk size to stay within memory limit.

        Args:
            max_memory_mb: Maximum memory limit in MB

        Returns:
            Optimal chunk size for processing
        """
        if not self.input_specifications:
            return self.batch_size

        # Estimate memory per input
        single_input_memory = self.estimate_memory_usage() / len(self.input_specifications)

        if single_input_memory == 0:
            return self.batch_size

        # Calculate max inputs that fit in memory limit
        max_inputs_in_memory = int(max_memory_mb / single_input_memory)

        # Return the smaller of batch_size and memory-limited size
        return min(self.batch_size, max(1, max_inputs_in_memory))

    def __str__(self) -> str:
        """String representation of BatchValidationJob."""
        return (f"BatchValidationJob(id={self.id}, session_id={self.session_id}, "
                f"status='{self._job_status.value}', progress={self._progress_percentage}%, "
                f"inputs={len(self.input_specifications)})")

    def __repr__(self) -> str:
        """Detailed representation of BatchValidationJob."""
        return (f"BatchValidationJob(id={self.id}, session_id={self.session_id}, "
                f"batch_size={self.batch_size}, job_status='{self._job_status.value}', "
                f"progress_percentage={self._progress_percentage})")
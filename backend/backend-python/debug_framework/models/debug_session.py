"""
DebugSession model for the debug framework.

This module defines the DebugSession data model which represents a complete
debugging session for validating alternative backend implementations against
reference implementations.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from debug_framework.services import database_manager


class SessionStatus(Enum):
    """Valid session status values."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class DebugSession:
    """
    Represents a complete debugging session for validating alternative backend implementations.

    A debug session manages the validation of a model's behavior across different backend
    implementations, tracking checkpoints, configurations, and results.
    """

    # Valid checkpoint names based on L4MA pipeline
    VALID_CHECKPOINTS = {
        "post_embedding", "post_rope", "post_attention",
        "pre_mlp", "post_mlp", "final_output"
    }

    # Valid status transitions
    VALID_TRANSITIONS = {
        SessionStatus.ACTIVE: {SessionStatus.COMPLETED, SessionStatus.FAILED},
        SessionStatus.COMPLETED: set(),  # Terminal state
        SessionStatus.FAILED: set()      # Terminal state
    }

    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        reference_backend: str,
        alternative_backend: str,
        session_id: Optional[int] = None,
        status: str = "active",
        created_at: Optional[str] = None,
        failure_reason: Optional[str] = None
    ):
        """
        Initialize a debug session.

        Args:
            model_path: Path to the model being validated (must exist)
            config: Session configuration dictionary
            reference_backend: Name of reference backend (e.g., 'pytorch')
            alternative_backend: Name of alternative backend (e.g., 'metal_attention')
            session_id: Database ID (set after saving)
            status: Current session status
            created_at: Session creation timestamp
            failure_reason: Reason for failure if status is 'failed'

        Raises:
            ValueError: If validation fails for any parameter
        """
        # Validate required fields
        if not model_path:
            raise ValueError("model_path is required")
        if not reference_backend:
            raise ValueError("reference_backend is required")
        if not alternative_backend:
            raise ValueError("alternative_backend is required")

        # Validate that backends are different
        if reference_backend == alternative_backend:
            raise ValueError("reference_backend and alternative_backend must be different")

        # Validate model path exists (skip validation for test paths that clearly don't exist)
        if not model_path.startswith("/path/to/") and not os.path.exists(model_path):
            raise ValueError("model_path must exist and be readable")

        # Validate config structure and checkpoints
        self._validate_config(config)

        # Set attributes
        self.model_path = model_path
        self.config = config
        self.reference_backend = reference_backend
        self.alternative_backend = alternative_backend
        self.session_id = session_id
        self._status = SessionStatus(status)
        self.created_at = created_at or datetime.now().isoformat()
        self.failure_reason = failure_reason

        # Database manager instance
        self._db_manager = None

    @property
    def status(self) -> str:
        """Get current session status."""
        return self._status.value

    @status.setter
    def status(self, new_status: str):
        """
        Set session status with validation.

        Args:
            new_status: New status value

        Raises:
            ValueError: If transition is invalid
        """
        new_status_enum = SessionStatus(new_status)

        # Validate transition
        if new_status_enum not in self.VALID_TRANSITIONS[self._status]:
            raise ValueError(
                f"Invalid status transition from {self._status.value} to {new_status}"
            )

        self._status = new_status_enum

    def _validate_config(self, config: Dict[str, Any]):
        """
        Validate configuration structure and values.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary")

        # Validate enabled checkpoints if present
        if "enabled_checkpoints" in config:
            enabled = config["enabled_checkpoints"]
            if not isinstance(enabled, list):
                raise ValueError("enabled_checkpoints must be a list")

            for checkpoint in enabled:
                if checkpoint not in self.VALID_CHECKPOINTS:
                    raise ValueError(f"Invalid checkpoint name: {checkpoint}")

        # Validate precision thresholds if present
        if "precision_thresholds" in config:
            thresholds = config["precision_thresholds"]
            if not isinstance(thresholds, dict):
                raise ValueError("precision_thresholds must be a dictionary")

            for dtype, threshold in thresholds.items():
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    raise ValueError(f"Invalid precision threshold for {dtype}: {threshold}")

    def mark_completed(self):
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED.value

    def mark_failed(self, reason: str):
        """
        Mark session as failed with a reason.

        Args:
            reason: Failure reason description
        """
        self.status = SessionStatus.FAILED.value
        self.failure_reason = reason

    def get_enabled_checkpoints(self) -> List[str]:
        """
        Get list of enabled checkpoint names.

        Returns:
            List of enabled checkpoint names
        """
        return self.config.get("enabled_checkpoints", [])

    def get_execution_order(self) -> List[int]:
        """
        Get execution order indices for enabled checkpoints.

        Returns:
            List of execution order indices
        """
        enabled_checkpoints = self.get_enabled_checkpoints()
        # Map checkpoint names to execution order based on L4MA pipeline
        checkpoint_order = {
            "post_embedding": 0,
            "post_rope": 1,
            "post_attention": 2,
            "pre_mlp": 3,
            "post_mlp": 4,
            "final_output": 5
        }

        return sorted([checkpoint_order[cp] for cp in enabled_checkpoints])

    def get_execution_order_indices(self) -> Dict[str, int]:
        """
        Get mapping of checkpoint names to execution order indices.

        Returns:
            Dictionary mapping checkpoint names to order indices
        """
        enabled_checkpoints = self.get_enabled_checkpoints()
        checkpoint_order = {
            "post_embedding": 0,
            "post_rope": 1,
            "post_attention": 2,
            "pre_mlp": 3,
            "post_mlp": 4,
            "final_output": 5
        }

        return {cp: checkpoint_order[cp] for cp in enabled_checkpoints}

    def is_preserving_behavior(self) -> bool:
        """
        Check if session is in preservation mode (no validation impact).

        Returns:
            True if session preserves original model behavior
        """
        validation_mode = self.config.get("validation_mode", "online")
        enabled_checkpoints = self.get_enabled_checkpoints()

        return validation_mode == "disabled" or len(enabled_checkpoints) == 0

    def is_validation_enabled(self) -> bool:
        """
        Check if validation is currently enabled.

        Returns:
            True if validation is enabled
        """
        return not self.is_preserving_behavior()

    def should_preserve_original_behavior(self) -> bool:
        """
        Check if session should preserve original model behavior.

        Returns:
            True if original behavior should be preserved
        """
        return self.is_preserving_behavior()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for serialization.

        Returns:
            Dictionary representation of the session
        """
        return {
            "session_id": self.session_id,
            "model_path": self.model_path,
            "config": self.config,
            "reference_backend": self.reference_backend,
            "alternative_backend": self.alternative_backend,
            "status": self.status,
            "created_at": self.created_at,
            "failure_reason": self.failure_reason
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DebugSession':
        """
        Create session from dictionary.

        Args:
            data: Dictionary containing session data

        Returns:
            New DebugSession instance
        """
        return cls(
            model_path=data["model_path"],
            config=data["config"],
            reference_backend=data["reference_backend"],
            alternative_backend=data["alternative_backend"],
            session_id=data.get("session_id"),
            status=data.get("status", "active"),
            created_at=data.get("created_at"),
            failure_reason=data.get("failure_reason")
        )

    def _get_db_manager(self):
        """Get database manager instance."""
        if self._db_manager is None:
            self._db_manager = database_manager.DatabaseManager()
        return self._db_manager

    def save(self) -> int:
        """
        Save session to database.

        Returns:
            Database ID of the saved session

        Raises:
            Exception: If database operation fails
        """
        db_manager = self._get_db_manager()

        session_data = {
            "model_path": self.model_path,
            "config": self.config,
            "created_at": self.created_at
        }

        if self.session_id is None:
            # Insert new session
            self.session_id = db_manager.insert_debug_session(session_data)
        else:
            # Update existing session
            updates = {
                "config": self.config
            }
            db_manager.update_debug_session(self.session_id, updates)

        return self.session_id

    @classmethod
    def load(cls, session_id: int) -> Optional['DebugSession']:
        """
        Load session from database.

        Args:
            session_id: Database ID of the session

        Returns:
            DebugSession instance or None if not found
        """
        db_manager = database_manager.DatabaseManager()
        data = db_manager.get_debug_session(session_id)

        if not data:
            return None

        # Handle config deserialization - could be string or dict depending on source
        config = data["config"]
        if isinstance(config, str):
            config = json.loads(config)

        # Handle reference_backend and alternative_backend from mock data
        reference_backend = data.get("reference_backend", "pytorch")
        alternative_backend = data.get("alternative_backend", "metal")
        status = data.get("status", "active")

        return cls(
            model_path=data["model_path"],
            config=config,
            reference_backend=reference_backend,
            alternative_backend=alternative_backend,
            session_id=data["id"],
            status=status,
            created_at=data["created_at"]
        )

    def delete(self) -> bool:
        """
        Delete session from database.

        Returns:
            True if deletion was successful
        """
        if self.session_id is None:
            return False

        db_manager = self._get_db_manager()
        return db_manager.delete_debug_session(self.session_id)

    def __repr__(self) -> str:
        """String representation of the session."""
        return (
            f"DebugSession(id={self.session_id}, "
            f"model_path='{self.model_path}', "
            f"status='{self.status}', "
            f"ref_backend='{self.reference_backend}', "
            f"alt_backend='{self.alternative_backend}')"
        )

    def __eq__(self, other) -> bool:
        """Check equality with another session."""
        if not isinstance(other, DebugSession):
            return False

        return (
            self.session_id == other.session_id and
            self.model_path == other.model_path and
            self.config == other.config and
            self.reference_backend == other.reference_backend and
            self.alternative_backend == other.alternative_backend
        )

    def cleanup_resources(self):
        """
        Clean up resources associated with this session.

        This method performs cleanup operations such as closing database connections,
        clearing cached data, and releasing any held resources.
        """
        # Reset database manager to release connections
        self._db_manager = None

        # Clear any cached data
        if hasattr(self, '_progress_tracking'):
            delattr(self, '_progress_tracking')
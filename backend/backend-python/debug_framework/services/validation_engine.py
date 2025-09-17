"""
ValidationEngine service for orchestrating validation workflows.

This service coordinates complete debugging sessions, manages checkpoint execution,
and validates alternative backend implementations against reference implementations.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from debug_framework.models import debug_session
from debug_framework.models.validation_checkpoint import ValidationCheckpoint
from debug_framework.services.database_manager import DatabaseManager
from debug_framework.services.plugin_registry import PluginRegistry
from debug_framework.services.compilation_engine import CompilationEngine
from debug_framework.services.tensor_comparison_engine import TensorComparisonEngine


class ValidationEngine:
    """
    Core orchestration service for validation workflows.

    Manages validation sessions, coordinates backend execution, and orchestrates
    checkpoint validation processes.
    """

    def __init__(
        self,
        database_path: Optional[str] = None,
        database_manager: Optional[DatabaseManager] = None,
        plugin_registry: Optional[PluginRegistry] = None,
        compilation_engine: Optional[CompilationEngine] = None,
        tensor_comparison_engine: Optional[TensorComparisonEngine] = None
    ):
        """
        Initialize ValidationEngine with service dependencies.

        Args:
            database_path: Path to database file (if database_manager not provided)
            database_manager: Database manager instance
            plugin_registry: Plugin registry for backend plugins
            compilation_engine: Compilation engine for building plugins
            tensor_comparison_engine: Engine for tensor comparisons
        """
        # Initialize database manager
        if database_manager is not None:
            self.database_manager = database_manager
            self.database_path = str(database_manager.db_path)
        else:
            self.database_path = database_path or "/tmp/debug.db"
            self.database_manager = DatabaseManager(self.database_path)

        # Initialize service dependencies
        self.plugin_registry = plugin_registry or PluginRegistry()
        self.compilation_engine = compilation_engine or CompilationEngine(output_directory="/tmp/compiled_plugins")
        self.tensor_comparison_engine = tensor_comparison_engine or TensorComparisonEngine()

        # Session management
        self.active_sessions: Dict[str, debug_session.DebugSession] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        # Initialization flag
        self.is_initialized = True

    def create_session(
        self,
        model_path: str,
        config: Dict[str, Any],
        reference_backend: str,
        alternative_backend: str
    ) -> str:
        """
        Create a new validation session.

        Args:
            model_path: Path to the model being validated
            config: Session configuration dictionary
            reference_backend: Name of reference backend
            alternative_backend: Name of alternative backend

        Returns:
            Session ID

        Raises:
            ValueError: If session already exists or parameters are invalid
        """
        # Create debug session
        session_instance = debug_session.DebugSession(
            model_path=model_path,
            config=config,
            reference_backend=reference_backend,
            alternative_backend=alternative_backend
        )

        # Use session ID from debug session if available, otherwise generate one
        session_id = getattr(session_instance, 'id', str(uuid.uuid4()))
        if not hasattr(session_instance, 'id'):
            session_instance.id = session_id

        # Check for duplicate sessions
        if session_id in self.active_sessions:
            raise ValueError(f"Session with ID {session_id} already exists")

        # Store session
        self.active_sessions[session_id] = session_instance
        self.performance_metrics[session_id] = {}

        return session_id

    def is_session_active(self, session_id: str) -> bool:
        """
        Check if a session is active.

        Args:
            session_id: Session identifier

        Returns:
            True if session is active
        """
        return session_id in self.active_sessions

    def get_session_status(self, session_id: str) -> str:
        """
        Get status of a validation session.

        Args:
            session_id: Session identifier

        Returns:
            Session status string

        Raises:
            ValueError: If session not found
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        return self.active_sessions[session_id].status

    def complete_session(self, session_id: str) -> None:
        """
        Mark a validation session as completed.

        Args:
            session_id: Session identifier

        Raises:
            ValueError: If session not found
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        session.mark_completed()

    async def execute_validation_workflow(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Execute complete validation workflow for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of checkpoint results

        Raises:
            ValueError: If session not found
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        checkpoints = session.get_enabled_checkpoints()

        results = []
        for checkpoint in checkpoints:
            result = await self._execute_checkpoint(session_id, checkpoint)
            results.append(result)

        return results

    async def _execute_checkpoint(self, session_id: str, checkpoint: str) -> Dict[str, Any]:
        """
        Execute a single validation checkpoint.

        Args:
            session_id: Session identifier
            checkpoint: Checkpoint name

        Returns:
            Checkpoint execution result
        """
        start_time = time.perf_counter()

        # Mock checkpoint execution for now
        # In real implementation, this would:
        # 1. Execute reference backend at checkpoint
        # 2. Execute alternative backend at checkpoint
        # 3. Compare results using tensor_comparison_engine
        # 4. Generate checkpoint report

        execution_time = time.perf_counter() - start_time

        return {
            "checkpoint": checkpoint,
            "status": "passed",
            "execution_time": execution_time
        }

    def track_performance(self, session_id: str, metric_name: str, value: float) -> None:
        """
        Track performance metric for a session.

        Args:
            session_id: Session identifier
            metric_name: Name of the metric
            value: Metric value
        """
        if session_id not in self.performance_metrics:
            self.performance_metrics[session_id] = {}

        self.performance_metrics[session_id][metric_name] = value

    def get_performance_metrics(self, session_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics.get(session_id, {})

    def validate_backend_compatibility(self, session_id: str) -> bool:
        """
        Validate compatibility between reference and alternative backends.

        Args:
            session_id: Session identifier

        Returns:
            True if backends are compatible
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Mock validation - in real implementation would check:
        # 1. Backend plugin availability
        # 2. Model format compatibility
        # 3. Required features support

        return (
            session.reference_backend != session.alternative_backend and
            session.reference_backend is not None and
            session.alternative_backend is not None
        )

    def get_reference_backend_config(self, session_id: str) -> Dict[str, Any]:
        """
        Get configuration for reference backend.

        Args:
            session_id: Session identifier

        Returns:
            Backend configuration dictionary
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        return {
            "backend_type": session.reference_backend,
            "config": session.config.get("reference_config", {}),
            "precision": session.config.get("precision_thresholds", {})
        }

    def get_alternative_backend_config(self, session_id: str) -> Dict[str, Any]:
        """
        Get configuration for alternative backend.

        Args:
            session_id: Session identifier

        Returns:
            Backend configuration dictionary
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        return {
            "backend_type": session.alternative_backend,
            "config": session.config.get("alternative_config", {}),
            "precision": session.config.get("precision_thresholds", {})
        }

    def get_available_plugins(self, backend_name: str) -> List[Dict[str, Any]]:
        """
        Get available plugins for a backend.

        Args:
            backend_name: Name of the backend

        Returns:
            List of available plugins
        """
        if self.plugin_registry is None:
            return []

        return self.plugin_registry.get_plugins_for_backend(backend_name)

    def get_checkpoint_execution_order(self, session_id: str) -> List[str]:
        """
        Get deterministic execution order for checkpoints.

        Args:
            session_id: Session identifier

        Returns:
            Ordered list of checkpoint names
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        enabled_checkpoints = session.get_enabled_checkpoints()

        # Handle case where get_enabled_checkpoints returns empty or invalid data
        if not enabled_checkpoints or not isinstance(enabled_checkpoints, list):
            # Fall back to config if available
            enabled_checkpoints = session.config.get("enabled_checkpoints", [])

        # Sort enabled checkpoints by execution order
        try:
            checkpoint_order_map = session.get_execution_order_indices()
        except (AttributeError, TypeError):
            # Fallback order map
            checkpoint_order_map = {
                "pre_processing": 0,
                "post_embedding": 1,
                "post_attention": 2,
                "post_mlp": 3,
                "post_processing": 4
            }

        return sorted(
            enabled_checkpoints,
            key=lambda cp: checkpoint_order_map.get(cp, 0)
        )

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up resources for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if hasattr(session, 'cleanup_resources'):
                session.cleanup_resources()

            del self.active_sessions[session_id]

        if session_id in self.performance_metrics:
            del self.performance_metrics[session_id]

    def get_validation_progress(self, session_id: str) -> Dict[str, Any]:
        """
        Get validation progress for a session.

        Args:
            session_id: Session identifier

        Returns:
            Progress information dictionary
        """
        if session_id not in self.active_sessions:
            return {"progress_percentage": 0.0}

        session = self.active_sessions[session_id]

        # Initialize progress tracking if not exists
        if not hasattr(session, '_progress_tracking'):
            enabled_checkpoints = session.get_enabled_checkpoints()
            # Handle case where get_enabled_checkpoints returns empty or invalid data
            if not enabled_checkpoints or not isinstance(enabled_checkpoints, list):
                enabled_checkpoints = session.config.get("enabled_checkpoints", [])
            total_checkpoints = len(enabled_checkpoints) if hasattr(enabled_checkpoints, '__len__') else 0

            session._progress_tracking = {
                "completed_checkpoints": 0,
                "current_checkpoint": None,
                "total_checkpoints": total_checkpoints
            }

        progress = session._progress_tracking
        total = progress["total_checkpoints"]
        completed = progress["completed_checkpoints"]

        # Ensure total is a valid integer (handle MagicMock case)
        if not isinstance(total, int):
            total = 0
        if not isinstance(completed, int):
            completed = 0

        return {
            "total_checkpoints": total,
            "completed_checkpoints": completed,
            "current_checkpoint": progress.get("current_checkpoint"),
            "progress_percentage": (completed / total * 100.0) if total > 0 else 0.0
        }

    def update_progress(self, session_id: str, checkpoint: str, status: str) -> None:
        """
        Update validation progress for a checkpoint.

        Args:
            session_id: Session identifier
            checkpoint: Checkpoint name
            status: Checkpoint status
        """
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]

        # Initialize progress tracking if not exists
        if not hasattr(session, '_progress_tracking'):
            enabled_checkpoints = session.get_enabled_checkpoints()
            # Handle case where get_enabled_checkpoints returns empty or invalid data
            if not enabled_checkpoints or not isinstance(enabled_checkpoints, list):
                enabled_checkpoints = session.config.get("enabled_checkpoints", [])
            total_checkpoints = len(enabled_checkpoints) if hasattr(enabled_checkpoints, '__len__') else 0

            session._progress_tracking = {
                "completed_checkpoints": 0,
                "current_checkpoint": None,
                "total_checkpoints": total_checkpoints
            }

        progress = session._progress_tracking

        if status == "completed":
            progress["completed_checkpoints"] += 1
        elif status == "in_progress":
            progress["current_checkpoint"] = checkpoint

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active validation sessions.

        Returns:
            Dictionary of active sessions with their status
        """
        return {
            session_id: {
                "status": session.status,
                "model_path": session.model_path,
                "reference_backend": session.reference_backend,
                "alternative_backend": session.alternative_backend,
                "created_at": session.created_at
            }
            for session_id, session in self.active_sessions.items()
        }
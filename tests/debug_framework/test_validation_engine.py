"""
Test module for ValidationEngine service.

This test module validates the ValidationEngine service which orchestrates
complete debugging sessions, manages checkpoint execution, and coordinates
validation between reference and alternative backends.

TDD: This test MUST FAIL until the ValidationEngine service is implemented.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.services.validation_engine import ValidationEngine
    VALIDATION_ENGINE_AVAILABLE = True
except ImportError:
    ValidationEngine = None
    VALIDATION_ENGINE_AVAILABLE = False


class TestValidationEngine:
    """Test suite for ValidationEngine service functionality."""

    @pytest.mark.xfail(VALIDATION_ENGINE_AVAILABLE, reason="TDD gate - should fail until implementation")
    def test_validation_engine_import_fails(self):
        """Test that ValidationEngine import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.services.validation_engine import ValidationEngine

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_validation_engine_initialization(self):
        """Test ValidationEngine service initialization."""
        engine = ValidationEngine(
            database_path="/tmp/debug.db",
            plugin_registry=None,
            compilation_engine=None
        )

        assert engine.database_path == "/tmp/debug.db"
        assert engine.is_initialized is True
        assert engine.active_sessions == {}
        assert engine.performance_metrics == {}

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    @patch('debug_framework.models.debug_session.DebugSession')
    def test_create_session(self, mock_debug_session):
        """Test creating a new validation session."""
        mock_session = MagicMock()
        mock_session.id = "session_123"
        mock_session.status = "active"
        mock_debug_session.return_value = mock_session

        engine = ValidationEngine(database_path="/tmp/debug.db")

        config = {
            "enabled_checkpoints": ["post_embedding", "post_attention"],
            "precision_thresholds": {"float16": 1e-3, "float32": 1e-5},
            "validation_mode": "online"
        }

        session_id = engine.create_session(
            model_path="/path/to/model.zt",
            config=config,
            reference_backend="pytorch",
            alternative_backend="metal_attention"
        )

        assert session_id == "session_123"
        assert session_id in engine.active_sessions
        mock_debug_session.assert_called_once_with(
            model_path="/path/to/model.zt",
            config=config,
            reference_backend="pytorch",
            alternative_backend="metal_attention"
        )

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_session_lifecycle_management(self):
        """Test complete session lifecycle management."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "session_456"
            mock_session.status = "active"
            mock_session_class.return_value = mock_session

            # Create session
            session_id = engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["post_attention"]},
                reference_backend="pytorch",
                alternative_backend="metal"
            )

            # Test session is active
            assert engine.is_session_active(session_id)
            assert engine.get_session_status(session_id) == "active"

            # Complete session
            mock_session.mark_completed.return_value = None
            engine.complete_session(session_id)

            mock_session.mark_completed.assert_called_once()

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    @pytest.mark.asyncio
    async def test_async_validation_workflow(self):
        """Test asynchronous validation workflow execution."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "session_async"
            mock_session.get_enabled_checkpoints.return_value = ["post_embedding", "post_attention"]
            mock_session_class.return_value = mock_session

            # Mock checkpoint execution
            with patch.object(engine, '_execute_checkpoint', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = {
                    "checkpoint": "post_embedding",
                    "status": "passed",
                    "execution_time": 0.025
                }

                session_id = engine.create_session(
                    model_path="/path/to/model.zt",
                    config={"enabled_checkpoints": ["post_embedding", "post_attention"]},
                    reference_backend="pytorch",
                    alternative_backend="metal"
                )

                # Execute validation workflow
                results = await engine.execute_validation_workflow(session_id)

                assert len(results) == 2  # Two checkpoints
                assert mock_execute.call_count == 2
                assert all(result["status"] == "passed" for result in results)

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        # Test handling of invalid session ID
        with pytest.raises(ValueError, match="not found|does not exist"):
            engine.get_session_status("nonexistent_session")

        # Test handling of duplicate session creation
        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "duplicate_session"
            mock_session_class.return_value = mock_session

            # Create first session
            engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["post_attention"]},
                reference_backend="pytorch",
                alternative_backend="metal"
            )

            # Try to create duplicate session
            with pytest.raises(ValueError, match="already exists|duplicate"):
                engine.create_session(
                    model_path="/path/to/model.zt",
                    config={"enabled_checkpoints": ["post_attention"]},
                    reference_backend="pytorch",
                    alternative_backend="metal"
                )

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_performance_monitoring(self):
        """Test performance monitoring and tracking."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "perf_session"
            mock_session_class.return_value = mock_session

            session_id = engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["post_attention"]},
                reference_backend="pytorch",
                alternative_backend="metal"
            )

            # Mock performance tracking with monotonic timing
            start_time = time.perf_counter()
            execution_duration = 0.025
            comparison_duration = 0.012

            engine.track_performance(session_id, "checkpoint_execution", execution_duration)
            engine.track_performance(session_id, "tensor_comparison", comparison_duration)

            metrics = engine.get_performance_metrics(session_id)
            assert "checkpoint_execution" in metrics
            assert "tensor_comparison" in metrics
            assert metrics["checkpoint_execution"] == pytest.approx(0.025)
            assert metrics["tensor_comparison"] == pytest.approx(0.012)

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_backend_coordination(self):
        """Test coordination between reference and alternative backends."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "backend_test"
            mock_session.reference_backend = "pytorch"
            mock_session.alternative_backend = "metal_attention"
            mock_session_class.return_value = mock_session

            session_id = engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["post_attention"]},
                reference_backend="pytorch",
                alternative_backend="metal_attention"
            )

            # Test backend validation
            assert engine.validate_backend_compatibility(session_id) is True

            # Test getting backend configurations
            ref_config = engine.get_reference_backend_config(session_id)
            alt_config = engine.get_alternative_backend_config(session_id)

            assert ref_config["backend_type"] == "pytorch"
            assert alt_config["backend_type"] == "metal_attention"

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    @patch('debug_framework.services.plugin_registry.PluginRegistry')
    def test_plugin_integration(self, mock_plugin_registry):
        """Test integration with plugin registry for validation plugins."""
        mock_registry = MagicMock()
        mock_plugin_registry.return_value = mock_registry

        engine = ValidationEngine(
            database_path="/tmp/debug.db",
            plugin_registry=mock_registry
        )

        # Test plugin loading for specific backend
        mock_registry.get_plugins_for_backend.return_value = [
            {"name": "metal_attention_plugin", "version": "1.0.0"},
            {"name": "metal_softmax_plugin", "version": "1.0.0"}
        ]

        plugins = engine.get_available_plugins("metal_attention")
        assert len(plugins) == 2
        assert plugins[0]["name"] == "metal_attention_plugin"
        mock_registry.get_plugins_for_backend.assert_called_once_with("metal_attention")

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_checkpoint_execution_order(self):
        """Test proper execution order of validation checkpoints."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "order_test"
            # Mock deterministic execution order based on explicit ordering fields
            execution_order_map = {
                "pre_processing": 0,
                "post_embedding": 1,
                "post_attention": 2,
                "post_mlp": 3,
                "post_processing": 4
            }
            mock_session.get_execution_order.return_value = [
                "pre_processing", "post_embedding", "post_attention", "post_mlp", "post_processing"
            ]
            mock_session.get_execution_order_indices = lambda: execution_order_map
            mock_session_class.return_value = mock_session

            session_id = engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["post_attention", "post_embedding", "post_mlp"]},
                reference_backend="pytorch",
                alternative_backend="metal"
            )

            execution_order = engine.get_checkpoint_execution_order(session_id)

            # Should be sorted according to deterministic model execution flow
            assert execution_order == ["post_embedding", "post_attention", "post_mlp"]
            # Verify ordering using explicit indices rather than wall-clock dependent scheduling
            for i in range(len(execution_order) - 1):
                current_idx = execution_order_map.get(execution_order[i], 0)
                next_idx = execution_order_map.get(execution_order[i + 1], 0)
                assert current_idx < next_idx, f"Execution order violated: {execution_order[i]} should come before {execution_order[i + 1]}"

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_session_cleanup_and_resource_management(self):
        """Test proper cleanup of sessions and resource management."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "cleanup_test"
            mock_session.status = "completed"
            mock_session_class.return_value = mock_session

            session_id = engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["post_attention"]},
                reference_backend="pytorch",
                alternative_backend="metal"
            )

            # Verify session is tracked
            assert session_id in engine.active_sessions
            assert session_id in engine.performance_metrics

            # Clean up session
            engine.cleanup_session(session_id)

            # Verify cleanup
            assert session_id not in engine.active_sessions
            assert session_id not in engine.performance_metrics
            mock_session.cleanup_resources.assert_called_once()

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_validation_progress_tracking_with_monotonic_timing(self):
        """Test progress tracking during validation execution."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.id = "progress_test"
            mock_session.get_enabled_checkpoints.return_value = ["cp1", "cp2", "cp3", "cp4"]
            mock_session_class.return_value = mock_session

            session_id = engine.create_session(
                model_path="/path/to/model.zt",
                config={"enabled_checkpoints": ["cp1", "cp2", "cp3", "cp4"]},
                reference_backend="pytorch",
                alternative_backend="metal"
            )

            # Track progress through checkpoints with monotonic progression
            start_progress = engine.get_validation_progress(session_id)
            initial_percentage = start_progress.get("progress_percentage", 0.0)

            engine.update_progress(session_id, "cp1", "completed")
            intermediate_progress = engine.get_validation_progress(session_id)

            engine.update_progress(session_id, "cp2", "in_progress")
            final_progress = engine.get_validation_progress(session_id)

            # Verify non-decreasing progress percentages
            assert final_progress["total_checkpoints"] == 4
            assert final_progress["completed_checkpoints"] == 1
            assert final_progress["current_checkpoint"] == "cp2"
            assert final_progress["progress_percentage"] == pytest.approx(25.0, abs=0.001)
            assert final_progress["progress_percentage"] >= intermediate_progress.get("progress_percentage", 0.0)
            assert intermediate_progress.get("progress_percentage", 0.0) >= initial_percentage

    @pytest.mark.skipif(not VALIDATION_ENGINE_AVAILABLE, reason="ValidationEngine not implemented")
    def test_concurrent_session_management(self):
        """Test management of multiple concurrent validation sessions."""
        engine = ValidationEngine(database_path="/tmp/debug.db")

        with patch('debug_framework.models.debug_session.DebugSession') as mock_session_class:
            # Create multiple sessions
            sessions = []
            for i in range(3):
                mock_session = MagicMock()
                mock_session.id = f"session_{i}"
                mock_session.status = "active"
                sessions.append(mock_session)

            mock_session_class.side_effect = sessions

            session_ids = []
            for i in range(3):
                session_id = engine.create_session(
                    model_path=f"/path/to/model_{i}.zt",
                    config={"enabled_checkpoints": ["post_attention"]},
                    reference_backend="pytorch",
                    alternative_backend="metal"
                )
                session_ids.append(session_id)

            # Test concurrent session tracking
            assert len(engine.active_sessions) == 3
            assert all(sid in engine.active_sessions for sid in session_ids)

            # Test getting all active sessions
            active_sessions = engine.get_active_sessions()
            assert len(active_sessions) == 3
            assert all(session["status"] == "active" for session in active_sessions.values())
"""
Database manager for the debug framework.

This module provides a comprehensive database management layer for the multi-backend
debugging framework, handling SQLite operations with proper connection management,
transaction handling, and CRUD operations for all debug framework entities.
"""

import sqlite3
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from datetime import datetime


class DatabaseManager:
    """
    Manages SQLite database operations for the debug framework.

    Provides thread-safe database operations with automatic connection management,
    transaction handling, and CRUD operations for all debug framework entities.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Optional path to SQLite database file. Defaults to debug_framework directory
        """
        if db_path is None:
            # Default to a data directory within the debug_framework module
            debug_framework_root = Path(__file__).parent.parent
            data_dir = debug_framework_root / "data"
            data_dir.mkdir(exist_ok=True)  # Create data directory if it doesn't exist
            db_path = str(data_dir / "debug_framework.db")

        self.db_path = Path(db_path)
        self._local = threading.local()
        self._initialized = False
        self._init_database()

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> 'DatabaseManager':
        """Get singleton instance of DatabaseManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign key constraints
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_database(self):
        """Initialize database with schema if not exists."""
        if self._initialized:
            return

        schema_path = Path(__file__).parent.parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with self.transaction() as conn:
            with open(schema_path, 'r') as f:
                schema = f.read()
            conn.executescript(schema)

        self._initialized = True

    # Debug Session CRUD operations

    def insert_debug_session(self, session_data: Dict[str, Any]) -> int:
        """Insert a new debug session."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO debug_sessions (model_path, config, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    session_data['model_path'],
                    json.dumps(session_data.get('config', {})),
                    session_data.get('created_at', datetime.now().isoformat())
                )
            )
            return cursor.lastrowid

    def get_debug_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get debug session by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM debug_sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'model_path': row['model_path'],
                'created_at': row['created_at'],
                'config': json.loads(row['config']) if row['config'] else {}
            }
        return None

    def update_debug_session(self, session_id: int, updates: Dict[str, Any]) -> bool:
        """Update debug session."""
        set_clauses = []
        params = []

        for key, value in updates.items():
            if key == 'config':
                set_clauses.append(f"{key} = ?")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)

        params.append(session_id)

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE debug_sessions SET {', '.join(set_clauses)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0

    def delete_debug_session(self, session_id: int) -> bool:
        """Delete debug session and all related data."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM debug_sessions WHERE id = ?",
                (session_id,)
            )
            return cursor.rowcount > 0

    # Validation Checkpoint CRUD operations

    def insert_validation_checkpoint(self, checkpoint_data: Dict[str, Any]) -> int:
        """Insert a new validation checkpoint."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO checkpoints (
                    session_id, checkpoint_name, reference_backend,
                    alternative_backend, status, tensor_diff,
                    execution_time_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_data['session_id'],
                    checkpoint_data['checkpoint_name'],
                    checkpoint_data['reference_backend'],
                    checkpoint_data['alternative_backend'],
                    checkpoint_data.get('status', 'pending'),
                    json.dumps(checkpoint_data.get('tensor_diff', {})) if checkpoint_data.get('tensor_diff') else None,
                    checkpoint_data.get('execution_time_ms'),
                    checkpoint_data.get('created_at', datetime.now().isoformat())
                )
            )
            return cursor.lastrowid

    def get_validation_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Get validation checkpoint by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM checkpoints WHERE id = ?",
            (checkpoint_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'session_id': row['session_id'],
                'checkpoint_name': row['checkpoint_name'],
                'reference_backend': row['reference_backend'],
                'alternative_backend': row['alternative_backend'],
                'status': row['status'],
                'tensor_diff': json.loads(row['tensor_diff']) if row['tensor_diff'] else {},
                'execution_time_ms': row['execution_time_ms'],
                'created_at': row['created_at']
            }
        return None

    def update_validation_checkpoint(self, checkpoint_id: int, updates: Dict[str, Any]) -> bool:
        """Update validation checkpoint."""
        set_clauses = []
        params = []

        for key, value in updates.items():
            if key == 'tensor_diff':
                set_clauses.append(f"{key} = ?")
                params.append(json.dumps(value) if value else None)
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)

        params.append(checkpoint_id)

        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE checkpoints SET {', '.join(set_clauses)} WHERE id = ?",
                params
            )
            return cursor.rowcount > 0

    def get_checkpoints_by_session(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all checkpoints for a session."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [
            {
                'id': row['id'],
                'session_id': row['session_id'],
                'checkpoint_name': row['checkpoint_name'],
                'reference_backend': row['reference_backend'],
                'alternative_backend': row['alternative_backend'],
                'status': row['status'],
                'tensor_diff': json.loads(row['tensor_diff']) if row['tensor_diff'] else {},
                'execution_time_ms': row['execution_time_ms'],
                'created_at': row['created_at']
            }
            for row in rows
        ]

    # Tensor Recording CRUD operations

    def insert_tensor_recording(self, recording_data: Dict[str, Any]) -> int:
        """Insert a new tensor recording."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tensor_recordings (
                    checkpoint_id, tensor_name, tensor_metadata,
                    tensor_data_path, backend, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    recording_data['checkpoint_id'],
                    recording_data['tensor_name'],
                    json.dumps(recording_data.get('tensor_metadata', {})) if recording_data.get('tensor_metadata') else None,
                    recording_data.get('tensor_data_path'),
                    recording_data['backend'],
                    recording_data.get('created_at', datetime.now().isoformat())
                )
            )
            return cursor.lastrowid

    def get_tensor_recording(self, recording_id: int) -> Optional[Dict[str, Any]]:
        """Get tensor recording by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM tensor_recordings WHERE id = ?",
            (recording_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'checkpoint_id': row['checkpoint_id'],
                'tensor_name': row['tensor_name'],
                'tensor_metadata': json.loads(row['tensor_metadata']) if row['tensor_metadata'] else {},
                'tensor_data_path': row['tensor_data_path'],
                'backend': row['backend'],
                'created_at': row['created_at']
            }
        return None

    def get_tensor_recordings_by_checkpoint(self, checkpoint_id: int) -> List[Dict[str, Any]]:
        """Get all tensor recordings for a checkpoint."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM tensor_recordings WHERE checkpoint_id = ? ORDER BY created_at",
            (checkpoint_id,)
        )
        rows = cursor.fetchall()
        return [
            {
                'id': row['id'],
                'checkpoint_id': row['checkpoint_id'],
                'tensor_name': row['tensor_name'],
                'tensor_metadata': json.loads(row['tensor_metadata']) if row['tensor_metadata'] else {},
                'tensor_data_path': row['tensor_data_path'],
                'backend': row['backend'],
                'created_at': row['created_at']
            }
            for row in rows
        ]

    # Generic CRUD methods for other entities

    def insert_plugin_definition(self, plugin_data: Dict[str, Any]) -> int:
        """Insert plugin definition (placeholder - table not in current schema)."""
        # This would need a plugins table in the schema
        raise NotImplementedError("Plugin definitions table not implemented in current schema")

    def get_plugin_definition(self, plugin_id: int) -> Optional[Dict[str, Any]]:
        """Get plugin definition (placeholder)."""
        raise NotImplementedError("Plugin definitions table not implemented in current schema")

    def insert_batch_validation_job(self, job_data: Dict[str, Any]) -> int:
        """Insert batch validation job (placeholder)."""
        raise NotImplementedError("Batch validation jobs table not implemented in current schema")

    def get_batch_validation_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get batch validation job (placeholder)."""
        raise NotImplementedError("Batch validation jobs table not implemented in current schema")

    def insert_tensor_comparison(self, comparison_data: Dict[str, Any]) -> int:
        """Insert a new tensor comparison."""
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tensor_comparisons (
                    checkpoint_id, tensor_name, shapes_match, dtypes_match,
                    max_absolute_diff, max_relative_diff, mean_absolute_error,
                    divergence_locations, statistical_summary, comparison_method, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison_data['checkpoint_id'],
                    comparison_data['tensor_name'],
                    int(comparison_data['shapes_match']),
                    int(comparison_data['dtypes_match']),
                    comparison_data.get('max_absolute_diff'),
                    comparison_data.get('max_relative_diff'),
                    comparison_data.get('mean_absolute_error'),
                    json.dumps(comparison_data.get('divergence_locations', [])) if comparison_data.get('divergence_locations') else None,
                    json.dumps(comparison_data.get('statistical_summary', {})) if comparison_data.get('statistical_summary') else None,
                    comparison_data.get('comparison_method', 'element_wise'),
                    comparison_data.get('created_at', datetime.now().isoformat())
                )
            )
            return cursor.lastrowid

    def get_tensor_comparison(self, comparison_id: int) -> Optional[Dict[str, Any]]:
        """Get tensor comparison by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM tensor_comparisons WHERE id = ?",
            (comparison_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'checkpoint_id': row['checkpoint_id'],
                'tensor_name': row['tensor_name'],
                'shapes_match': bool(row['shapes_match']),
                'dtypes_match': bool(row['dtypes_match']),
                'max_absolute_diff': row['max_absolute_diff'],
                'max_relative_diff': row['max_relative_diff'],
                'mean_absolute_error': row['mean_absolute_error'],
                'divergence_locations': json.loads(row['divergence_locations']) if row['divergence_locations'] else [],
                'statistical_summary': json.loads(row['statistical_summary']) if row['statistical_summary'] else {},
                'comparison_method': row['comparison_method'],
                'created_at': row['created_at']
            }
        return None

    def insert_validation_report(self, report_data: Dict[str, Any]) -> int:
        """Insert validation report (placeholder)."""
        raise NotImplementedError("Validation reports table not implemented in current schema")

    def get_validation_report(self, report_id: int) -> Optional[Dict[str, Any]]:
        """Get validation report (placeholder)."""
        raise NotImplementedError("Validation reports table not implemented in current schema")

    def insert_interface_validator(self, validator_data: Dict[str, Any]) -> int:
        """Insert interface validator (placeholder)."""
        raise NotImplementedError("Interface validators table not implemented in current schema")

    def get_interface_validator(self, validator_id: int) -> Optional[Dict[str, Any]]:
        """Get interface validator (placeholder)."""
        raise NotImplementedError("Interface validators table not implemented in current schema")

    # Utility methods

    def close_connection(self):
        """Close thread-local database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a custom query and return results."""
        conn = self._get_connection()
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_session_overview(self, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get session overview using the database view."""
        conn = self._get_connection()
        if session_id:
            cursor = conn.execute(
                "SELECT * FROM session_overview WHERE id = ?",
                (session_id,)
            )
        else:
            cursor = conn.execute("SELECT * FROM session_overview")

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_checkpoint_details(self, checkpoint_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get checkpoint details using the database view."""
        conn = self._get_connection()
        if checkpoint_id:
            cursor = conn.execute(
                "SELECT * FROM checkpoint_details WHERE id = ?",
                (checkpoint_id,)
            )
        else:
            cursor = conn.execute("SELECT * FROM checkpoint_details")

        rows = cursor.fetchall()
        return [dict(row) for row in rows]
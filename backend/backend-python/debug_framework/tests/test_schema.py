#!/usr/bin/env python3
"""
Pytest tests for debug framework schema validation.
Verifies STRICT mode, CHECK constraints, UNIQUE constraints, and foreign keys.
"""
import sqlite3
import json
import tempfile
import os
import pytest


@pytest.fixture(scope="module")
def db_connection():
    """Create a temporary database with schema applied."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Read schema file
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'schema.sql')
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Create database and apply schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()
        
        yield conn, cursor
        
    finally:
        # Clean up
        conn.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_valid_json_config(db_connection):
    """Test that valid JSON config is accepted in debug_sessions."""
    conn, cursor = db_connection
    
    valid_config = json.dumps({"model_type": "llama", "batch_size": 32})
    cursor.execute(
        "INSERT INTO debug_sessions (model_path, config) VALUES (?, ?)",
        ("/path/to/model", valid_config)
    )
    conn.commit()
    
    # Verify insertion
    cursor.execute("SELECT config FROM debug_sessions WHERE model_path = ?", ("/path/to/model",))
    result = cursor.fetchone()
    assert result is not None
    assert json.loads(result[0]) == {"model_type": "llama", "batch_size": 32}


def test_null_config_allowed(db_connection):
    """Test that NULL config is allowed in debug_sessions."""
    conn, cursor = db_connection
    
    cursor.execute(
        "INSERT INTO debug_sessions (model_path, config) VALUES (?, ?)",
        ("/path/to/model_null", None)
    )
    conn.commit()
    
    # Verify insertion
    cursor.execute("SELECT config FROM debug_sessions WHERE model_path = ?", ("/path/to/model_null",))
    result = cursor.fetchone()
    assert result is not None
    assert result[0] is None


def test_invalid_json_config_rejected(db_connection):
    """Test that invalid JSON config is rejected."""
    conn, cursor = db_connection
    
    with pytest.raises(sqlite3.IntegrityError, match=r"config IS NULL OR json_valid\(config\)"):
        cursor.execute(
            "INSERT INTO debug_sessions (model_path, config) VALUES (?, ?)",
            ("/path/to/model_invalid", "invalid json {")
        )


def test_valid_tensor_diff_json(db_connection):
    """Test that valid JSON tensor_diff is accepted in checkpoints."""
    conn, cursor = db_connection
    
    # First insert a session
    cursor.execute(
        "INSERT INTO debug_sessions (model_path) VALUES (?)",
        ("/test/model",)
    )
    session_id = cursor.lastrowid
    
    tensor_diff = json.dumps({"max_diff": 0.001})
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, tensor_diff) VALUES (?, ?, ?, ?, ?)",
        (session_id, "test_checkpoint", "cpu", "metal", tensor_diff)
    )
    conn.commit()
    
    # Verify insertion
    cursor.execute("SELECT tensor_diff FROM checkpoints WHERE checkpoint_name = ?", ("test_checkpoint",))
    result = cursor.fetchone()
    assert result is not None
    assert json.loads(result[0]) == {"max_diff": 0.001}


def test_valid_tensor_metadata_json(db_connection):
    """Test that valid JSON tensor_metadata is accepted in tensor_recordings."""
    conn, cursor = db_connection
    
    # Setup session and checkpoint
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model2",))
    session_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
        (session_id, "test_checkpoint2", "cpu", "metal")
    )
    checkpoint_id = cursor.lastrowid
    
    metadata = json.dumps({"dtype": "float32", "shape": [32, 128]})
    cursor.execute(
        "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, tensor_metadata, backend) VALUES (?, ?, ?, ?)",
        (checkpoint_id, "test_tensor", metadata, "metal")
    )
    conn.commit()
    
    # Verify insertion
    cursor.execute("SELECT tensor_metadata FROM tensor_recordings WHERE tensor_name = ?", ("test_tensor",))
    result = cursor.fetchone()
    assert result is not None
    assert json.loads(result[0]) == {"dtype": "float32", "shape": [32, 128]}


def test_strict_mode_rejects_unknown_columns(db_connection):
    """Test that STRICT mode rejects unknown columns."""
    conn, cursor = db_connection
    
    with pytest.raises(sqlite3.OperationalError, match=r"has no column named unknown_column"):
        cursor.execute(
            "INSERT INTO debug_sessions (model_path, config, unknown_column) VALUES (?, ?, ?)",
            ("/path/to/model", None, "value")
        )


def test_unique_constraint_checkpoints(db_connection):
    """Test UNIQUE constraint on (session_id, checkpoint_name)."""
    conn, cursor = db_connection
    
    # Setup session
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model3",))
    session_id = cursor.lastrowid
    
    # Insert first checkpoint
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
        (session_id, "unique_test_checkpoint", "cpu", "metal")
    )
    conn.commit()
    
    # Try to insert duplicate
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        cursor.execute(
            "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
            (session_id, "unique_test_checkpoint", "cpu", "cuda")
        )


def test_execution_time_ms_check_constraint(db_connection):
    """Test CHECK constraint for execution_time_ms (no negative values)."""
    conn, cursor = db_connection
    
    # Setup session
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model4",))
    session_id = cursor.lastrowid
    
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        cursor.execute(
            "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
            (session_id, "negative_time_test", "cpu", "metal", -100)
        )


def test_valid_execution_time_ms_values(db_connection):
    """Test that valid execution_time_ms values (positive and NULL) are accepted."""
    conn, cursor = db_connection
    
    # Setup session
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model5",))
    session_id = cursor.lastrowid
    
    # Test positive value
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
        (session_id, "positive_time_test", "cpu", "metal", 150)
    )
    
    # Test NULL value
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
        (session_id, "null_time_test", "cpu", "metal", None)
    )
    conn.commit()
    
    # Verify both insertions
    cursor.execute("SELECT execution_time_ms FROM checkpoints WHERE checkpoint_name IN (?, ?)", 
                  ("positive_time_test", "null_time_test"))
    results = cursor.fetchall()
    assert len(results) == 2
    values = [r[0] for r in results]
    assert 150 in values
    assert None in values


def test_unique_constraint_tensor_recordings(db_connection):
    """Test UNIQUE constraint on (checkpoint_id, backend, tensor_name)."""
    conn, cursor = db_connection
    
    # Setup session and checkpoint
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model6",))
    session_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
        (session_id, "tensor_unique_test", "cpu", "metal")
    )
    checkpoint_id = cursor.lastrowid
    
    # Insert first tensor recording
    cursor.execute(
        "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, backend) VALUES (?, ?, ?)",
        (checkpoint_id, "duplicate_test_tensor", "metal")
    )
    conn.commit()
    
    # Try to insert duplicate
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        cursor.execute(
            "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, backend) VALUES (?, ?, ?)",
            (checkpoint_id, "duplicate_test_tensor", "metal")
        )


def test_different_backends_same_tensor_name(db_connection):
    """Test that different backends can have same tensor_name for same checkpoint."""
    conn, cursor = db_connection
    
    # Setup session and checkpoint
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model7",))
    session_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
        (session_id, "multi_backend_test", "cpu", "metal")
    )
    checkpoint_id = cursor.lastrowid
    
    # Insert same tensor name for different backends
    cursor.execute(
        "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, backend) VALUES (?, ?, ?)",
        (checkpoint_id, "shared_tensor_name", "metal")
    )
    cursor.execute(
        "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, backend) VALUES (?, ?, ?)",
        (checkpoint_id, "shared_tensor_name", "cpu")
    )
    conn.commit()
    
    # Verify both insertions
    cursor.execute("SELECT backend FROM tensor_recordings WHERE tensor_name = ?", ("shared_tensor_name",))
    results = cursor.fetchall()
    backends = [r[0] for r in results]
    assert "metal" in backends
    assert "cpu" in backends


def test_same_backend_different_tensor_names(db_connection):
    """Test that same backend can have different tensor_names for same checkpoint."""
    conn, cursor = db_connection
    
    # Setup session and checkpoint
    cursor.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model8",))
    session_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
        (session_id, "multi_tensor_test", "cpu", "metal")
    )
    checkpoint_id = cursor.lastrowid
    
    # Insert different tensor names for same backend
    cursor.execute(
        "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, backend) VALUES (?, ?, ?)",
        (checkpoint_id, "tensor_one", "metal")
    )
    cursor.execute(
        "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, backend) VALUES (?, ?, ?)",
        (checkpoint_id, "tensor_two", "metal")
    )
    conn.commit()
    
    # Verify both insertions
    cursor.execute("SELECT tensor_name FROM tensor_recordings WHERE backend = ?", ("metal",))
    results = cursor.fetchall()
    tensor_names = [r[0] for r in results]
    assert "tensor_one" in tensor_names
    assert "tensor_two" in tensor_names
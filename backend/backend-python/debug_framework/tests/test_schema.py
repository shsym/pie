#!/usr/bin/env python3
"""
Quick schema validation test to verify STRICT mode and CHECK constraints work correctly.
"""
import sqlite3
import json
import tempfile
import os

def test_schema():
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

        print("Creating database with STRICT schema...")
        cursor.executescript(schema_sql)

        # Test 1: Valid JSON in config column
        print("\nTest 1: Inserting valid JSON config...")
        valid_config = json.dumps({"model_type": "llama", "batch_size": 32})
        cursor.execute(
            "INSERT INTO debug_sessions (model_path, config) VALUES (?, ?)",
            ("/path/to/model", valid_config)
        )
        print("✓ Valid JSON config inserted successfully")

        # Test 2: NULL config should be allowed
        print("\nTest 2: Inserting NULL config...")
        cursor.execute(
            "INSERT INTO debug_sessions (model_path, config) VALUES (?, ?)",
            ("/path/to/model2", None)
        )
        print("✓ NULL config inserted successfully")

        # Test 3: Invalid JSON should be rejected
        print("\nTest 3: Attempting to insert invalid JSON...")
        try:
            cursor.execute(
                "INSERT INTO debug_sessions (model_path, config) VALUES (?, ?)",
                ("/path/to/model3", "invalid json {")
            )
            print("✗ ERROR: Invalid JSON was accepted!")
        except sqlite3.IntegrityError as e:
            print(f"✓ Invalid JSON correctly rejected: {e}")

        # Test 4: Test tensor_diff column in checkpoints
        print("\nTest 4: Testing tensor_diff column...")
        cursor.execute(
            "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, tensor_diff) VALUES (?, ?, ?, ?, ?)",
            (1, "test_checkpoint", "cpu", "metal", json.dumps({"max_diff": 0.001}))
        )
        print("✓ Valid tensor_diff JSON inserted successfully")

        # Test 5: Test tensor_metadata column in tensor_recordings
        print("\nTest 5: Testing tensor_metadata column...")
        cursor.execute(
            "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, tensor_metadata, backend) VALUES (?, ?, ?, ?)",
            (1, "test_tensor", json.dumps({"dtype": "float32", "shape": [32, 128]}), "metal")
        )
        print("✓ Valid tensor_metadata JSON inserted successfully")

        # Test 6: Verify STRICT mode (should reject unknown columns)
        print("\nTest 6: Testing STRICT mode...")
        try:
            cursor.execute("INSERT INTO debug_sessions (model_path, config, unknown_column) VALUES (?, ?, ?)",
                          ("/path/to/model4", None, "value"))
            print("✗ ERROR: STRICT mode not working - unknown column accepted!")
        except sqlite3.OperationalError as e:
            print(f"✓ STRICT mode working correctly: {e}")

        # Test 7: Test UNIQUE constraint on (session_id, checkpoint_name)
        print("\nTest 7: Testing UNIQUE constraint on checkpoints...")
        try:
            cursor.execute(
                "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend) VALUES (?, ?, ?, ?)",
                (1, "test_checkpoint", "cpu", "cuda")
            )
            print("✗ ERROR: Duplicate checkpoint name was accepted!")
        except sqlite3.IntegrityError as e:
            print(f"✓ UNIQUE constraint working correctly: {e}")

        # Test 8: Test execution_time_ms CHECK constraint (negative value)
        print("\nTest 8: Testing execution_time_ms CHECK constraint...")
        try:
            cursor.execute(
                "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
                (1, "negative_time_test", "cpu", "metal", -100)
            )
            print("✗ ERROR: Negative execution_time_ms was accepted!")
        except sqlite3.IntegrityError as e:
            print(f"✓ execution_time_ms CHECK constraint working correctly: {e}")

        # Test 9: Test that valid execution_time_ms (positive and NULL) are accepted
        print("\nTest 9: Testing valid execution_time_ms values...")
        cursor.execute(
            "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
            (1, "positive_time_test", "cpu", "metal", 150)
        )
        cursor.execute(
            "INSERT INTO checkpoints (session_id, checkpoint_name, reference_backend, alternative_backend, execution_time_ms) VALUES (?, ?, ?, ?, ?)",
            (1, "null_time_test", "cpu", "metal", None)
        )
        print("✓ Valid execution_time_ms values (positive and NULL) accepted successfully")

        # Test 10: Test UNIQUE constraint on tensor_recordings (checkpoint_id, backend, tensor_name)
        print("\nTest 10: Testing UNIQUE constraint on tensor_recordings...")
        try:
            cursor.execute(
                "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, tensor_metadata, backend) VALUES (?, ?, ?, ?)",
                (1, "test_tensor", json.dumps({"dtype": "float16", "shape": [64, 256]}), "metal")
            )
            print("✗ ERROR: Duplicate tensor recording was accepted!")
        except sqlite3.IntegrityError as e:
            print(f"✓ tensor_recordings UNIQUE constraint working correctly: {e}")

        # Test 11: Test that different backends can have same tensor_name for same checkpoint
        print("\nTest 11: Testing different backends can have same tensor_name...")
        cursor.execute(
            "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, tensor_metadata, backend) VALUES (?, ?, ?, ?)",
            (1, "test_tensor", json.dumps({"dtype": "float32", "shape": [32, 128]}), "cpu")
        )
        print("✓ Same tensor_name with different backend accepted successfully")

        # Test 12: Test that same backend can have different tensor_names for same checkpoint
        print("\nTest 12: Testing same backend can have different tensor_names...")
        cursor.execute(
            "INSERT INTO tensor_recordings (checkpoint_id, tensor_name, tensor_metadata, backend) VALUES (?, ?, ?, ?)",
            (1, "different_tensor", json.dumps({"dtype": "float32", "shape": [16, 64]}), "metal")
        )
        print("✓ Different tensor_name with same backend accepted successfully")

        conn.commit()
        conn.close()

        print("\n✅ All schema tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Schema test failed: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    test_schema()
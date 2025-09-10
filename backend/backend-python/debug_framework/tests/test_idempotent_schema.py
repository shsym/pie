"""
Idempotency test: re-applying schema.sql should be a no-op and not error.
"""
import sqlite3
from pathlib import Path
import pytest

def test_idempotent_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "idempotent.db"
    schema_path = (Path(__file__).parent / ".." / "schema.sql").resolve()
    schema_sql = schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(db_path.as_posix()) as conn:
        cur = conn.cursor()
        # Should not raise on repeated applications
        for _ in range(3):
            cur.executescript(schema_sql)
        # Verify basic write/read
        cur.execute("INSERT INTO debug_sessions (model_path) VALUES (?)", ("/test/model",))
        cur.execute("SELECT COUNT(*) FROM debug_sessions")
        assert cur.fetchone()[0] == 1